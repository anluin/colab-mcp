"""Stable stdio supervisor for hot-swapping the Colab MCP worker."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .logging_config import configure_logging

logger = logging.getLogger("colab_mcp.supervisor")

CONNECTOR_TOOL = "colab_connector"
CONNECTOR_SCHEMA: dict[str, Any] = {
    "name": CONNECTOR_TOOL,
    "description": (
        "Inspect or hot-reload the local colab-mcp worker without restarting the MCP client. "
        "Use status after patching to compare active and available source fingerprints, then use "
        "reload. Reload drains in-flight MCP requests, initializes and health-checks a replacement "
        "worker before switching, preserves Colab assignments/durable processes, and rolls back "
        "to the current worker if validation fails. It reloads MCP implementation code only; "
        "plugin skills, manifests, and this stable supervisor still require a client refresh."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "reload"],
                "description": (
                    "status reports the active worker and source fingerprints without changing "
                    "state; reload performs a validated worker replacement."
                ),
            },
            "expected_source_fingerprint": {
                "type": ["string", "null"],
                "description": (
                    "Optional SHA-256 from a prior status.available_source_fingerprint. Reload "
                    "fails before spawning if the source changed again; null accepts current source."
                ),
            },
            "source_root": {
                "type": ["string", "null"],
                "description": (
                    "Optional absolute checkout of https://github.com/anluin/colab-mcp to bind "
                    "during reload. The supervisor verifies its local Git origin and required "
                    "layout; null keeps the fixed current root. Use only after cloning with gh."
                ),
            },
            "drain_timeout_seconds": {
                "type": "number",
                "minimum": 1,
                "maximum": 3600,
                "default": 300,
                "description": (
                    "Seconds to wait for already submitted MCP calls before reloading; defaults "
                    "to 300. Calls are never cancelled merely to reload."
                ),
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
    "annotations": {
        "title": "Inspect or reload Colab MCP",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}


def source_fingerprint(root: Path) -> str:
    """Hash worker source deterministically without including runtime or VCS state."""
    digest = hashlib.sha256()
    # The supervisor intentionally stays stable for the client connection. Its
    # own source cannot take effect through a worker swap and is excluded so a
    # status result never claims otherwise.
    sources = sorted(path for path in (root / "src").glob("*.py") if path.name != "supervisor.py")
    if not sources:
        raise RuntimeError(f"hot_reload_source_missing: no Python sources under {root / 'src'}")
    for path in sources:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class Worker:
    def __init__(self, process: asyncio.subprocess.Process, fingerprint: str) -> None:
        self.process = process
        self.fingerprint = fingerprint

    @property
    def pid(self) -> int | None:
        return self.process.pid


class ColabMcpSupervisor:
    """Own Codex's stdio connection while replaceable workers own Colab behavior."""

    def __init__(self, source_root: Path | None = None) -> None:
        configured = source_root or Path(
            os.environ.get("COLAB_MCP_HOT_RELOAD_ROOT", Path(__file__).resolve().parents[1])
        )
        self.source_root = configured.expanduser().resolve()
        if not (self.source_root / "src" / "cli.py").is_file():
            raise RuntimeError(
                "hot_reload_source_invalid: expected src/cli.py under " + str(self.source_root)
            )
        self.worker: Worker | None = None
        self._worker_reader: asyncio.Task[None] | None = None
        self._client_write_lock = asyncio.Lock()
        self._worker_write_lock = asyncio.Lock()
        self._pending: set[str] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._list_requests: set[str] = set()
        self._initialize_request: dict[str, Any] | None = None
        self._initialized_notification: dict[str, Any] | None = None
        self.generation = 0

    @staticmethod
    def _request_key(request_id: Any) -> str:
        return json.dumps(request_id, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _validated_source_root(value: str) -> Path:
        candidate = Path(value).expanduser().resolve()
        if not candidate.is_absolute() or not (candidate / "src" / "cli.py").is_file():
            raise RuntimeError(
                "hot_reload_source_invalid: expected an absolute root with src/cli.py"
            )
        try:
            completed = subprocess.run(
                ["git", "-C", str(candidate), "remote", "get-url", "origin"],
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                "hot_reload_source_origin_unverifiable: Git is required for source binding"
            ) from error
        origin = completed.stdout.strip().rstrip("/")
        if origin.endswith(".git"):
            origin = origin[:-4]
        allowed = {
            "https://github.com/anluin/colab-mcp",
            "git@github.com:anluin/colab-mcp",
        }
        if completed.returncode != 0 or origin.lower() not in allowed:
            raise RuntimeError(
                "hot_reload_source_origin_rejected: source_root must be the anluin/colab-mcp checkout"
            )
        return candidate

    async def _spawn_worker(self, source_root: Path | None = None) -> Worker:
        root = source_root or self.source_root
        fingerprint = source_fingerprint(root)
        environment = dict(os.environ)
        environment["COLAB_MCP_SUPERVISED"] = "1"
        environment["COLAB_MCP_WORKER_SOURCE_FINGERPRINT"] = fingerprint
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "src.cli",
            "serve-worker",
            cwd=str(root),
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
        )
        if process.stdin is None or process.stdout is None:
            process.kill()
            raise RuntimeError("hot_reload_worker_pipe_failed")
        return Worker(process, fingerprint)

    async def _write_worker(self, worker: Worker, message: dict[str, Any]) -> None:
        stream = worker.process.stdin
        if stream is None or stream.is_closing():
            raise RuntimeError("hot_reload_worker_transport_closed")
        encoded = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        async with self._worker_write_lock:
            stream.write(encoded)
            await stream.drain()

    async def _write_client(self, message: dict[str, Any]) -> None:
        encoded = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        async with self._client_write_lock:
            await asyncio.to_thread(self._write_client_sync, encoded)

    @staticmethod
    def _write_client_sync(encoded: bytes) -> None:
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()

    @staticmethod
    async def _read_message(stream: asyncio.StreamReader, timeout: float) -> dict[str, Any]:
        while True:
            line = await asyncio.wait_for(stream.readline(), timeout=timeout)
            if not line:
                raise RuntimeError("hot_reload_candidate_exited_during_validation")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("worker_stdout_non_json_ignored bytes=%d", len(line))
                continue
            if isinstance(message, dict):
                return message

    async def _candidate_request(
        self, worker: Worker, method: str, params: dict[str, Any], timeout: float = 30
    ) -> dict[str, Any]:
        request_id = f"__colab_supervisor_{uuid.uuid4().hex}"
        await self._write_worker(
            worker,
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        )
        if worker.process.stdout is None:
            raise RuntimeError("candidate worker stdout is unavailable")
        while True:
            response = await self._read_message(worker.process.stdout, timeout)
            if response.get("id") == request_id:
                if "error" in response:
                    raise RuntimeError(f"hot_reload_candidate_{method}_failed: {response['error']}")
                return response.get("result") or {}
            logger.info("candidate_validation_notification method=%s", response.get("method"))

    async def _validate_candidate(self, worker: Worker) -> int:
        if self._initialize_request is None:
            raise RuntimeError("hot_reload_unavailable_before_initialize")
        initialize = dict(self._initialize_request)
        params = initialize.get("params") or {}
        await self._candidate_request(worker, "initialize", params)
        notification = self._initialized_notification or {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        await self._write_worker(worker, notification)
        tools = await self._candidate_request(worker, "tools/list", {})
        names = {
            item.get("name")
            for item in tools.get("tools", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if "colab_health" not in names or "colab_stop" not in names:
            raise RuntimeError("hot_reload_candidate_required_tools_missing")
        health = await self._candidate_request(
            worker, "tools/call", {"name": "colab_health", "arguments": {}}
        )
        if health.get("isError"):
            raise RuntimeError("hot_reload_candidate_health_failed")
        return len(names)

    async def _stop_worker(self, worker: Worker, timeout: float = 10) -> None:
        stream = worker.process.stdin
        if stream is not None and not stream.is_closing():
            stream.close()
            with contextlib.suppress(Exception):
                await stream.wait_closed()
        try:
            await asyncio.wait_for(worker.process.wait(), timeout=timeout)
        except TimeoutError:
            worker.process.terminate()
            try:
                await asyncio.wait_for(worker.process.wait(), timeout=5)
            except TimeoutError:
                worker.process.kill()
                await worker.process.wait()

    def status(self) -> dict[str, Any]:
        worker = self.worker
        source_error = None
        try:
            available = source_fingerprint(self.source_root)
        except (OSError, RuntimeError) as error:
            available = None
            source_error = str(error)[:1_000]
        result = {
            "supervised": True,
            "generation": self.generation,
            "worker_pid": worker.pid if worker else None,
            "worker_running": bool(worker and worker.process.returncode is None),
            "source_root": str(self.source_root),
            "active_source_fingerprint": worker.fingerprint if worker else None,
            "available_source_fingerprint": available,
            "reload_available": bool(worker and worker.fingerprint != available),
            "in_flight_requests": len(self._pending),
            "preserves_remote_assignments": True,
            "reload_scope": "MCP worker implementation; not plugin skills/manifest/supervisor",
        }
        if source_error:
            result["available_source_error"] = source_error
        return result

    async def reload(
        self,
        expected_source_fingerprint: str | None,
        drain_timeout_seconds: float,
        source_root: str | None = None,
    ) -> dict[str, Any]:
        if not 1 <= drain_timeout_seconds <= 3600:
            raise ValueError("drain_timeout_seconds must be between 1 and 3600")
        target_root = self._validated_source_root(source_root) if source_root else self.source_root
        available = source_fingerprint(target_root)
        if expected_source_fingerprint is not None and expected_source_fingerprint != available:
            raise RuntimeError(
                "hot_reload_source_changed: expected fingerprint does not match current source"
            )
        await asyncio.wait_for(self._idle.wait(), timeout=drain_timeout_seconds)
        previous = self.worker
        if previous is None:
            raise RuntimeError("hot_reload_worker_missing")
        candidate = await self._spawn_worker(target_root)
        try:
            tool_count = await self._validate_candidate(candidate)
        except BaseException:
            await self._stop_worker(candidate)
            raise

        old_reader = self._worker_reader
        if old_reader is not None:
            old_reader.cancel()
            await asyncio.gather(old_reader, return_exceptions=True)
        self.worker = candidate
        self.source_root = target_root
        self.generation += 1
        self._worker_reader = asyncio.create_task(self._read_worker(candidate))
        await self._stop_worker(previous)
        result = self.status()
        result.update(
            {
                "reloaded": True,
                "previous_worker_pid": previous.pid,
                "validated_tool_count": tool_count + 1,
                "rollback_used": False,
            }
        )
        return result

    @staticmethod
    def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                }
            ],
            "structuredContent": {"result": payload},
            "isError": is_error,
        }

    async def _handle_connector(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        arguments = (message.get("params") or {}).get("arguments") or {}
        try:
            action = arguments.get("action")
            if action == "status":
                payload = self.status()
            elif action == "reload":
                payload = await self.reload(
                    arguments.get("expected_source_fingerprint"),
                    float(arguments.get("drain_timeout_seconds", 300)),
                    arguments.get("source_root"),
                )
            else:
                raise ValueError("action must be status or reload")
        except Exception as error:
            logger.warning("connector_action_failed error_type=%s", type(error).__name__)
            payload = {
                "reloaded": False,
                "error": {
                    "code": "hot_reload_failed",
                    "message": str(error)[:2_000],
                    "resolution": (
                        "The current worker remains active. Fix the source or wait for in-flight "
                        "calls, call status, then retry reload with its available fingerprint."
                    ),
                },
                "rollback_used": True,
                **self.status(),
            }
            result = self._tool_result(payload, is_error=True)
        else:
            result = self._tool_result(payload)
        await self._write_client({"jsonrpc": "2.0", "id": request_id, "result": result})
        if payload.get("reloaded"):
            await self._write_client(
                {"jsonrpc": "2.0", "method": "notifications/tools/list_changed", "params": {}}
            )

    @staticmethod
    def _augment_tools(message: dict[str, Any]) -> None:
        result = message.get("result")
        if not isinstance(result, dict):
            return
        tools = result.get("tools")
        if not isinstance(tools, list):
            return
        if not any(isinstance(tool, dict) and tool.get("name") == CONNECTOR_TOOL for tool in tools):
            tools.append(CONNECTOR_SCHEMA)

    async def _read_worker(self, worker: Worker) -> None:
        if worker.process.stdout is None:
            raise RuntimeError("active worker stdout is unavailable")
        while True:
            line = await worker.process.stdout.readline()
            if not line:
                returncode = await worker.process.wait()
                if worker is self.worker:
                    logger.error("worker_exited returncode=%s", returncode)
                    pending = list(self._pending)
                    self._pending.clear()
                    self._list_requests.clear()
                    self._idle.set()
                    for key in pending:
                        await self._write_client(
                            {
                                "jsonrpc": "2.0",
                                "id": json.loads(key),
                                "error": {
                                    "code": -32001,
                                    "message": (
                                        "colab_mcp_worker_exited_response_unknown: call "
                                        "colab_connector status/reload; inspect durable process or "
                                        "file state before retrying a mutation"
                                    ),
                                    "data": {"worker_returncode": returncode},
                                },
                            }
                        )
                return
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("worker_stdout_non_json_ignored bytes=%d", len(line))
                continue
            if not isinstance(message, dict):
                continue
            if "id" in message:
                key = self._request_key(message["id"])
                if key in self._list_requests:
                    self._list_requests.discard(key)
                    self._augment_tools(message)
                self._pending.discard(key)
                if not self._pending:
                    self._idle.set()
            await self._write_client(message)

    async def _read_client(self) -> None:
        while True:
            line = await asyncio.to_thread(sys.stdin.buffer.readline)
            if not line:
                return
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("client_stdin_non_json_ignored bytes=%d", len(line))
                continue
            if not isinstance(message, dict):
                continue
            method = message.get("method")
            if method == "initialize" and "id" in message:
                self._initialize_request = message
            elif method == "notifications/initialized":
                self._initialized_notification = message
            if method == "tools/call" and (
                (message.get("params") or {}).get("name") == CONNECTOR_TOOL
            ):
                await self._handle_connector(message)
                continue
            worker = self.worker
            if worker is None:
                raise RuntimeError("hot_reload_worker_missing")
            if "id" in message and method:
                key = self._request_key(message["id"])
                self._pending.add(key)
                self._idle.clear()
                if method == "tools/list":
                    self._list_requests.add(key)
            await self._write_worker(worker, message)

    async def run(self) -> None:
        self.worker = await self._spawn_worker()
        self.generation = 1
        self._worker_reader = asyncio.create_task(self._read_worker(self.worker))
        logger.info(
            "supervisor_started worker_pid=%s source_root=%s",
            self.worker.pid,
            self.source_root,
        )
        try:
            await self._read_client()
        finally:
            reader = self._worker_reader
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            if self.worker is not None:
                await self._stop_worker(self.worker)
            logger.info("supervisor_stopped assignments_persist_for_recovery=true")


def main() -> None:
    configure_logging()
    asyncio.run(ColabMcpSupervisor().run())


if __name__ == "__main__":
    main()
