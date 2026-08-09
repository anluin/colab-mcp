from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import jupyter_kernel_client
import nbformat
from colab_cli.auth import TOKEN_CONFIG_PATH, AuthProvider, get_credentials
from colab_cli.client import Accelerator, Client, Prod, Variant
from colab_cli.state import SessionState, StateStore
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from nbformat.v4 import new_output

from .remote import (
    DEFAULT_OUTPUT_LIMIT,
    DEFAULT_PROCESS_OUTPUT_LIMIT,
    MAX_OUTPUT_LIMIT,
    RemoteOperationError,
    RuntimeReplacedError,
    build_remote_code,
    parse_remote_result,
    validate_argv,
    validate_environment,
    validate_output_limit,
    validate_process_output_limit,
    validate_timeout,
)

GPU_TYPES = {"T4", "L4", "G4", "H100", "A100"}
COMPUTE_UNITS_URL = "https://colab.research.google.com/signup"
logger = logging.getLogger("colab_mcp.manager")


class KernelConnectionError(RuntimeError):
    """The runtime operation was not sent because kernel connection setup failed."""


class ManagedSessionState(SessionState):
    """Persisted assignment ownership plus this backend incarnation's marker."""

    runtime_fingerprint: str | None = None
    runtime_replaced_at: str | None = None
    runtime_replaced_reason: str | None = None


def _secure_permissions(path: Path, mode: int, platform: str = os.name) -> None:
    """Apply owner-only POSIX permissions; Windows relies on the user profile ACL."""
    if platform != "nt" and path.exists():
        path.chmod(mode)


class SecureStateStore(StateStore):
    """Upstream state storage with explicit protection for runtime proxy tokens."""

    def _load_raw(self, handle: Any) -> dict[str, SessionState]:
        try:
            handle.seek(0)
            content = handle.read()
            if not content or content.isspace():
                return {}
            data = json.loads(content)
            return {key: ManagedSessionState(**value) for key, value in data.items()}
        except Exception:
            return {}

    def add(self, state: SessionState) -> None:
        super().add(state)
        _secure_permissions(Path(self.path), 0o600)

    def remove(self, name: str) -> None:
        super().remove(name)
        _secure_permissions(Path(self.path), 0o600)


class ProcessJournal:
    """Persist last-known managed-process state outside the ephemeral runtime."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, records: dict[str, dict[str, dict[str, Any]]]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
        _secure_permissions(temporary, 0o600)
        temporary.replace(self.path)
        _secure_permissions(self.path, 0o600)

    def update(self, session: str, process: dict[str, Any]) -> dict[str, Any]:
        process_id = str(process["process_id"])
        with self._lock:
            records = self._read()
            previous = records.setdefault(session, {}).get(process_id, {})
            current = {
                **previous,
                **process,
                "process_id": process_id,
                "last_observed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            records[session][process_id] = current
            self._write(records)
            return current

    def get(self, session: str, process_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read().get(session, {}).get(process_id)

    def list(self, session: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._read().get(session, {}).values())

    def remove_session(self, session: str) -> None:
        with self._lock:
            records = self._read()
            if records.pop(session, None) is not None:
                self._write(records)


def require_local_file(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Local file does not exist: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transfer_bounds(chunk_size: int, max_total_bytes: int, max_files: int) -> None:
    if not 1 <= chunk_size <= MAX_OUTPUT_LIMIT:
        raise ValueError(f"chunk_size must be between 1 and {MAX_OUTPUT_LIMIT}")
    if not 1 <= max_total_bytes <= 10_000_000_000:
        raise ValueError("max_total_bytes must be between 1 and 10000000000")
    if not 1 <= max_files <= 100_000:
        raise ValueError("max_files must be between 1 and 100000")


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _bound_outputs(outputs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    encoded = json.dumps(outputs, ensure_ascii=False).encode("utf-8")
    if len(encoded) <= limit:
        return outputs
    remaining = max(0, limit - 200)
    result: list[dict[str, Any]] = []
    for output in outputs:
        if remaining <= 0:
            break
        item = dict(output)
        if item.get("output_type") == "stream":
            text = str(item.get("text", ""))
            raw = text.encode("utf-8", errors="replace")
            item["text"] = raw[:remaining].decode("utf-8", errors="replace")
        elif item.get("output_type") == "error":
            traceback = "\n".join(str(line) for line in item.get("traceback", []))
            item["traceback"] = [
                traceback.encode("utf-8", errors="replace")[:remaining].decode(
                    "utf-8", errors="replace"
                )
            ]
        else:
            item = {
                "output_type": "stream",
                "name": "stderr",
                "text": "[colab-mcp: rich output omitted by output limit]",
            }
        item_size = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
        if item_size > remaining:
            break
        result.append(item)
        remaining -= item_size
    result.append(
        {
            "output_type": "stream",
            "name": "stderr",
            "text": "[colab-mcp: output truncated]",
        }
    )
    return result


def _save_outputs(cell: Any, outputs: list[dict[str, Any]]) -> None:
    cell.outputs = []
    for out in outputs:
        kind = out.get("output_type")
        if kind == "stream":
            cell.outputs.append(
                new_output("stream", name=out.get("name", "stdout"), text=out.get("text", ""))
            )
        elif kind == "error":
            cell.outputs.append(
                new_output(
                    "error",
                    ename=out.get("ename", "Error"),
                    evalue=out.get("evalue", ""),
                    traceback=out.get("traceback", []),
                )
            )
        elif "data" in out:
            cell.outputs.append(
                new_output(
                    kind or "display_data", data=out["data"], metadata=out.get("metadata", {})
                )
            )


class ColabManager:
    """Cross-platform orchestration using portable Google Colab client components."""

    def __init__(self) -> None:
        config_dir = Path(
            os.environ.get("COLAB_MCP_STATE_DIR", Path.home() / ".config" / "colab-mcp")
        )
        # StateStore constructs its SQLite-backed lock before its own directory
        # creation hook runs, so a fresh installation must create the parent.
        config_dir.mkdir(parents=True, exist_ok=True)
        _secure_permissions(config_dir, 0o700)
        self.store = SecureStateStore(str(config_dir / "sessions.json"))
        self.process_journal = ProcessJournal(config_dir / "processes.json")
        self.suspended_path = config_dir / "suspended.json"
        self._suspended_lock = threading.Lock()
        self.auth_provider = AuthProvider(os.environ.get("COLAB_MCP_AUTH", "oauth2"))
        self.oauth_config = os.environ.get("COLAB_MCP_OAUTH_CONFIG")
        self._client: Client | None = None
        self._keepalives: dict[str, asyncio.Task] = {}
        self.keepalive_seconds = int(os.environ.get("COLAB_MCP_KEEPALIVE_SECONDS", "60"))

    def _read_suspended(self) -> dict[str, dict[str, Any]]:
        with self._suspended_lock:
            if not self.suspended_path.exists():
                return {}
            try:
                return json.loads(self.suspended_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}

    def _write_suspended(self, records: dict[str, dict[str, Any]]) -> None:
        with self._suspended_lock:
            temporary = self.suspended_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
            _secure_permissions(temporary, 0o600)
            temporary.replace(self.suspended_path)
            _secure_permissions(self.suspended_path, 0o600)

    @property
    def authenticated(self) -> bool:
        return self.auth_provider is AuthProvider.ADC or Path(TOKEN_CONFIG_PATH).exists()

    def client(self) -> Client:
        if not self.authenticated:
            raise RuntimeError(
                "Colab OAuth is not initialized. Run `colab-mcp auth` in a terminal."
            )
        if self._client is None:
            if self.auth_provider is AuthProvider.OAUTH2:
                try:
                    credentials = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH)
                    if credentials.expired and credentials.refresh_token:
                        credentials.refresh(Request())
                    if not credentials.valid:
                        raise RuntimeError("credentials are invalid")
                    session = AuthorizedSession(credentials)
                except Exception as error:
                    raise RuntimeError(
                        "Colab credentials are unavailable or could not be refreshed. "
                        "Run `colab-mcp auth` in a terminal; MCP agent sessions never prompt for OAuth."
                    ) from error
            else:
                session = get_credentials(self.oauth_config, self.auth_provider)
            self._client = Client(Prod(), session)
        return self._client

    def resolve(self, name: str | None) -> SessionState:
        sessions = self.store.list()
        if name:
            session = sessions.get(name)
        elif len(sessions) == 1:
            session = next(iter(sessions.values()))
        else:
            raise ValueError("Specify a session name when zero or multiple sessions are stored")
        if session is None:
            raise ValueError(f"Unknown session: {name}")
        return session

    async def _keepalive_loop(self, name: str, endpoint: str) -> None:
        try:
            while True:
                await asyncio.sleep(self.keepalive_seconds)
                current = self.store.get(name)
                if current is None or current.endpoint != endpoint:
                    return
                await asyncio.to_thread(self.client().keep_alive_assignment, endpoint)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A later status/execute call will surface a stale assignment clearly.
            return

    def ensure_keepalive(self, session: SessionState) -> None:
        task = self._keepalives.get(session.name)
        if task is None or task.done():
            self._keepalives[session.name] = asyncio.create_task(
                self._keepalive_loop(session.name, session.endpoint)
            )

    async def start(self, name: str, gpu: str | None) -> SessionState:
        if self.store.get(name):
            raise ValueError(f"Session already exists: {name}")
        if gpu is not None and gpu not in GPU_TYPES:
            raise ValueError(f"Unsupported GPU {gpu!r}; choose one of {sorted(GPU_TYPES)}")
        variant = Variant.GPU if gpu else Variant.DEFAULT
        accelerator = Accelerator(gpu) if gpu else Accelerator.NONE
        response = await asyncio.to_thread(self.client().assign, uuid.uuid4(), variant, accelerator)
        proxy = response.runtime_proxy_info
        session = ManagedSessionState(
            name=name,
            token=proxy.token,
            url=proxy.url,
            endpoint=response.endpoint,
            variant=variant.value,
            accelerator=accelerator.value,
            runtime_fingerprint=uuid.uuid4().hex,
        )
        # Persist immediately after assignment. If either preflight or cleanup
        # fails, a later reconciliation still knows which endpoint we own.
        self.store.add(session)
        try:
            await asyncio.to_thread(self.client().keep_alive_assignment, session.endpoint)
            await self._initialize_runtime_incarnation(session)
        except Exception as preflight_error:
            try:
                await asyncio.to_thread(self.client().unassign, session.endpoint)
            except Exception as cleanup_error:
                raise RuntimeError(
                    f"Runtime {name!r} was allocated but preflight and cleanup both failed. "
                    "It remains tracked; run colab_reconcile with release_orphans=false, "
                    "then colab_stop once connectivity returns."
                ) from cleanup_error
            self.store.remove(session.name)
            raise RuntimeError(
                f"Runtime {name!r} failed preflight and was released"
            ) from preflight_error
        self.ensure_keepalive(session)
        logger.info("runtime_allocated session=%s accelerator=%s", name, accelerator.value)
        return session

    async def _initialize_runtime_incarnation(self, session: ManagedSessionState) -> None:
        """Write the marker before exposing a newly allocated runtime to tools."""
        code = build_remote_code(
            "incarnation_init", {"runtime_fingerprint": session.runtime_fingerprint}
        )
        outputs = await self.execute(code, session.name, 120, output_limit=3_000_000)
        result = parse_remote_result(outputs)
        if result.get("runtime_fingerprint") != session.runtime_fingerprint:
            raise RuntimeError("Runtime incarnation initialization returned an invalid marker")

    def create_notebook(self, path: str, cells: list[str] | None = None) -> dict[str, Any]:
        """Create a local Colab-ready notebook without allocating compute."""
        output = Path(path).expanduser().resolve()
        if output.suffix.lower() != ".ipynb":
            raise ValueError("Notebook path must end in .ipynb")
        if output.exists():
            raise ValueError(f"Refusing to overwrite existing notebook: {output}")
        notebook = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell(source) for source in (cells or [])],
            metadata={
                "colab": {"provenance": []},
                "kernelspec": {"display_name": "Python 3", "name": "python3"},
            },
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
        return {"notebook_path": str(output), "cells": len(notebook.cells)}

    async def sessions(self) -> list[dict[str, Any]]:
        remote = await asyncio.to_thread(self.client().list_assignments)
        remote_endpoints = {item.endpoint for item in remote}
        result = []
        for session in self.store.list().values():
            active = session.endpoint in remote_endpoints
            if active:
                self.ensure_keepalive(session)
            result.append({**session.model_dump(exclude={"token"}), "active": active})
        return result

    async def reconcile(
        self, forget_stale: bool = False, release_orphans: bool = False
    ) -> dict[str, Any]:
        """Compare persisted ownership with live assignments and optionally clean up."""
        remote = await asyncio.to_thread(self.client().list_assignments)
        remote_endpoints = {item.endpoint for item in remote}
        local = self.store.list()
        local_endpoints = {session.endpoint for session in local.values()}
        stale = [
            {"session": session.name, "endpoint": session.endpoint}
            for session in local.values()
            if session.endpoint not in remote_endpoints
        ]
        orphans = sorted(remote_endpoints - local_endpoints)
        forgotten: list[str] = []
        released: list[str] = []
        errors: list[dict[str, str]] = []
        if forget_stale:
            for item in stale:
                task = self._keepalives.pop(item["session"], None)
                if task:
                    task.cancel()
                self.store.remove(item["session"])
                self.process_journal.remove_session(item["session"])
                forgotten.append(item["session"])
        if release_orphans:
            for endpoint in orphans:
                try:
                    await asyncio.to_thread(self.client().unassign, endpoint)
                    released.append(endpoint)
                except Exception as error:
                    errors.append({"endpoint": endpoint, "error": str(error)})
        result = {
            "stale_sessions": stale,
            "orphan_endpoints": orphans,
            "forgotten_sessions": forgotten,
            "released_orphans": released,
            "errors": errors,
        }
        logger.info(
            "runtime_reconciled stale=%d orphans=%d forgotten=%d released=%d errors=%d",
            len(stale),
            len(orphans),
            len(forgotten),
            len(released),
            len(errors),
        )
        return result

    async def execute(
        self, code: str, name: str | None, timeout: float, output_limit: int = 100_000
    ) -> list[dict[str, Any]]:
        if not 1 <= output_limit <= 3_000_000:
            raise ValueError("output_limit must be between 1 and 3000000")
        session = self.resolve(name)
        self.ensure_keepalive(session)

        def run() -> list[dict[str, Any]]:
            # Use the Colab-specialized client directly. This avoids the CLI's
            # legacy KernelClient feature-detection path, which is unreliable
            # during circular package initialization on Windows.
            kernel = None
            try:
                try:
                    kernel = jupyter_kernel_client.ColabKernelClient(
                        server_url=session.url,
                        proxy_token=session.token,
                        kernel_id=session.kernel_id,
                        client_kwargs={
                            "extra_params": {"colab-runtime-proxy-token": session.token}
                        },
                    )
                    kernel.start(timeout=60)
                    if not session.kernel_id and kernel.id:
                        session.kernel_id = kernel.id
                        self.store.add(session)
                    kernel.execute(
                        "import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')",
                        timeout=60,
                    )
                except Exception as error:
                    raise KernelConnectionError(
                        "Kernel connection failed before the requested operation was sent"
                    ) from error
                reply = kernel.execute(code, timeout=timeout)
                return reply.get("outputs", []) if reply else []
            finally:
                if kernel is not None:
                    with contextlib.suppress(Exception):
                        kernel.stop(shutdown_kernel=False)

        for attempt in range(2):
            try:
                raw_outputs = await asyncio.to_thread(run)
                return _bound_outputs(_json_safe(raw_outputs), output_limit)
            except KernelConnectionError:
                if attempt:
                    raise
                session.kernel_id = None
                self.store.add(session)
                logger.warning(
                    "kernel_connection_retry session=%s operation_not_sent=true", session.name
                )
        raise AssertionError("unreachable")

    async def execute_python(
        self,
        code: str,
        name: str | None,
        timeout: float = 900,
        output_limit: int = 100_000,
    ) -> list[dict[str, Any]]:
        if not code or len(code.encode("utf-8")) > 1_000_000:
            raise ValueError("code must contain between 1 and 1000000 UTF-8 bytes")
        timeout = validate_timeout(timeout)
        output_limit = validate_output_limit(output_limit)
        return await self.execute(code, name, timeout, output_limit)

    async def _remote_operation(
        self, operation: str, payload: dict[str, Any], name: str | None, timeout: float = 120
    ) -> Any:
        session = self.resolve(name)
        fingerprint = getattr(session, "runtime_fingerprint", None)
        if not fingerprint:
            raise self._replacement_error(
                session,
                "session has no verifiable runtime incarnation; stop it and start a new session",
                payload.get("process_id"),
            )
        if getattr(session, "runtime_replaced_at", None):
            raise self._replacement_error(
                session,
                session.runtime_replaced_reason or "runtime fingerprint previously changed",
                payload.get("process_id"),
            )
        payload = {**payload, "runtime_fingerprint": fingerprint}
        code = build_remote_code(operation, payload)
        try:
            try:
                outputs = await self.execute(code, name, timeout, output_limit=3_000_000)
            except (TimeoutError, OSError):
                retryable = {
                    "process_status",
                    "process_list",
                    "process_output",
                    "fs_list",
                    "fs_stat",
                    "fs_read",
                    "inspect",
                }
                if operation not in retryable:
                    raise
                session.kernel_id = None
                self.store.add(session)
                logger.warning("kernel_reconnect_retry operation=%s session=%s", operation, name)
                outputs = await self.execute(code, name, timeout, output_limit=3_000_000)
            return parse_remote_result(outputs)
        except RuntimeReplacedError as error:
            session.runtime_replaced_at = datetime.datetime.now(datetime.UTC).isoformat()
            session.runtime_replaced_reason = str(error)
            self.store.add(session)
            logger.warning("runtime_replaced session=%s operation=%s", session.name, operation)
            raise self._replacement_error(session, str(error), payload.get("process_id")) from error

    def _replacement_error(
        self, session: SessionState, reason: str, process_id: str | None = None
    ) -> RuntimeReplacedError:
        last_known = self.process_journal.get(session.name, process_id) if process_id else None
        details = {
            "code": "runtime_replaced",
            "session": session.name,
            "process_id": process_id,
            "probable_cause": "colab_runtime_recycle_or_runtime_oom",
            "message": reason,
            "last_known_process": last_known,
        }
        return RuntimeReplacedError(
            "runtime_replaced: " + json.dumps(details, separators=(",", ":")), details
        )

    def _lost_process(
        self,
        session: SessionState,
        process_id: str,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        last_known = self.process_journal.get(session.name, process_id)
        return {
            **(last_known or {"process_id": process_id}),
            "process_id": process_id,
            "status": "lost",
            "last_known_process": last_known,
            "diagnostic": {
                "code": code,
                "probable_cause": (
                    "colab_runtime_recycle_or_runtime_oom"
                    if code == "runtime_replaced"
                    else "remote_process_state_lost_or_runtime_oom"
                ),
                "message": message,
            },
        }

    async def run_command(
        self,
        argv: list[str],
        name: str | None,
        cwd: str = "/content",
        environment: dict[str, str] | None = None,
        timeout: float = 300,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
    ) -> dict[str, Any]:
        """Wait briefly for a durable process, handing back control when time expires."""
        validate_argv(argv)
        environment = validate_environment(environment)
        timeout = validate_timeout(timeout)
        output_limit = validate_output_limit(output_limit)
        started = time.monotonic()
        process = await self.process_start(argv, name, cwd, environment, output_limit)
        process_id = process["process_id"]
        status = process
        delay = 0.1
        while time.monotonic() - started < timeout:
            status = await self.process_status(process_id, name)
            if status["status"] != "running":
                break
            await asyncio.sleep(min(delay, max(0, timeout - (time.monotonic() - started))))
            delay = min(delay * 2, 5.0)
        wait_expired = status["status"] == "running"
        stdout = await self.process_output(process_id, name, "stdout", 0, output_limit)
        stderr = await self.process_output(process_id, name, "stderr", 0, output_limit)
        return {
            "process_id": process_id,
            "argv": argv,
            "cwd": process["cwd"],
            "status": status["status"],
            "exit_code": status.get("exit_code"),
            "timed_out": wait_expired,
            "process_continues": wait_expired,
            "stdout": stdout["data"],
            "stderr": stderr["data"],
            "stdout_truncated": stdout["more_available"] or stdout.get("truncated", False),
            "stderr_truncated": stderr["more_available"] or stderr.get("truncated", False),
            "duration_seconds": status.get(
                "duration_seconds", round(time.monotonic() - started, 3)
            ),
        }

    async def process_start(
        self,
        argv: list[str],
        name: str | None,
        cwd: str = "/content",
        environment: dict[str, str] | None = None,
        output_limit: int = DEFAULT_PROCESS_OUTPUT_LIMIT,
    ) -> dict[str, Any]:
        """Start a detached process whose metadata and logs live on the runtime."""
        validate_argv(argv)
        environment = validate_environment(environment)
        output_limit = validate_process_output_limit(output_limit)
        result = await self._remote_operation(
            "process_start",
            {
                "process_id": uuid.uuid4().hex,
                "argv": argv,
                "cwd": cwd,
                "environment": environment,
                "output_limit": output_limit,
            },
            name,
        )
        session = self.resolve(name)
        self.process_journal.update(
            session.name,
            {
                **result,
                "session": session.name,
                "runtime_fingerprint": getattr(session, "runtime_fingerprint", None),
            },
        )
        logger.info("process_started session=%s process_id=%s", name, result["process_id"])
        return result

    async def process_status(self, process_id: str, name: str | None) -> dict[str, Any]:
        session = self.resolve(name)
        try:
            result = await self._remote_operation(
                "process_status", {"process_id": process_id}, session.name
            )
        except RuntimeReplacedError as error:
            return self._lost_process(
                session, process_id, error.code, error.details.get("message", str(error))
            )
        except RemoteOperationError as error:
            if not str(error).startswith("FileNotFoundError: Unknown process_id:"):
                raise
            return self._lost_process(session, process_id, "process_state_lost", str(error))
        return self.process_journal.update(session.name, result)

    async def process_list(self, name: str | None) -> list[dict[str, Any]]:
        session = self.resolve(name)
        try:
            result = await self._remote_operation("process_list", {}, session.name)
        except RuntimeReplacedError as error:
            return [
                self._lost_process(
                    session,
                    item["process_id"],
                    error.code,
                    error.details.get("message", str(error)),
                )
                for item in self.process_journal.list(session.name)
            ]
        return [self.process_journal.update(session.name, item) for item in result]

    async def process_output(
        self,
        process_id: str,
        name: str | None,
        stream: str = "stdout",
        offset: int = 0,
        limit: int = 65_536,
    ) -> dict[str, Any]:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        if offset < 0:
            raise ValueError("offset must be zero or greater")
        limit = validate_output_limit(limit)
        session = self.resolve(name)
        try:
            result = await self._remote_operation(
                "process_output",
                {"process_id": process_id, "stream": stream, "offset": offset, "limit": limit},
                session.name,
            )
        except RuntimeReplacedError as error:
            lost = self._lost_process(
                session, process_id, error.code, error.details.get("message", str(error))
            )
            return {
                **lost,
                "stream": stream,
                "offset": offset,
                "next_offset": offset,
                "data": "",
                "more_available": False,
                "eof": True,
            }
        except RemoteOperationError as error:
            if not str(error).startswith("FileNotFoundError: Unknown process_id:"):
                raise
            lost = self._lost_process(session, process_id, "process_state_lost", str(error))
            return {
                **lost,
                "stream": stream,
                "offset": offset,
                "next_offset": offset,
                "data": "",
                "more_available": False,
                "eof": True,
            }
        self.process_journal.update(
            session.name,
            {
                "process_id": process_id,
                "status": result["status"],
                f"{stream}_next_offset": result["next_offset"],
            },
        )
        return result

    async def process_signal(
        self, process_id: str, name: str | None, signal: str = "TERM"
    ) -> dict[str, Any]:
        if signal not in {"TERM", "KILL", "INT"}:
            raise ValueError("signal must be TERM, KILL, or INT")
        session = self.resolve(name)
        try:
            result = await self._remote_operation(
                "process_signal", {"process_id": process_id, "signal": signal}, session.name
            )
        except RuntimeReplacedError as error:
            return self._lost_process(
                session, process_id, error.code, error.details.get("message", str(error))
            )
        except RemoteOperationError as error:
            if not str(error).startswith("FileNotFoundError: Unknown process_id:"):
                raise
            return self._lost_process(session, process_id, "process_state_lost", str(error))
        self.process_journal.update(session.name, {"process_id": process_id, **result})
        return result

    async def filesystem_list(
        self, path: str, name: str | None, limit: int = 1_000
    ) -> dict[str, Any]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        return await self._remote_operation("fs_list", {"path": path, "limit": limit}, name)

    async def filesystem_stat(
        self, path: str, name: str | None, checksum: bool = False
    ) -> dict[str, Any]:
        return await self._remote_operation("fs_stat", {"path": path, "checksum": checksum}, name)

    async def filesystem_read(
        self, path: str, name: str | None, offset: int = 0, limit: int = 262_144
    ) -> dict[str, Any]:
        if offset < 0:
            raise ValueError("offset must be zero or greater")
        limit = validate_output_limit(limit)
        return await self._remote_operation(
            "fs_read", {"path": path, "offset": offset, "limit": limit}, name
        )

    async def filesystem_write(
        self,
        path: str,
        data_base64: str,
        name: str | None,
        append: bool = False,
        create_parents: bool = False,
    ) -> dict[str, Any]:
        try:
            decoded = base64.b64decode(data_base64, validate=True)
        except ValueError as error:
            raise ValueError("data_base64 must be valid base64") from error
        if len(decoded) > MAX_OUTPUT_LIMIT:
            raise ValueError(f"decoded data must not exceed {MAX_OUTPUT_LIMIT} bytes per request")
        return await self._remote_operation(
            "fs_write",
            {
                "path": path,
                "data_base64": data_base64,
                "append": append,
                "create_parents": create_parents,
            },
            name,
        )

    async def filesystem_mkdir(
        self, path: str, name: str | None, parents: bool = True, exist_ok: bool = True
    ) -> dict[str, Any]:
        return await self._remote_operation(
            "fs_mkdir", {"path": path, "parents": parents, "exist_ok": exist_ok}, name
        )

    async def filesystem_move(
        self, source: str, destination: str, name: str | None, overwrite: bool = False
    ) -> dict[str, Any]:
        return await self._remote_operation(
            "fs_move",
            {"source": source, "destination": destination, "overwrite": overwrite},
            name,
        )

    async def filesystem_remove(
        self,
        path: str,
        name: str | None,
        recursive: bool = False,
        missing_ok: bool = False,
    ) -> dict[str, Any]:
        return await self._remote_operation(
            "fs_remove",
            {"path": path, "recursive": recursive, "missing_ok": missing_ok},
            name,
        )

    async def inspect_runtime(
        self,
        name: str | None,
        tools: list[str] | None = None,
        process_limit: int = 100,
    ) -> dict[str, Any]:
        tools = tools or ["bash", "git", "python", "pip", "uv", "gcc", "make", "nvidia-smi"]
        if len(tools) > 100 or any(not tool or len(tool) > 200 for tool in tools):
            raise ValueError("tools must contain at most 100 non-empty names")
        if not 1 <= process_limit <= 1_000:
            raise ValueError("process_limit must be between 1 and 1000")
        return await self._remote_operation(
            "inspect", {"tools": tools, "process_limit": process_limit}, name
        )

    async def _remote_stat_or_none(
        self, path: str, name: str | None, checksum: bool = False
    ) -> dict[str, Any] | None:
        try:
            return await self.filesystem_stat(path, name, checksum)
        except RemoteOperationError as error:
            if str(error).startswith("FileNotFoundError:"):
                return None
            raise

    async def transfer_upload(
        self,
        local_path: str,
        remote_path: str,
        name: str | None,
        overwrite: bool = False,
        sync: bool = True,
        chunk_size: int = 524_288,
        max_total_bytes: int = 100_000_000,
        max_files: int = 10_000,
    ) -> dict[str, Any]:
        """Upload files through staged, checksummed runtime filesystem writes."""
        _transfer_bounds(chunk_size, max_total_bytes, max_files)
        source = Path(local_path).expanduser().resolve()
        if not source.exists() or (not source.is_file() and not source.is_dir()):
            raise ValueError(f"Local file or directory does not exist: {source}")
        files = (
            [source]
            if source.is_file()
            else sorted(path for path in source.rglob("*") if path.is_file())
        )
        if len(files) > max_files:
            raise ValueError(f"Transfer contains {len(files)} files; max_files is {max_files}")
        total = sum(path.stat().st_size for path in files)
        if total > max_total_bytes:
            raise ValueError(
                f"Transfer contains {total} bytes; max_total_bytes is {max_total_bytes}"
            )
        remote_root = PurePosixPath(remote_path)
        transferred: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for source_file in files:
            relative = (
                PurePosixPath(source_file.name)
                if source.is_file()
                else PurePosixPath(*source_file.relative_to(source).parts)
            )
            destination = str(remote_root if source.is_file() else remote_root / relative)
            checksum = await asyncio.to_thread(_file_sha256, source_file)
            existing = await self._remote_stat_or_none(destination, name, checksum=True)
            if existing and sync and existing.get("sha256") == checksum:
                skipped.append(
                    {"local_path": str(source_file), "remote_path": destination, "sha256": checksum}
                )
                continue
            if existing and not overwrite:
                raise FileExistsError(f"Remote destination exists: {destination}")
            temporary = destination + ".colab-mcp-part-" + uuid.uuid4().hex
            first = True
            try:
                with source_file.open("rb") as handle:
                    while True:
                        chunk = handle.read(chunk_size)
                        if not chunk and not first:
                            break
                        await self.filesystem_write(
                            temporary,
                            base64.b64encode(chunk).decode(),
                            name,
                            append=not first,
                            create_parents=True,
                        )
                        first = False
                        if not chunk:
                            break
                remote_stat = await self.filesystem_stat(temporary, name, checksum=True)
                if remote_stat.get("sha256") != checksum:
                    raise RuntimeError(f"Checksum mismatch while uploading {source_file}")
                await self.filesystem_move(temporary, destination, name, overwrite=overwrite)
                transferred.append(
                    {
                        "local_path": str(source_file),
                        "remote_path": destination,
                        "size": source_file.stat().st_size,
                        "sha256": checksum,
                    }
                )
            finally:
                await self.filesystem_remove(temporary, name, missing_ok=True)
        return {"files_transferred": transferred, "files_skipped": skipped, "total_bytes": total}

    async def _remote_files(
        self, path: str, name: str | None, max_files: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        root = await self.filesystem_stat(path, name)
        if root["kind"] == "file":
            return root, [root]
        pending = [root["path"]]
        files: list[dict[str, Any]] = []
        while pending:
            listing = await self.filesystem_list(pending.pop(), name, limit=min(max_files, 10_000))
            if listing["truncated"]:
                raise ValueError("Remote directory exceeds max_files listing bound")
            for entry in listing["entries"]:
                if entry["kind"] == "directory":
                    pending.append(entry["path"])
                elif entry["kind"] == "file":
                    files.append(entry)
                    if len(files) > max_files:
                        raise ValueError(f"Remote transfer exceeds max_files={max_files}")
        return root, files

    async def transfer_download(
        self,
        remote_path: str,
        local_path: str,
        name: str | None,
        overwrite: bool = False,
        sync: bool = True,
        chunk_size: int = 524_288,
        max_total_bytes: int = 100_000_000,
        max_files: int = 10_000,
    ) -> dict[str, Any]:
        """Download files through checksummed chunks and atomic local replacement."""
        _transfer_bounds(chunk_size, max_total_bytes, max_files)
        root, files = await self._remote_files(remote_path, name, max_files)
        total = sum(int(item["size"]) for item in files)
        if total > max_total_bytes:
            raise ValueError(
                f"Transfer contains {total} bytes; max_total_bytes is {max_total_bytes}"
            )
        destination_root = Path(local_path).expanduser().resolve()
        if root["kind"] == "directory":
            destination_root.mkdir(parents=True, exist_ok=True)
        transferred: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        remote_root = PurePosixPath(root["path"])
        for item in files:
            relative = (
                PurePosixPath(item["path"]).relative_to(remote_root)
                if root["kind"] == "directory"
                else None
            )
            destination = (
                destination_root
                if relative is None
                else (destination_root / Path(*relative.parts)).resolve()
            )
            if (
                relative is not None
                and destination != destination_root
                and destination_root not in destination.parents
            ):
                raise ValueError("Remote path would escape the local destination")
            remote_stat = await self.filesystem_stat(item["path"], name, checksum=True)
            checksum = remote_stat["sha256"]
            if (
                destination.exists()
                and sync
                and destination.is_file()
                and await asyncio.to_thread(_file_sha256, destination) == checksum
            ):
                skipped.append(
                    {
                        "remote_path": item["path"],
                        "local_path": str(destination),
                        "sha256": checksum,
                    }
                )
                continue
            if destination.exists() and not overwrite:
                raise FileExistsError(f"Local destination exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                destination.name + ".colab-mcp-part-" + uuid.uuid4().hex
            )
            try:
                offset = 0
                with temporary.open("wb") as handle:
                    while True:
                        chunk = await self.filesystem_read(item["path"], name, offset, chunk_size)
                        handle.write(base64.b64decode(chunk["data_base64"], validate=True))
                        offset = chunk["next_offset"]
                        if chunk["eof"]:
                            break
                if await asyncio.to_thread(_file_sha256, temporary) != checksum:
                    raise RuntimeError(f"Checksum mismatch while downloading {item['path']}")
                temporary.replace(destination)
                transferred.append(
                    {
                        "remote_path": item["path"],
                        "local_path": str(destination),
                        "size": item["size"],
                        "sha256": checksum,
                    }
                )
            finally:
                temporary.unlink(missing_ok=True)
        return {"files_transferred": transferred, "files_skipped": skipped, "total_bytes": total}

    async def execute_notebook(
        self, source: str, output: str, name: str | None, timeout: float
    ) -> dict[str, Any]:
        source_path = require_local_file(source)
        output_path = Path(output).expanduser().resolve()
        notebook = nbformat.read(source_path, as_version=4)
        executed = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            outputs = await self.execute(cell.source, name, timeout)
            _save_outputs(cell, outputs)
            cell.execution_count = executed + 1
            executed += 1
            if any(item.get("output_type") == "error" for item in outputs):
                break
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output_path)
        return {"output_path": str(output_path), "executed_cells": executed}

    async def upload(self, local: str, remote: str, name: str | None) -> dict[str, Any]:
        """Compatibility alias for the bounded, checksummed transfer path."""
        return await self.transfer_upload(local, remote, name)

    async def download(self, remote: str, local: str, name: str | None) -> dict[str, Any]:
        """Compatibility alias for the bounded, checksummed transfer path."""
        return await self.transfer_download(remote, local, name)

    async def stop(self, name: str | None) -> dict[str, Any]:
        session = self.resolve(name)
        task = self._keepalives.pop(session.name, None)
        if task:
            task.cancel()
        assignments = await asyncio.to_thread(self.client().list_assignments)
        was_active = session.endpoint in {item.endpoint for item in assignments}
        if was_active:
            await asyncio.to_thread(self.client().unassign, session.endpoint)
        self.store.remove(session.name)
        self.process_journal.remove_session(session.name)
        logger.info("runtime_released session=%s was_active=%s", session.name, was_active)
        return {"stopped": session.name, "runtime_was_active": was_active}

    async def pause(self, name: str | None, notebook_path: str) -> dict[str, Any]:
        """Record a notebook checkpoint and release its runtime/GPU allocation."""
        session = self.resolve(name)
        checkpoint = require_local_file(notebook_path)
        if checkpoint.suffix.lower() != ".ipynb":
            raise ValueError("Pause checkpoint must be an .ipynb notebook")
        record = {
            "session": session.name,
            "notebook_path": str(checkpoint),
            "accelerator": session.accelerator,
            "variant": session.variant,
        }
        await self.stop(session.name)
        records = self._read_suspended()
        records[session.name] = record
        self._write_suspended(records)
        return {**record, "state": "paused", "runtime_released": True}

    async def resume(
        self,
        name: str,
        execute_notebook: bool = False,
        output_path: str | None = None,
        cell_timeout: float = 900,
    ) -> dict[str, Any]:
        """Allocate a new runtime from a paused record, optionally rerunning its notebook."""
        records = self._read_suspended()
        record = records.get(name)
        if record is None:
            raise ValueError(f"No paused notebook session named {name!r}")
        accelerator = record.get("accelerator")
        gpu = accelerator if accelerator in GPU_TYPES else None
        session = await self.start(name, gpu)
        result: dict[str, Any] = {
            "session": session.model_dump(exclude={"token"}),
            "notebook_path": record["notebook_path"],
            "state": "resumed",
            "fresh_runtime": True,
        }
        if execute_notebook:
            if not output_path:
                source = Path(record["notebook_path"])
                output_path = str(source.with_name(f"{source.stem}.resumed.ipynb"))
            result["execution"] = await self.execute_notebook(
                record["notebook_path"], output_path, name, cell_timeout
            )
        records.pop(name, None)
        self._write_suspended(records)
        return result

    def suspended(self) -> list[dict[str, Any]]:
        return list(self._read_suspended().values())
