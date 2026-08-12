from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import gzip
import hashlib
import io
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
import zlib
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import quote

import nbformat
import requests
from colab_cli.auth import TOKEN_CONFIG_PATH, AuthProvider, get_credentials
from colab_cli.client import Accelerator, Client, Prod, Variant
from colab_cli.state import SessionState, StateStore
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from nbformat.v4 import new_output
from pydantic import BaseModel, Field, field_validator

from .colab_adapter import binary_upload, kernel_client
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
MAX_TRANSFER_CHUNK = 2_000_000
ALREADY_COMPRESSED_SUFFIXES = frozenset(
    {
        ".7z",
        ".aac",
        ".avif",
        ".avi",
        ".flac",
        ".gif",
        ".gz",
        ".heic",
        ".heif",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".ogg",
        ".opus",
        ".png",
        ".rar",
        ".webm",
        ".webp",
        ".wmv",
        ".zip",
    }
)
# Colab's runtime proxy regularly needs more than five seconds to establish a
# fresh websocket even when the allocation is healthy.  Keep the phase bounded,
# but allow enough time for normal cross-region cold/reconnect latency.
LEASED_CONNECTION_TIMEOUT_SECONDS = 20
logger = logging.getLogger("colab_mcp.manager")

SYNC_ALWAYS_EXCLUDED = (
    ".git/**",
    ".venv/**",
    "venv/**",
    "node_modules/**",
    "**/__pycache__/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    ".env",
    ".env.*",
    "**/*.key",
    "**/*.pem",
)


class AutoExportRule(BaseModel):
    """A durable local export selected by the owned process's exit code."""

    remote_path: Annotated[
        str,
        Field(description="File or directory under /content to download after process exit."),
    ]
    local_path: Annotated[
        str,
        Field(description="Absolute or host-relative destination; normalized to an absolute path."),
    ]
    exit_codes: Annotated[
        list[int] | None,
        Field(description="Matching exit codes; null matches every exit code. Defaults to [0]."),
    ] = Field(default_factory=lambda: [0])
    overwrite: Annotated[
        bool,
        Field(description="Replace an existing destination file; existing directories stay safe."),
    ] = False
    chunk_size: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_TRANSFER_CHUNK,
            description="Download chunk size in bytes; defaults to 524,288.",
        ),
    ] = 524_288
    max_total_bytes: Annotated[
        int,
        Field(
            ge=1,
            le=10_000_000_000,
            description="Hard artifact size limit in bytes; defaults to 100,000,000.",
        ),
    ] = 100_000_000
    max_files: Annotated[
        int,
        Field(
            ge=1,
            le=100_000,
            description="Hard directory file-count limit; defaults to 10,000.",
        ),
    ] = 10_000
    compression: Annotated[
        Literal["auto", "gzip", "none"],
        Field(description="Wire compression: auto (default), forced gzip, or none."),
    ] = "auto"
    compression_min_bytes: Annotated[
        int,
        Field(
            ge=0,
            le=10_000_000_000,
            description="Auto mode only considers files at least this large; defaults to 1 MiB.",
        ),
    ] = 1_048_576
    compression_min_savings: Annotated[
        float,
        Field(
            ge=0,
            lt=1,
            description="Minimum fractional wire-byte saving required in auto mode; defaults to 0.10.",
        ),
    ] = 0.10

    @field_validator("remote_path", "local_path")
    @classmethod
    def non_empty_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be empty")
        return value

    @field_validator("exit_codes")
    @classmethod
    def bounded_exit_codes(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and (not value or len(value) > 256 or len(value) != len(set(value))):
            raise ValueError("exit_codes must contain 1-256 unique integers or be null")
        return value


class KernelConnectionError(RuntimeError):
    """The runtime operation was not sent because kernel connection setup failed."""

    code = "kernel_connection_failed_request_not_submitted"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.details = {
            "code": self.code,
            "phase": "kernel_connection",
            "request_submission": "not_submitted",
            "message": message,
            **(details or {}),
        }
        super().__init__(self.code + ": " + json.dumps(self.details, separators=(",", ":")))


class OperationLeaseError(RuntimeError):
    """An operation-bound lease is absent, stale, expired, or no longer assigned."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.details = {"code": code, "message": message, **(details or {})}
        super().__init__(code + ": " + json.dumps(self.details, separators=(",", ":")))


class RequestOutcomeUnknownError(RuntimeError):
    """The synchronous upstream adapter raised after submission may have begun."""

    def __init__(self, code: str, message: str, details: dict[str, Any]) -> None:
        self.code = code
        self.details = {"code": code, "message": message, **details}
        super().__init__(code + ": " + json.dumps(self.details, separators=(",", ":")))


class TransferError(RuntimeError):
    """A resumable transfer failed with explicit staging and submission state."""

    def __init__(self, code: str, message: str, details: dict[str, Any]) -> None:
        self.code = code
        self.details = {"code": code, "message": message, **details}
        super().__init__(code + ": " + json.dumps(self.details, separators=(",", ":")))


ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
CONTROL_TIMING_PREFIX = "__COLAB_MCP_CONTROL_TIMING__"


def _guarded_python_source(code: str, fingerprint: str, lease_token: str) -> str:
    """Validate incarnation and lease in the same kernel request before user code starts."""
    return f"""
import datetime as __cm_datetime
import json as __cm_json
import pathlib as __cm_pathlib
import time as __cm_time
__cm_guard_started = __cm_time.monotonic()
__cm_state = __cm_pathlib.Path('/content/.colab-mcp')
__cm_actual = (__cm_state / 'runtime-incarnation').read_text(encoding='ascii').strip() if (__cm_state / 'runtime-incarnation').is_file() else None
if __cm_actual != {fingerprint!r}:
    raise RuntimeError('runtime_replaced: guarded execution observed a different incarnation')
__cm_lease = __cm_json.loads((__cm_state / 'operation-lease.json').read_text(encoding='utf-8'))
__cm_candidates = __cm_lease.get('leases')
if not isinstance(__cm_candidates, list):
    __cm_candidates = [{{'token': __cm_lease.get('token'), 'expires_at': __cm_lease.get('expires_at')}}]
__cm_match = next((item for item in __cm_candidates if isinstance(item, dict) and item.get('token') == {lease_token!r}), None)
if __cm_match is None or __cm_lease.get('runtime_fingerprint') != __cm_actual:
    raise RuntimeError('operation_lease_stale: guarded execution lease does not match')
if __cm_datetime.datetime.fromisoformat(__cm_match['expires_at']) <= __cm_datetime.datetime.now(__cm_datetime.timezone.utc):
    raise RuntimeError('operation_lease_expired: guarded execution lease expired')
print({CONTROL_TIMING_PREFIX!r} + __cm_json.dumps({{'fingerprint_validation_seconds': round(__cm_time.monotonic() - __cm_guard_started, 6)}}))
del __cm_datetime, __cm_json, __cm_pathlib, __cm_time, __cm_guard_started, __cm_state, __cm_actual, __cm_lease, __cm_candidates, __cm_match
{code}
"""


def _extract_control_timing(outputs: list[dict[str, Any]]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for output in outputs:
        if output.get("output_type") != "stream":
            continue
        kept = []
        for line in str(output.get("text", "")).splitlines(keepends=True):
            stripped = line.rstrip("\r\n")
            if stripped.startswith(CONTROL_TIMING_PREFIX):
                value = json.loads(stripped[len(CONTROL_TIMING_PREFIX) :])
                timings.update({key: float(item) for key, item in value.items()})
            else:
                kept.append(line)
        output["text"] = "".join(kept)
    return timings


class ManagedSessionState(SessionState):
    """Persisted assignment ownership plus this backend incarnation's marker."""

    runtime_fingerprint: str | None = None
    runtime_replaced_at: str | None = None
    runtime_replaced_reason: str | None = None
    keepalive_status: str | None = None
    last_keepalive_at: str | None = None
    last_keepalive_error: str | None = None
    consecutive_keepalive_failures: int = 0
    operation_lease_token: str | None = None
    operation_lease_issued_at: str | None = None
    operation_lease_expires_at: str | None = None


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
    if not 1 <= chunk_size <= MAX_TRANSFER_CHUNK:
        raise ValueError(f"chunk_size must be between 1 and {MAX_TRANSFER_CHUNK}")
    if not 1 <= max_total_bytes <= 10_000_000_000:
        raise ValueError("max_total_bytes must be between 1 and 10000000000")
    if not 1 <= max_files <= 100_000:
        raise ValueError("max_files must be between 1 and 100000")


def _compression_settings(
    compression: str, compression_min_bytes: int, compression_min_savings: float
) -> tuple[str, int, float]:
    if compression not in {"auto", "gzip", "none"}:
        raise ValueError("compression must be auto, gzip, or none")
    if not 0 <= compression_min_bytes <= 10_000_000_000:
        raise ValueError("compression_min_bytes must be between 0 and 10000000000")
    if not 0 <= compression_min_savings < 1:
        raise ValueError("compression_min_savings must be at least 0 and less than 1")
    return compression, compression_min_bytes, compression_min_savings


def _gzip_local_file(source: Path) -> tuple[Path, int, str]:
    descriptor, temporary_name = tempfile.mkstemp(prefix="colab-mcp-gzip-", suffix=".gz")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_output, compresslevel=6, mtime=0
            ) as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        return temporary, temporary.stat().st_size, _file_sha256(temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _use_compressed_wire(
    mode: str, content_bytes: int, wire_bytes: int, minimum_savings: float
) -> bool:
    return mode == "gzip" or (
        mode == "auto"
        and wire_bytes < content_bytes
        and (content_bytes - wire_bytes) / max(1, content_bytes) >= minimum_savings
    )


def _auto_compression_candidate(path: Path | PurePosixPath) -> bool:
    """Avoid full gzip trial passes for formats that are already compressed."""
    return path.suffix.lower() not in ALREADY_COMPRESSED_SUFFIXES


def _raw_download_to_file(
    session: Any,
    remote_path: str,
    destination: Path,
    codec: str,
    content_bytes: int,
    wire_bytes: int,
    wire_checksum: str,
    content_checksum: str,
    max_total_bytes: int,
    chunk_size: int,
) -> None:
    """Stream authenticated raw bytes from Jupyter's files endpoint."""
    quoted = quote(remote_path.strip("/"), safe="/")
    url = session.url.rstrip("/") + "/files/" + quoted
    response = requests.get(
        url,
        params={"authuser": "0", "colab-runtime-proxy-token": session.token},
        headers={
            "X-Colab-Client-Agent": "colab-mcp",
            "X-Colab-Runtime-Proxy-Token": session.token,
        },
        stream=True,
        timeout=(30, 120),
    )
    response.raise_for_status()
    wire_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS) if codec == "gzip" else None
    wire_read = 0
    content_written = 0
    with destination.open("wb") as handle:
        for wire_data in response.iter_content(chunk_size):
            if not wire_data:
                continue
            wire_read += len(wire_data)
            if wire_read > wire_bytes:
                raise RuntimeError("Raw download exceeded declared wire size")
            wire_digest.update(wire_data)
            content_data = decompressor.decompress(wire_data) if decompressor else wire_data
            content_written += len(content_data)
            if content_written > content_bytes or content_written > max_total_bytes:
                raise RuntimeError("Decompressed transfer exceeded declared size bound")
            content_digest.update(content_data)
            handle.write(content_data)
        if decompressor:
            final_data = decompressor.flush()
            content_written += len(final_data)
            content_digest.update(final_data)
            handle.write(final_data)
            if not decompressor.eof:
                raise RuntimeError("Compressed transfer ended before the gzip stream")
        handle.flush()
        os.fsync(handle.fileno())
    if wire_read != wire_bytes or wire_digest.hexdigest() != wire_checksum:
        raise RuntimeError("Raw download wire checksum mismatch")
    if content_written != content_bytes or content_digest.hexdigest() != content_checksum:
        raise RuntimeError("Raw download content checksum mismatch")


def _workspace_manifest(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": PurePosixPath(*path.relative_to(root).parts).as_posix(),
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in files
    ]


def _sync_pattern_matches(path: str, pattern: str) -> bool:
    """Match normalized POSIX paths with a small, predictable git-style subset."""
    pattern = pattern.strip().replace("\\", "/").lstrip("/")
    if not pattern or ".." in PurePosixPath(pattern).parts:
        raise ValueError(f"Invalid sync include pattern: {pattern!r}")
    candidate = PurePosixPath(path)
    if pattern.startswith("**/") and pattern.endswith("/**"):
        directory = pattern[3:-3].strip("/")
        return (
            path == directory or path.startswith(directory + "/") or f"/{directory}/" in f"/{path}/"
        )
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    return candidate.match(pattern) or ("/" not in pattern and candidate.name == pattern)


def _sync_selected(path: str, include: tuple[str, ...]) -> tuple[bool, str | None]:
    for pattern in SYNC_ALWAYS_EXCLUDED:
        if _sync_pattern_matches(path, pattern):
            return False, f"built_in:{pattern}"
    if include and not any(_sync_pattern_matches(path, pattern) for pattern in include):
        return False, "not_in_include"
    return True, None


def _build_workspace_bundle(
    root: Path,
    files_by_relative: dict[str, Path],
    manifest: list[dict[str, Any]],
    compression: str,
    compression_min_savings: float,
) -> tuple[Path, str, int, str]:
    """Build a deterministic, safe tar bundle for one workspace delta."""
    descriptor, archive_name = tempfile.mkstemp(prefix="colab-mcp-workspace-", suffix=".tar")
    os.close(descriptor)
    archive = Path(archive_name)
    compressed: Path | None = None
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    try:
        with tarfile.open(archive, "w") as bundle:
            metadata = tarfile.TarInfo("manifest.json")
            metadata.size = len(manifest_bytes)
            metadata.mtime = metadata.uid = metadata.gid = 0
            metadata.mode = 0o600
            bundle.addfile(metadata, io.BytesIO(manifest_bytes))
            for item in manifest:
                source = files_by_relative[item["path"]]
                info = tarfile.TarInfo("files/" + item["path"])
                info.size = item["size"]
                info.mtime = info.uid = info.gid = 0
                info.mode = 0o600
                with source.open("rb") as handle:
                    bundle.addfile(info, handle)
        codec = "none"
        wire = archive
        should_try_compression = compression == "gzip" or (
            compression == "auto"
            and any(
                _auto_compression_candidate(files_by_relative[item["path"]]) for item in manifest
            )
        )
        if should_try_compression:
            candidate, candidate_size, _ = _gzip_local_file(archive)
            if _use_compressed_wire(
                compression, archive.stat().st_size, candidate_size, compression_min_savings
            ):
                compressed, wire, codec = candidate, candidate, "gzip"
                archive.unlink()
            else:
                candidate.unlink(missing_ok=True)
        return wire, codec, wire.stat().st_size, _file_sha256(wire)
    except BaseException:
        archive.unlink(missing_ok=True)
        if compressed is not None:
            compressed.unlink(missing_ok=True)
        raise


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
        self.sync_history_path = config_dir / "sync-throughput.json"
        self._suspended_lock = threading.Lock()
        self.auth_provider = AuthProvider(os.environ.get("COLAB_MCP_AUTH", "oauth2"))
        self.oauth_config = os.environ.get("COLAB_MCP_OAUTH_CONFIG")
        self._client: Client | None = None
        # Keep one live Jupyter channel per owned runtime.  The upstream Colab
        # runtime object does the same: reconnecting for every MCP request is
        # both slow and substantially more failure-prone on the runtime proxy.
        self._kernel_clients: dict[str, Any] = {}
        self._kernel_locks: dict[str, asyncio.Lock] = {}
        self._kernel_clients_guard = threading.Lock()
        # Several verified transfers may overlap within one runtime. A newer
        # probe must not invalidate an already-running transfer locally; the
        # runtime enforces the same bounded, expiring lease set.
        self._operation_leases: dict[str, dict[str, str]] = {}
        self._keepalives: dict[str, asyncio.Task] = {}
        self.keepalive_seconds = int(os.environ.get("COLAB_MCP_KEEPALIVE_SECONDS", "60"))
        self._export_watchers: dict[tuple[str, str], asyncio.Task] = {}
        self.export_poll_seconds = float(os.environ.get("COLAB_MCP_EXPORT_POLL_SECONDS", "5"))

    def _sync_speed_estimate(self, direction: str, transfer_bytes: int) -> dict[str, Any]:
        samples: list[float] = []
        try:
            records = json.loads(self.sync_history_path.read_text(encoding="utf-8"))
            samples = [
                float(item["mib_per_second"])
                for item in records
                if item.get("direction") == direction and float(item.get("mib_per_second", 0)) > 0
            ][-20:]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        if not samples:
            return {
                "status": "insufficient_history",
                "estimated_seconds": None,
                "estimated_mib_per_second": None,
            }
        ordered = sorted(samples)
        low = ordered[max(0, math.floor((len(ordered) - 1) * 0.2))]
        typical = ordered[len(ordered) // 2]
        high = ordered[min(len(ordered) - 1, math.ceil((len(ordered) - 1) * 0.8))]
        mib = transfer_bytes / (1024 * 1024)
        return {
            "status": "estimated_from_observed_transfers",
            "sample_count": len(samples),
            "estimated_mib_per_second": round(typical, 2),
            "range_mib_per_second": [round(low, 2), round(high, 2)],
            "estimated_seconds": round(mib / typical, 2) if typical else None,
            "range_seconds": [
                round(mib / high, 2) if high else None,
                round(mib / low, 2) if low else None,
            ],
        }

    def _record_sync_speed(self, direction: str, wire_bytes: int, seconds: float) -> None:
        if wire_bytes <= 0 or seconds <= 0:
            return
        try:
            records = json.loads(self.sync_history_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = []
        except (OSError, json.JSONDecodeError):
            records = []
        records.append(
            {
                "direction": direction,
                "mib_per_second": wire_bytes / (1024 * 1024) / seconds,
                "wire_bytes": wire_bytes,
                "seconds": seconds,
                "observed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
        temporary = self.sync_history_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(records[-40:], indent=2), encoding="utf-8")
        _secure_permissions(temporary, 0o600)
        temporary.replace(self.sync_history_path)
        _secure_permissions(self.sync_history_path, 0o600)

    def _kernel_lock(self, name: str) -> asyncio.Lock:
        with self._kernel_clients_guard:
            return self._kernel_locks.setdefault(name, asyncio.Lock())

    def _cached_kernel(self, name: str) -> Any | None:
        with self._kernel_clients_guard:
            return self._kernel_clients.get(name)

    def _cache_kernel(self, name: str, kernel: Any) -> None:
        with self._kernel_clients_guard:
            self._kernel_clients[name] = kernel

    async def close_kernel_channel(self, name: str) -> None:
        """Close a local channel without shutting down the remote Colab kernel."""
        async with self._kernel_lock(name):
            await self._close_kernel_channel_unlocked(name)

    async def _close_kernel_channel_unlocked(self, name: str) -> None:
        with self._kernel_clients_guard:
            kernel = self._kernel_clients.pop(name, None)
        if kernel is not None:
            await asyncio.to_thread(self._stop_kernel_safely, kernel)

    @staticmethod
    def _stop_kernel_safely(kernel: Any) -> None:
        with contextlib.suppress(Exception):
            kernel.stop(shutdown_kernel=False)

    async def shutdown_kernel_channels(self) -> None:
        """Close all local channels while preserving tracked runtime ownership."""
        with self._kernel_clients_guard:
            kernels = list(self._kernel_clients.values())
            self._kernel_clients.clear()
            self._kernel_locks.clear()
        if kernels:
            await asyncio.gather(
                *(asyncio.to_thread(self._stop_kernel_safely, kernel) for kernel in kernels),
                return_exceptions=True,
            )

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

    def _persist_keepalive(
        self,
        name: str,
        endpoint: str,
        *,
        status: str,
        error: str | None,
        succeeded: bool,
    ) -> ManagedSessionState | None:
        current = self.store.get(name)
        if current is None or current.endpoint != endpoint:
            return None
        current.keepalive_status = status
        current.last_keepalive_error = error
        if succeeded:
            current.last_keepalive_at = datetime.datetime.now(datetime.UTC).isoformat()
            current.consecutive_keepalive_failures = 0
        else:
            current.consecutive_keepalive_failures += 1
        self.store.add(current)
        return current

    async def _keepalive_once(self, session: SessionState) -> bool:
        try:
            await asyncio.to_thread(self.client().keep_alive_assignment, session.endpoint)
        except Exception as error:
            current = self._persist_keepalive(
                session.name,
                session.endpoint,
                status="degraded",
                error=str(error)[:1_000],
                succeeded=False,
            )
            logger.warning(
                "keepalive_error session=%s failures=%d error_type=%s",
                session.name,
                current.consecutive_keepalive_failures if current else 0,
                type(error).__name__,
            )
            return False
        self._persist_keepalive(
            session.name,
            session.endpoint,
            status="healthy",
            error=None,
            succeeded=True,
        )
        return True

    async def _keepalive_loop(self, name: str, endpoint: str) -> None:
        while True:
            current = self.store.get(name)
            if current is None or current.endpoint != endpoint:
                return
            succeeded = await self._keepalive_once(current)
            current = self.store.get(name)
            if current is None or current.endpoint != endpoint:
                return
            if not succeeded and current.consecutive_keepalive_failures >= 2:
                try:
                    assignments = await asyncio.to_thread(self.client().list_assignments)
                except Exception:
                    pass
                else:
                    if endpoint not in {item.endpoint for item in assignments}:
                        self._persist_keepalive(
                            name,
                            endpoint,
                            status="lease_lost",
                            error="tracked assignment is no longer active",
                            succeeded=False,
                        )
                        return
            await asyncio.sleep(self.keepalive_seconds)

    def ensure_keepalive(self, session: SessionState) -> None:
        task = self._keepalives.get(session.name)
        if task is None or task.done():
            self._keepalives[session.name] = asyncio.create_task(
                self._keepalive_loop(session.name, session.endpoint)
            )

    async def recover_keepalives(self) -> dict[str, Any]:
        """Restore heartbeats for persisted assignments when the MCP server restarts."""
        local = self.store.list()
        if not local:
            return {"recovered": [], "lease_lost": [], "error": None}
        try:
            assignments = await asyncio.to_thread(self.client().list_assignments)
        except Exception as error:
            logger.warning("keepalive_recovery_error error_type=%s", type(error).__name__)
            return {"recovered": [], "lease_lost": [], "error": str(error)[:1_000]}
        endpoints = {item.endpoint for item in assignments}
        recovered: list[str] = []
        lease_lost: list[str] = []
        for session in local.values():
            if session.endpoint in endpoints:
                self.ensure_keepalive(session)
                recovered.append(session.name)
            else:
                self._persist_keepalive(
                    session.name,
                    session.endpoint,
                    status="lease_lost",
                    error="tracked assignment was absent during server startup recovery",
                    succeeded=False,
                )
                lease_lost.append(session.name)
        logger.info("keepalive_recovered active=%d lease_lost=%d", len(recovered), len(lease_lost))
        return {"recovered": recovered, "lease_lost": lease_lost, "error": None}

    async def shutdown_keepalives(self) -> None:
        """Stop only local heartbeat tasks; runtime ownership remains persisted."""
        tasks = list(self._keepalives.values())
        self._keepalives.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _normalize_auto_exports(self, rules: list[AutoExportRule] | None) -> list[dict[str, Any]]:
        if rules is None:
            return []
        if not 1 <= len(rules) <= 32:
            raise ValueError("export_on_exit must contain between 1 and 32 rules")
        normalized: list[dict[str, Any]] = []
        destinations: set[str] = set()
        for index, rule_value in enumerate(rules):
            rule = (
                rule_value
                if isinstance(rule_value, AutoExportRule)
                else AutoExportRule.model_validate(rule_value)
            )
            destination = str(Path(rule.local_path).expanduser().resolve())
            if destination in destinations:
                raise ValueError("export_on_exit local_path destinations must be unique")
            destinations.add(destination)
            normalized.append(
                {
                    "rule_id": f"export-{index}",
                    "remote_path": rule.remote_path,
                    "local_path": destination,
                    "exit_codes": rule.exit_codes,
                    "overwrite": rule.overwrite,
                    "chunk_size": rule.chunk_size,
                    "max_total_bytes": rule.max_total_bytes,
                    "max_files": rule.max_files,
                    "compression": rule.compression,
                    "compression_min_bytes": rule.compression_min_bytes,
                    "compression_min_savings": rule.compression_min_savings,
                }
            )
        return normalized

    def _update_auto_export(
        self, session: str, process_id: str, **changes: Any
    ) -> dict[str, Any] | None:
        process = self.process_journal.get(session, process_id)
        if process is None or "auto_export" not in process:
            return None
        state = {**process["auto_export"], **changes}
        self.process_journal.update(session, {"process_id": process_id, "auto_export": state})
        return state

    def ensure_process_export_watcher(self, session: str, process_id: str) -> None:
        process = self.process_journal.get(session, process_id)
        state = process.get("auto_export") if process else None
        if not state or state.get("status") in {"completed", "held", "skipped"}:
            return
        key = (session, process_id)
        task = self._export_watchers.get(key)
        if task is None or task.done():
            self._export_watchers[key] = asyncio.create_task(
                self._watch_process_exports(session, process_id)
            )

    async def _watch_process_exports(self, session: str, process_id: str) -> None:
        delay = max(0.1, self.export_poll_seconds)
        while True:
            process = self.process_journal.get(session, process_id)
            state = process.get("auto_export") if process else None
            if not state:
                return
            try:
                status = await self.process_status(process_id, session)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._update_auto_export(
                    session,
                    process_id,
                    status="degraded",
                    last_error=str(error)[:1_000],
                    last_attempt_at=datetime.datetime.now(datetime.UTC).isoformat(),
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if status.get("status") == "running":
                await asyncio.sleep(max(0.1, self.export_poll_seconds))
                continue
            if status.get("status") != "exited":
                self._update_auto_export(
                    session,
                    process_id,
                    status="held",
                    last_error="process state was lost before automatic export",
                    finished_process=status,
                )
                return

            exit_code = status.get("exit_code")
            results = dict(state.get("results") or {})
            pending = False
            for rule in state["rules"]:
                rule_id = rule["rule_id"]
                if results.get(rule_id, {}).get("status") in {"exported", "skipped"}:
                    continue
                exit_codes = rule.get("exit_codes")
                if exit_codes is not None and exit_code not in exit_codes:
                    results[rule_id] = {
                        "status": "skipped",
                        "reason": f"exit_code {exit_code} did not match {exit_codes}",
                    }
                    continue
                try:
                    exported = await self.process_export(
                        process_id,
                        rule["remote_path"],
                        rule["local_path"],
                        session,
                        release_on_success=False,
                        overwrite=bool(rule["overwrite"]),
                        chunk_size=int(rule.get("chunk_size", 524_288)),
                        max_total_bytes=int(rule.get("max_total_bytes", 100_000_000)),
                        max_files=int(rule.get("max_files", 10_000)),
                        compression=str(rule.get("compression", "auto")),
                        compression_min_bytes=int(rule.get("compression_min_bytes", 1_048_576)),
                        compression_min_savings=float(rule.get("compression_min_savings", 0.10)),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    exported = {
                        "exported": False,
                        "error": {"code": "auto_export_error", "message": str(error)[:1_000]},
                    }
                if exported.get("exported"):
                    transfer = exported.get("transfer") or {}
                    results[rule_id] = {
                        "status": "exported",
                        "local_path": exported.get("local_path"),
                        "total_bytes": transfer.get("total_bytes"),
                        "wire_bytes": transfer.get("wire_bytes"),
                        "compression": transfer.get("compression"),
                        "files_transferred": len(transfer.get("files_transferred") or []),
                        "files_skipped": len(transfer.get("files_skipped") or []),
                    }
                else:
                    pending = True
                    results[rule_id] = {
                        "status": "degraded",
                        "error": exported.get("error"),
                        "recoverable_export": exported.get("recoverable_export"),
                    }
            complete = not pending and all(
                item.get("status") in {"exported", "skipped"} for item in results.values()
            )
            self._update_auto_export(
                session,
                process_id,
                status="completed" if complete else "degraded",
                exit_code=exit_code,
                results=results,
                last_error=None if complete else "one or more automatic exports will be retried",
                last_attempt_at=datetime.datetime.now(datetime.UTC).isoformat(),
            )
            if complete:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    async def recover_process_export_watchers(self) -> dict[str, list[str]]:
        recovered: list[str] = []
        for session in self.store.list().values():
            if getattr(session, "keepalive_status", None) == "lease_lost":
                continue
            for process in self.process_journal.list(session.name):
                state = process.get("auto_export")
                if state and state.get("status") not in {"completed", "held", "skipped"}:
                    self.ensure_process_export_watcher(session.name, process["process_id"])
                    recovered.append(process["process_id"])
        return {"recovered_process_ids": recovered}

    async def shutdown_process_export_watchers(self) -> None:
        tasks = list(self._export_watchers.values())
        self._export_watchers.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def keepalive(self, name: str | None, refresh: bool = True) -> dict[str, Any]:
        """Report heartbeat health and optionally send an immediate upstream ping."""
        session = self.resolve(name)
        refreshed = None
        if refresh:
            refreshed = await self._keepalive_once(session)
        current = self.resolve(session.name)
        self.ensure_keepalive(current)
        task = self._keepalives.get(current.name)
        return {
            "session": current.name,
            "status": getattr(current, "keepalive_status", None) or "scheduled",
            "refresh_succeeded": refreshed,
            "last_keepalive_at": getattr(current, "last_keepalive_at", None),
            "last_error": getattr(current, "last_keepalive_error", None),
            "consecutive_failures": getattr(current, "consecutive_keepalive_failures", 0),
            "interval_seconds": self.keepalive_seconds,
            "background_task_running": task is not None and not task.done(),
            "guarantees_runtime_persistence": False,
        }

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
            if not await self._keepalive_once(session):
                raise RuntimeError("initial Colab assignment keep-alive failed")
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
            result.append(
                {
                    **session.model_dump(exclude={"token", "operation_lease_token"}),
                    "active": active,
                }
            )
        return result

    async def _refresh_runtime_proxy(self, session: ManagedSessionState) -> bool:
        """Refresh a rotated runtime-proxy credential without changing ownership."""
        assignments = await asyncio.to_thread(self.client().list_assignments)
        assignment = next((item for item in assignments if item.endpoint == session.endpoint), None)
        if assignment is None:
            return False
        return self._adopt_runtime_proxy(session, assignment)

    def _adopt_runtime_proxy(self, session: ManagedSessionState, assignment: Any) -> bool:
        """Persist the current proxy URL and token returned for an owned assignment."""
        proxy = getattr(assignment, "runtime_proxy_info", None)
        if proxy is None:
            return False
        changed = session.token != proxy.token or session.url != proxy.url
        if changed:
            session.token = proxy.token
            session.url = proxy.url
            self.store.add(session)
        return changed

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
                await self.close_kernel_channel(item["session"])
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

    async def _execute_detailed(
        self,
        code: str,
        name: str | None,
        timeout: float,
        output_limit: int = 100_000,
        connection_timeout: float = 60,
        connection_attempts: int = 2,
    ) -> dict[str, Any]:
        if not 1 <= output_limit <= 3_000_000:
            raise ValueError("output_limit must be between 1 and 3000000")
        session = self.resolve(name)

        overall_started = time.monotonic()

        def connect(connection_abandoned: threading.Event) -> tuple[Any, dict[str, Any]]:
            # Use the Colab-specialized client directly. This avoids the CLI's
            # legacy KernelClient feature-detection path, which is unreliable
            # during circular package initialization on Windows.
            connection_started = time.monotonic()
            attempt_timings: dict[str, Any] = {}
            kernel = self._cached_kernel(session.name)
            reused = kernel is not None
            try:
                if reused:
                    assert kernel is not None
                    # A harmless request verifies the cached websocket before the
                    # caller's operation is submitted.  Failure here is therefore
                    # always safe to reconnect and retry, including process_start.
                    preflight_started = time.monotonic()
                    kernel.execute("import os; os.chdir('/content')", timeout=connection_timeout)
                    attempt_timings.update(
                        {
                            "kernel_connection_seconds": 0.0,
                            "kernel_connection_reused": True,
                            "kernel_preflight_seconds": round(
                                time.monotonic() - preflight_started, 3
                            ),
                        }
                    )
                    return kernel, attempt_timings
                kernel = kernel_client(
                    connection_timeout=connection_timeout,
                    server_url=session.url,
                    proxy_token=session.token,
                    kernel_id=session.kernel_id,
                    client_kwargs={"extra_params": {"colab-runtime-proxy-token": session.token}},
                )
                kernel.start(timeout=connection_timeout)
                attempt_timings["kernel_connection_seconds"] = round(
                    time.monotonic() - connection_started, 3
                )
                attempt_timings["kernel_connection_reused"] = False
                if not session.kernel_id and kernel.id:
                    session.kernel_id = kernel.id
                    self.store.add(session)
                preflight_started = time.monotonic()
                kernel.execute(
                    "import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')",
                    timeout=connection_timeout,
                )
                if connection_abandoned.is_set():
                    with contextlib.suppress(Exception):
                        kernel.stop(shutdown_kernel=False)
                    raise TimeoutError("connection completed after local deadline")
                attempt_timings["kernel_preflight_seconds"] = round(
                    time.monotonic() - preflight_started, 3
                )
                self._cache_kernel(session.name, kernel)
                return kernel, attempt_timings
            except Exception as error:
                if kernel is not None:
                    self._stop_kernel_safely(kernel)
                with self._kernel_clients_guard:
                    if self._kernel_clients.get(session.name) is kernel:
                        self._kernel_clients.pop(session.name, None)
                response = getattr(error, "response", None)
                http_status_code = getattr(response, "status_code", None)
                details = {
                    "kernel_connection_seconds": round(time.monotonic() - connection_started, 3),
                    "connection_timeout_seconds": connection_timeout,
                }
                if isinstance(http_status_code, int):
                    details["http_status_code"] = http_status_code
                raise KernelConnectionError(
                    "Kernel connection failed before the requested operation was sent",
                    details,
                ) from error

        def submit(
            kernel: Any, attempt_timings: dict[str, Any]
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            request_started = time.monotonic()
            try:
                reply = kernel.execute(code, timeout=timeout)
            except TimeoutError as error:
                raise RequestOutcomeUnknownError(
                    "operation_timed_out_submission_outcome_unknown",
                    "The kernel adapter timed out after its execute call began.",
                    {
                        "phase": "request_submission_to_output_retrieval",
                        "request_submission": "outcome_unknown",
                        "elapsed_seconds": round(time.monotonic() - request_started, 3),
                        "timeout_seconds": timeout,
                    },
                ) from error
            except OSError as error:
                raise RequestOutcomeUnknownError(
                    "request_submission_outcome_unknown_response_lost",
                    "The kernel channel failed after its execute call began.",
                    {
                        "phase": "request_submission_to_output_retrieval",
                        "request_submission": "outcome_unknown",
                        "elapsed_seconds": round(time.monotonic() - request_started, 3),
                    },
                ) from error
            attempt_timings["request_submission_to_output_retrieval_seconds"] = round(
                time.monotonic() - request_started, 3
            )
            attempt_timings["request_submission_confirmed"] = reply is not None
            return (reply.get("outputs", []) if reply else []), attempt_timings

        if connection_attempts not in {1, 2}:
            raise ValueError("connection_attempts must be 1 or 2")
        attempts: list[dict[str, Any]] = []
        async with self._kernel_lock(session.name):
            for attempt in range(connection_attempts):
                connection_abandoned = threading.Event()
                try:
                    # Upstream connection setup spans HTTP model lookup, websocket startup,
                    # and a preflight request. Some layers ignore their nominal timeout, so
                    # enforce one bounded deadline around the complete pre-submission phase.
                    try:
                        kernel, attempt_timing = await asyncio.wait_for(
                            asyncio.to_thread(connect, connection_abandoned),
                            timeout=connection_timeout + 1,
                        )
                    except TimeoutError as error:
                        connection_abandoned.set()
                        await self._close_kernel_channel_unlocked(session.name)
                        raise KernelConnectionError(
                            "Kernel connection exceeded the bounded local deadline before submission",
                            {
                                "kernel_connection_seconds": round(connection_timeout + 1, 3),
                                "connection_timeout_seconds": connection_timeout,
                                "local_deadline_seconds": connection_timeout + 1,
                            },
                        ) from error
                    try:
                        raw_outputs, attempt_timing = await asyncio.to_thread(
                            submit, kernel, attempt_timing
                        )
                    except BaseException:
                        await self._close_kernel_channel_unlocked(session.name)
                        raise
                    attempts.append({"attempt": attempt + 1, **attempt_timing})
                    output_started = time.monotonic()
                    outputs = _bound_outputs(_json_safe(raw_outputs), output_limit)
                    output_processing_seconds = round(time.monotonic() - output_started, 3)
                    return {
                        "outputs": outputs,
                        "timings": {
                            "assignment_lookup_seconds": None,
                            "fingerprint_validation_seconds": None,
                            "attempts": attempts,
                            "retries": attempt,
                            "retry_backoff_seconds": 0.0,
                            "local_output_processing_seconds": output_processing_seconds,
                            "total_seconds": round(time.monotonic() - overall_started, 3),
                            "upstream_limitation": (
                                "The pinned kernel client exposes submission, remote execution, and "
                                "I/O collection as one synchronous interval. That combined interval is "
                                "reported without inventing a split."
                            ),
                        },
                    }
                except KernelConnectionError as error:
                    if attempt + 1 >= connection_attempts:
                        error.details["retries"] = attempt
                        raise
                    proxy_refreshed = False
                    if error.details.get("http_status_code") in {401, 403, 404}:
                        proxy_refreshed = await self._refresh_runtime_proxy(session)
                    logger.warning(
                        "kernel_connection_retry session=%s operation_not_sent=true "
                        "preserve_kernel_id=true proxy_refreshed=%s",
                        session.name,
                        proxy_refreshed,
                    )
        raise AssertionError("unreachable")

    async def execute(
        self,
        code: str,
        name: str | None,
        timeout: float,
        output_limit: int = 100_000,
        connection_timeout: float = 60,
        connection_attempts: int = 2,
    ) -> list[dict[str, Any]]:
        result = await self._execute_detailed(
            code,
            name,
            timeout,
            output_limit,
            connection_timeout,
            connection_attempts,
        )
        return result["outputs"]

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

    async def execute_python_detailed(
        self,
        code: str,
        name: str | None,
        timeout: float = 900,
        output_limit: int = 100_000,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute Python and expose honest control-plane/kernel phase timings."""
        if not code or len(code.encode("utf-8")) > 1_000_000:
            raise ValueError("code must contain between 1 and 1000000 UTF-8 bytes")
        timeout = validate_timeout(timeout)
        output_limit = validate_output_limit(output_limit)
        session, lease = await self._operation_lease(name, lease_token)
        result = await self._execute_detailed(
            _guarded_python_source(
                code, str(session.runtime_fingerprint), str(lease["lease_token"])
            ),
            session.name,
            timeout,
            output_limit,
            connection_timeout=LEASED_CONNECTION_TIMEOUT_SECONDS,
            connection_attempts=1,
        )
        control_timings = _extract_control_timing(result["outputs"])
        result["timings"]["assignment_lookup_seconds"] = lease["assignment_lookup_seconds"]
        result["timings"].update(control_timings)
        result["lease"] = lease
        return result

    async def _remote_operation(
        self,
        operation: str,
        payload: dict[str, Any],
        name: str | None,
        timeout: float = 120,
        lease_token: str | None = None,
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
        if lease_token is not None:
            self._validate_operation_lease(session, lease_token)
        payload = {
            **payload,
            "runtime_fingerprint": fingerprint,
            **({"operation_lease_token": lease_token} if lease_token else {}),
        }
        code = build_remote_code(operation, payload)
        try:
            try:
                if lease_token:
                    outputs = await self.execute(
                        code,
                        name,
                        timeout,
                        output_limit=3_000_000,
                        connection_timeout=LEASED_CONNECTION_TIMEOUT_SECONDS,
                        connection_attempts=1,
                    )
                else:
                    outputs = await self.execute(code, name, timeout, output_limit=3_000_000)
            except KernelConnectionError as error:
                # Connection/preflight failed before this operation was sent. It is
                # safe to retry every operation with the same lease and fingerprint.
                proxy_refreshed = False
                if error.details.get("http_status_code") in {401, 403, 404}:
                    proxy_refreshed = await self._refresh_runtime_proxy(session)
                logger.warning(
                    "kernel_reconnect_retry operation=%s session=%s operation_not_sent=true "
                    "proxy_refreshed=%s",
                    operation,
                    name,
                    proxy_refreshed,
                )
                if lease_token:
                    await self._operation_lease(session.name, lease_token)
                    outputs = await self.execute(
                        code,
                        name,
                        timeout,
                        output_limit=3_000_000,
                        connection_timeout=LEASED_CONNECTION_TIMEOUT_SECONDS,
                        connection_attempts=1,
                    )
                else:
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
                    "lease_probe",
                    # Offset-checked chunks are idempotent. A confirmed
                    # pre-submission failure can safely reconnect and retry
                    # with the same runtime lease and payload.
                    "transfer_upload_chunk",
                }
                if operation not in retryable:
                    raise
                logger.warning(
                    "kernel_reconnect_retry operation=%s session=%s preserve_kernel_id=true",
                    operation,
                    name,
                )
                # A pre-submission failure must not consume or replace the lease. Recheck
                # local token validity and assignment ownership, while retaining the exact
                # token. The retried remote request atomically verifies its fingerprint.
                if lease_token:
                    await self._operation_lease(session.name, lease_token)
                if lease_token:
                    outputs = await self.execute(
                        code,
                        name,
                        timeout,
                        output_limit=3_000_000,
                        connection_timeout=LEASED_CONNECTION_TIMEOUT_SECONDS,
                        connection_attempts=1,
                    )
                else:
                    outputs = await self.execute(code, name, timeout, output_limit=3_000_000)
            return parse_remote_result(outputs)
        except RuntimeReplacedError as error:
            await self.close_kernel_channel(session.name)
            session.runtime_replaced_at = datetime.datetime.now(datetime.UTC).isoformat()
            session.runtime_replaced_reason = str(error)
            self.store.add(session)
            logger.warning("runtime_replaced session=%s operation=%s", session.name, operation)
            raise self._replacement_error(session, str(error), payload.get("process_id")) from error

    async def allocation_probe(
        self,
        name: str | None,
        observations: int = 2,
        interval: float = 0.25,
    ) -> dict[str, Any]:
        """Verify that an owned assignment and its runtime incarnation remain stable."""
        if not 2 <= observations <= 5:
            raise ValueError("observations must be between 2 and 5")
        if not 0 <= interval <= 5:
            raise ValueError("interval must be between 0 and 5 seconds")
        session = self.resolve(name)
        observed: list[str] = []
        for index in range(observations):
            assignments = await asyncio.to_thread(self.client().list_assignments)
            assignment = next(
                (item for item in assignments if item.endpoint == session.endpoint), None
            )
            current = self.store.get(session.name)
            if current is None or current.endpoint != session.endpoint:
                raise RuntimeError(
                    "allocation_lease_changed: local session ownership changed during probe"
                )
            if assignment is None:
                raise RuntimeError(
                    "allocation_lease_lost: the tracked Colab assignment is no longer active"
                )
            self._adopt_runtime_proxy(session, assignment)
            observed.append(session.endpoint)
            if index + 1 < observations and interval:
                await asyncio.sleep(interval)
        lease_token = uuid.uuid4().hex
        issued_at = datetime.datetime.now(datetime.UTC)
        expires_at = issued_at + datetime.timedelta(hours=1)
        incarnation = await self._remote_operation(
            "lease_probe",
            {
                "issue_lease_token": lease_token,
                "lease_expires_at": expires_at.isoformat(),
            },
            session.name,
        )
        session.operation_lease_token = lease_token
        session.operation_lease_issued_at = issued_at.isoformat()
        session.operation_lease_expires_at = expires_at.isoformat()
        self.store.add(session)
        lease_pool = self._operation_leases.setdefault(session.name, {})
        now = datetime.datetime.now(datetime.UTC)
        retained = {
            token: expiry
            for token, expiry in lease_pool.items()
            if datetime.datetime.fromisoformat(expiry) > now
        }
        lease_pool.clear()
        lease_pool.update(dict(list(retained.items())[-7:]))
        lease_pool[lease_token] = expires_at.isoformat()
        return {
            "status": "stable",
            "session": session.name,
            "endpoint": session.endpoint,
            "observations": len(observed),
            "runtime_fingerprint": incarnation["runtime_fingerprint"],
            "observed_at": incarnation["observed_at"],
            "lease_token": lease_token,
            "lease_expires_at": expires_at.isoformat(),
            "heartbeat": "background",
        }

    def _validate_operation_lease(self, session: SessionState, lease_token: str) -> None:
        expected = getattr(session, "operation_lease_token", None)
        expires_at = getattr(session, "operation_lease_expires_at", None)
        if lease_token != expected:
            expires_at = self._operation_leases.get(session.name, {}).get(lease_token)
        if not expires_at:
            raise OperationLeaseError(
                "operation_lease_stale",
                "The opaque lease does not match the latest probe for this session.",
                {"session": session.name},
            )
        if not expires_at or datetime.datetime.fromisoformat(expires_at) <= datetime.datetime.now(
            datetime.UTC
        ):
            raise OperationLeaseError(
                "operation_lease_expired",
                "The operation lease expired; call colab_allocation_probe again.",
                {"session": session.name, "lease_expires_at": expires_at},
            )

    async def _operation_lease(
        self, name: str | None, lease_token: str | None
    ) -> tuple[SessionState, dict[str, Any]]:
        """Resolve an explicit lease or issue one, then recheck assignment ownership quickly."""
        session = self.resolve(name)
        if lease_token is None:
            lease = await self.allocation_probe(session.name)
            lease_token = lease["lease_token"]
        else:
            self._validate_operation_lease(session, lease_token)
            lease = {
                "status": "accepted",
                "session": session.name,
                "endpoint": session.endpoint,
                "runtime_fingerprint": getattr(session, "runtime_fingerprint", None),
                "lease_token": lease_token,
                "lease_expires_at": getattr(session, "operation_lease_expires_at", None),
            }
        started = time.monotonic()
        try:
            assignments = await asyncio.wait_for(
                asyncio.to_thread(self.client().list_assignments), timeout=5
            )
        except TimeoutError as error:
            raise OperationLeaseError(
                "assignment_lookup_timed_out",
                "Colab assignment lookup did not complete within five seconds.",
                {"session": session.name, "elapsed_seconds": round(time.monotonic() - started, 3)},
            ) from error
        assignment = next((item for item in assignments if item.endpoint == session.endpoint), None)
        if assignment is None:
            raise OperationLeaseError(
                "assignment_no_longer_exists",
                "The tracked Colab assignment is no longer active.",
                {"session": session.name, "elapsed_seconds": round(time.monotonic() - started, 3)},
            )
        self._adopt_runtime_proxy(session, assignment)
        lease["assignment_lookup_seconds"] = round(time.monotonic() - started, 3)
        return session, lease

    async def _critical_heartbeat(self, session: SessionState, stop: asyncio.Event) -> None:
        while not stop.is_set():
            with contextlib.suppress(Exception):
                await self._keepalive_once(session)
            try:
                await asyncio.wait_for(stop.wait(), timeout=min(self.keepalive_seconds, 15))
            except TimeoutError:
                pass

    def _replacement_error(
        self, session: SessionState, reason: str, process_id: str | None = None
    ) -> RuntimeReplacedError:
        last_known = self.process_journal.get(session.name, process_id) if process_id else None
        probable_cause = (
            "runtime_incarnation_marker_missing_after_backend_reset_or_reclamation"
            if "observed missing" in reason
            else "endpoint_now_points_to_a_different_runtime_incarnation"
        )
        details = {
            "code": "runtime_replaced",
            "session": session.name,
            "process_id": process_id,
            "probable_cause": probable_cause,
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
                    "runtime_incarnation_changed; process OOM is not inferred"
                    if code == "runtime_replaced"
                    else "process_state_missing; inspect runtime memory and system logs for OOM evidence"
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
        export_on_exit: list[AutoExportRule] | None = None,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        """Start a detached process whose metadata and logs live on the runtime."""
        validate_argv(argv)
        environment = validate_environment(environment)
        output_limit = validate_process_output_limit(output_limit)
        export_rules = self._normalize_auto_exports(export_on_exit)
        operation_started = time.monotonic()
        session, lease = await self._operation_lease(name, lease_token)
        submission_started = time.monotonic()
        result = await self._remote_operation(
            "process_start",
            {
                "process_id": uuid.uuid4().hex,
                "argv": argv,
                "cwd": cwd,
                "environment": environment,
                "output_limit": output_limit,
            },
            session.name,
            lease_token=lease["lease_token"],
        )
        result["timings"] = {
            "assignment_lookup_seconds": lease["assignment_lookup_seconds"],
            "request_and_remote_start_seconds": round(time.monotonic() - submission_started, 3),
            "total_seconds": round(time.monotonic() - operation_started, 3),
        }
        journaled = self.process_journal.update(
            session.name,
            {
                **result,
                "session": session.name,
                "runtime_fingerprint": getattr(session, "runtime_fingerprint", None),
                **(
                    {
                        "auto_export": {
                            "status": "watching",
                            "rules": export_rules,
                            "results": {},
                            "configured_at": datetime.datetime.now(datetime.UTC).isoformat(),
                        }
                    }
                    if export_rules
                    else {}
                ),
            },
        )
        if export_rules:
            self.ensure_process_export_watcher(session.name, result["process_id"])
        logger.info("process_started session=%s process_id=%s", name, result["process_id"])
        return {
            **result,
            **({"auto_export": journaled["auto_export"]} if export_rules else {}),
        }

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
                f"{stream}_stored_bytes": result["stored_bytes"],
                f"{stream}_total_bytes": result["total_bytes"],
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

    async def _binary_upload_file(
        self,
        source: Path,
        remote_path: str,
        offset: int,
        session: ManagedSessionState,
        lease_token: str,
        timeout: float = 900,
    ) -> dict[str, Any]:
        """Upload through one native-buffer websocket stream under the operation lease."""

        def connect_and_upload() -> dict[str, Any]:
            kernel = self._cached_kernel(session.name)
            try:
                if kernel is None:
                    kernel = kernel_client(
                        connection_timeout=LEASED_CONNECTION_TIMEOUT_SECONDS,
                        server_url=session.url,
                        proxy_token=session.token,
                        kernel_id=session.kernel_id,
                        client_kwargs={
                            "extra_params": {"colab-runtime-proxy-token": session.token}
                        },
                    )
                    kernel.start(timeout=LEASED_CONNECTION_TIMEOUT_SECONDS)
                    if not session.kernel_id and kernel.id:
                        session.kernel_id = kernel.id
                        self.store.add(session)
                    self._cache_kernel(session.name, kernel)
                return binary_upload(
                    kernel,
                    source,
                    remote_path,
                    offset,
                    session.runtime_fingerprint or "",
                    lease_token,
                    timeout,
                )
            except BaseException:
                if kernel is not None:
                    self._stop_kernel_safely(kernel)
                with self._kernel_clients_guard:
                    if self._kernel_clients.get(session.name) is kernel:
                        self._kernel_clients.pop(session.name, None)
                raise

        async with self._kernel_lock(session.name):
            return await asyncio.to_thread(connect_and_upload)

    async def _raw_download_file(
        self,
        session: ManagedSessionState,
        remote_path: str,
        destination: Path,
        codec: str,
        content_bytes: int,
        wire_bytes: int,
        wire_checksum: str,
        content_checksum: str,
        max_total_bytes: int,
        chunk_size: int,
    ) -> None:
        for attempt in range(2):
            try:
                await asyncio.to_thread(
                    _raw_download_to_file,
                    session,
                    remote_path,
                    destination,
                    codec,
                    content_bytes,
                    wire_bytes,
                    wire_checksum,
                    content_checksum,
                    max_total_bytes,
                    chunk_size,
                )
                return
            except requests.RequestException:
                if attempt:
                    raise
                refreshed = await self._refresh_runtime_proxy(session)
                logger.warning(
                    "raw_download_retry session=%s proxy_refreshed=%s",
                    session.name,
                    refreshed,
                )

    async def _remote_stat_or_none(
        self,
        path: str,
        name: str | None,
        checksum: bool = False,
        lease_token: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            if lease_token:
                return await self._remote_operation(
                    "fs_stat",
                    {"path": path, "checksum": checksum},
                    name,
                    lease_token=lease_token,
                )
            return await self.filesystem_stat(path, name, checksum)
        except RemoteOperationError as error:
            if str(error).startswith("FileNotFoundError:"):
                return None
            raise

    async def workspace_sync_selection(
        self,
        direction: Literal["push", "pull"],
        local_folder: str,
        remote_folder: str,
        name: str | None,
        include: list[str] | None = None,
        max_files: int = 10_000,
        max_total_bytes: int = 1_000_000_000,
        lease_token: str | None = None,
    ) -> tuple[list[str], list[str], str, dict[str, Any]]:
        """Select the safe source set and changed paths for one workspace sync."""
        normalized_include = tuple(dict.fromkeys(include or ()))
        if len(normalized_include) > 100:
            raise ValueError("include accepts at most 100 patterns")
        session, lease = await self._operation_lease(name, lease_token)
        lease_token = lease["lease_token"]
        local_root = Path(local_folder).expanduser().resolve()
        if direction == "push" and not local_root.is_dir():
            raise ValueError("local_folder must be an existing directory for push")

        remote = await self._remote_operation(
            "workspace_manifest",
            {
                "path": remote_folder,
                "max_files": max_files,
                "max_total_bytes": 10_000_000_000,
                "include": list(normalized_include),
                "exclude": list(SYNC_ALWAYS_EXCLUDED),
            },
            session.name,
            lease_token=lease_token,
        )
        if "files" not in remote:
            raise ValueError("remote_folder must be an existing directory")
        remote_manifest = remote["files"]
        local_candidates = (
            sorted(
                path for path in local_root.rglob("*") if path.is_file() and not path.is_symlink()
            )
            if local_root.is_dir()
            else []
        )
        local_files: list[Path] = []
        for path in local_candidates:
            relative = PurePosixPath(*path.relative_to(local_root).parts).as_posix()
            accepted, reason = _sync_selected(relative, normalized_include)
            if accepted:
                local_files.append(path)
        local_manifest = await asyncio.to_thread(_workspace_manifest, local_root, local_files)
        source_all = local_manifest if direction == "push" else remote_manifest
        destination = remote_manifest if direction == "push" else local_manifest
        selected = source_all
        if not selected:
            raise ValueError(
                "Sync selection is empty; adjust local_folder, remote_folder, or include"
            )
        if len(selected) > max_files:
            raise ValueError(
                f"Sync selection contains {len(selected)} files; max_files is {max_files}"
            )
        selected_bytes = sum(int(item["size"]) for item in selected)
        if selected_bytes > max_total_bytes:
            raise ValueError(
                f"Sync selection contains {selected_bytes} bytes; max_total_bytes is {max_total_bytes}"
            )
        destination_hashes = {item["path"]: item.get("sha256") for item in destination}
        changed_paths = [
            item["path"]
            for item in selected
            if destination_hashes.get(item["path"]) != item["sha256"]
        ]
        return [item["path"] for item in selected], changed_paths, session.name, lease

    async def workspace_upload(
        self,
        local_folder: str,
        remote_folder: str,
        name: str | None,
        chunk_size: int = 2_000_000,
        max_total_bytes: int = 1_000_000_000,
        max_files: int = 10_000,
        compression: str = "auto",
        compression_min_savings: float = 0.10,
        lease_token: str | None = None,
        progress: ProgressCallback | None = None,
        selected_paths: set[str] | None = None,
        changed_paths: set[str] | None = None,
    ) -> dict[str, Any]:
        """Push a directory delta as one verified bundle instead of per-file RPCs."""
        _transfer_bounds(chunk_size, max_total_bytes, max_files)
        compression, _, compression_min_savings = _compression_settings(
            compression, 0, compression_min_savings
        )
        root = Path(local_folder).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("local_folder must be an existing directory for push")
        files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
        if selected_paths is not None:
            files = [
                path
                for path in files
                if PurePosixPath(*path.relative_to(root).parts).as_posix() in selected_paths
            ]
        if len(files) > max_files:
            raise ValueError(f"Transfer contains {len(files)} files; max_files is {max_files}")
        total = sum(path.stat().st_size for path in files)
        if total > max_total_bytes:
            raise ValueError(
                f"Transfer contains {total} bytes; max_total_bytes is {max_total_bytes}"
            )
        operation_started = time.monotonic()
        session, lease = await self._operation_lease(name, lease_token)
        lease_token = lease["lease_token"]
        local_manifest = await asyncio.to_thread(_workspace_manifest, root, files)
        local_by_relative = {
            PurePosixPath(*path.relative_to(root).parts).as_posix(): path for path in files
        }
        heartbeat_stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._critical_heartbeat(session, heartbeat_stop))
        bundle: Path | None = None
        staging_path: str | None = None
        staged_offset = 0
        chunks_completed = 0
        progress_count = 0
        try:
            manifest_started = time.monotonic()
            if changed_paths is None:
                remote = await self._remote_operation(
                    "workspace_manifest",
                    {
                        "path": remote_folder,
                        "max_files": max_files,
                        "max_total_bytes": 10_000_000_000,
                        "include": sorted(selected_paths) if selected_paths is not None else [],
                        "exclude": list(SYNC_ALWAYS_EXCLUDED),
                    },
                    session.name,
                    lease_token=lease_token,
                )
                remote_by_relative = {item["path"]: item for item in remote["files"]}
                changed = [
                    item
                    for item in local_manifest
                    if remote_by_relative.get(item["path"], {}).get("sha256") != item["sha256"]
                ]
            else:
                changed = [item for item in local_manifest if item["path"] in changed_paths]
            skipped = [
                {
                    "local_path": str(local_by_relative[item["path"]]),
                    "remote_path": str(PurePosixPath(remote_folder) / item["path"]),
                    "sha256": item["sha256"],
                }
                for item in local_manifest
                if item not in changed
            ]
            manifest_seconds = time.monotonic() - manifest_started
            if not changed:
                return {
                    "transfer_id": None,
                    "files_transferred": [],
                    "files_skipped": skipped,
                    "total_bytes": total,
                    "wire_bytes": 0,
                    "compression": compression,
                    "lease": lease,
                    "progress_events_emitted": 0,
                    "timings": {
                        "assignment_lookup_seconds": lease["assignment_lookup_seconds"],
                        "manifest_seconds": round(manifest_seconds, 3),
                        "total_seconds": round(time.monotonic() - operation_started, 3),
                    },
                    "strategy": "verified_bundle_delta",
                    "staging_cleanup": "no staging required",
                }
            bundle, codec, wire_bytes, wire_checksum = await asyncio.to_thread(
                _build_workspace_bundle,
                root,
                local_by_relative,
                changed,
                compression,
                compression_min_savings,
            )
            assert bundle is not None
            logical_manifest = json.dumps(changed, sort_keys=True, separators=(",", ":"))
            transfer_id = hashlib.sha256(
                (remote_folder + "\0" + logical_manifest + "\0" + wire_checksum).encode()
            ).hexdigest()[:32]
            suffix = ".tar.gz" if codec == "gzip" else ".tar"
            staging_path = f"/content/.colab-mcp/workspace.colab-mcp-wire-{transfer_id}{suffix}"
            staged = await self._remote_stat_or_none(
                staging_path,
                session.name,
                checksum=True,
                lease_token=lease_token,
            )
            if staged:
                staged_offset = int(staged["size"])
                if staged_offset > wire_bytes:
                    raise TransferError(
                        "transfer_resume_conflict",
                        "Remote workspace bundle is larger than this upload.",
                        {"transfer_id": transfer_id, "staging_path": staging_path},
                    )
                prefix_digest = hashlib.sha256()
                with bundle.open("rb") as handle:
                    remaining = staged_offset
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        prefix_digest.update(chunk)
                        remaining -= len(chunk)
                if prefix_digest.hexdigest() != staged["sha256"]:
                    raise TransferError(
                        "transfer_resume_conflict",
                        "Remote workspace bundle differs from this source.",
                        {"transfer_id": transfer_id, "staging_path": staging_path},
                    )
            transfer_started = time.monotonic()
            resumed_from = staged_offset
            result = await self._binary_upload_file(
                bundle, staging_path, staged_offset, session, lease_token
            )
            staged_offset = int(result["offset"])
            if staged_offset != wire_bytes or result["sha256"] != wire_checksum:
                raise RuntimeError("Binary workspace upload checksum mismatch")
            chunks_completed = 1
            progress_count = 1
            if progress is not None:
                await progress(
                    {
                        "phase": "bundle_transfer",
                        "transfer_id": transfer_id,
                        "bytes_sent": staged_offset,
                        "total_bytes": wire_bytes,
                        "chunk_number": chunks_completed,
                        "elapsed_seconds": round(time.monotonic() - operation_started, 3),
                        "resumed": resumed_from > 0,
                    }
                )
            transfer_seconds = time.monotonic() - transfer_started
            publication_started = time.monotonic()
            published = await self._remote_operation(
                "workspace_bundle_publish",
                {
                    "archive_path": staging_path,
                    "workspace_path": remote_folder,
                    "wire_sha256": wire_checksum,
                    "compression": codec,
                    "transfer_id": transfer_id,
                    "max_files": max_files,
                    "max_total_bytes": max_total_bytes,
                },
                session.name,
                lease_token=lease_token,
            )
            publication_seconds = time.monotonic() - publication_started
            transferred = [
                {
                    "local_path": str(local_by_relative[item["path"]]),
                    "remote_path": str(PurePosixPath(remote_folder) / item["path"]),
                    "size": item["size"],
                    "sha256": item["sha256"],
                    "compression": codec,
                }
                for item in published["files"]
            ]
            return {
                "transfer_id": transfer_id,
                "files_transferred": transferred,
                "files_skipped": skipped,
                "total_bytes": total,
                "wire_bytes": wire_bytes,
                "compression": codec,
                "lease": lease,
                "progress_events_emitted": progress_count,
                "timings": {
                    "assignment_lookup_seconds": lease["assignment_lookup_seconds"],
                    "manifest_seconds": round(manifest_seconds, 3),
                    "data_transfer_seconds": round(transfer_seconds, 3),
                    "publication_seconds": round(publication_seconds, 3),
                    "total_seconds": round(time.monotonic() - operation_started, 3),
                },
                "strategy": "verified_bundle_delta",
                "staging_cleanup": "published bundle removed; failed bundle preserved for resume",
            }
        except Exception as error:
            if isinstance(error, (TransferError, OperationLeaseError, RuntimeReplacedError)):
                raise
            code = getattr(error, "code", "transfer_failed_staging_preserved")
            raise TransferError(
                code,
                str(error),
                {
                    "session": session.name,
                    "runtime_fingerprint": getattr(session, "runtime_fingerprint", None),
                    "staging_path": staging_path,
                    "staged_bytes": staged_offset,
                    "chunks_completed": chunks_completed,
                    "request_submission": (
                        "not_submitted"
                        if isinstance(error, KernelConnectionError)
                        else "unknown_after_error"
                    ),
                    "safe_to_resume": staging_path is not None,
                    "resume_requires_same_incarnation": True,
                },
            ) from error
        finally:
            if bundle is not None:
                bundle.unlink(missing_ok=True)
            heartbeat_stop.set()
            await asyncio.gather(heartbeat, return_exceptions=True)

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
        compression: str = "auto",
        compression_min_bytes: int = 1_048_576,
        compression_min_savings: float = 0.10,
        lease_token: str | None = None,
        transfer_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Upload resumable files under one operation-bound runtime lease."""
        _transfer_bounds(chunk_size, max_total_bytes, max_files)
        compression, compression_min_bytes, compression_min_savings = _compression_settings(
            compression, compression_min_bytes, compression_min_savings
        )
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
        operation_started = time.monotonic()
        session, lease = await self._operation_lease(name, lease_token)
        lease_token = lease["lease_token"]
        if transfer_id is not None and (
            len(transfer_id) != 32 or any(char not in "0123456789abcdef" for char in transfer_id)
        ):
            raise ValueError("transfer_id must be 32 lowercase hexadecimal characters")
        transfer_id = transfer_id or uuid.uuid4().hex
        phase_timings = {"assignment_lookup_seconds": lease["assignment_lookup_seconds"]}
        progress_count = 0

        async def report(event: dict[str, Any]) -> None:
            nonlocal progress_count
            progress_count += 1
            if progress is not None:
                await progress(event)

        heartbeat_stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._critical_heartbeat(session, heartbeat_stop))
        remote_root = PurePosixPath(remote_path)
        transferred: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        try:
            for source_file in files:
                relative = (
                    PurePosixPath(source_file.name)
                    if source.is_file()
                    else PurePosixPath(*source_file.relative_to(source).parts)
                )
                destination = str(remote_root if source.is_file() else remote_root / relative)
                checksum = await asyncio.to_thread(_file_sha256, source_file)
                existing = await self._remote_stat_or_none(
                    destination, session.name, checksum=True, lease_token=lease_token
                )
                if existing and sync and existing.get("sha256") == checksum:
                    skipped.append(
                        {
                            "local_path": str(source_file),
                            "remote_path": destination,
                            "sha256": checksum,
                        }
                    )
                    continue
                if existing and not overwrite:
                    raise FileExistsError(f"Remote destination exists: {destination}")
                file_transfer_id = hashlib.sha256(
                    f"{transfer_id}:{relative.as_posix()}".encode()
                ).hexdigest()[:32]
                wire_temporary = destination + ".colab-mcp-wire-" + file_transfer_id
                content_temporary = destination + ".colab-mcp-part-" + file_transfer_id
                local_compressed: Path | None = None
                staged_offset = 0
                chunks_completed = 0
                try:
                    content_bytes = source_file.stat().st_size
                    wire_source = source_file
                    wire_bytes = content_bytes
                    wire_checksum = checksum
                    codec = "none"
                    if compression == "gzip" or (
                        compression == "auto"
                        and content_bytes >= compression_min_bytes
                        and _auto_compression_candidate(source_file)
                    ):
                        candidate, candidate_bytes, candidate_checksum = await asyncio.to_thread(
                            _gzip_local_file, source_file
                        )
                        if _use_compressed_wire(
                            compression,
                            content_bytes,
                            candidate_bytes,
                            compression_min_savings,
                        ):
                            local_compressed = candidate
                            wire_source = candidate
                            wire_bytes = candidate_bytes
                            wire_checksum = candidate_checksum
                            codec = "gzip"
                        else:
                            candidate.unlink(missing_ok=True)
                    connection_started = time.monotonic()
                    staged = await self._remote_stat_or_none(
                        wire_temporary,
                        session.name,
                        checksum=True,
                        lease_token=lease_token,
                    )
                    phase_timings["connection_and_staging_seconds"] = round(
                        phase_timings.get("connection_and_staging_seconds", 0.0)
                        + time.monotonic()
                        - connection_started,
                        3,
                    )
                    if staged:
                        staged_offset = int(staged["size"])
                        if staged_offset > wire_bytes:
                            raise TransferError(
                                "transfer_resume_conflict",
                                "Remote staging file is larger than this upload.",
                                {
                                    "transfer_id": transfer_id,
                                    "staging_path": wire_temporary,
                                    "staged_bytes": staged_offset,
                                    "wire_bytes": wire_bytes,
                                },
                            )
                        prefix_digest = hashlib.sha256()
                        with wire_source.open("rb") as prefix_handle:
                            remaining = staged_offset
                            while remaining:
                                prefix = prefix_handle.read(min(1024 * 1024, remaining))
                                prefix_digest.update(prefix)
                                remaining -= len(prefix)
                        if prefix_digest.hexdigest() != staged.get("sha256"):
                            raise TransferError(
                                "transfer_resume_conflict",
                                "Remote staging bytes do not match this local source.",
                                {
                                    "transfer_id": transfer_id,
                                    "staging_path": wire_temporary,
                                    "staged_bytes": staged_offset,
                                },
                            )
                    transfer_started = time.monotonic()
                    resumed_from = staged_offset
                    result = await self._binary_upload_file(
                        wire_source, wire_temporary, staged_offset, session, lease_token
                    )
                    staged_offset = int(result["offset"])
                    if staged_offset != wire_bytes or result["sha256"] != wire_checksum:
                        raise RuntimeError(f"Wire checksum mismatch while uploading {source_file}")
                    chunks_completed += 1
                    await report(
                        {
                            "phase": "transfer",
                            "transfer_id": transfer_id,
                            "file": relative.as_posix(),
                            "bytes_sent": staged_offset,
                            "total_bytes": wire_bytes,
                            "chunk_number": chunks_completed,
                            "chunk_seconds": round(float(result["seconds"]), 3),
                            "elapsed_seconds": round(time.monotonic() - operation_started, 3),
                            "resumed": resumed_from > 0,
                        }
                    )
                    phase_timings["data_transfer_seconds"] = round(
                        phase_timings.get("data_transfer_seconds", 0.0)
                        + time.monotonic()
                        - transfer_started,
                        3,
                    )
                    verify_started = time.monotonic()
                    remote_stat = await self._remote_operation(
                        "fs_stat",
                        {"path": wire_temporary, "checksum": True},
                        session.name,
                        lease_token=lease_token,
                    )
                    if remote_stat.get("sha256") != wire_checksum:
                        raise RuntimeError(f"Wire checksum mismatch while uploading {source_file}")
                    phase_timings["verification_seconds"] = round(
                        phase_timings.get("verification_seconds", 0.0)
                        + time.monotonic()
                        - verify_started,
                        3,
                    )
                    publication_started = time.monotonic()
                    publication_source = wire_temporary
                    if codec == "gzip":
                        await self._remote_operation(
                            "fs_gzip_decompress",
                            {
                                "source": wire_temporary,
                                "destination": content_temporary,
                                "expected_size": content_bytes,
                                "expected_sha256": checksum,
                                "max_output_bytes": max_total_bytes,
                            },
                            session.name,
                            lease_token=lease_token,
                        )
                        publication_source = content_temporary
                    await self._remote_operation(
                        "fs_move",
                        {
                            "source": publication_source,
                            "destination": destination,
                            "overwrite": overwrite,
                        },
                        session.name,
                        lease_token=lease_token,
                    )
                    phase_timings["publication_seconds"] = round(
                        phase_timings.get("publication_seconds", 0.0)
                        + time.monotonic()
                        - publication_started,
                        3,
                    )
                    transferred.append(
                        {
                            "local_path": str(source_file),
                            "remote_path": destination,
                            "size": content_bytes,
                            "sha256": checksum,
                            "compression": codec,
                            "content_bytes": content_bytes,
                            "wire_bytes": wire_bytes,
                            "wire_sha256": wire_checksum,
                            "wire_ratio": round(wire_bytes / max(1, content_bytes), 6),
                            "resumed_from_bytes": int(staged.get("size", 0)) if staged else 0,
                        }
                    )
                    with contextlib.suppress(Exception):
                        await self._remote_operation(
                            "fs_remove",
                            {"path": wire_temporary, "recursive": False, "missing_ok": True},
                            session.name,
                            lease_token=lease_token,
                        )
                except Exception as error:
                    if isinstance(
                        error, (TransferError, OperationLeaseError, RuntimeReplacedError)
                    ):
                        raise
                    code = getattr(error, "code", "transfer_failed_staging_preserved")
                    raise TransferError(
                        code,
                        str(error),
                        {
                            "transfer_id": transfer_id,
                            "session": session.name,
                            "runtime_fingerprint": getattr(session, "runtime_fingerprint", None),
                            "staging_path": wire_temporary,
                            "staged_bytes": staged_offset,
                            "chunks_completed": chunks_completed,
                            "request_submission": (
                                "not_submitted"
                                if isinstance(error, KernelConnectionError)
                                else "unknown_after_error"
                            ),
                            "safe_to_resume": True,
                            "resume_requires_same_incarnation": True,
                            "elapsed_seconds": round(time.monotonic() - operation_started, 3),
                        },
                    ) from error
                finally:
                    if local_compressed is not None:
                        local_compressed.unlink(missing_ok=True)
        finally:
            heartbeat_stop.set()
            await asyncio.gather(heartbeat, return_exceptions=True)
        wire_total = sum(int(item["wire_bytes"]) for item in transferred)
        phase_timings["total_seconds"] = round(time.monotonic() - operation_started, 3)
        return {
            "transfer_id": transfer_id,
            "files_transferred": transferred,
            "files_skipped": skipped,
            "total_bytes": total,
            "wire_bytes": wire_total,
            "compression": compression,
            "lease": lease,
            "progress_events_emitted": progress_count,
            "timings": phase_timings,
            "staging_cleanup": "published staging removed; failed staging preserved for resume",
        }

    async def transfer_cleanup(
        self, staging_paths: list[str], name: str | None, lease_token: str | None = None
    ) -> dict[str, Any]:
        """Explicitly remove only colab-mcp transfer staging paths under an owned lease."""
        if not staging_paths or len(staging_paths) > 1_000:
            raise ValueError("staging_paths must contain 1-1000 paths")
        if any(
            ".colab-mcp-wire-" not in path and ".colab-mcp-part-" not in path
            for path in staging_paths
        ):
            raise ValueError("Every path must be a colab-mcp wire or publication staging path")
        session, lease = await self._operation_lease(name, lease_token)
        removed = []
        missing = []
        for path in staging_paths:
            existing = await self._remote_stat_or_none(
                path, session.name, lease_token=lease["lease_token"]
            )
            if existing is None:
                missing.append(path)
                continue
            await self._remote_operation(
                "fs_remove",
                {"path": path, "recursive": False, "missing_ok": True},
                session.name,
                lease_token=lease["lease_token"],
            )
            removed.append(path)
        return {"removed": removed, "already_missing": missing, "lease": lease}

    async def _remote_files(
        self, path: str, name: str | None, max_files: int, lease_token: str | None = None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if lease_token:
            root = await self._remote_operation(
                "fs_stat", {"path": path, "checksum": False}, name, lease_token=lease_token
            )
        else:
            root = await self.filesystem_stat(path, name)
        if root["kind"] == "file":
            return root, [root]
        pending = [root["path"]]
        files: list[dict[str, Any]] = []
        while pending:
            pending_path = pending.pop()
            if lease_token:
                listing = await self._remote_operation(
                    "fs_list",
                    {"path": pending_path, "limit": min(max_files, 10_000)},
                    name,
                    lease_token=lease_token,
                )
            else:
                listing = await self.filesystem_list(
                    pending_path, name, limit=min(max_files, 10_000)
                )
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
        compression: str = "auto",
        compression_min_bytes: int = 1_048_576,
        compression_min_savings: float = 0.10,
        lease_token: str | None = None,
        selected_paths: set[str] | None = None,
    ) -> dict[str, Any]:
        """Download files through checksummed chunks and atomic local replacement."""
        _transfer_bounds(chunk_size, max_total_bytes, max_files)
        compression, compression_min_bytes, compression_min_savings = _compression_settings(
            compression, compression_min_bytes, compression_min_savings
        )
        operation_started = time.monotonic()
        session, lease = await self._operation_lease(name, lease_token)
        lease_token = lease["lease_token"]
        root, files = await self._remote_files(
            remote_path, session.name, max_files, lease_token=lease_token
        )
        remote_root = PurePosixPath(root["path"])
        if selected_paths is not None:
            files = [
                item
                for item in files
                if (
                    PurePosixPath(item["path"]).relative_to(remote_root).as_posix()
                    if root["kind"] == "directory"
                    else PurePosixPath(item["path"]).name
                )
                in selected_paths
            ]
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
        transfer_seconds = 0.0
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
            remote_stat = await self._remote_operation(
                "fs_stat",
                {"path": item["path"], "checksum": True},
                session.name,
                lease_token=lease_token,
            )
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
            remote_compressed: str | None = None
            try:
                content_bytes = int(item["size"])
                wire_path = item["path"]
                wire_bytes = content_bytes
                wire_checksum = checksum
                codec = "none"
                if compression == "gzip" or (
                    compression == "auto"
                    and content_bytes >= compression_min_bytes
                    and _auto_compression_candidate(PurePosixPath(item["path"]))
                ):
                    remote_compressed = "/content/.colab-mcp/transfers/" + uuid.uuid4().hex + ".gz"
                    candidate = await self._remote_operation(
                        "fs_gzip_compress",
                        {
                            "source": item["path"],
                            "destination": remote_compressed,
                            "max_input_bytes": max_total_bytes,
                        },
                        session.name,
                        lease_token=lease_token,
                    )
                    candidate_bytes = int(candidate["size"])
                    if _use_compressed_wire(
                        compression, content_bytes, candidate_bytes, compression_min_savings
                    ):
                        wire_path = remote_compressed
                        wire_bytes = candidate_bytes
                        wire_checksum = candidate["sha256"]
                        codec = "gzip"
                    else:
                        await self._remote_operation(
                            "fs_remove",
                            {"path": remote_compressed, "recursive": False, "missing_ok": True},
                            session.name,
                            lease_token=lease_token,
                        )
                        remote_compressed = None
                transfer_started = time.monotonic()
                await self._raw_download_file(
                    session,
                    wire_path,
                    temporary,
                    codec,
                    content_bytes,
                    wire_bytes,
                    wire_checksum,
                    checksum,
                    max_total_bytes,
                    chunk_size,
                )
                transfer_seconds += time.monotonic() - transfer_started
                temporary.replace(destination)
                transferred.append(
                    {
                        "remote_path": item["path"],
                        "local_path": str(destination),
                        "size": item["size"],
                        "sha256": checksum,
                        "compression": codec,
                        "content_bytes": content_bytes,
                        "wire_bytes": wire_bytes,
                        "wire_sha256": wire_checksum,
                        "wire_ratio": round(wire_bytes / max(1, content_bytes), 6),
                    }
                )
            finally:
                temporary.unlink(missing_ok=True)
                if remote_compressed is not None:
                    await self._remote_operation(
                        "fs_remove",
                        {"path": remote_compressed, "recursive": False, "missing_ok": True},
                        session.name,
                        lease_token=lease_token,
                    )
        wire_total = sum(int(item["wire_bytes"]) for item in transferred)
        return {
            "files_transferred": transferred,
            "files_skipped": skipped,
            "total_bytes": total,
            "wire_bytes": wire_total,
            "compression": compression,
            "lease": lease,
            "timings": {
                "assignment_lookup_seconds": lease["assignment_lookup_seconds"],
                "data_transfer_seconds": round(transfer_seconds, 3),
                "total_seconds": round(time.monotonic() - operation_started, 3),
            },
        }

    @staticmethod
    def _process_export_stage(process_id: str, remote_path: str, destination: Path) -> Path:
        identity = hashlib.sha256(
            f"{process_id}\0{remote_path}\0{destination}".encode()
        ).hexdigest()[:32]
        return destination.with_name(destination.name + ".colab-mcp-export-" + identity)

    async def process_export_cleanup(
        self, process_id: str, remote_path: str, local_path: str, name: str | None
    ) -> dict[str, Any]:
        """Discard only the deterministic local stage for one owned process export."""
        session = self.resolve(name)
        if self.process_journal.get(session.name, process_id) is None:
            raise ValueError(f"Unknown owned process_id: {process_id}")
        watcher = self._export_watchers.pop((session.name, process_id), None)
        if watcher is not None and watcher is not asyncio.current_task():
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        self._update_auto_export(
            session.name,
            process_id,
            status="held",
            last_error="automatic export explicitly abandoned and staging cleaned",
            last_attempt_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        destination = Path(local_path).expanduser().resolve()
        staged = self._process_export_stage(process_id, remote_path, destination)
        existed = staged.exists()
        if staged.is_dir():
            shutil.rmtree(staged)
        else:
            staged.unlink(missing_ok=True)
        return {
            "process_id": process_id,
            "session": session.name,
            "staging_path": str(staged),
            "removed": existed,
            "watcher_stopped": watcher is not None,
        }

    async def process_export(
        self,
        process_id: str,
        remote_path: str,
        local_path: str,
        name: str | None,
        release_on_success: bool = False,
        overwrite: bool = False,
        chunk_size: int = 524_288,
        max_total_bytes: int = 100_000_000,
        max_files: int = 10_000,
        compression: str = "auto",
        compression_min_bytes: int = 1_048_576,
        compression_min_savings: float = 0.10,
    ) -> dict[str, Any]:
        """Atomically publish completed-process output locally, or retain its runtime."""
        session = self.resolve(name)
        status = await self.process_status(process_id, session.name)
        if status.get("status") != "exited":
            return {
                "process_id": process_id,
                "session": session.name,
                "exported": False,
                "disposition": "held",
                "runtime_released": False,
                "error": {
                    "code": "process_not_finished",
                    "message": "Owned process must have status=exited before export",
                },
                "last_known_process": status,
            }

        destination = Path(local_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = self._process_export_stage(process_id, remote_path, destination)
        transfer: dict[str, Any] | None = None
        published = False
        try:
            if destination.exists() and not overwrite:
                raise FileExistsError(f"Local destination exists: {destination}")
            if destination.is_dir():
                raise IsADirectoryError(
                    "Atomic overwrite of an existing directory is not supported; choose a new path"
                )
            transfer = await self.transfer_download(
                remote_path,
                str(staged),
                session.name,
                overwrite=False,
                sync=True,
                chunk_size=chunk_size,
                max_total_bytes=max_total_bytes,
                max_files=max_files,
                compression=compression,
                compression_min_bytes=compression_min_bytes,
                compression_min_savings=compression_min_savings,
            )
            staged.replace(destination)
            published = True
            for item in transfer["files_transferred"]:
                staged_item = Path(item["local_path"])
                relative = staged_item.relative_to(staged) if staged_item != staged else None
                item["local_path"] = str(
                    destination if relative is None else destination / relative
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return {
                "process_id": process_id,
                "session": session.name,
                "exported": False,
                "disposition": "held",
                "runtime_released": False,
                "error": {"code": "export_failed_runtime_held", "message": str(error)},
                "recoverable_export": {
                    "staging_path": str(staged),
                    "staging_exists": staged.exists(),
                    "resume": "Retry colab_process_export with the same process, paths, and limits.",
                    "cleanup": "Use colab_process_export_cleanup to explicitly discard staging.",
                },
                "last_known_process": status,
            }
        finally:
            if published:
                if staged.is_dir():
                    shutil.rmtree(staged, ignore_errors=True)
                else:
                    staged.unlink(missing_ok=True)

        result = {
            "process_id": process_id,
            "session": session.name,
            "exported": True,
            "local_path": str(destination),
            "transfer": transfer,
            "disposition": "held",
            "runtime_released": False,
        }
        if not release_on_success:
            return result
        try:
            stopped = await self.stop(session.name)
        except Exception as error:
            return {
                **result,
                "error": {"code": "release_failed_runtime_held", "message": str(error)},
            }
        return {
            **result,
            "disposition": "released",
            "runtime_released": True,
            "release": stopped,
        }

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

    async def stop(self, name: str | None) -> dict[str, Any]:
        session = self.resolve(name)
        watcher_tasks = [
            task
            for (watcher_session, _process_id), task in self._export_watchers.items()
            if watcher_session == session.name
        ]
        self._export_watchers = {
            key: task for key, task in self._export_watchers.items() if key[0] != session.name
        }
        for watcher in watcher_tasks:
            watcher.cancel()
        if watcher_tasks:
            await asyncio.gather(*watcher_tasks, return_exceptions=True)
        task = self._keepalives.pop(session.name, None)
        if task:
            task.cancel()
        await self.close_kernel_channel(session.name)
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
