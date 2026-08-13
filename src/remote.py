"""Structured command and process primitives executed inside a Colab runtime."""

from __future__ import annotations

import base64
import json
import re
from typing import Any

RESULT_PREFIX = "__COLAB_MCP_RESULT__"
DEFAULT_OUTPUT_LIMIT = 100_000
MAX_OUTPUT_LIMIT = 1_000_000
MAX_COMMAND_TIMEOUT = 21_600.0
ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_PROCESS_OUTPUT_LIMIT = 10_000_000
MAX_PROCESS_OUTPUT_LIMIT = 1_000_000_000

PROCESS_RUNNER_SOURCE = """\
import datetime
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import traceback

launch = pathlib.Path(sys.argv[1])
payload = json.loads(launch.read_text(encoding="utf-8"))
launch.unlink()
directory = pathlib.Path(payload["directory"])
environment = os.environ.copy()
environment.update(payload["environment"])
output_limit = int(payload["output_limit"])
started = time.time()


def drain(stream, path, truncated_path):
    total = 0
    truncated = False
    with path.open("wb", buffering=0) as handle:
        while True:
            # BufferedReader.read(size) tries to fill the requested size and can
            # retain flushed, short writes until 64 KiB or EOF.  Read the pipe
            # directly so each available write reaches the durable spool while
            # the child is still running.
            chunk = os.read(stream.fileno(), 65_536)
            if not chunk:
                break
            remaining = max(0, output_limit - min(total, output_limit))
            if remaining:
                handle.write(chunk[:remaining])
            total += len(chunk)
            if total > output_limit and not truncated:
                truncated_path.write_text("true", encoding="ascii")
                truncated = True
    return total, truncated


try:
    child = subprocess.Popen(
        payload["argv"],
        cwd=payload["cwd"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        shell=False,
        start_new_session=True,
    )
    (directory / "target_pid").write_text(str(child.pid), encoding="ascii")
    drain_results = {}

    def drain_to(key, stream):
        drain_results[key] = drain(
            stream,
            directory / (key + ".log"),
            directory / (key + ".truncated"),
        )

    threads = [
        threading.Thread(target=drain_to, args=("stdout", child.stdout), daemon=True),
        threading.Thread(target=drain_to, args=("stderr", child.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()
    code = child.wait()
    for thread in threads:
        thread.join()
    result = {
        "exit_code": code,
        "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "stdout_total_bytes": drain_results["stdout"][0],
        "stderr_total_bytes": drain_results["stderr"][0],
        "stdout_truncated": drain_results["stdout"][1],
        "stderr_truncated": drain_results["stderr"][1],
    }
except BaseException as exc:
    result = {
        "exit_code": None,
        "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "error": str(exc),
        "traceback": traceback.format_exc()[-4000:],
    }
temporary = directory / "exit.tmp"
temporary.write_text(json.dumps(result), encoding="utf-8")
temporary.replace(directory / "exit.json")
"""


class RemoteOperationError(RuntimeError):
    """A structured operation failed inside the remote runtime."""


class RuntimeReplacedError(RemoteOperationError):
    """The Colab endpoint now refers to a different ephemeral backend."""

    code = "runtime_replaced"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.details = details or {"code": self.code, "message": message}
        super().__init__(message)


def validate_argv(argv: list[str]) -> None:
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise ValueError("argv must contain at least one non-empty string")
    if sum(len(value) for value in argv) > 100_000:
        raise ValueError("argv is too large")


def validate_environment(environment: dict[str, str] | None) -> dict[str, str]:
    result = environment or {}
    for key, value in result.items():
        if not ENVIRONMENT_KEY.fullmatch(key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
        if not isinstance(value, str):
            raise ValueError(f"Environment value for {key!r} must be a string")
    return result


def validate_output_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_OUTPUT_LIMIT:
        raise ValueError(f"output_limit must be between 1 and {MAX_OUTPUT_LIMIT}")
    return limit


def validate_process_output_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_PROCESS_OUTPUT_LIMIT:
        raise ValueError(f"process output_limit must be between 1 and {MAX_PROCESS_OUTPUT_LIMIT}")
    return limit


def validate_timeout(timeout: float) -> float:
    if not 0.1 <= timeout <= MAX_COMMAND_TIMEOUT:
        raise ValueError(f"timeout must be between 0.1 and {MAX_COMMAND_TIMEOUT} seconds")
    return timeout


def build_remote_code(operation: str, payload: dict[str, Any]) -> str:
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    return f"""
import base64 as _cm_base64
import datetime as _cm_datetime
import gzip as _cm_gzip
import hashlib as _cm_hashlib
import json as _cm_json
import os as _cm_os
import pathlib as _cm_pathlib
import platform as _cm_platform
import shutil as _cm_shutil
import signal as _cm_signal
import subprocess as _cm_subprocess
import sys as _cm_sys
import tarfile as _cm_tarfile
import time as _cm_time
import traceback as _cm_traceback
import uuid as _cm_uuid

_cm_payload = _cm_json.loads(_cm_base64.b64decode({encoded!r}).decode())
_cm_root = _cm_pathlib.Path('/content').resolve()
_cm_state_root = _cm_root / '.colab-mcp'
_cm_incarnation_path = _cm_state_root / 'runtime-incarnation'
_cm_operation_lease_path = _cm_state_root / 'operation-lease.json'
_cm_process_root = _cm_state_root / 'processes'

def _cm_path(value):
    candidate = (_cm_root / (value or '.')).resolve() if not _cm_pathlib.Path(value or '.').is_absolute() else _cm_pathlib.Path(value).resolve()
    if candidate != _cm_root and _cm_root not in candidate.parents:
        raise ValueError('Remote paths must stay within /content')
    return candidate

def _cm_metadata(process_id):
    if not isinstance(process_id, str) or not process_id or not all(c.isalnum() or c in '-_' for c in process_id):
        raise ValueError('Invalid process_id')
    directory = _cm_process_root / process_id
    metadata_path = directory / 'metadata.json'
    if not metadata_path.is_file():
        raise FileNotFoundError('Unknown process_id: ' + process_id)
    return directory, _cm_json.loads(metadata_path.read_text(encoding='utf-8'))

def _cm_status(directory, metadata):
    exit_path = directory / 'exit.json'
    if exit_path.is_file():
        exit_data = _cm_json.loads(exit_path.read_text(encoding='utf-8'))
        return {{**metadata, **exit_data, 'status': 'exited'}}
    pid = int(metadata['pid'])
    try:
        _cm_os.kill(pid, 0)
        return {{**metadata, 'status': 'running'}}
    except ProcessLookupError:
        return {{**metadata, 'status': 'lost', 'error': 'process disappeared without exit metadata'}}

def _cm_stat(path):
    value = path.stat()
    kind = 'directory' if path.is_dir() else ('file' if path.is_file() else 'other')
    return {{
        'path': str(path), 'name': path.name, 'kind': kind,
        'size': value.st_size, 'modified_at': _cm_datetime.datetime.fromtimestamp(
            value.st_mtime, _cm_datetime.timezone.utc).isoformat(),
    }}

class RuntimeReplacedError(RuntimeError):
    pass

try:
    _cm_operation = {operation!r}
    if _cm_operation == 'incarnation_init':
        _cm_fingerprint = _cm_payload.get('runtime_fingerprint')
        if not isinstance(_cm_fingerprint, str) or len(_cm_fingerprint) != 32 or not all(
                char in '0123456789abcdef' for char in _cm_fingerprint):
            raise ValueError('runtime_fingerprint must be 32 lowercase hexadecimal characters')
        _cm_state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _cm_temporary_incarnation = _cm_incarnation_path.with_suffix('.tmp')
        _cm_temporary_incarnation.write_text(_cm_fingerprint, encoding='ascii')
        _cm_temporary_incarnation.replace(_cm_incarnation_path)
        _cm_result = {{'runtime_fingerprint': _cm_fingerprint}}
    else:
        _cm_expected_fingerprint = _cm_payload.get('runtime_fingerprint')
        _cm_actual_fingerprint = (
            _cm_incarnation_path.read_text(encoding='ascii').strip()
            if _cm_incarnation_path.is_file() else None)
        if _cm_actual_fingerprint != _cm_expected_fingerprint:
            _cm_actual_label = _cm_actual_fingerprint or 'missing'
            raise RuntimeReplacedError(
                'runtime_replaced: expected incarnation ' + str(_cm_expected_fingerprint)
                + ', observed ' + _cm_actual_label
                + '; Colab recycled the backend, so start a new session')
        _cm_operation_lease_token = _cm_payload.get('operation_lease_token')
        if _cm_operation_lease_token is not None:
            if not _cm_operation_lease_path.is_file():
                raise RuntimeError('operation_lease_missing: probe again before retrying')
            _cm_operation_lease = _cm_json.loads(
                _cm_operation_lease_path.read_text(encoding='utf-8'))
            _cm_lease_candidates = _cm_operation_lease.get('leases')
            if not isinstance(_cm_lease_candidates, list):
                _cm_lease_candidates = [{{
                    'token': _cm_operation_lease.get('token'),
                    'expires_at': _cm_operation_lease.get('expires_at'),
                }}]
            _cm_matching_lease = next((
                item for item in _cm_lease_candidates
                if isinstance(item, dict) and item.get('token') == _cm_operation_lease_token
            ), None)
            if (_cm_matching_lease is None
                    or _cm_operation_lease.get('runtime_fingerprint') != _cm_actual_fingerprint):
                raise RuntimeError('operation_lease_stale: lease does not own this incarnation')
            _cm_lease_expiry = _cm_datetime.datetime.fromisoformat(
                _cm_matching_lease['expires_at'])
            if _cm_lease_expiry <= _cm_datetime.datetime.now(_cm_datetime.timezone.utc):
                raise RuntimeError('operation_lease_expired: probe again before retrying')
        _cm_process_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if _cm_operation == 'incarnation_init':
        pass
    elif _cm_operation == 'lease_probe':
        _cm_issued_token = _cm_payload.get('issue_lease_token')
        _cm_expires_at = _cm_payload.get('lease_expires_at')
        if (not isinstance(_cm_issued_token, str) or len(_cm_issued_token) != 32
                or not all(char in '0123456789abcdef' for char in _cm_issued_token)):
            raise ValueError('issue_lease_token must be 32 hexadecimal characters')
        _cm_operation_lease_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _cm_now = _cm_datetime.datetime.now(_cm_datetime.timezone.utc)
        _cm_retained_leases = []
        if _cm_operation_lease_path.is_file():
            try:
                _cm_prior_lease = _cm_json.loads(
                    _cm_operation_lease_path.read_text(encoding='utf-8'))
                _cm_prior_candidates = _cm_prior_lease.get('leases')
                if not isinstance(_cm_prior_candidates, list):
                    _cm_prior_candidates = [{{
                        'token': _cm_prior_lease.get('token'),
                        'expires_at': _cm_prior_lease.get('expires_at'),
                    }}]
                if _cm_prior_lease.get('runtime_fingerprint') == _cm_actual_fingerprint:
                    _cm_retained_leases = [
                        item for item in _cm_prior_candidates
                        if isinstance(item, dict)
                        and item.get('token') != _cm_issued_token
                        and isinstance(item.get('expires_at'), str)
                        and _cm_datetime.datetime.fromisoformat(item['expires_at']) > _cm_now
                    ][-7:]
            except Exception:
                _cm_retained_leases = []
        _cm_retained_leases.append({{
            'token': _cm_issued_token,
            'expires_at': _cm_expires_at,
        }})
        _cm_lease_temporary = _cm_operation_lease_path.with_suffix('.tmp')
        _cm_lease_temporary.write_text(_cm_json.dumps({{
            'token': _cm_issued_token,
            'runtime_fingerprint': _cm_actual_fingerprint,
            'expires_at': _cm_expires_at,
            'leases': _cm_retained_leases,
        }}), encoding='utf-8')
        _cm_lease_temporary.replace(_cm_operation_lease_path)
        _cm_result = {{
            'status': 'stable',
            'runtime_fingerprint': _cm_actual_fingerprint,
            'observed_at': _cm_datetime.datetime.now(_cm_datetime.timezone.utc).isoformat(),
            'lease_expires_at': _cm_expires_at,
        }}
    elif _cm_operation == 'workspace_manifest':
        _cm_workspace = _cm_path(_cm_payload['path'])
        _cm_max_files = int(_cm_payload['max_files'])
        _cm_max_total = int(_cm_payload['max_total_bytes'])
        _cm_include = tuple(_cm_payload.get('include', []))
        _cm_exclude = tuple(_cm_payload.get('exclude', []))
        def _cm_matches(_cm_relative, _cm_pattern):
            _cm_pattern = _cm_pattern.strip().replace('\\\\', '/').lstrip('/')
            if _cm_pattern.startswith('**/') and _cm_pattern.endswith('/**'):
                _cm_directory = _cm_pattern[3:-3].strip('/')
                return _cm_relative == _cm_directory or _cm_relative.startswith(_cm_directory + '/') or f'/{{_cm_directory}}/' in f'/{{_cm_relative}}/'
            if _cm_pattern.endswith('/**'):
                _cm_prefix = _cm_pattern[:-3].rstrip('/')
                return _cm_relative == _cm_prefix or _cm_relative.startswith(_cm_prefix + '/')
            _cm_candidate = _cm_pathlib.PurePosixPath(_cm_relative)
            return _cm_candidate.match(_cm_pattern) or ('/' not in _cm_pattern and _cm_candidate.name == _cm_pattern)
        if not _cm_workspace.exists():
            _cm_result = {{'path': str(_cm_workspace), 'files': [], 'total_bytes': 0, 'excluded': []}}
        elif not _cm_workspace.is_dir():
            raise NotADirectoryError(str(_cm_workspace))
        else:
            _cm_workspace_files = []
            _cm_workspace_excluded = []
            for _cm_item in sorted(_cm_workspace.rglob('*')):
                if not _cm_item.is_file() or _cm_item.is_symlink():
                    continue
                _cm_relative = _cm_item.relative_to(_cm_workspace).as_posix()
                _cm_reason = next((f'built_in:{{p}}' for p in _cm_exclude if _cm_matches(_cm_relative, p)), None)
                if _cm_reason is None and _cm_include and not any(_cm_matches(_cm_relative, p) for p in _cm_include):
                    _cm_reason = 'not_in_include'
                if _cm_reason is not None:
                    _cm_workspace_excluded.append({{'path': _cm_relative, 'reason': _cm_reason}})
                else:
                    _cm_workspace_files.append(_cm_item)
            if len(_cm_workspace_files) > _cm_max_files:
                raise ValueError('workspace exceeds max_files')
            _cm_workspace_total = sum(item.stat().st_size for item in _cm_workspace_files)
            if _cm_workspace_total > _cm_max_total:
                raise ValueError('workspace exceeds max_total_bytes')
            _cm_workspace_manifest = []
            for _cm_item in _cm_workspace_files:
                _cm_digest = _cm_hashlib.sha256()
                with _cm_item.open('rb') as _cm_handle:
                    for _cm_chunk in iter(lambda: _cm_handle.read(1024 * 1024), b''):
                        _cm_digest.update(_cm_chunk)
                _cm_workspace_manifest.append({{
                    'path': _cm_item.relative_to(_cm_workspace).as_posix(),
                    'size': _cm_item.stat().st_size,
                    'sha256': _cm_digest.hexdigest(),
                }})
            _cm_result = {{
                'path': str(_cm_workspace), 'files': _cm_workspace_manifest,
                'total_bytes': _cm_workspace_total, 'excluded': _cm_workspace_excluded,
            }}
    elif _cm_operation == 'workspace_bundle_publish':
        _cm_archive = _cm_path(_cm_payload['archive_path'])
        _cm_workspace = _cm_path(_cm_payload['workspace_path'])
        _cm_max_files = int(_cm_payload['max_files'])
        _cm_max_total = int(_cm_payload['max_total_bytes'])
        if not _cm_archive.is_file():
            raise FileNotFoundError(str(_cm_archive))
        _cm_wire_digest = _cm_hashlib.sha256()
        with _cm_archive.open('rb') as _cm_handle:
            for _cm_chunk in iter(lambda: _cm_handle.read(1024 * 1024), b''):
                _cm_wire_digest.update(_cm_chunk)
        if _cm_wire_digest.hexdigest() != _cm_payload['wire_sha256']:
            raise ValueError('workspace bundle wire checksum mismatch')
        _cm_publish_root = _cm_state_root / ('workspace-publish-' + _cm_payload['transfer_id'])
        _cm_shutil.rmtree(_cm_publish_root, ignore_errors=True)
        _cm_publish_root.mkdir(mode=0o700, parents=True)
        _cm_published = []
        _cm_publish_success = False
        try:
            _cm_tar_mode = 'r:gz' if _cm_payload['compression'] == 'gzip' else 'r:'
            with _cm_tarfile.open(_cm_archive, _cm_tar_mode) as _cm_bundle:
                _cm_members = _cm_bundle.getmembers()
                if len(_cm_members) > _cm_max_files + 1:
                    raise ValueError('workspace bundle exceeds max_files')
                _cm_manifest_member = _cm_bundle.getmember('manifest.json')
                if not _cm_manifest_member.isfile() or _cm_manifest_member.size > 10_000_000:
                    raise ValueError('invalid workspace bundle manifest')
                _cm_manifest_handle = _cm_bundle.extractfile(_cm_manifest_member)
                if _cm_manifest_handle is None:
                    raise ValueError('workspace bundle manifest is unreadable')
                _cm_manifest = _cm_json.loads(_cm_manifest_handle.read().decode('utf-8'))
                if not isinstance(_cm_manifest, list) or len(_cm_manifest) > _cm_max_files:
                    raise ValueError('invalid workspace bundle manifest entries')
                _cm_expected_names = {{'manifest.json'}}
                _cm_seen_paths = set()
                _cm_declared_total = 0
                for _cm_entry in _cm_manifest:
                    _cm_relative = _cm_entry.get('path') if isinstance(_cm_entry, dict) else None
                    if (not isinstance(_cm_relative, str) or not _cm_relative
                            or _cm_relative.startswith('/') or chr(92) in _cm_relative):
                        raise ValueError('invalid workspace bundle path')
                    _cm_parts = _cm_pathlib.PurePosixPath(_cm_relative).parts
                    if any(part in ('', '.', '..') for part in _cm_parts):
                        raise ValueError('workspace bundle path escapes its root')
                    if _cm_relative in _cm_seen_paths:
                        raise ValueError('duplicate workspace bundle path')
                    _cm_seen_paths.add(_cm_relative)
                    _cm_size = int(_cm_entry['size'])
                    if _cm_size < 0:
                        raise ValueError('invalid workspace bundle file size')
                    _cm_declared_total += _cm_size
                    _cm_expected_names.add('files/' + _cm_relative)
                if _cm_declared_total > _cm_max_total:
                    raise ValueError('workspace bundle exceeds max_total_bytes')
                if (len(_cm_members) != len(_cm_expected_names)
                        or {{member.name for member in _cm_members}} != _cm_expected_names):
                    raise ValueError('workspace bundle contains undeclared members')
                for _cm_entry in _cm_manifest:
                    _cm_relative = _cm_entry['path']
                    _cm_member = _cm_bundle.getmember('files/' + _cm_relative)
                    if not _cm_member.isfile() or _cm_member.size != int(_cm_entry['size']):
                        raise ValueError('workspace bundle member metadata mismatch')
                    _cm_source = _cm_bundle.extractfile(_cm_member)
                    if _cm_source is None:
                        raise ValueError('workspace bundle member is unreadable')
                    _cm_staged_file = _cm_publish_root.joinpath(*_cm_pathlib.PurePosixPath(_cm_relative).parts)
                    _cm_staged_file.parent.mkdir(parents=True, exist_ok=True)
                    _cm_digest = _cm_hashlib.sha256()
                    _cm_written = 0
                    with _cm_staged_file.open('wb') as _cm_output:
                        while True:
                            _cm_chunk = _cm_source.read(1024 * 1024)
                            if not _cm_chunk:
                                break
                            _cm_written += len(_cm_chunk)
                            if _cm_written > int(_cm_entry['size']):
                                raise ValueError('workspace bundle member exceeds declared size')
                            _cm_digest.update(_cm_chunk)
                            _cm_output.write(_cm_chunk)
                    if (_cm_written != int(_cm_entry['size'])
                            or _cm_digest.hexdigest() != _cm_entry['sha256']):
                        raise ValueError('workspace bundle content checksum mismatch')
                _cm_destinations = {{}}
                for _cm_entry in _cm_manifest:
                    _cm_relative = _cm_entry['path']
                    _cm_destination = _cm_workspace.joinpath(*_cm_pathlib.PurePosixPath(_cm_relative).parts)
                    _cm_destination = _cm_path(str(_cm_destination))
                    if _cm_destination.exists() and _cm_destination.is_dir():
                        raise IsADirectoryError(str(_cm_destination))
                    _cm_destinations[_cm_relative] = _cm_destination
                for _cm_entry in _cm_manifest:
                    _cm_relative = _cm_entry['path']
                    _cm_staged_file = _cm_publish_root.joinpath(*_cm_pathlib.PurePosixPath(_cm_relative).parts)
                    _cm_destination = _cm_destinations[_cm_relative]
                    _cm_destination.parent.mkdir(parents=True, exist_ok=True)
                    _cm_staged_file.replace(_cm_destination)
                    _cm_published.append({{
                        'path': _cm_relative, 'size': int(_cm_entry['size']),
                        'sha256': _cm_entry['sha256'],
                    }})
            _cm_publish_success = True
            _cm_result = {{'files': _cm_published, 'total_bytes': _cm_declared_total}}
        finally:
            _cm_shutil.rmtree(_cm_publish_root, ignore_errors=True)
            if _cm_publish_success:
                _cm_archive.unlink(missing_ok=True)
    elif _cm_operation == 'process_start':
        _cm_cwd = _cm_path(_cm_payload.get('cwd'))
        if not _cm_cwd.is_dir():
            raise NotADirectoryError(str(_cm_cwd))
        _cm_process_id = _cm_payload.get('process_id') or _cm_uuid.uuid4().hex
        _cm_directory = _cm_process_root / _cm_process_id
        _cm_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        _cm_runner = _cm_directory / 'runner.py'
        _cm_runner_payload = _cm_base64.b64encode(_cm_json.dumps({{
            'argv': _cm_payload['argv'], 'cwd': str(_cm_cwd),
            'environment': _cm_payload.get('environment') or {{}},
            'directory': str(_cm_directory),
            'output_limit': int(_cm_payload['output_limit']),
        }}).encode()).decode()
        _cm_launch = _cm_directory / 'launch.json'
        _cm_launch.write_text(_cm_base64.b64decode(_cm_runner_payload).decode(), encoding='utf-8')
        _cm_launch.chmod(0o600)
        _cm_runner.write_text({PROCESS_RUNNER_SOURCE!r}, encoding='utf-8')
        _cm_runner_process = _cm_subprocess.Popen(
            [_cm_sys.executable, str(_cm_runner), str(_cm_launch)],
            cwd=str(_cm_cwd), stdin=_cm_subprocess.DEVNULL,
            stdout=_cm_subprocess.DEVNULL, stderr=_cm_subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )
        _cm_metadata_value = {{
            'process_id': _cm_process_id, 'pid': _cm_runner_process.pid,
            'argv': _cm_payload['argv'], 'cwd': str(_cm_cwd),
            'created_at': _cm_datetime.datetime.now(_cm_datetime.timezone.utc).isoformat(),
            'environment_keys': sorted((_cm_payload.get('environment') or {{}}).keys()),
            'output_limit': int(_cm_payload['output_limit']),
        }}
        (_cm_directory / 'metadata.json').write_text(_cm_json.dumps(_cm_metadata_value), encoding='utf-8')
        for _cm_attempt in range(100):
            if not _cm_launch.exists() and ((_cm_directory / 'target_pid').exists() or (_cm_directory / 'exit.json').exists()):
                break
            _cm_time.sleep(0.05)
        if _cm_launch.exists() or not ((_cm_directory / 'target_pid').exists() or (_cm_directory / 'exit.json').exists()):
            try:
                _cm_os.killpg(_cm_runner_process.pid, _cm_signal.SIGKILL)
            except ProcessLookupError:
                pass
            _cm_launch.unlink(missing_ok=True)
            raise RuntimeError('Remote process runner did not start securely')
        _cm_result = {{**_cm_metadata_value, 'status': 'running'}}
    elif _cm_operation == 'process_status':
        _cm_directory, _cm_metadata_value = _cm_metadata(_cm_payload['process_id'])
        _cm_result = _cm_status(_cm_directory, _cm_metadata_value)
    elif _cm_operation == 'process_list':
        _cm_result = []
        for _cm_directory in sorted(_cm_process_root.iterdir()):
            if (_cm_directory / 'metadata.json').is_file():
                _cm_metadata_value = _cm_json.loads((_cm_directory / 'metadata.json').read_text(encoding='utf-8'))
                _cm_result.append(_cm_status(_cm_directory, _cm_metadata_value))
    elif _cm_operation == 'process_output':
        _cm_directory, _cm_metadata_value = _cm_metadata(_cm_payload['process_id'])
        _cm_stream = _cm_payload.get('stream', 'stdout')
        if _cm_stream not in ('stdout', 'stderr'):
            raise ValueError('stream must be stdout or stderr')
        _cm_path_value = _cm_directory / (_cm_stream + '.log')
        _cm_offset = max(0, int(_cm_payload.get('offset', 0)))
        _cm_limit = int(_cm_payload['limit'])
        if _cm_path_value.exists():
            with _cm_path_value.open('rb') as _cm_handle:
                _cm_handle.seek(_cm_offset)
                _cm_data = _cm_handle.read(_cm_limit)
                _cm_next = _cm_handle.tell()
                _cm_more = bool(_cm_handle.read(1))
        else:
            _cm_data, _cm_next, _cm_more = b'', _cm_offset, False
        _cm_status_value = _cm_status(_cm_directory, _cm_metadata_value)
        _cm_stored_bytes = _cm_path_value.stat().st_size if _cm_path_value.exists() else 0
        _cm_total_bytes = _cm_status_value.get(_cm_stream + '_total_bytes')
        if _cm_total_bytes is None:
            # The final byte count is written to exit.json.  Until then the
            # persisted spool size is the strongest observable lower bound and
            # must not be reported as null.
            _cm_total_bytes = _cm_stored_bytes
        _cm_result = {{
            'process_id': _cm_payload['process_id'], 'stream': _cm_stream,
            'offset': _cm_offset, 'next_offset': _cm_next,
            'data': _cm_data.decode('utf-8', errors='replace'),
            'more_available': _cm_more,
            'eof': _cm_status_value['status'] != 'running' and not _cm_more,
            'status': _cm_status_value['status'],
            'truncated': (_cm_directory / (_cm_stream + '.truncated')).exists(),
            'stored_bytes': _cm_stored_bytes,
            'total_bytes': _cm_total_bytes,
            'total_bytes_final': _cm_status_value['status'] != 'running',
            'output_limit': _cm_metadata_value.get('output_limit'),
        }}
    elif _cm_operation == 'process_signal':
        _cm_directory, _cm_metadata_value = _cm_metadata(_cm_payload['process_id'])
        _cm_before = _cm_status(_cm_directory, _cm_metadata_value)
        _cm_signal_name = _cm_payload.get('signal', 'TERM')
        _cm_signals = {{'TERM': _cm_signal.SIGTERM, 'KILL': _cm_signal.SIGKILL, 'INT': _cm_signal.SIGINT}}
        if _cm_signal_name not in _cm_signals:
            raise ValueError('signal must be TERM, KILL, or INT')
        if _cm_before['status'] == 'running':
            _cm_target_path = _cm_directory / 'target_pid'
            if not _cm_target_path.is_file():
                raise RuntimeError('process is still starting; retry the signal request')
            _cm_os.killpg(int(_cm_target_path.read_text()), _cm_signals[_cm_signal_name])
        _cm_result = {{'process_id': _cm_payload['process_id'], 'signal': _cm_signal_name, 'previous_status': _cm_before['status']}}
    elif _cm_operation == 'fs_list':
        _cm_target = _cm_path(_cm_payload.get('path'))
        if not _cm_target.is_dir():
            raise NotADirectoryError(str(_cm_target))
        _cm_limit = int(_cm_payload['limit'])
        _cm_entries = sorted(_cm_target.iterdir(), key=lambda item: item.name)
        _cm_result = {{
            'path': str(_cm_target),
            'entries': [_cm_stat(item) for item in _cm_entries[:_cm_limit]],
            'truncated': len(_cm_entries) > _cm_limit,
        }}
    elif _cm_operation == 'fs_stat':
        _cm_target = _cm_path(_cm_payload['path'])
        if not _cm_target.exists():
            raise FileNotFoundError(str(_cm_target))
        _cm_result = _cm_stat(_cm_target)
        if _cm_target.is_file() and _cm_payload.get('checksum'):
            _cm_digest = _cm_hashlib.sha256()
            with _cm_target.open('rb') as _cm_handle:
                for _cm_chunk in iter(lambda: _cm_handle.read(1024 * 1024), b''):
                    _cm_digest.update(_cm_chunk)
            _cm_result['sha256'] = _cm_digest.hexdigest()
    elif _cm_operation == 'fs_read':
        _cm_target = _cm_path(_cm_payload['path'])
        if not _cm_target.is_file():
            raise FileNotFoundError(str(_cm_target))
        _cm_offset = int(_cm_payload.get('offset', 0))
        _cm_limit = int(_cm_payload['limit'])
        with _cm_target.open('rb') as _cm_handle:
            _cm_handle.seek(_cm_offset)
            _cm_data = _cm_handle.read(_cm_limit)
            _cm_next = _cm_handle.tell()
            _cm_more = bool(_cm_handle.read(1))
        _cm_result = {{
            'path': str(_cm_target), 'offset': _cm_offset, 'next_offset': _cm_next,
            'data_base64': _cm_base64.b64encode(_cm_data).decode(),
            'bytes_read': len(_cm_data), 'more_available': _cm_more, 'eof': not _cm_more,
        }}
    elif _cm_operation == 'fs_write':
        _cm_target = _cm_path(_cm_payload['path'])
        _cm_target.parent.mkdir(parents=True, exist_ok=True) if _cm_payload.get('create_parents') else None
        _cm_data = _cm_base64.b64decode(_cm_payload['data_base64'], validate=True)
        if _cm_payload.get('append'):
            with _cm_target.open('ab') as _cm_handle:
                _cm_handle.write(_cm_data)
        else:
            _cm_temporary = _cm_target.with_name(_cm_target.name + '.colab-mcp-tmp-' + _cm_uuid.uuid4().hex)
            try:
                _cm_temporary.write_bytes(_cm_data)
                _cm_temporary.replace(_cm_target)
            finally:
                _cm_temporary.unlink(missing_ok=True)
        _cm_result = {{**_cm_stat(_cm_target), 'bytes_written': len(_cm_data)}}
    elif _cm_operation == 'fs_gzip_compress':
        _cm_source = _cm_path(_cm_payload['source'])
        _cm_destination = _cm_path(_cm_payload['destination'])
        if not _cm_source.is_file():
            raise FileNotFoundError(str(_cm_source))
        _cm_max_input = int(_cm_payload['max_input_bytes'])
        if _cm_source.stat().st_size > _cm_max_input:
            raise ValueError('gzip source exceeds max_input_bytes')
        _cm_destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _cm_source.open('rb') as _cm_input, _cm_gzip.open(
                    _cm_destination, 'wb', compresslevel=6) as _cm_output:
                _cm_shutil.copyfileobj(_cm_input, _cm_output, length=1024 * 1024)
            _cm_digest = _cm_hashlib.sha256()
            with _cm_destination.open('rb') as _cm_handle:
                for _cm_chunk in iter(lambda: _cm_handle.read(1024 * 1024), b''):
                    _cm_digest.update(_cm_chunk)
            _cm_result = {{
                **_cm_stat(_cm_destination),
                'sha256': _cm_digest.hexdigest(),
                'content_bytes': _cm_source.stat().st_size,
            }}
        except BaseException:
            _cm_destination.unlink(missing_ok=True)
            raise
    elif _cm_operation == 'fs_gzip_decompress':
        _cm_source = _cm_path(_cm_payload['source'])
        _cm_destination = _cm_path(_cm_payload['destination'])
        if not _cm_source.is_file():
            raise FileNotFoundError(str(_cm_source))
        _cm_expected_size = int(_cm_payload['expected_size'])
        _cm_max_output = int(_cm_payload['max_output_bytes'])
        if _cm_expected_size > _cm_max_output:
            raise ValueError('expected gzip output exceeds max_output_bytes')
        _cm_destination.parent.mkdir(parents=True, exist_ok=True)
        _cm_digest = _cm_hashlib.sha256()
        _cm_written = 0
        try:
            with _cm_gzip.open(_cm_source, 'rb') as _cm_input, _cm_destination.open('wb') as _cm_output:
                while True:
                    _cm_chunk = _cm_input.read(1024 * 1024)
                    if not _cm_chunk:
                        break
                    _cm_written += len(_cm_chunk)
                    if _cm_written > _cm_max_output or _cm_written > _cm_expected_size:
                        raise ValueError('gzip output exceeded declared size bound')
                    _cm_digest.update(_cm_chunk)
                    _cm_output.write(_cm_chunk)
            if _cm_written != _cm_expected_size:
                raise ValueError('gzip output size did not match expected_size')
            if _cm_digest.hexdigest() != _cm_payload['expected_sha256']:
                raise ValueError('gzip output checksum did not match expected_sha256')
            _cm_result = {{
                **_cm_stat(_cm_destination),
                'sha256': _cm_digest.hexdigest(),
                'content_bytes': _cm_written,
                'wire_bytes': _cm_source.stat().st_size,
            }}
        except BaseException:
            _cm_destination.unlink(missing_ok=True)
            raise
    elif _cm_operation == 'fs_mkdir':
        _cm_target = _cm_path(_cm_payload['path'])
        _cm_target.mkdir(parents=bool(_cm_payload.get('parents')), exist_ok=bool(_cm_payload.get('exist_ok')))
        _cm_result = _cm_stat(_cm_target)
    elif _cm_operation == 'fs_move':
        _cm_source = _cm_path(_cm_payload['source'])
        _cm_destination = _cm_path(_cm_payload['destination'])
        if not _cm_source.exists():
            raise FileNotFoundError(str(_cm_source))
        if _cm_destination.exists() and not _cm_payload.get('overwrite'):
            raise FileExistsError(str(_cm_destination))
        _cm_destination.parent.mkdir(parents=True, exist_ok=True)
        if _cm_destination.exists():
            if _cm_destination.is_dir():
                _cm_shutil.rmtree(_cm_destination)
            else:
                _cm_destination.unlink()
        _cm_shutil.move(str(_cm_source), str(_cm_destination))
        _cm_result = _cm_stat(_cm_destination)
    elif _cm_operation == 'fs_remove':
        _cm_target = _cm_path(_cm_payload['path'])
        if _cm_target == _cm_root:
            raise ValueError('Refusing to remove /content')
        if not _cm_target.exists():
            if not _cm_payload.get('missing_ok'):
                raise FileNotFoundError(str(_cm_target))
        elif _cm_target.is_dir():
            if not _cm_payload.get('recursive'):
                _cm_target.rmdir()
            else:
                _cm_shutil.rmtree(_cm_target)
        else:
            _cm_target.unlink()
        _cm_result = {{'removed': str(_cm_target), 'recursive': bool(_cm_payload.get('recursive'))}}
    elif _cm_operation == 'p2p_ranges':
        _cm_target = _cm_path(_cm_payload['path'])
        if not _cm_target.is_file():
            raise FileNotFoundError(str(_cm_target))
        _cm_expected_size = int(_cm_payload['expected_size'])
        _cm_max_total = int(_cm_payload['max_total_bytes'])
        if (_cm_target.stat().st_size != _cm_expected_size
                or not 0 <= _cm_expected_size <= _cm_max_total):
            raise ValueError('WebRTC range source size changed or exceeds its bound')
        _cm_ranges = _cm_payload['ranges']
        if not isinstance(_cm_ranges, list) or not 1 <= len(_cm_ranges) <= 16:
            raise ValueError('WebRTC transfer requires 1-16 ranges')
        _cm_range_result = []
        _cm_cursor = 0
        with _cm_target.open('rb') as _cm_handle:
            for _cm_range in _cm_ranges:
                _cm_offset = int(_cm_range['offset'])
                _cm_size = int(_cm_range['size'])
                if _cm_offset != _cm_cursor or _cm_size <= 0 or _cm_offset + _cm_size > _cm_expected_size:
                    raise ValueError('WebRTC ranges must be contiguous and bounded')
                _cm_handle.seek(_cm_offset)
                _cm_digest = _cm_hashlib.sha256()
                _cm_remaining = _cm_size
                while _cm_remaining:
                    _cm_chunk = _cm_handle.read(min(1024 * 1024, _cm_remaining))
                    if not _cm_chunk:
                        raise ValueError('WebRTC range source ended early')
                    _cm_digest.update(_cm_chunk)
                    _cm_remaining -= len(_cm_chunk)
                _cm_range_result.append({{
                    'offset': _cm_offset, 'size': _cm_size,
                    'sha256': _cm_digest.hexdigest(),
                }})
                _cm_cursor += _cm_size
        if _cm_cursor != _cm_expected_size:
            raise ValueError('WebRTC ranges do not cover the complete source')
        _cm_result = {{'path': str(_cm_target), 'ranges': _cm_range_result}}
    elif _cm_operation == 'p2p_assemble':
        _cm_target = _cm_path(_cm_payload['target'])
        if '.colab-mcp-wire-' not in _cm_target.name:
            raise ValueError('WebRTC assembly requires a transfer staging path')
        _cm_prefix_size = int(_cm_payload['prefix_size'])
        _cm_expected_size = int(_cm_payload['expected_size'])
        _cm_max_total = int(_cm_payload['max_total_bytes'])
        if not 0 <= _cm_prefix_size <= _cm_expected_size <= _cm_max_total:
            raise ValueError('invalid WebRTC assembly size')
        if _cm_prefix_size and (not _cm_target.is_file()
                                or _cm_target.stat().st_size != _cm_prefix_size):
            raise ValueError('WebRTC assembly prefix changed')
        _cm_parts = _cm_payload['parts']
        if not isinstance(_cm_parts, list) or not 1 <= len(_cm_parts) <= 16:
            raise ValueError('WebRTC assembly requires 1-16 parts')
        _cm_temporary = _cm_target.with_name(
            _cm_target.name + '.colab-mcp-assemble-' + _cm_uuid.uuid4().hex)
        _cm_digest = _cm_hashlib.sha256()
        _cm_written = 0
        _cm_part_paths = []
        try:
            with _cm_temporary.open('wb') as _cm_output:
                if _cm_prefix_size:
                    with _cm_target.open('rb') as _cm_prefix:
                        _cm_remaining = _cm_prefix_size
                        while _cm_remaining:
                            _cm_chunk = _cm_prefix.read(min(1024 * 1024, _cm_remaining))
                            if not _cm_chunk:
                                raise ValueError('WebRTC assembly prefix ended early')
                            _cm_output.write(_cm_chunk)
                            _cm_digest.update(_cm_chunk)
                            _cm_written += len(_cm_chunk)
                            _cm_remaining -= len(_cm_chunk)
                for _cm_part in _cm_parts:
                    _cm_part_path = _cm_path(_cm_part['path'])
                    if '.colab-mcp-wire-' not in _cm_part_path.name or not _cm_part_path.is_file():
                        raise ValueError('invalid WebRTC assembly part')
                    _cm_part_size = int(_cm_part['size'])
                    if _cm_part_path.stat().st_size != _cm_part_size:
                        raise ValueError('WebRTC assembly part size mismatch')
                    _cm_part_digest = _cm_hashlib.sha256()
                    with _cm_part_path.open('rb') as _cm_input:
                        for _cm_chunk in iter(lambda: _cm_input.read(1024 * 1024), b''):
                            _cm_output.write(_cm_chunk)
                            _cm_digest.update(_cm_chunk)
                            _cm_part_digest.update(_cm_chunk)
                            _cm_written += len(_cm_chunk)
                            if _cm_written > _cm_expected_size or _cm_written > _cm_max_total:
                                raise ValueError('WebRTC assembly exceeded declared size')
                    if _cm_part_digest.hexdigest() != _cm_part['sha256']:
                        raise ValueError('WebRTC assembly part checksum mismatch')
                    _cm_part_paths.append(_cm_part_path)
                _cm_output.flush()
                _cm_os.fsync(_cm_output.fileno())
            if (_cm_written != _cm_expected_size
                    or _cm_digest.hexdigest() != _cm_payload['expected_sha256']):
                raise ValueError('WebRTC assembly checksum mismatch')
            _cm_temporary.replace(_cm_target)
            for _cm_part_path in _cm_part_paths:
                _cm_part_path.unlink(missing_ok=True)
            _cm_result = {{
                **_cm_stat(_cm_target), 'sha256': _cm_digest.hexdigest(),
                'parts': len(_cm_parts),
            }}
        finally:
            _cm_temporary.unlink(missing_ok=True)
    elif _cm_operation == 'p2p_prepare':
        _cm_source = _cm_payload['source']
        if not isinstance(_cm_source, str) or not 1 <= len(_cm_source.encode('utf-8')) <= 200_000:
            raise ValueError('invalid WebRTC endpoint source')
        if _cm_hashlib.sha256(_cm_source.encode('utf-8')).hexdigest() != _cm_payload['sha256']:
            raise ValueError('WebRTC endpoint source checksum mismatch')
        _cm_endpoint = _cm_state_root / 'p2p-endpoint.py'
        _cm_existing = (
            _cm_hashlib.sha256(_cm_endpoint.read_bytes()).hexdigest()
            if _cm_endpoint.is_file() else None)
        if _cm_existing != _cm_payload['sha256']:
            _cm_endpoint.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _cm_temporary = _cm_endpoint.with_suffix('.tmp')
            _cm_temporary.write_text(_cm_source, encoding='utf-8')
            _cm_os.chmod(_cm_temporary, 0o600)
            _cm_temporary.replace(_cm_endpoint)
        _cm_requirement = _cm_payload['requirement']
        if not isinstance(_cm_requirement, str) or not _cm_requirement.startswith('aiortc=='):
            raise ValueError('invalid WebRTC dependency requirement')
        _cm_version = None
        try:
            import importlib.metadata as _cm_metadata_module
            _cm_version = _cm_metadata_module.version('aiortc')
        except Exception:
            _cm_version = None
        _cm_required_version = _cm_requirement.split('==', 1)[1]
        _cm_installed = False
        if _cm_version != _cm_required_version:
            _cm_install = _cm_subprocess.run(
                [_cm_sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                 '--quiet', _cm_requirement], capture_output=True, text=True, timeout=240)
            if _cm_install.returncode != 0:
                raise RuntimeError('WebRTC dependency installation failed: '
                                   + _cm_install.stderr[-1000:])
            _cm_installed = True
            _cm_version = _cm_required_version
        _cm_result = {{
            'ready': True, 'endpoint_sha256': _cm_payload['sha256'],
            'aiortc_version': _cm_version, 'installed': _cm_installed,
        }}
    elif _cm_operation == 'p2p_start':
        _cm_request_id = _cm_payload['request_id']
        if (not isinstance(_cm_request_id, str) or len(_cm_request_id) != 32
                or any(char not in '0123456789abcdef' for char in _cm_request_id)):
            raise ValueError('request_id must be 32 lowercase hexadecimal characters')
        _cm_direction = _cm_payload['direction']
        if _cm_direction not in ('upload', 'download'):
            raise ValueError('WebRTC direction must be upload or download')
        _cm_transfer_path = _cm_path(_cm_payload['path'])
        if _cm_direction == 'upload' and '.colab-mcp-wire-' not in _cm_transfer_path.name:
            raise ValueError('WebRTC uploads require a transfer staging path')
        if _cm_direction == 'download' and not _cm_transfer_path.is_file():
            raise FileNotFoundError(str(_cm_transfer_path))
        _cm_size = int(_cm_payload['size'])
        _cm_offset = int(_cm_payload.get('offset', 0))
        if not 0 <= _cm_offset <= _cm_size <= int(_cm_payload['max_total_bytes']):
            raise ValueError('invalid WebRTC transfer size or offset')
        _cm_source_offset = int(_cm_payload.get('source_offset', 0))
        if _cm_source_offset < 0:
            raise ValueError('invalid WebRTC source offset')
        if (_cm_direction == 'download'
                and _cm_source_offset + _cm_size > _cm_transfer_path.stat().st_size):
            raise ValueError('WebRTC download range exceeds the source')
        _cm_sha256 = _cm_payload['sha256']
        _cm_secret = _cm_payload['secret']
        if (not isinstance(_cm_sha256, str) or len(_cm_sha256) != 64
                or any(char not in '0123456789abcdef' for char in _cm_sha256)):
            raise ValueError('invalid WebRTC transfer checksum')
        if (not isinstance(_cm_secret, str) or len(_cm_secret) != 64
                or any(char not in '0123456789abcdef' for char in _cm_secret)):
            raise ValueError('invalid WebRTC transfer secret')
        _cm_offer = _cm_payload['offer']
        if (not isinstance(_cm_offer, dict) or _cm_offer.get('type') != 'offer'
                or not isinstance(_cm_offer.get('sdp'), str)
                or len(_cm_offer['sdp']) > 200_000):
            raise ValueError('invalid WebRTC offer')
        _cm_ice_servers = _cm_payload['ice_servers']
        if not isinstance(_cm_ice_servers, list) or not 0 <= len(_cm_ice_servers) <= 8:
            raise ValueError('invalid WebRTC ICE server list')
        _cm_endpoint = _cm_state_root / 'p2p-endpoint.py'
        if not _cm_endpoint.is_file():
            raise RuntimeError('WebRTC endpoint is not prepared')
        _cm_request_root = _cm_state_root / 'p2p' / _cm_request_id
        if _cm_request_root.exists():
            raise FileExistsError('WebRTC request already exists')
        _cm_request_root.mkdir(mode=0o700, parents=True)
        _cm_config_path = _cm_request_root / 'config.json'
        _cm_answer_path = _cm_request_root / 'answer.json'
        _cm_result_path = _cm_request_root / 'result.json'
        _cm_config = {{
            'runtime_fingerprint': _cm_actual_fingerprint,
            'lease_token': _cm_operation_lease_token,
            'fingerprint_path': str(_cm_incarnation_path),
            'lease_path': str(_cm_operation_lease_path),
            'content_root': str(_cm_root),
            'direction': _cm_direction,
            'path': str(_cm_transfer_path),
            'size': _cm_size,
            'offset': _cm_offset,
            'source_offset': _cm_source_offset,
            'sha256': _cm_sha256,
            'secret': _cm_secret,
            'offer': _cm_offer,
            'ice_servers': _cm_ice_servers,
            'connect_timeout': float(_cm_payload['connect_timeout']),
            'transfer_timeout': float(_cm_payload['transfer_timeout']),
            'answer_path': str(_cm_answer_path),
            'result_path': str(_cm_result_path),
        }}
        _cm_config_path.write_text(_cm_json.dumps(_cm_config), encoding='utf-8')
        _cm_os.chmod(_cm_config_path, 0o600)
        _cm_stdout = (_cm_request_root / 'stdout.log').open('wb')
        _cm_stderr = (_cm_request_root / 'stderr.log').open('wb')
        _cm_process = _cm_subprocess.Popen(
            [_cm_sys.executable, str(_cm_endpoint), str(_cm_config_path)],
            stdin=_cm_subprocess.DEVNULL, stdout=_cm_stdout, stderr=_cm_stderr,
            start_new_session=True)
        _cm_stdout.close()
        _cm_stderr.close()
        (_cm_request_root / 'pid').write_text(str(_cm_process.pid), encoding='ascii')
        _cm_deadline = _cm_time.monotonic() + float(_cm_payload['connect_timeout'])
        while not _cm_answer_path.is_file() and _cm_time.monotonic() < _cm_deadline:
            if _cm_result_path.is_file():
                _cm_early_result = _cm_json.loads(_cm_result_path.read_text(encoding='utf-8'))
                raise RuntimeError(_cm_early_result.get('error', 'WebRTC endpoint exited early'))
            if _cm_process.poll() is not None:
                _cm_failure = (_cm_request_root / 'stderr.log').read_text(
                    encoding='utf-8', errors='replace')[-1000:]
                raise RuntimeError('WebRTC endpoint exited before answering: ' + _cm_failure)
            _cm_time.sleep(0.05)
        if not _cm_answer_path.is_file():
            _cm_os.killpg(_cm_process.pid, _cm_signal.SIGTERM)
            raise TimeoutError('WebRTC endpoint answer timed out')
        _cm_answer = _cm_json.loads(_cm_answer_path.read_text(encoding='utf-8'))
        _cm_result = {{
            'request_id': _cm_request_id, 'answer': _cm_answer,
            'endpoint_pid': _cm_process.pid, 'protocol_version': 1,
        }}
    elif _cm_operation == 'p2p_finish':
        _cm_request_id = _cm_payload['request_id']
        if (not isinstance(_cm_request_id, str) or len(_cm_request_id) != 32
                or any(char not in '0123456789abcdef' for char in _cm_request_id)):
            raise ValueError('invalid WebRTC request_id')
        _cm_request_root = _cm_state_root / 'p2p' / _cm_request_id
        _cm_result_path = _cm_request_root / 'result.json'
        _cm_deadline = _cm_time.monotonic() + float(_cm_payload.get('wait_seconds', 15))
        while not _cm_result_path.is_file() and _cm_time.monotonic() < _cm_deadline:
            _cm_time.sleep(0.05)
        if not _cm_result_path.is_file():
            raise TimeoutError('WebRTC endpoint completion timed out')
        _cm_p2p_result = _cm_json.loads(_cm_result_path.read_text(encoding='utf-8'))
        if _cm_p2p_result.get('ok'):
            _cm_shutil.rmtree(_cm_request_root, ignore_errors=True)
        _cm_result = _cm_p2p_result
    elif _cm_operation == 'p2p_abort':
        _cm_request_id = _cm_payload['request_id']
        if (not isinstance(_cm_request_id, str) or len(_cm_request_id) != 32
                or any(char not in '0123456789abcdef' for char in _cm_request_id)):
            raise ValueError('invalid WebRTC request_id')
        _cm_request_root = _cm_state_root / 'p2p' / _cm_request_id
        _cm_pid_path = _cm_request_root / 'pid'
        if _cm_pid_path.is_file():
            try:
                _cm_os.killpg(int(_cm_pid_path.read_text(encoding='ascii')), _cm_signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
        _cm_shutil.rmtree(_cm_request_root, ignore_errors=True)
        _cm_result = {{'request_id': _cm_request_id, 'aborted': True}}
    elif _cm_operation == 'inspect':
        _cm_disk = _cm_shutil.disk_usage(_cm_root)
        _cm_memory = {{}}
        _cm_meminfo = _cm_pathlib.Path('/proc/meminfo')
        if _cm_meminfo.is_file():
            for _cm_line in _cm_meminfo.read_text().splitlines():
                _cm_key, _cm_value = _cm_line.split(':', 1)
                if _cm_key in ('MemTotal', 'MemAvailable'):
                    _cm_memory[_cm_key] = int(_cm_value.strip().split()[0]) * 1024
        _cm_tools = {{name: _cm_shutil.which(name) for name in _cm_payload.get('tools', [])}}
        _cm_gpu = []
        _cm_nvidia = _cm_shutil.which('nvidia-smi')
        _cm_driver = None
        if _cm_nvidia:
            _cm_query = _cm_subprocess.run([
                _cm_nvidia, '--query-gpu=index,name,memory.total,memory.free,driver_version',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
            if _cm_query.returncode == 0:
                for _cm_line in _cm_query.stdout.splitlines():
                    _cm_parts = [part.strip() for part in _cm_line.split(',')]
                    if len(_cm_parts) == 5:
                        _cm_driver = _cm_parts[4]
                        _cm_gpu.append({{
                            'index': int(_cm_parts[0]), 'name': _cm_parts[1],
                            'vram_total_mib': int(_cm_parts[2]),
                            'vram_free_mib': int(_cm_parts[3]), 'driver_version': _cm_parts[4],
                        }})
        _cm_cuda_version = None
        _cm_cuda_file = _cm_pathlib.Path('/usr/local/cuda/version.json')
        if _cm_cuda_file.is_file():
            try:
                _cm_cuda_version = _cm_json.loads(_cm_cuda_file.read_text()).get('cuda', {{}}).get('version')
            except Exception:
                _cm_cuda_version = None
        if _cm_cuda_version is None and _cm_nvidia:
            _cm_summary = _cm_subprocess.run(
                [_cm_nvidia], capture_output=True, text=True, timeout=15)
            _cm_marker = 'CUDA Version:'
            if _cm_summary.returncode == 0 and _cm_marker in _cm_summary.stdout:
                _cm_cuda_version = _cm_summary.stdout.split(_cm_marker, 1)[1].strip().split()[0]
        _cm_processes = _cm_subprocess.run(
            ['ps', '-eo', 'pid=,ppid=,stat=,etimes=,comm=', '--sort=pid'],
            capture_output=True, text=True, timeout=10)
        _cm_process_lines = _cm_processes.stdout.splitlines()[:int(_cm_payload['process_limit'])]
        _cm_result = {{
            'os': {{'system': _cm_platform.system(), 'release': _cm_platform.release(),
                   'machine': _cm_platform.machine(), 'platform': _cm_platform.platform()}},
            'python': {{'version': _cm_platform.python_version(), 'executable': _cm_sys.executable}},
            'cpu': {{'logical_count': _cm_os.cpu_count(), 'model': _cm_platform.processor() or None}},
            'memory': {{'total_bytes': _cm_memory.get('MemTotal'),
                       'available_bytes': _cm_memory.get('MemAvailable')}},
            'disk': {{'path': str(_cm_root), 'total_bytes': _cm_disk.total,
                     'used_bytes': _cm_disk.used, 'free_bytes': _cm_disk.free}},
            'gpu': _cm_gpu,
            'cuda': {{'version': _cm_cuda_version, 'driver_version': _cm_driver,
                     'nvidia_smi': _cm_nvidia}},
            'tools': _cm_tools,
            'processes': _cm_process_lines,
            'processes_truncated': len(_cm_processes.stdout.splitlines()) > len(_cm_process_lines),
        }}
    else:
        raise ValueError('Unknown remote operation: ' + _cm_operation)
    print({RESULT_PREFIX!r} + _cm_json.dumps({{'ok': True, 'result': _cm_result}}, separators=(',', ':')))
except BaseException as _cm_error:
    print({RESULT_PREFIX!r} + _cm_json.dumps({{
        'ok': False, 'error': {{'type': type(_cm_error).__name__, 'message': str(_cm_error)}}
    }}, separators=(',', ':')))
"""


def parse_remote_result(outputs: list[dict[str, Any]]) -> Any:
    for output in reversed(outputs):
        if output.get("output_type") != "stream":
            continue
        for line in reversed(str(output.get("text", "")).splitlines()):
            if not line.startswith(RESULT_PREFIX):
                continue
            payload = json.loads(line[len(RESULT_PREFIX) :])
            if payload.get("ok"):
                return payload.get("result")
            error = payload.get("error") or {}
            error_type = error.get("type", "RemoteError")
            message = error.get("message", "operation failed")
            if error_type == "RuntimeReplacedError" or str(message).startswith("runtime_replaced:"):
                raise RuntimeReplacedError(message)
            raise RemoteOperationError(f"{error_type}: {message}")
    raise RemoteOperationError("Remote runtime returned no structured result")
