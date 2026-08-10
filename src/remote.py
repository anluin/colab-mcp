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
import time as _cm_time
import traceback as _cm_traceback
import uuid as _cm_uuid

_cm_payload = _cm_json.loads(_cm_base64.b64decode({encoded!r}).decode())
_cm_root = _cm_pathlib.Path('/content').resolve()
_cm_state_root = _cm_root / '.colab-mcp'
_cm_incarnation_path = _cm_state_root / 'runtime-incarnation'
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
        _cm_process_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if _cm_operation == 'incarnation_init':
        pass
    elif _cm_operation == 'lease_probe':
        _cm_result = {{
            'status': 'stable',
            'runtime_fingerprint': _cm_actual_fingerprint,
            'observed_at': _cm_datetime.datetime.now(_cm_datetime.timezone.utc).isoformat(),
        }}
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
