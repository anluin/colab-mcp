import asyncio
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from colab_cli.state import SessionState

from src.manager import ColabManager, ManagedSessionState
from src.remote import (
    MAX_OUTPUT_LIMIT,
    MAX_PROCESS_OUTPUT_LIMIT,
    PROCESS_RUNNER_SOURCE,
    RESULT_PREFIX,
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


def make_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ColabManager:
    monkeypatch.setenv("COLAB_MCP_STATE_DIR", str(tmp_path / "state"))
    return ColabManager()


def stream_result(result: object) -> list[dict]:
    payload = json.dumps({"ok": True, "result": result})
    return [{"output_type": "stream", "text": f"noise\n{RESULT_PREFIX}{payload}\n"}]


def stream_error(error_type: str, message: str) -> list[dict]:
    payload = json.dumps({"ok": False, "error": {"type": error_type, "message": message}})
    return [{"output_type": "stream", "text": RESULT_PREFIX + payload}]


def test_remote_payload_is_encoded_not_interpolated():
    dangerous = "'\"; raise RuntimeError('injected')"
    code = build_remote_code(
        "process_start",
        {"argv": [dangerous], "environment": {}, "output_limit": 100},
    )
    assert dangerous not in code
    assert "b64decode" in code


def test_process_environment_is_not_passed_in_runner_command_line():
    code = build_remote_code(
        "process_start",
        {
            "argv": ["env"],
            "environment": {"PRIVATE_VALUE": "secret"},
            "output_limit": 100,
        },
    )
    assert "launch.json" in code
    assert "str(_cm_launch)" in code
    assert "_cm_runner_payload]" not in code


def test_process_output_reconciles_exit_count_with_live_spool_size():
    code = build_remote_code(
        "process_output", {"process_id": "process", "stream": "stdout", "limit": 100}
    )
    assert "_cm_status_value.get(_cm_stream + '_total_bytes')" in code
    assert "_cm_path_value.stat().st_size" in code
    assert "'stored_bytes': _cm_stored_bytes" in code
    assert "_cm_total_bytes = _cm_stored_bytes" in code
    assert "'total_bytes_final': _cm_status_value['status'] != 'running'" in code


@pytest.mark.parametrize(
    "operation",
    [
        "incarnation_init",
        "lease_probe",
        "process_start",
        "process_status",
        "process_list",
        "process_output",
        "process_signal",
        "fs_list",
        "fs_stat",
        "fs_read",
        "fs_write",
        "fs_mkdir",
        "fs_move",
        "fs_remove",
        "inspect",
    ],
)
def test_generated_remote_program_compiles(operation):
    compile(build_remote_code(operation, {}), "<colab-mcp-remote>", "exec")


def test_parse_remote_result_and_error():
    assert parse_remote_result(stream_result({"exit_code": 0})) == {"exit_code": 0}
    failure = json.dumps(
        {"ok": False, "error": {"type": "FileNotFoundError", "message": "missing"}}
    )
    with pytest.raises(RemoteOperationError, match="FileNotFoundError: missing"):
        parse_remote_result([{"output_type": "stream", "text": RESULT_PREFIX + failure}])
    with pytest.raises(RemoteOperationError, match="no structured result"):
        parse_remote_result([])


def test_parse_runtime_replaced_error_has_stable_type_and_code():
    failure = json.dumps(
        {
            "ok": False,
            "error": {
                "type": "RuntimeReplacedError",
                "message": "runtime_replaced: expected old, observed missing",
            },
        }
    )
    with pytest.raises(RuntimeReplacedError, match="runtime_replaced") as caught:
        parse_remote_result([{"output_type": "stream", "text": RESULT_PREFIX + failure}])
    assert caught.value.code == "runtime_replaced"


def test_remote_operations_verify_incarnation_before_process_or_filesystem_state():
    code = build_remote_code(
        "process_status",
        {
            "process_id": "old-process",
            "runtime_fingerprint": "a" * 32,
        },
    )
    assert code.index("_cm_actual_fingerprint != _cm_expected_fingerprint") < code.index(
        "_cm_metadata(_cm_payload['process_id'])"
    )


@pytest.mark.parametrize("argv", [[], [""], [1]])
def test_invalid_argv_is_rejected(argv):
    with pytest.raises(ValueError, match="argv"):
        validate_argv(argv)


def test_invalid_environment_is_rejected():
    with pytest.raises(ValueError, match="Invalid environment"):
        validate_environment({"BAD-NAME": "value"})
    with pytest.raises(ValueError, match="must be a string"):
        validate_environment({"VALID": 3})


@pytest.mark.parametrize("value", [0, MAX_OUTPUT_LIMIT + 1])
def test_invalid_output_limit_is_rejected(value):
    with pytest.raises(ValueError, match="output_limit"):
        validate_output_limit(value)


@pytest.mark.parametrize("value", [0, MAX_PROCESS_OUTPUT_LIMIT + 1])
def test_invalid_process_output_limit_is_rejected(value):
    with pytest.raises(ValueError, match="output_limit"):
        validate_process_output_limit(value)


def test_process_runner_caps_both_streams_and_reports_truncation(tmp_path):
    process_directory = tmp_path / "process"
    process_directory.mkdir()
    runner = process_directory / "runner.py"
    runner.write_text(PROCESS_RUNNER_SOURCE, encoding="utf-8")
    launch = process_directory / "launch.json"
    launch.write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-c",
                    "import sys;sys.stdout.write('o'*4096);sys.stderr.write('e'*3072)",
                ],
                "cwd": str(tmp_path),
                "environment": {},
                "directory": str(process_directory),
                "output_limit": 128,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run([sys.executable, str(runner), str(launch)], check=False, timeout=10)

    assert completed.returncode == 0
    assert not launch.exists()
    assert (process_directory / "stdout.log").stat().st_size == 128
    assert (process_directory / "stderr.log").stat().st_size == 128
    assert (process_directory / "stdout.truncated").is_file()
    assert (process_directory / "stderr.truncated").is_file()
    result = json.loads((process_directory / "exit.json").read_text(encoding="utf-8"))
    assert result["exit_code"] == 0
    assert result["stdout_total_bytes"] == 4096
    assert result["stderr_total_bytes"] == 3072
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True


def test_process_runner_spools_flushed_short_writes_before_exit(tmp_path):
    process_directory = tmp_path / "live-process"
    process_directory.mkdir()
    runner = process_directory / "runner.py"
    runner.write_text(PROCESS_RUNNER_SOURCE, encoding="utf-8")
    launch = process_directory / "launch.json"
    launch.write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-c",
                    "import time; print('{\"step\": 180}', flush=True); time.sleep(2)",
                ],
                "cwd": str(tmp_path),
                "environment": {},
                "directory": str(process_directory),
                "output_limit": 1024,
            }
        ),
        encoding="utf-8",
    )

    running = subprocess.Popen([sys.executable, str(runner), str(launch)])
    spool = process_directory / "stdout.log"
    try:
        expires = time.monotonic() + 1.5
        while (not spool.exists() or spool.stat().st_size == 0) and time.monotonic() < expires:
            time.sleep(0.02)
        assert running.poll() is None
        assert spool.read_text(encoding="utf-8") == '{"step": 180}\n'
    finally:
        running.wait(timeout=5)


@pytest.mark.parametrize("value", [0, 21_601])
def test_invalid_timeout_is_rejected(value):
    with pytest.raises(ValueError, match="timeout"):
        validate_timeout(value)


def test_run_command_returns_completed_durable_process(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)

    async def start(*_args, **_kwargs):
        return {"process_id": "p1", "cwd": "/content", "status": "running"}

    async def status(*_args, **_kwargs):
        return {"status": "exited", "exit_code": 0, "duration_seconds": 0.1}

    async def output(process_id, name, stream="stdout", offset=0, limit=65_536):
        return {"data": "ok\n" if stream == "stdout" else "", "more_available": False}

    instance.process_start = start
    instance.process_status = status
    instance.process_output = output
    result = asyncio.run(
        instance.run_command(
            ["python", "-c", "print('ok')"],
            "runtime",
            environment={"GREETING": "hello"},
            timeout=2,
        )
    )
    assert result["exit_code"] == 0
    assert result["stdout"] == "ok\n"
    assert result["process_id"] == "p1"
    assert result["process_continues"] is False


def test_run_command_timeout_hands_back_live_process(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)

    async def start(*_args, **_kwargs):
        return {"process_id": "p1", "cwd": "/content", "status": "running"}

    async def status(*_args, **_kwargs):
        return {"status": "running"}

    async def output(*_args, **_kwargs):
        return {"data": "partial\n", "more_available": False}

    instance.process_start = start
    instance.process_status = status
    instance.process_output = output
    result = asyncio.run(instance.run_command(["sleep", "10"], "runtime", timeout=0.1))
    assert result["timed_out"] is True
    assert result["process_continues"] is True
    assert result["status"] == "running"
    assert result["process_id"] == "p1"


def test_run_command_request_cancellation_does_not_signal_process(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    polled = asyncio.Event()
    signals = []

    async def start(*_args, **_kwargs):
        return {"process_id": "p1", "cwd": "/content", "status": "running"}

    async def status(*_args, **_kwargs):
        polled.set()
        return {"status": "running"}

    async def signal(*args, **kwargs):
        signals.append((args, kwargs))

    async def scenario():
        instance.process_start = start
        instance.process_status = status
        instance.process_signal = signal
        task = asyncio.create_task(instance.run_command(["sleep", "10"], "runtime", timeout=10))
        await polled.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert signals == []


def test_process_operations_cross_request_boundaries(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    instance.store.add(
        ManagedSessionState(
            name="runtime",
            token="secret",
            url="https://runtime",
            endpoint="endpoint",
            runtime_fingerprint="a" * 32,
        )
    )
    operations = []

    async def fake_remote(operation, payload, name, timeout=120):
        operations.append((operation, payload, name, timeout))
        if operation == "process_output":
            return {
                "operation": operation,
                "process_id": payload["process_id"],
                "status": "running",
                "next_offset": payload["offset"],
                "stored_bytes": 180,
                "total_bytes": 180,
            }
        return {
            "operation": operation,
            "process_id": payload["process_id"],
            **({"status": "running"} if operation == "process_status" else {}),
        }

    instance._remote_operation = fake_remote
    started = asyncio.run(instance.process_start(["sleep", "10"], "runtime"))
    asyncio.run(instance.process_status("abc", "runtime"))
    asyncio.run(instance.process_output("abc", "runtime", offset=4, limit=10))
    asyncio.run(instance.process_signal("abc", "runtime", "TERM"))
    assert started["operation"] == "process_start"
    assert [item[0] for item in operations] == [
        "process_start",
        "process_status",
        "process_output",
        "process_signal",
    ]
    assert operations[0][1]["process_id"]
    assert operations[0][1]["output_limit"] == 10_000_000
    journaled = instance.process_journal.get("runtime", "abc")
    assert journaled["stdout_stored_bytes"] == 180
    assert journaled["stdout_total_bytes"] == 180


def test_process_arguments_fail_before_remote_call(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="offset"):
        asyncio.run(instance.process_output("abc", None, offset=-1))
    with pytest.raises(ValueError, match="signal"):
        asyncio.run(instance.process_signal("abc", None, "STOP"))


def test_filesystem_operations_are_session_scoped(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    operations = []

    async def fake_remote(operation, payload, name, timeout=120):
        operations.append((operation, payload, name))
        return {"operation": operation}

    instance._remote_operation = fake_remote
    encoded = base64.b64encode(b"hello").decode()
    asyncio.run(instance.filesystem_list("/content/project", "runtime"))
    asyncio.run(instance.filesystem_stat("/content/project/file", "runtime", True))
    asyncio.run(instance.filesystem_read("/content/project/file", "runtime", 2, 10))
    asyncio.run(
        instance.filesystem_write("/content/project/file", encoded, "runtime", create_parents=True)
    )
    asyncio.run(instance.filesystem_mkdir("/content/dir", "runtime"))
    asyncio.run(instance.filesystem_move("/content/a", "/content/b", "runtime"))
    asyncio.run(instance.filesystem_remove("/content/b", "runtime", recursive=True))
    assert [item[0] for item in operations] == [
        "fs_list",
        "fs_stat",
        "fs_read",
        "fs_write",
        "fs_mkdir",
        "fs_move",
        "fs_remove",
    ]
    assert all(item[2] == "runtime" for item in operations)
    assert operations[3][1]["data_base64"] == encoded


def test_filesystem_input_bounds_fail_before_remote_call(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="base64"):
        asyncio.run(instance.filesystem_write("/content/file", "%%%", None))
    with pytest.raises(ValueError, match="offset"):
        asyncio.run(instance.filesystem_read("/content/file", None, offset=-1))
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(instance.filesystem_list("/content", None, limit=0))


def test_runtime_inspection_is_bounded_and_forwarded(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    seen = {}

    async def fake_remote(operation, payload, name, timeout=120):
        seen.update(operation=operation, payload=payload, name=name)
        return {"gpu": []}

    instance._remote_operation = fake_remote
    assert asyncio.run(instance.inspect_runtime("runtime", ["git"], 12)) == {"gpu": []}
    assert seen == {
        "operation": "inspect",
        "payload": {"tools": ["git"], "process_limit": 12},
        "name": "runtime",
    }
    with pytest.raises(ValueError, match="process_limit"):
        asyncio.run(instance.inspect_runtime(None, process_limit=0))
    with pytest.raises(ValueError, match="at most 100"):
        asyncio.run(instance.inspect_runtime(None, tools=["tool"] * 101))


def test_read_only_remote_operation_reconnects_kernel_once(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    instance.store.add(
        ManagedSessionState(
            name="runtime",
            token="secret",
            url="https://runtime",
            endpoint="endpoint",
            kernel_id="stale-kernel",
            runtime_fingerprint="a" * 32,
        )
    )
    calls = []

    async def execute(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise TimeoutError("kernel channel stalled")
        return stream_result({"path": "/content/file"})

    instance.execute = execute
    result = asyncio.run(
        instance._remote_operation("fs_stat", {"path": "/content/file"}, "runtime")
    )
    assert result == {"path": "/content/file"}
    assert len(calls) == 2
    assert instance.store.get("runtime").kernel_id is None


def test_mutating_remote_operation_is_never_retried(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    instance.store.add(
        ManagedSessionState(
            name="runtime",
            token="secret",
            url="https://runtime",
            endpoint="endpoint",
            runtime_fingerprint="a" * 32,
        )
    )
    calls = []

    async def execute(*_args, **_kwargs):
        calls.append(True)
        raise TimeoutError("unknown write outcome")

    instance.execute = execute
    with pytest.raises(TimeoutError, match="unknown write outcome"):
        asyncio.run(instance._remote_operation("fs_write", {}, "runtime"))
    assert len(calls) == 1


def test_remote_operation_rejects_legacy_session_before_execution(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    instance.store.add(
        SessionState(name="legacy", token="secret", url="https://runtime", endpoint="endpoint")
    )

    async def execute(*_args, **_kwargs):
        raise AssertionError("unverified runtime must not execute an operation")

    instance.execute = execute
    with pytest.raises(RuntimeReplacedError, match="stop it and start a new session"):
        asyncio.run(instance._remote_operation("fs_list", {"path": "/content"}, "legacy"))


def test_remote_operation_injects_persisted_fingerprint(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    instance.store.add(
        ManagedSessionState(
            name="runtime",
            token="secret",
            url="https://runtime",
            endpoint="endpoint",
            runtime_fingerprint="b" * 32,
        )
    )
    captured = {}

    def build(operation, payload):
        captured.update(operation=operation, payload=payload)
        return "remote code"

    async def execute(*_args, **_kwargs):
        return stream_result({"entries": []})

    monkeypatch.setattr("src.manager.build_remote_code", build)
    instance.execute = execute
    asyncio.run(instance._remote_operation("fs_list", {"path": "/content"}, "runtime"))
    assert captured["payload"]["runtime_fingerprint"] == "b" * 32


def test_replaced_runtime_returns_last_process_metadata_and_then_fails_fast(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    instance.store.add(
        ManagedSessionState(
            name="runtime",
            token="secret",
            url="https://runtime",
            endpoint="endpoint",
            runtime_fingerprint="a" * 32,
        )
    )
    instance.process_journal.update(
        "runtime",
        {
            "process_id": "process-1",
            "argv": ["python", "job.py"],
            "cwd": "/content/project",
            "status": "running",
            "pid": 123,
        },
    )
    calls = []

    async def execute(*_args, **_kwargs):
        calls.append(True)
        return stream_error(
            "RuntimeReplacedError",
            "runtime_replaced: expected incarnation a, observed missing",
        )

    instance.execute = execute
    first = asyncio.run(instance.process_status("process-1", "runtime"))
    second = asyncio.run(instance.process_status("process-1", "runtime"))

    assert len(calls) == 1
    assert first["status"] == second["status"] == "lost"
    assert first["last_known_process"]["argv"] == ["python", "job.py"]
    assert first["last_known_process"]["cwd"] == "/content/project"
    assert first["diagnostic"]["code"] == "runtime_replaced"
    assert first["diagnostic"]["probable_cause"] == "colab_runtime_recycle_or_runtime_oom"
    recovered = instance.store.get("runtime")
    assert recovered.runtime_replaced_at is not None


def test_file_operations_fail_fast_after_first_fingerprint_mismatch(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    instance.store.add(
        ManagedSessionState(
            name="runtime",
            token="secret",
            url="https://runtime",
            endpoint="endpoint",
            runtime_fingerprint="a" * 32,
        )
    )
    calls = []

    async def execute(*_args, **_kwargs):
        calls.append(True)
        return stream_error(
            "RuntimeReplacedError",
            "runtime_replaced: expected incarnation a, observed f",
        )

    instance.execute = execute
    with pytest.raises(RuntimeReplacedError):
        asyncio.run(instance.filesystem_list("/content", "runtime"))
    with pytest.raises(RuntimeReplacedError) as second:
        asyncio.run(instance.filesystem_stat("/content/file", "runtime"))

    assert len(calls) == 1
    assert second.value.details["probable_cause"] == "colab_runtime_recycle_or_runtime_oom"


def test_unknown_remote_process_returns_journaled_metadata_and_diagnostic(tmp_path, monkeypatch):
    instance = make_manager(tmp_path, monkeypatch)
    instance.store.add(
        ManagedSessionState(
            name="runtime",
            token="secret",
            url="https://runtime",
            endpoint="endpoint",
            runtime_fingerprint="a" * 32,
        )
    )
    instance.process_journal.update(
        "runtime",
        {"process_id": "process-1", "argv": ["worker"], "status": "running", "pid": 9},
    )

    async def execute(*_args, **_kwargs):
        return stream_error("FileNotFoundError", "Unknown process_id: process-1")

    instance.execute = execute
    result = asyncio.run(instance.process_status("process-1", "runtime"))

    assert result["status"] == "lost"
    assert result["last_known_process"]["pid"] == 9
    assert result["diagnostic"]["code"] == "process_state_lost"
    assert result["diagnostic"]["probable_cause"] == "remote_process_state_lost_or_runtime_oom"
