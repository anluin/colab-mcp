import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.supervisor import ColabMcpSupervisor, Worker, source_fingerprint


def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    source = root / "src"
    source.mkdir(parents=True)
    (source / "cli.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "worker.py").write_text("VALUE = 2\n", encoding="utf-8")
    return root


def fake_worker(pid: int, fingerprint: str) -> Worker:
    process = SimpleNamespace(pid=pid, returncode=None)
    return Worker(process, fingerprint)  # type: ignore[arg-type]


def test_source_fingerprint_is_deterministic_and_changes_with_worker_code(tmp_path: Path):
    root = source_root(tmp_path)
    first = source_fingerprint(root)
    second = source_fingerprint(root)
    (root / "src" / "worker.py").write_text("VALUE = 3\n", encoding="utf-8")

    assert first == second
    assert source_fingerprint(root) != first


def test_invalid_hot_reload_root_fails_before_starting_transport(tmp_path: Path):
    with pytest.raises(RuntimeError, match="hot_reload_source_invalid"):
        ColabMcpSupervisor(tmp_path)


def test_status_remains_available_while_source_files_are_being_repaired(tmp_path: Path):
    root = source_root(tmp_path)
    instance = ColabMcpSupervisor(root)
    instance.worker = fake_worker(100, "old")
    for path in (root / "src").glob("*.py"):
        path.unlink()

    status = instance.status()
    assert status["worker_running"] is True
    assert status["available_source_fingerprint"] is None
    assert "hot_reload_source_missing" in status["available_source_error"]


def test_candidate_validation_failure_keeps_current_worker(tmp_path: Path, monkeypatch):
    instance = ColabMcpSupervisor(source_root(tmp_path))
    active = fake_worker(100, "old")
    candidate = fake_worker(200, "new")
    instance.worker = active
    stopped = []

    async def spawn(_root=None):
        return candidate

    async def validate(_worker):
        raise RuntimeError("candidate health failed")

    async def stop(worker, timeout=10):
        stopped.append((worker, timeout))

    monkeypatch.setattr(instance, "_spawn_worker", spawn)
    monkeypatch.setattr(instance, "_validate_candidate", validate)
    monkeypatch.setattr(instance, "_stop_worker", stop)

    with pytest.raises(RuntimeError, match="candidate health failed"):
        asyncio.run(instance.reload(None, 5))

    assert instance.worker is active
    assert stopped == [(candidate, 10)]
    assert instance.generation == 0


def test_expected_fingerprint_rejects_changed_source_before_spawn(tmp_path: Path, monkeypatch):
    instance = ColabMcpSupervisor(source_root(tmp_path))
    instance.worker = fake_worker(100, "old")
    spawned = False

    async def spawn(_root=None):
        nonlocal spawned
        spawned = True
        return fake_worker(200, "new")

    monkeypatch.setattr(instance, "_spawn_worker", spawn)
    with pytest.raises(RuntimeError, match="hot_reload_source_changed"):
        asyncio.run(instance.reload("0" * 64, 5))
    assert spawned is False
    assert instance.worker.pid == 100


def test_source_binding_rejects_non_project_git_origin(tmp_path: Path, monkeypatch):
    root = source_root(tmp_path)

    def completed(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="https://github.com/example/other.git\n")

    monkeypatch.setattr("src.supervisor.subprocess.run", completed)
    with pytest.raises(RuntimeError, match="hot_reload_source_origin_rejected"):
        ColabMcpSupervisor._validated_source_root(str(root))


def test_worker_crash_fails_pending_requests_and_keeps_supervisor_recoverable(tmp_path: Path):
    instance = ColabMcpSupervisor(source_root(tmp_path))

    class EmptyStdout:
        async def readline(self):
            return b""

    class ExitedProcess:
        pid = 100
        returncode = 7
        stdout = EmptyStdout()

        async def wait(self):
            return self.returncode

    worker = Worker(ExitedProcess(), "old")  # type: ignore[arg-type]
    instance.worker = worker
    key = instance._request_key(42)
    instance._pending.add(key)
    instance._idle.clear()
    responses = []

    async def write(message):
        responses.append(message)

    instance._write_client = write  # type: ignore[method-assign]
    asyncio.run(instance._read_worker(worker))

    assert responses[0]["id"] == 42
    assert responses[0]["error"]["code"] == -32001
    assert "response_unknown" in responses[0]["error"]["message"]
    assert instance._pending == set()
    assert instance._idle.is_set()
