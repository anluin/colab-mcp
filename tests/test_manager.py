import asyncio
import base64
import gzip
import hashlib
import json
import os
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli import (
    _client_executable,
    _client_has_server,
    _ensure_codex_timeouts,
    claude_desktop_config_path,
    config_json,
    install_claude_desktop,
    install_command,
    isolated_server_command,
    server_command,
)
from src.manager import (
    AutoExportRule,
    ColabManager,
    KernelConnectionError,
    ManagedSessionState,
    OperationLeaseError,
    RequestOutcomeUnknownError,
    TransferError,
    _bound_outputs,
    _extract_control_timing,
    _json_safe,
    _secure_permissions,
    require_local_file,
)


def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ColabManager:
    monkeypatch.setenv("COLAB_MCP_STATE_DIR", str(tmp_path / "state"))
    instance = ColabManager()

    async def stable_probe(*_args, **_kwargs):
        return {"status": "stable", "observations": 2, "lease_token": "b" * 32}

    async def stable_operation_lease(name, lease_token=None):
        runtime = SimpleNamespace(
            name=name or "runtime",
            endpoint="endpoint",
            runtime_fingerprint="a" * 32,
        )
        return runtime, {
            "status": "stable",
            "lease_token": lease_token or "b" * 32,
            "assignment_lookup_seconds": 0.001,
        }

    async def quiet_heartbeat(_session, stop):
        await stop.wait()

    instance.allocation_probe = stable_probe
    instance._operation_lease = stable_operation_lease
    instance._critical_heartbeat = quiet_heartbeat
    return instance


def test_require_local_file(tmp_path: Path):
    file = tmp_path / "experiment.py"
    file.write_text("print('ok')")
    assert require_local_file(str(file)) == file.resolve()


def test_require_local_file_rejects_missing(tmp_path: Path):
    with pytest.raises(ValueError):
        require_local_file(str(tmp_path / "missing.py"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not Windows ACLs")
def test_posix_state_permissions_are_owner_only(tmp_path: Path):
    directory = tmp_path / "state"
    directory.mkdir()
    state = directory / "sessions.json"
    state.write_text("{}", encoding="utf-8")

    _secure_permissions(directory, 0o700, platform="posix")
    _secure_permissions(state, 0o600, platform="posix")

    assert directory.stat().st_mode & 0o777 == 0o700
    assert state.stat().st_mode & 0o777 == 0o600


def test_windows_permission_helper_is_a_safe_noop(tmp_path: Path):
    state = tmp_path / "sessions.json"
    state.write_text("{}", encoding="utf-8")
    before = state.stat().st_mode

    _secure_permissions(state, 0o600, platform="nt")

    assert state.stat().st_mode == before


def test_json_safe_replaces_bytes():
    assert _json_safe({"data": b"abc"}) == {"data": "<3 bytes>"}


def test_kernel_outputs_are_bounded_with_explicit_marker():
    outputs = [{"output_type": "stream", "name": "stdout", "text": "x" * 10_000}]
    bounded = _bound_outputs(outputs, 1_000)
    assert bounded[-1]["text"] == "[colab-mcp: output truncated]"
    assert len(json.dumps(bounded).encode()) <= 1_000


def test_create_notebook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    target = tmp_path / "experiment.ipynb"
    result = instance.create_notebook(str(target), ["print('GPU experiment')"])
    assert result["cells"] == 1
    assert target.exists()


def test_create_notebook_does_not_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    target = tmp_path / "experiment.ipynb"
    target.write_text("keep me")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        instance.create_notebook(str(target))


def test_codex_registration_command_is_portable(tmp_path: Path):
    command = install_command("codex", "codex", "uv", tmp_path, "colab")
    assert command[:4] == ["codex", "mcp", "add", "colab"]
    assert command[-6:] == ["--directory", str(tmp_path), "run", "--locked", "colab-mcp", "serve"]


def test_windows_codex_prefers_launchable_app_cli_over_windowsapps_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cli = tmp_path / "OpenAI" / "Codex" / "bin" / "build" / "codex.exe"
    cli.parent.mkdir(parents=True)
    cli.write_bytes(b"")
    monkeypatch.setattr(
        "src.cli.shutil.which", lambda _name: r"C:\Program Files\WindowsApps\Codex\codex.exe"
    )

    assert _client_executable("codex", system="Windows", localappdata=str(tmp_path)) == str(cli)


def test_windows_codex_honors_explicit_cli_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cli = tmp_path / "codex.exe"
    cli.write_bytes(b"")
    monkeypatch.setenv("CODEX_CLI_PATH", str(cli))

    assert _client_executable("codex", system="Windows", localappdata=str(tmp_path)) == str(cli)


def test_codex_timeout_setup_preserves_config_and_is_idempotent(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text(
        '[mcp_servers.colab]\ncommand = "uv"\n\n[mcp_servers.other]\ncommand = "other"\n',
        encoding="utf-8",
    )

    _ensure_codex_timeouts("colab", config)
    _ensure_codex_timeouts("colab", config)

    text = config.read_text(encoding="utf-8")
    assert text.count("startup_timeout_sec = 30") == 1
    assert text.count("tool_timeout_sec = 21600") == 1
    assert '[mcp_servers.other]\ncommand = "other"' in text


def test_claude_registration_uses_user_scoped_stdio(tmp_path: Path):
    command = install_command("claude", "claude", "uv", tmp_path, "colab")
    assert command[:8] == [
        "claude",
        "mcp",
        "add",
        "--transport",
        "stdio",
        "--scope",
        "user",
        "--env",
    ]
    assert command[-6:] == ["--directory", str(tmp_path), "run", "--locked", "colab-mcp", "serve"]


def test_grok_registration_uses_isolated_user_scope(tmp_path: Path):
    command = install_command("grok", "grok", "uv", tmp_path, "colab")
    assert command[:6] == ["grok", "mcp", "add", "--scope", "user", "colab"]
    assert "-e" in command
    assert "COLAB_MCP_AUTH=oauth2" in command
    assert command[-7:] == [
        "--directory",
        str(tmp_path),
        "run",
        "--isolated",
        "--locked",
        "colab-mcp",
        "serve",
    ]


def test_generic_json_config(tmp_path: Path):
    text = config_json("uv", tmp_path)
    assert '"colab"' in text
    assert server_command("uv", tmp_path)[1:] == [
        "--directory",
        str(tmp_path),
        "run",
        "--locked",
        "colab-mcp",
        "serve",
    ]


def test_claude_desktop_paths_are_cross_platform(tmp_path: Path):
    assert claude_desktop_config_path(
        "Windows",
        tmp_path,
        str(tmp_path / "roaming"),
        str(tmp_path / "local"),
    ) == (tmp_path / "roaming" / "Claude" / "claude_desktop_config.json")
    assert claude_desktop_config_path("Darwin", tmp_path) == (
        tmp_path / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    )
    assert claude_desktop_config_path("Linux", tmp_path) == (
        tmp_path / ".config" / "Claude" / "claude_desktop_config.json"
    )


def test_claude_desktop_prefers_microsoft_store_virtualized_config(tmp_path: Path):
    packaged = (
        tmp_path
        / "local"
        / "Packages"
        / "Claude_package"
        / "LocalCache"
        / "Roaming"
        / "Claude"
        / "claude_desktop_config.json"
    )
    packaged.parent.mkdir(parents=True)
    packaged.write_text('{"mcpServers": {}}', encoding="utf-8")

    assert (
        claude_desktop_config_path(
            "Windows",
            tmp_path,
            str(tmp_path / "roaming"),
            str(tmp_path / "local"),
        )
        == packaged
    )


def test_claude_desktop_install_preserves_existing_config(tmp_path: Path, monkeypatch, capsys):
    config = tmp_path / "claude_desktop_config.json"
    config.write_text(json.dumps({"theme": "dark", "mcpServers": {"existing": {}}}))
    monkeypatch.setattr("src.cli.claude_desktop_config_path", lambda: config)
    monkeypatch.setattr("src.cli._uv", lambda: "uv")
    monkeypatch.setattr("src.cli._project", lambda: tmp_path / "project")

    install_claude_desktop("colab", False)

    result = json.loads(config.read_text(encoding="utf-8"))
    assert result["theme"] == "dark"
    assert "existing" in result["mcpServers"]
    assert result["mcpServers"]["colab"]["command"] == "uv"
    assert (
        result["mcpServers"]["colab"]["args"]
        == isolated_server_command("uv", tmp_path / "project")[1:]
    )
    assert "--isolated" in result["mcpServers"]["colab"]["args"]
    assert "Restart Claude Desktop" in capsys.readouterr().out


def test_client_has_server_parses_grok_list_json(monkeypatch: pytest.MonkeyPatch):
    class Result:
        returncode = 0
        stdout = json.dumps(
            [
                {
                    "name": "colab",
                    "scope": "user",
                    "command": "uv",
                    "args": ["run", "colab-mcp", "serve"],
                    "enabled": True,
                }
            ]
        )

    def fake_run(command, **_kwargs):
        assert command[:3] == ["grok", "mcp", "list"]
        return Result()

    monkeypatch.setattr("src.cli.subprocess.run", fake_run)
    assert _client_has_server("grok", "grok", "colab")
    assert not _client_has_server("grok", "grok", "other")


def test_server_credentials_never_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    token = tmp_path / "token.json"
    token.write_text("not valid credentials")
    monkeypatch.setattr("src.manager.TOKEN_CONFIG_PATH", str(token))
    monkeypatch.setattr(
        "src.manager.Credentials.from_authorized_user_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid")),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not prompt")),
    )
    with pytest.raises(RuntimeError, match="never prompt for OAuth"):
        instance.client()


def session(name: str, endpoint: str) -> ManagedSessionState:
    return ManagedSessionState(
        name=name,
        token="secret",
        url="https://runtime",
        endpoint=endpoint,
        runtime_fingerprint="a" * 32,
    )


def test_runtime_fingerprint_survives_state_store_restart(tmp_path, monkeypatch):
    first = manager(tmp_path, monkeypatch)
    first.store.add(session("runtime", "endpoint"))

    recovered = manager(tmp_path, monkeypatch).store.get("runtime")

    assert isinstance(recovered, ManagedSessionState)
    assert recovered.runtime_fingerprint == "a" * 32


def test_reconcile_reports_and_explicitly_cleans_stale_and_orphaned_assignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("active", "endpoint-active"))
    instance.store.add(session("stale", "endpoint-stale"))

    class FakeClient:
        def __init__(self):
            self.released = []

        def list_assignments(self):
            return [SimpleNamespace(endpoint="endpoint-active"), SimpleNamespace(endpoint="orphan")]

        def unassign(self, endpoint):
            self.released.append(endpoint)

    client = FakeClient()
    instance.client = lambda: client
    audit = asyncio.run(instance.reconcile())
    assert audit["stale_sessions"] == [{"session": "stale", "endpoint": "endpoint-stale"}]
    assert audit["orphan_endpoints"] == ["orphan"]
    assert instance.store.get("stale") is not None
    cleaned = asyncio.run(instance.reconcile(forget_stale=True, release_orphans=True))
    assert cleaned["forgotten_sessions"] == ["stale"]
    assert cleaned["released_orphans"] == ["orphan"]
    assert client.released == ["orphan"]
    assert instance.store.get("stale") is None


def test_reconcile_retains_failed_orphan_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)

    class FakeClient:
        def list_assignments(self):
            return [SimpleNamespace(endpoint="orphan")]

        def unassign(self, endpoint):
            raise OSError("network down")

    instance.client = lambda: FakeClient()
    result = asyncio.run(instance.reconcile(release_orphans=True))
    assert result["released_orphans"] == []
    assert result["errors"] == [{"endpoint": "orphan", "error": "network down"}]


def test_stop_is_idempotent_for_stale_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("stale", "endpoint-stale"))

    class FakeClient:
        def list_assignments(self):
            return []

        def unassign(self, endpoint):
            raise AssertionError("must not unassign an already absent endpoint")

    instance.client = lambda: FakeClient()
    result = asyncio.run(instance.stop("stale"))
    assert result == {"stopped": "stale", "runtime_was_active": False}
    assert instance.store.get("stale") is None


def test_start_retains_assignment_when_preflight_and_cleanup_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)

    class FakeClient:
        def assign(self, *_args):
            proxy = SimpleNamespace(token="secret", url="https://runtime")
            return SimpleNamespace(endpoint="allocated", runtime_proxy_info=proxy)

        def keep_alive_assignment(self, endpoint):
            raise OSError("preflight down")

        def unassign(self, endpoint):
            raise OSError("cleanup down")

    instance.client = lambda: FakeClient()
    with pytest.raises(RuntimeError, match="remains tracked"):
        asyncio.run(instance.start("recoverable", None))
    assert instance.store.get("recoverable").endpoint == "allocated"


def test_start_releases_and_forgets_failed_preflight_when_cleanup_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)

    class FakeClient:
        def __init__(self):
            self.released = []

        def assign(self, *_args):
            proxy = SimpleNamespace(token="secret", url="https://runtime")
            return SimpleNamespace(endpoint="allocated", runtime_proxy_info=proxy)

        def keep_alive_assignment(self, endpoint):
            raise OSError("preflight down")

        def unassign(self, endpoint):
            self.released.append(endpoint)

    client = FakeClient()
    instance.client = lambda: client
    with pytest.raises(RuntimeError, match="failed preflight and was released"):
        asyncio.run(instance.start("failed", None))
    assert client.released == ["allocated"]
    assert instance.store.get("failed") is None


def test_start_initializes_and_persists_runtime_incarnation(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    initialized = []

    class FakeClient:
        def assign(self, *_args):
            proxy = SimpleNamespace(token="secret", url="https://runtime")
            return SimpleNamespace(endpoint="allocated", runtime_proxy_info=proxy)

        def keep_alive_assignment(self, endpoint):
            return None

    async def initialize(runtime):
        initialized.append(runtime.runtime_fingerprint)

    instance.client = lambda: FakeClient()
    instance._initialize_runtime_incarnation = initialize
    created = asyncio.run(instance.start("runtime", None))

    assert len(initialized) == 1
    assert initialized[0] == created.runtime_fingerprint
    assert len(created.runtime_fingerprint) == 32
    assert instance.store.get("runtime").runtime_fingerprint == created.runtime_fingerprint


def test_start_releases_runtime_when_incarnation_initialization_fails(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    released = []

    class FakeClient:
        def assign(self, *_args):
            proxy = SimpleNamespace(token="secret", url="https://runtime")
            return SimpleNamespace(endpoint="allocated", runtime_proxy_info=proxy)

        def keep_alive_assignment(self, endpoint):
            return None

        def unassign(self, endpoint):
            released.append(endpoint)

    async def initialize(_runtime):
        raise OSError("kernel unavailable")

    instance.client = lambda: FakeClient()
    instance._initialize_runtime_incarnation = initialize
    with pytest.raises(RuntimeError, match="failed preflight and was released"):
        asyncio.run(instance.start("runtime", None))

    assert released == ["allocated"]
    assert instance.store.get("runtime") is None


def test_keepalive_retries_after_transient_error_and_persists_health(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    instance.keepalive_seconds = 0.01
    attempts = []

    class FakeClient:
        def keep_alive_assignment(self, endpoint):
            attempts.append(endpoint)
            if len(attempts) == 1:
                raise OSError("temporary network failure")

        def list_assignments(self):
            return [SimpleNamespace(endpoint="endpoint")]

    async def scenario():
        instance.client = lambda: FakeClient()
        instance.ensure_keepalive(instance.resolve("runtime"))
        deadline = asyncio.get_running_loop().time() + 1
        while asyncio.get_running_loop().time() < deadline:
            current = instance.store.get("runtime")
            if len(attempts) >= 2 and current.keepalive_status == "healthy":
                break
            await asyncio.sleep(0.01)
        await instance.shutdown_keepalives()

    asyncio.run(scenario())
    current = instance.store.get("runtime")
    assert len(attempts) >= 2
    assert current.keepalive_status == "healthy"
    assert current.last_keepalive_at is not None
    assert current.last_keepalive_error is None
    assert current.consecutive_keepalive_failures == 0


def test_keepalive_status_can_refresh_immediately(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    class FakeClient:
        def keep_alive_assignment(self, endpoint):
            assert endpoint == "endpoint"

    async def scenario():
        instance.client = lambda: FakeClient()
        result = await instance.keepalive("runtime", refresh=True)
        await instance.shutdown_keepalives()
        return result

    result = asyncio.run(scenario())
    assert result["status"] == "healthy"
    assert result["refresh_succeeded"] is True
    assert result["background_task_running"] is True
    assert result["guarantees_runtime_persistence"] is False


def test_server_restart_recovers_active_keepalives_and_marks_lost_leases(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("active", "endpoint-active"))
    instance.store.add(session("lost", "endpoint-lost"))
    recovered = []

    class FakeClient:
        def list_assignments(self):
            return [SimpleNamespace(endpoint="endpoint-active")]

    instance.client = lambda: FakeClient()
    instance.ensure_keepalive = lambda value: recovered.append(value.name)
    result = asyncio.run(instance.recover_keepalives())

    assert result == {"recovered": ["active"], "lease_lost": ["lost"], "error": None}
    assert recovered == ["active"]
    assert instance.store.get("lost").keepalive_status == "lease_lost"


def test_keepalive_recovery_with_no_sessions_never_contacts_colab(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.client = lambda: (_ for _ in ()).throw(AssertionError("must stay offline"))

    assert asyncio.run(instance.recover_keepalives()) == {
        "recovered": [],
        "lease_lost": [],
        "error": None,
    }


def test_allocation_probe_observes_owned_lease_and_runtime_incarnation(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    keepalives = []

    class FakeClient:
        def list_assignments(self):
            return [SimpleNamespace(endpoint="endpoint")]

        def keep_alive_assignment(self, endpoint):
            keepalives.append(endpoint)

    async def remote(operation, payload, name, timeout=120, **_kwargs):
        assert operation == "lease_probe"
        assert name == "runtime"
        assert len(payload["issue_lease_token"]) == 32
        assert payload["lease_expires_at"]
        return {
            "status": "stable",
            "runtime_fingerprint": "a" * 32,
            "observed_at": "2026-08-10T00:00:00+00:00",
        }

    instance.client = lambda: FakeClient()
    instance._remote_operation = remote
    result = asyncio.run(
        ColabManager.allocation_probe(instance, "runtime", observations=3, interval=0)
    )

    assert result["status"] == "stable"
    assert result["observations"] == 3
    assert result["runtime_fingerprint"] == "a" * 32
    assert len(result["lease_token"]) == 32
    assert instance.store.get("runtime").operation_lease_token == result["lease_token"]
    assert keepalives == []
    assert result["heartbeat"] == "background"


def test_allocation_probe_does_not_block_on_slow_keepalive_endpoint(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    class FakeClient:
        def list_assignments(self):
            return [SimpleNamespace(endpoint="endpoint")]

        def keep_alive_assignment(self, _endpoint):
            raise AssertionError("probe must use the existing background heartbeat")

    async def remote(*_args, **_kwargs):
        return {
            "runtime_fingerprint": "a" * 32,
            "observed_at": "2026-08-10T00:00:00+00:00",
        }

    instance.client = lambda: FakeClient()
    instance._remote_operation = remote
    result = asyncio.run(ColabManager.allocation_probe(instance, "runtime", interval=0))
    assert result["status"] == "stable"


def test_allocation_probe_fails_before_remote_access_when_lease_is_lost(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    class FakeClient:
        def list_assignments(self):
            return []

    instance.client = lambda: FakeClient()
    with pytest.raises(RuntimeError, match="allocation_lease_lost"):
        asyncio.run(ColabManager.allocation_probe(instance, "runtime", interval=0))


def test_explicit_operation_lease_fails_fast_when_assignment_vanished(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    runtime = session("runtime", "endpoint")
    runtime.operation_lease_token = "b" * 32
    runtime.operation_lease_expires_at = "2099-01-01T00:00:00+00:00"
    instance.store.add(runtime)

    class MissingClient:
        def list_assignments(self):
            return []

    instance.client = lambda: MissingClient()
    started = time.monotonic()
    with pytest.raises(OperationLeaseError) as caught:
        asyncio.run(ColabManager._operation_lease(instance, "runtime", "b" * 32))
    assert caught.value.code == "assignment_no_longer_exists"
    assert time.monotonic() - started < 5


def test_stale_operation_lease_is_rejected_before_assignment_lookup(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    runtime = session("runtime", "endpoint")
    runtime.operation_lease_token = "b" * 32
    runtime.operation_lease_expires_at = "2099-01-01T00:00:00+00:00"
    instance.store.add(runtime)
    with pytest.raises(OperationLeaseError) as caught:
        ColabManager._validate_operation_lease(instance, runtime, "c" * 32)
    assert caught.value.code == "operation_lease_stale"


def test_guard_timing_is_extracted_without_leaking_control_output():
    outputs = [
        {
            "output_type": "stream",
            "text": (
                '__COLAB_MCP_CONTROL_TIMING__{"fingerprint_validation_seconds":0.012}\n'
                "user output\n"
            ),
        }
    ]
    assert _extract_control_timing(outputs) == {"fingerprint_validation_seconds": 0.012}
    assert outputs[0]["text"] == "user output\n"


def test_detailed_execute_is_lease_guarded_and_reports_control_timings(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    captured = []

    async def detailed(code, *_args, **_kwargs):
        captured.append(code)
        return {
            "outputs": [
                {
                    "output_type": "stream",
                    "text": (
                        '__COLAB_MCP_CONTROL_TIMING__{"fingerprint_validation_seconds":0.002}\nok\n'
                    ),
                }
            ],
            "timings": {"assignment_lookup_seconds": None, "total_seconds": 1.0},
        }

    instance._execute_detailed = detailed
    result = asyncio.run(instance.execute_python_detailed("print('ok')", "runtime"))
    assert captured and captured[0].index("operation-lease.json") < captured[0].index("print('ok')")
    assert result["outputs"][0]["text"] == "ok\n"
    assert result["timings"]["assignment_lookup_seconds"] == 0.001
    assert result["timings"]["fingerprint_validation_seconds"] == 0.002


def test_request_timeout_is_classified_as_submission_outcome_unknown(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    class TimeoutKernel:
        id = "kernel"

        def __init__(self, **_kwargs):
            pass

        def start(self, timeout):
            pass

        def execute(self, code, timeout):
            if code.startswith("import os;"):
                return {"outputs": []}
            raise TimeoutError("late")

        def stop(self, shutdown_kernel=False):
            pass

    monkeypatch.setattr("src.manager.kernel_client", TimeoutKernel)
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        asyncio.run(instance.execute("print('x')", "runtime", 1, connection_attempts=1))
    assert caught.value.code == "operation_timed_out_submission_outcome_unknown"
    assert caught.value.details["request_submission"] == "outcome_unknown"


def test_kernel_failure_always_closes_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    class FakeKernel:
        id = "kernel"

        def __init__(self):
            self.execute_count = 0
            self.stopped = False

        def start(self, timeout):
            return None

        def execute(self, code, timeout):
            self.execute_count += 1
            if self.execute_count == 2:
                raise OSError("kernel disappeared")
            return {"outputs": []}

        def stop(self, shutdown_kernel=False):
            self.stopped = True

    kernel = FakeKernel()
    monkeypatch.setattr("src.manager.kernel_client", lambda **_kwargs: kernel)
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        asyncio.run(instance.execute_python("print('hello')", "runtime"))
    assert caught.value.code == "request_submission_outcome_unknown_response_lost"
    assert kernel.stopped is True


def test_successive_tools_reuse_one_verified_kernel_channel(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    created = []

    class FakeKernel:
        id = "kernel"

        def __init__(self):
            self.started = 0
            self.stopped = 0
            self.executed = []

        def start(self, timeout):
            self.started += 1

        def execute(self, code, timeout):
            self.executed.append(code)
            return {"outputs": [{"output_type": "stream", "text": "ok\n"}]}

        def stop(self, shutdown_kernel=False):
            self.stopped += 1

    def factory(**_kwargs):
        kernel = FakeKernel()
        created.append(kernel)
        return kernel

    monkeypatch.setattr("src.manager.kernel_client", factory)
    first = asyncio.run(instance._execute_detailed("first", "runtime", 10))
    second = asyncio.run(instance._execute_detailed("second", "runtime", 10))

    assert len(created) == 1
    assert created[0].started == 1
    assert created[0].stopped == 0
    assert created[0].executed == [
        "import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')",
        "first",
        "import os; os.chdir('/content')",
        "second",
    ]
    assert first["timings"]["attempts"][0]["kernel_connection_reused"] is False
    assert second["timings"]["attempts"][0]["kernel_connection_reused"] is True
    asyncio.run(instance.shutdown_kernel_channels())
    assert created[0].stopped == 1


def test_cached_channel_preflight_failure_reconnects_before_user_code(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    created = []
    user_code = []

    class FakeKernel:
        id = "kernel"

        def __init__(self, fail_later=False):
            self.fail_later = fail_later
            self.calls = 0

        def start(self, timeout):
            pass

        def execute(self, code, timeout):
            self.calls += 1
            if self.fail_later and self.calls == 3:
                raise OSError("cached websocket ended")
            if not code.startswith("import os;"):
                user_code.append(code)
            return {"outputs": []}

        def stop(self, shutdown_kernel=False):
            pass

    def factory(**_kwargs):
        kernel = FakeKernel(fail_later=not created)
        created.append(kernel)
        return kernel

    monkeypatch.setattr("src.manager.kernel_client", factory)
    asyncio.run(instance.execute("first", "runtime", 10))
    asyncio.run(instance.execute("second", "runtime", 10))
    assert len(created) == 2
    assert user_code == ["first", "second"]


def test_kernel_connection_failure_retries_before_user_code_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    attempts = []
    user_code = []

    class FakeKernel:
        id = "new-kernel"

        def start(self, timeout):
            return None

        def execute(self, code, timeout):
            if code.startswith("import os;"):
                return {"outputs": []}
            user_code.append(code)
            return {"outputs": [{"output_type": "stream", "text": "ok\n"}]}

        def stop(self, shutdown_kernel=False):
            return None

    def kernel_factory(**_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("proxy unavailable before send")
        return FakeKernel()

    monkeypatch.setattr("src.manager.kernel_client", kernel_factory)
    outputs = asyncio.run(instance.execute_python("print('once')", "runtime"))
    assert outputs == [{"output_type": "stream", "text": "ok\n"}]
    assert len(attempts) == 2
    assert user_code == ["print('once')"]
    assert instance.store.get("runtime").kernel_id == "new-kernel"


def test_kernel_connection_retry_never_clears_owned_incarnation_kernel_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    runtime = session("runtime", "endpoint")
    runtime.kernel_id = "owned-kernel"
    instance.store.add(runtime)
    received_ids = []

    class FakeKernel:
        id = "owned-kernel"

        def start(self, timeout):
            pass

        def execute(self, code, timeout):
            return {"outputs": [{"output_type": "stream", "text": "ok\n"}]}

        def stop(self, shutdown_kernel=False):
            pass

    def kernel_factory(**kwargs):
        received_ids.append(kwargs["kernel_id"])
        if len(received_ids) == 1:
            raise OSError("transient proxy failure")
        return FakeKernel()

    monkeypatch.setattr("src.manager.kernel_client", kernel_factory)
    asyncio.run(instance.execute_python("print('ok')", "runtime"))
    assert received_ids == ["owned-kernel", "owned-kernel"]
    assert instance.store.get("runtime").kernel_id == "owned-kernel"


def test_lease_bound_download_retries_pre_submission_failure_without_replacing_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    runtime = session("runtime", "endpoint")
    runtime.operation_lease_token = "b" * 32
    runtime.operation_lease_expires_at = "2099-01-01T00:00:00+00:00"
    instance.store.add(runtime)
    attempts = 0
    lease_checks = []

    async def execute(*_args, **kwargs):
        nonlocal attempts
        attempts += 1
        assert kwargs["connection_timeout"] == 20
        assert kwargs["connection_attempts"] == 1
        if attempts == 1:
            raise KernelConnectionError("proxy unavailable before send")
        return [
            {
                "output_type": "stream",
                "text": '__COLAB_MCP_RESULT__{"ok":true,"result":{"ok":true}}\n',
            }
        ]

    async def preserve_lease(name, token):
        lease_checks.append((name, token))
        return runtime, {"lease_token": token, "runtime_fingerprint": "a" * 32}

    instance.execute = execute
    instance._operation_lease = preserve_lease
    result = asyncio.run(
        instance._remote_operation(
            "fs_read",
            {"path": "/content/file", "offset": 0, "limit": 3},
            "runtime",
            lease_token="b" * 32,
        )
    )

    assert result == {"ok": True}
    assert attempts == 2
    assert lease_checks == [("runtime", "b" * 32)]
    assert instance.store.get("runtime").operation_lease_token == "b" * 32


def test_process_start_retries_when_connection_failed_before_submission(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    runtime = session("runtime", "endpoint")
    runtime.operation_lease_token = "b" * 32
    runtime.operation_lease_expires_at = "2099-01-01T00:00:00+00:00"
    instance.store.add(runtime)
    attempts = 0
    lease_checks = []

    async def execute(*_args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise KernelConnectionError("proxy unavailable before send")
        return [
            {
                "output_type": "stream",
                "text": '__COLAB_MCP_RESULT__{"ok":true,"result":{"process_id":"p1"}}\n',
            }
        ]

    async def preserve_lease(name, token):
        lease_checks.append((name, token))
        return runtime, {"lease_token": token, "runtime_fingerprint": "a" * 32}

    instance.execute = execute
    instance._operation_lease = preserve_lease
    result = asyncio.run(
        instance._remote_operation(
            "process_start",
            {"argv": ["python", "job.py"]},
            "runtime",
            lease_token="b" * 32,
        )
    )
    assert result == {"process_id": "p1"}
    assert attempts == 2
    assert lease_checks == [("runtime", "b" * 32)]


def test_lease_bound_upload_chunk_retries_confirmed_pre_submission_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    runtime = session("runtime", "endpoint")
    runtime.operation_lease_token = "b" * 32
    runtime.operation_lease_expires_at = "2099-01-01T00:00:00+00:00"
    instance.store.add(runtime)
    attempts = 0

    async def execute(*_args, **kwargs):
        nonlocal attempts
        attempts += 1
        assert kwargs["connection_timeout"] == 20
        assert kwargs["connection_attempts"] == 1
        if attempts == 1:
            raise KernelConnectionError("proxy unavailable before send")
        return [
            {
                "output_type": "stream",
                "text": '__COLAB_MCP_RESULT__{"ok":true,"result":{"offset":3}}\n',
            }
        ]

    async def preserve_lease(name, token):
        return runtime, {"lease_token": token, "runtime_fingerprint": "a" * 32}

    instance.execute = execute
    instance._operation_lease = preserve_lease
    result = asyncio.run(
        instance._remote_operation(
            "transfer_upload_chunk",
            {"path": "/content/file.part", "offset": 0, "data_base64": "YWJj"},
            "runtime",
            lease_token="b" * 32,
        )
    )

    assert result == {"offset": 3}
    assert attempts == 2


def test_lease_bound_read_does_not_retry_unknown_submission_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    attempts = 0

    async def execute(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RequestOutcomeUnknownError(
            "request_submission_outcome_unknown_response_lost",
            "response lost",
            {"request_submission": "outcome_unknown"},
        )

    instance.execute = execute
    with pytest.raises(RequestOutcomeUnknownError):
        asyncio.run(instance._remote_operation("fs_read", {}, "runtime"))
    assert attempts == 1


def test_connection_setup_has_hard_local_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    def blocked_kernel_client(**_kwargs):
        time.sleep(0.2)
        raise OSError("late failure")

    real_wait_for = asyncio.wait_for

    async def shortened_wait_for(awaitable, timeout):
        assert timeout == 6
        return await real_wait_for(awaitable, timeout=0.02)

    monkeypatch.setattr("src.manager.kernel_client", blocked_kernel_client)
    monkeypatch.setattr("src.manager.asyncio.wait_for", shortened_wait_for)
    with pytest.raises(KernelConnectionError) as caught:
        asyncio.run(
            instance.execute(
                "print('never sent')",
                "runtime",
                1,
                connection_timeout=5,
                connection_attempts=1,
            )
        )
    assert caught.value.details["request_submission"] == "not_submitted"
    assert caught.value.details["local_deadline_seconds"] == 6


def test_upload_is_chunked_verified_and_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdefgh")
    expected_checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    wire = bytearray()
    moves = []
    removes = []
    progress = []

    async def stat_or_none(*_args, **_kwargs):
        return None

    async def remote(operation, payload, _name, **_kwargs):
        if operation == "transfer_upload_chunk":
            data = base64.b64decode(payload["data_base64"])
            assert payload["offset"] == len(wire)
            wire.extend(data)
            return {"offset": len(wire), "already_applied": False}
        if operation == "fs_stat":
            return {"size": len(wire), "sha256": hashlib.sha256(wire).hexdigest()}
        if operation == "fs_move":
            moves.append(payload)
            return {}
        if operation == "fs_remove":
            removes.append(payload)
            return {}
        raise AssertionError(operation)

    async def on_progress(event):
        progress.append(event)

    instance._remote_stat_or_none = stat_or_none
    instance._remote_operation = remote
    result = asyncio.run(
        instance.transfer_upload(
            str(source),
            "/content/destination.bin",
            "runtime",
            chunk_size=3,
            progress=on_progress,
        )
    )
    assert bytes(wire) == b"abcdefgh"
    assert [item["bytes_sent"] for item in progress] == [3, 6, 8]
    assert moves[0]["destination"] == "/content/destination.bin"
    assert removes[0]["missing_ok"] is True
    assert result["files_transferred"][0]["sha256"] == expected_checksum
    assert result["lease"]["status"] == "stable"
    assert result["progress_events_emitted"] == 3
    assert result["timings"]["total_seconds"] >= 0


def test_upload_interruption_preserves_remote_partial_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdef")

    async def stat_or_none(*_args, **_kwargs):
        return None

    async def remote(operation, *_args, **_kwargs):
        if operation == "transfer_upload_chunk":
            raise KernelConnectionError("connection failed")
        raise AssertionError(operation)

    instance._remote_stat_or_none = stat_or_none
    instance._remote_operation = remote
    with pytest.raises(TransferError) as caught:
        asyncio.run(instance.transfer_upload(str(source), "/content/file", "runtime"))
    assert caught.value.code == "kernel_connection_failed_request_not_submitted"
    assert caught.value.details["request_submission"] == "not_submitted"
    assert caught.value.details["safe_to_resume"] is True
    assert ".colab-mcp-wire-" in caught.value.details["staging_path"]


def test_upload_resumes_verified_staging_on_same_incarnation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdefgh")
    transfer_id = "d" * 32
    staged = bytearray(b"abc")
    stat_calls = 0
    progress = []

    async def stat_or_none(*_args, **_kwargs):
        nonlocal stat_calls
        stat_calls += 1
        if stat_calls == 1:
            return None
        return {
            "size": len(staged),
            "sha256": hashlib.sha256(staged).hexdigest(),
        }

    async def remote(operation, payload, _name, **_kwargs):
        if operation == "transfer_upload_chunk":
            assert payload["offset"] == len(staged)
            staged.extend(base64.b64decode(payload["data_base64"]))
            return {"offset": len(staged), "already_applied": False}
        if operation == "fs_stat":
            return {"size": len(staged), "sha256": hashlib.sha256(staged).hexdigest()}
        if operation in {"fs_move", "fs_remove"}:
            return {}
        raise AssertionError(operation)

    async def on_progress(event):
        progress.append(event)

    instance._remote_stat_or_none = stat_or_none
    instance._remote_operation = remote
    result = asyncio.run(
        instance.transfer_upload(
            str(source),
            "/content/file",
            "runtime",
            chunk_size=3,
            transfer_id=transfer_id,
            progress=on_progress,
        )
    )
    assert bytes(staged) == b"abcdefgh"
    assert [event["bytes_sent"] for event in progress] == [6, 8]
    assert result["files_transferred"][0]["resumed_from_bytes"] == 3


def test_upload_forced_gzip_verifies_wire_and_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    content = b"compressible payload\n" * 200
    source = tmp_path / "source.bin"
    source.write_bytes(content)
    wire = bytearray()
    operations = []

    async def stat_or_none(*_args, **_kwargs):
        return None

    async def operation(operation, payload, _name, **_kwargs):
        operations.append((operation, payload))
        if operation == "transfer_upload_chunk":
            data = base64.b64decode(payload["data_base64"])
            wire.extend(data)
            return {"offset": len(wire), "already_applied": False}
        if operation == "fs_stat":
            return {"size": len(wire), "sha256": hashlib.sha256(wire).hexdigest()}
        return {}

    instance._remote_stat_or_none = stat_or_none
    instance._remote_operation = operation
    result = asyncio.run(
        instance.transfer_upload(
            str(source), "/content/file", "runtime", compression="gzip", chunk_size=41
        )
    )
    assert gzip.decompress(bytes(wire)) == content
    decompress = next(item for item in operations if item[0] == "fs_gzip_decompress")
    assert decompress[1]["expected_sha256"] == hashlib.sha256(content).hexdigest()
    assert result["files_transferred"][0]["compression"] == "gzip"
    assert result["wire_bytes"] < result["total_bytes"]


def test_download_is_chunked_verified_and_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    content = b"abcdefgh"
    checksum = hashlib.sha256(content).hexdigest()

    async def remote_files(*_args, **_kwargs):
        item = {"path": "/content/source.bin", "kind": "file", "size": len(content)}
        return item, [item]

    async def remote(operation, payload, _name, **_kwargs):
        if operation == "fs_stat":
            return {"sha256": checksum}
        if operation == "fs_read":
            offset = payload["offset"]
            data = content[offset : offset + payload["limit"]]
            next_offset = offset + len(data)
            return {
                "data_base64": base64.b64encode(data).decode(),
                "next_offset": next_offset,
                "eof": next_offset == len(content),
            }
        raise AssertionError(operation)

    instance._remote_files = remote_files
    instance._remote_operation = remote
    destination = tmp_path / "download.bin"
    result = asyncio.run(
        instance.transfer_download("/content/source.bin", str(destination), "runtime", chunk_size=3)
    )
    assert destination.read_bytes() == content
    assert result["files_transferred"][0]["sha256"] == checksum
    assert not list(tmp_path.glob("*.colab-mcp-part-*"))


def test_download_forced_gzip_streams_and_verifies_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    content = b"remote payload\n" * 200
    wire = gzip.compress(content, mtime=0)
    checksum = hashlib.sha256(content).hexdigest()
    wire_checksum = hashlib.sha256(wire).hexdigest()
    removed = []

    async def remote_files(*_args, **_kwargs):
        item = {"path": "/content/source.bin", "kind": "file", "size": len(content)}
        return item, [item]

    async def operation(operation, payload, _name, **_kwargs):
        if operation == "fs_stat":
            return {"sha256": checksum}
        if operation == "fs_gzip_compress":
            return {"size": len(wire), "sha256": wire_checksum}
        if operation == "fs_read":
            offset = payload["offset"]
            data = wire[offset : offset + payload["limit"]]
            return {
                "data_base64": base64.b64encode(data).decode(),
                "next_offset": offset + len(data),
                "eof": offset + len(data) == len(wire),
            }
        if operation == "fs_remove":
            removed.append(payload["path"])
            return {}
        raise AssertionError(operation)

    instance._remote_files = remote_files
    instance._remote_operation = operation
    destination = tmp_path / "download.bin"
    result = asyncio.run(
        instance.transfer_download(
            "/content/source.bin", str(destination), "runtime", compression="gzip", chunk_size=37
        )
    )
    assert destination.read_bytes() == content
    assert result["files_transferred"][0]["compression"] == "gzip"
    assert result["wire_bytes"] == len(wire)
    assert removed and removed[0].endswith(".gz")


def test_download_corrupt_gzip_cleans_partial_and_remote_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    content = b"expected content" * 20
    wire = b"not a gzip stream"
    removed = []

    async def remote_files(*_args, **_kwargs):
        item = {"path": "/content/source.bin", "kind": "file", "size": len(content)}
        return item, [item]

    async def operation(operation, payload, _name, **_kwargs):
        if operation == "fs_stat":
            return {"sha256": hashlib.sha256(content).hexdigest()}
        if operation == "fs_gzip_compress":
            return {"size": len(wire), "sha256": hashlib.sha256(wire).hexdigest()}
        if operation == "fs_read":
            offset = payload["offset"]
            data = wire[offset : offset + payload["limit"]]
            return {
                "data_base64": base64.b64encode(data).decode(),
                "next_offset": offset + len(data),
                "eof": offset + len(data) == len(wire),
            }
        if operation == "fs_remove":
            removed.append(payload["path"])
            return {}
        raise AssertionError(operation)

    instance._remote_files = remote_files
    instance._remote_operation = operation
    destination = tmp_path / "download.bin"
    with pytest.raises(zlib.error):
        asyncio.run(
            instance.transfer_download(
                "/content/source.bin", str(destination), "runtime", compression="gzip"
            )
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.colab-mcp-part-*"))
    assert removed


def test_transfer_bounds_and_overwrite_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    source = tmp_path / "source"
    source.write_bytes(b"data")

    async def existing(*_args, **_kwargs):
        return {"sha256": "different"}

    instance._remote_stat_or_none = existing
    with pytest.raises(FileExistsError, match="Remote destination exists"):
        asyncio.run(instance.transfer_upload(str(source), "/content/file", None))
    with pytest.raises(ValueError, match="max_total_bytes"):
        asyncio.run(instance.transfer_upload(str(source), "/content/file", None, max_total_bytes=3))


def test_remote_directory_walk_respects_listing_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    seen_limits = []

    async def stat(*_args, **_kwargs):
        return {"path": "/content/root", "kind": "directory", "size": 0}

    async def listing(path, name, limit=1_000):
        seen_limits.append(limit)
        return {"entries": [], "truncated": False}

    instance.filesystem_stat = stat
    instance.filesystem_list = listing
    root, files = asyncio.run(instance._remote_files("/content/root", None, 10_000))
    assert root["kind"] == "directory"
    assert files == []
    assert seen_limits == [10_000]


def test_legacy_transfer_aliases_use_bounded_verified_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    calls = []

    async def upload(local, remote, name):
        calls.append(("upload", local, remote, name))
        return {"files_transferred": []}

    async def download(remote, local, name):
        calls.append(("download", remote, local, name))
        return {"files_transferred": []}

    instance.transfer_upload = upload
    instance.transfer_download = download
    assert asyncio.run(instance.upload("local", "/content/remote", "runtime")) == {
        "files_transferred": []
    }
    assert asyncio.run(instance.download("/content/remote", "local", "runtime")) == {
        "files_transferred": []
    }
    assert calls == [
        ("upload", "local", "/content/remote", "runtime"),
        ("download", "/content/remote", "local", "runtime"),
    ]


def test_process_export_atomically_publishes_then_explicitly_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    stopped = []

    async def status(*_args):
        return {"process_id": "process", "status": "exited", "exit_code": 0}

    async def download(remote, local, name, **_kwargs):
        Path(local).write_bytes(b"artifact")
        return {
            "files_transferred": [
                {"remote_path": remote, "local_path": local, "size": 8, "sha256": "digest"}
            ],
            "files_skipped": [],
            "total_bytes": 8,
            "lease": {"status": "stable"},
        }

    async def stop(name):
        stopped.append(name)
        return {"stopped": name, "runtime_was_active": True}

    instance.process_status = status
    instance.transfer_download = download
    instance.stop = stop
    destination = tmp_path / "published.bin"
    result = asyncio.run(
        instance.process_export(
            "process", "/content/artifact.bin", str(destination), "runtime", True
        )
    )

    assert destination.read_bytes() == b"artifact"
    assert result["exported"] is True
    assert result["disposition"] == "released"
    assert result["transfer"]["files_transferred"][0]["local_path"] == str(destination)
    assert stopped == ["runtime"]
    assert not list(tmp_path.glob("*.colab-mcp-export-*"))


def test_process_export_failure_holds_runtime_and_preserves_recoverable_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    async def status(*_args):
        return {"process_id": "process", "status": "exited", "exit_code": 0}

    async def download(_remote, local, _name, **_kwargs):
        Path(local).write_bytes(b"partial")
        raise OSError("connection interrupted")

    async def stop(_name):
        raise AssertionError("failed export must not release the runtime")

    instance.process_status = status
    instance.transfer_download = download
    instance.stop = stop
    destination = tmp_path / "published.bin"
    result = asyncio.run(
        instance.process_export(
            "process", "/content/artifact.bin", str(destination), "runtime", True
        )
    )

    assert result["exported"] is False
    assert result["disposition"] == "held"
    assert result["error"]["code"] == "export_failed_runtime_held"
    assert not destination.exists()
    stages = list(tmp_path.glob("*.colab-mcp-export-*"))
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"partial"
    assert result["recoverable_export"]["staging_path"] == str(stages[0])
    assert result["recoverable_export"]["staging_exists"] is True


def test_process_export_retry_reuses_deterministic_stage_then_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    calls = []

    async def status(*_args):
        return {"process_id": "process", "status": "exited", "exit_code": 0}

    async def download(_remote, local, _name, **kwargs):
        calls.append((local, kwargs))
        stage = Path(local)
        stage.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            (stage / "checkpoint-00.bin").write_bytes(b"first")
            raise OSError("interrupted")
        assert (stage / "checkpoint-00.bin").read_bytes() == b"first"
        (stage / "checkpoint-01.bin").write_bytes(b"second")
        return {
            "files_transferred": [
                {
                    "remote_path": "/content/checkpoints/checkpoint-01.bin",
                    "local_path": str(stage / "checkpoint-01.bin"),
                }
            ],
            "files_skipped": [
                {
                    "remote_path": "/content/checkpoints/checkpoint-00.bin",
                    "local_path": str(stage / "checkpoint-00.bin"),
                }
            ],
            "total_bytes": 11,
            "lease": {"status": "stable"},
        }

    instance.process_status = status
    instance.transfer_download = download
    destination = tmp_path / "published"
    first = asyncio.run(
        instance.process_export("process", "/content/checkpoints", str(destination), "runtime")
    )
    second = asyncio.run(
        instance.process_export("process", "/content/checkpoints", str(destination), "runtime")
    )
    assert first["exported"] is False
    assert second["exported"] is True
    assert calls[0][0] == calls[1][0]
    assert calls[1][1]["sync"] is True
    assert (destination / "checkpoint-00.bin").read_bytes() == b"first"
    assert (destination / "checkpoint-01.bin").read_bytes() == b"second"
    assert not list(tmp_path.glob("*.colab-mcp-export-*"))


def test_process_export_cleanup_only_removes_owned_deterministic_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    instance.process_journal.update("runtime", {"process_id": "process", "status": "exited"})
    destination = tmp_path / "published"
    stage = instance._process_export_stage("process", "/content/checkpoints", destination.resolve())
    stage.mkdir()
    (stage / "partial").write_bytes(b"data")
    result = asyncio.run(
        instance.process_export_cleanup(
            "process", "/content/checkpoints", str(destination), "runtime"
        )
    )
    assert result["removed"] is True
    assert not stage.exists()


def test_process_export_holds_while_owned_process_is_running(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    async def status(*_args):
        return {"process_id": "process", "status": "running", "argv": ["worker"]}

    instance.process_status = status
    result = asyncio.run(
        instance.process_export(
            "process", "/content/artifact.bin", str(tmp_path / "artifact"), "runtime", True
        )
    )

    assert result["exported"] is False
    assert result["disposition"] == "held"
    assert result["error"]["code"] == "process_not_finished"
    assert result["last_known_process"]["argv"] == ["worker"]


def test_process_export_keeps_published_artifact_when_release_fails(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    async def status(*_args):
        return {"process_id": "process", "status": "exited", "exit_code": 0}

    async def download(remote, local, name, **_kwargs):
        Path(local).write_bytes(b"artifact")
        return {
            "files_transferred": [{"remote_path": remote, "local_path": local}],
            "files_skipped": [],
            "total_bytes": 8,
            "lease": {"status": "stable"},
        }

    async def stop(_name):
        raise OSError("quota API unavailable")

    instance.process_status = status
    instance.transfer_download = download
    instance.stop = stop
    destination = tmp_path / "published.bin"
    result = asyncio.run(
        instance.process_export(
            "process", "/content/artifact.bin", str(destination), "runtime", True
        )
    )

    assert destination.read_bytes() == b"artifact"
    assert result["exported"] is True
    assert result["disposition"] == "held"
    assert result["error"]["code"] == "release_failed_runtime_held"


def test_process_start_persists_typed_auto_export_rules(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    watchers = []

    async def remote(operation, payload, name, timeout=120, **_kwargs):
        assert operation == "process_start"
        return {
            "process_id": payload["process_id"],
            "pid": 123,
            "cwd": "/content",
            "status": "running",
        }

    instance._remote_operation = remote
    instance.ensure_process_export_watcher = lambda session_name, process_id: watchers.append(
        (session_name, process_id)
    )
    destination = tmp_path / "artifact.bin"
    result = asyncio.run(
        instance.process_start(
            ["worker"],
            "runtime",
            export_on_exit=[
                AutoExportRule(
                    remote_path="/content/artifact.bin",
                    local_path=str(destination),
                    exit_codes=[0, 2],
                )
            ],
        )
    )

    rule = result["auto_export"]["rules"][0]
    assert rule["local_path"] == str(destination.resolve())
    assert rule["exit_codes"] == [0, 2]
    assert result["auto_export"]["status"] == "watching"
    assert watchers == [("runtime", result["process_id"])]


def test_auto_export_watcher_selects_rules_by_exit_code(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    process_id = "process"
    instance.process_journal.update(
        "runtime",
        {
            "process_id": process_id,
            "auto_export": {
                "status": "watching",
                "rules": [
                    {
                        "rule_id": "success",
                        "remote_path": "/content/success.bin",
                        "local_path": str(tmp_path / "success.bin"),
                        "exit_codes": [0],
                        "overwrite": False,
                    },
                    {
                        "rule_id": "failure",
                        "remote_path": "/content/failure.log",
                        "local_path": str(tmp_path / "failure.log"),
                        "exit_codes": [1],
                        "overwrite": False,
                    },
                ],
                "results": {},
            },
        },
    )
    exports = []

    async def status(*_args):
        return {"process_id": process_id, "status": "exited", "exit_code": 0}

    async def export(_process_id, remote, local, _session, **_kwargs):
        exports.append((remote, local))
        return {"exported": True, "local_path": local}

    instance.process_status = status
    instance.process_export = export
    asyncio.run(instance._watch_process_exports("runtime", process_id))

    state = instance.process_journal.get("runtime", process_id)["auto_export"]
    assert state["status"] == "completed"
    assert state["results"]["success"]["status"] == "exported"
    assert state["results"]["failure"]["status"] == "skipped"
    assert exports == [("/content/success.bin", str(tmp_path / "success.bin"))]


def test_auto_export_retries_failure_and_recovers_without_agent_polling(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    instance.export_poll_seconds = 0.01
    process_id = "process"
    instance.process_journal.update(
        "runtime",
        {
            "process_id": process_id,
            "auto_export": {
                "status": "watching",
                "rules": [
                    {
                        "rule_id": "export-0",
                        "remote_path": "/content/artifact",
                        "local_path": str(tmp_path / "artifact"),
                        "exit_codes": None,
                        "overwrite": False,
                    }
                ],
                "results": {},
            },
        },
    )
    attempts = []

    async def status(*_args):
        return {"process_id": process_id, "status": "exited", "exit_code": 7}

    async def export(*_args, **_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            return {"exported": False, "error": {"code": "interrupted"}}
        return {"exported": True}

    instance.process_status = status
    instance.process_export = export
    asyncio.run(asyncio.wait_for(instance._watch_process_exports("runtime", process_id), 2))

    state = instance.process_journal.get("runtime", process_id)["auto_export"]
    assert len(attempts) == 2
    assert state["status"] == "completed"
    assert state["exit_code"] == 7


def test_auto_export_recovery_reschedules_unfinished_watchers(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))
    instance.process_journal.update(
        "runtime",
        {
            "process_id": "pending",
            "auto_export": {"status": "degraded", "rules": [], "results": {}},
        },
    )
    recovered = []
    instance.ensure_process_export_watcher = lambda session_name, process_id: recovered.append(
        (session_name, process_id)
    )

    result = asyncio.run(instance.recover_process_export_watchers())

    assert result == {"recovered_process_ids": ["pending"]}
    assert recovered == [("runtime", "pending")]


def test_auto_export_rejects_duplicate_local_destinations(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    destination = str(tmp_path / "same")
    rules = [
        AutoExportRule(remote_path="/content/a", local_path=destination),
        AutoExportRule(remote_path="/content/b", local_path=destination),
    ]

    with pytest.raises(ValueError, match="destinations must be unique"):
        instance._normalize_auto_exports(rules)
