import asyncio
import base64
import gzip
import hashlib
import json
import os
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli import (
    _client_has_server,
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
    ManagedSessionState,
    _bound_outputs,
    _json_safe,
    _secure_permissions,
    require_local_file,
)


def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ColabManager:
    monkeypatch.setenv("COLAB_MCP_STATE_DIR", str(tmp_path / "state"))
    instance = ColabManager()

    async def stable_probe(*_args, **_kwargs):
        return {"status": "stable", "observations": 2}

    instance.allocation_probe = stable_probe
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
        while len(attempts) < 2 and asyncio.get_running_loop().time() < deadline:
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

    async def remote(operation, payload, name, timeout=120):
        assert (operation, payload, name) == ("lease_probe", {}, "runtime")
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
    assert keepalives == ["endpoint", "endpoint", "endpoint"]


def test_allocation_probe_fails_before_remote_access_when_lease_is_lost(tmp_path, monkeypatch):
    instance = manager(tmp_path, monkeypatch)
    instance.store.add(session("runtime", "endpoint"))

    class FakeClient:
        def list_assignments(self):
            return []

    instance.client = lambda: FakeClient()
    with pytest.raises(RuntimeError, match="allocation_lease_lost"):
        asyncio.run(ColabManager.allocation_probe(instance, "runtime", interval=0))


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
    monkeypatch.setattr(
        "src.manager.jupyter_kernel_client.ColabKernelClient", lambda **_kwargs: kernel
    )
    with pytest.raises(OSError, match="kernel disappeared"):
        asyncio.run(instance.execute_python("print('hello')", "runtime"))
    assert kernel.stopped is True


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

    monkeypatch.setattr("src.manager.jupyter_kernel_client.ColabKernelClient", kernel_factory)
    outputs = asyncio.run(instance.execute_python("print('once')", "runtime"))
    assert outputs == [{"output_type": "stream", "text": "ok\n"}]
    assert len(attempts) == 2
    assert user_code == ["print('once')"]
    assert instance.store.get("runtime").kernel_id == "new-kernel"


def test_upload_is_chunked_verified_and_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdefgh")
    expected_checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    writes = []
    moves = []
    removes = []
    events = []

    async def probe(*_args, **_kwargs):
        events.append("probe")
        return {"status": "stable", "observations": 2}

    async def stat_or_none(*_args, **_kwargs):
        events.append("stat")
        return None

    async def write(path, data, name, append=False, create_parents=False):
        writes.append((path, base64.b64decode(data), append, create_parents))

    async def stat(path, name, checksum=False):
        return {"path": path, "kind": "file", "size": 8, "sha256": expected_checksum}

    async def move(source_path, destination, name, overwrite=False):
        moves.append((source_path, destination, overwrite))

    async def remove(path, name, recursive=False, missing_ok=False):
        removes.append((path, missing_ok))

    instance._remote_stat_or_none = stat_or_none
    instance.allocation_probe = probe
    instance.filesystem_write = write
    instance.filesystem_stat = stat
    instance.filesystem_move = move
    instance.filesystem_remove = remove
    result = asyncio.run(
        instance.transfer_upload(str(source), "/content/destination.bin", "runtime", chunk_size=3)
    )
    assert [item[1] for item in writes] == [b"abc", b"def", b"gh"]
    assert [item[2] for item in writes] == [False, True, True]
    assert moves[0][1:] == ("/content/destination.bin", False)
    assert removes[0][1] is True
    assert result["files_transferred"][0]["sha256"] == expected_checksum
    assert result["lease"]["status"] == "stable"
    assert events[:2] == ["probe", "stat"]


def test_upload_interruption_cleans_remote_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdef")
    removed = []

    async def stat_or_none(*_args, **_kwargs):
        return None

    async def write(*_args, **_kwargs):
        raise OSError("interrupted")

    async def remove(path, *_args, **_kwargs):
        removed.append(path)

    instance._remote_stat_or_none = stat_or_none
    instance.filesystem_write = write
    instance.filesystem_remove = remove
    with pytest.raises(OSError, match="interrupted"):
        asyncio.run(instance.transfer_upload(str(source), "/content/file", "runtime"))
    assert len(removed) == 2
    assert any(".colab-mcp-wire-" in path for path in removed)
    assert any(".colab-mcp-part-" in path for path in removed)


def test_upload_forced_gzip_verifies_wire_and_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    instance = manager(tmp_path, monkeypatch)
    content = b"compressible payload\n" * 200
    source = tmp_path / "source.bin"
    source.write_bytes(content)
    writes: list[bytes] = []
    operations = []

    async def stat_or_none(*_args, **_kwargs):
        return None

    async def write(_path, data, _name, append=False, create_parents=False):
        writes.append(base64.b64decode(data))

    async def stat(path, _name, checksum=False):
        wire = b"".join(writes)
        return {
            "path": path,
            "kind": "file",
            "size": len(wire),
            "sha256": hashlib.sha256(wire).hexdigest(),
        }

    async def operation(operation, payload, _name):
        operations.append((operation, payload))
        return {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    async def no_op(*_args, **_kwargs):
        return None

    instance._remote_stat_or_none = stat_or_none
    instance.filesystem_write = write
    instance.filesystem_stat = stat
    instance._remote_operation = operation
    instance.filesystem_move = no_op
    instance.filesystem_remove = no_op
    result = asyncio.run(
        instance.transfer_upload(
            str(source), "/content/file", "runtime", compression="gzip", chunk_size=41
        )
    )
    wire = b"".join(writes)
    assert gzip.decompress(wire) == content
    assert operations[0][0] == "fs_gzip_decompress"
    assert operations[0][1]["expected_sha256"] == hashlib.sha256(content).hexdigest()
    assert result["files_transferred"][0]["compression"] == "gzip"
    assert result["wire_bytes"] < result["total_bytes"]


def test_download_is_chunked_verified_and_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = manager(tmp_path, monkeypatch)
    content = b"abcdefgh"
    checksum = hashlib.sha256(content).hexdigest()

    async def remote_files(*_args):
        item = {"path": "/content/source.bin", "kind": "file", "size": len(content)}
        return item, [item]

    async def stat(*_args, **_kwargs):
        return {"sha256": checksum}

    async def read(path, name, offset=0, limit=262_144):
        data = content[offset : offset + limit]
        next_offset = offset + len(data)
        return {
            "data_base64": base64.b64encode(data).decode(),
            "next_offset": next_offset,
            "eof": next_offset == len(content),
        }

    instance._remote_files = remote_files
    instance.filesystem_stat = stat
    instance.filesystem_read = read
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

    async def remote_files(*_args):
        item = {"path": "/content/source.bin", "kind": "file", "size": len(content)}
        return item, [item]

    async def stat(*_args, **_kwargs):
        return {"sha256": checksum}

    async def operation(operation, _payload, _name):
        assert operation == "fs_gzip_compress"
        return {"size": len(wire), "sha256": wire_checksum}

    async def read(_path, _name, offset=0, limit=262_144):
        data = wire[offset : offset + limit]
        return {
            "data_base64": base64.b64encode(data).decode(),
            "next_offset": offset + len(data),
            "eof": offset + len(data) == len(wire),
        }

    async def remove(path, *_args, **_kwargs):
        removed.append(path)

    instance._remote_files = remote_files
    instance.filesystem_stat = stat
    instance._remote_operation = operation
    instance.filesystem_read = read
    instance.filesystem_remove = remove
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

    async def remote_files(*_args):
        item = {"path": "/content/source.bin", "kind": "file", "size": len(content)}
        return item, [item]

    async def stat(*_args, **_kwargs):
        return {"sha256": hashlib.sha256(content).hexdigest()}

    async def operation(*_args, **_kwargs):
        return {"size": len(wire), "sha256": hashlib.sha256(wire).hexdigest()}

    async def read(_path, _name, offset=0, limit=262_144):
        data = wire[offset : offset + limit]
        return {
            "data_base64": base64.b64encode(data).decode(),
            "next_offset": offset + len(data),
            "eof": offset + len(data) == len(wire),
        }

    async def remove(path, *_args, **_kwargs):
        removed.append(path)

    instance._remote_files = remote_files
    instance.filesystem_stat = stat
    instance._remote_operation = operation
    instance.filesystem_read = read
    instance.filesystem_remove = remove
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


def test_process_export_failure_holds_runtime_and_removes_staging(
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
    assert not list(tmp_path.glob("*.colab-mcp-export-*"))


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

    async def remote(operation, payload, name, timeout=120):
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
