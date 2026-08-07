import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from colab_cli.state import SessionState

from src.cli import (
    claude_desktop_config_path,
    config_json,
    install_claude_desktop,
    install_command,
    isolated_server_command,
    server_command,
)
from src.manager import (
    ColabManager,
    _bound_outputs,
    _json_safe,
    _secure_permissions,
    require_local_file,
)


def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ColabManager:
    monkeypatch.setenv("COLAB_MCP_STATE_DIR", str(tmp_path / "state"))
    return ColabManager()


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


def session(name: str, endpoint: str) -> SessionState:
    return SessionState(name=name, token="secret", url="https://runtime", endpoint=endpoint)


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

    async def stat_or_none(*_args, **_kwargs):
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
    assert len(removed) == 1
    assert ".colab-mcp-part-" in removed[0]


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
