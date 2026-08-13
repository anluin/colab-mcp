import asyncio
from types import SimpleNamespace

import pytest

from src import server


class Progress:
    async def report_progress(self, *_args):
        pass


def test_workspace_push_requires_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(server.manager, "allocation_probe", _stable_probe)
    with pytest.raises(ValueError, match="existing directory"):
        asyncio.run(
            server.colab_workspace_sync(
                "push", str(tmp_path / "missing"), Progress(), session="runtime"
            )
        )


def test_workspace_pull_rejects_single_remote_file(tmp_path, monkeypatch):
    runtime = SimpleNamespace(name="runtime")

    async def lease(_session, token):
        return runtime, {"lease_token": token}

    async def operation(*_args, **_kwargs):
        return {"kind": "file"}

    monkeypatch.setattr(server.manager, "_operation_lease", lease)
    monkeypatch.setattr(server.manager, "_remote_operation", operation)
    with pytest.raises(ValueError, match="remote_folder"):
        asyncio.run(
            server.colab_workspace_sync(
                "pull",
                str(tmp_path / "output"),
                Progress(),
                session="runtime",
                lease_token="b" * 32,
            )
        )


def test_workspace_transport_selection_reaches_manager(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    seen = {}

    async def selection(**_kwargs):
        return ["main.py"], ["main.py"], "runtime", {"lease_token": "b" * 32}

    async def upload(**kwargs):
        seen.update(kwargs)
        return {
            "wire_bytes": 10,
            "data_transport": "kernel_websocket",
            "timings": {"data_transfer_seconds": 1},
        }

    monkeypatch.setattr(server.manager, "workspace_sync_selection", selection)
    monkeypatch.setattr(server.manager, "workspace_upload", upload)
    monkeypatch.setattr(server.manager, "_record_sync_speed", lambda *_args: None)
    result = asyncio.run(
        server.colab_workspace_sync(
            "push",
            str(source),
            Progress(),
            session="runtime",
            lease_token="b" * 32,
            transport="kernel",
        )
    )
    assert seen["transport"] == "kernel"
    assert result["data_transport"] == "kernel_websocket"


async def _stable_probe(_session):
    return {"lease_token": "b" * 32}
