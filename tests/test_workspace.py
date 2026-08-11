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


async def _stable_probe(_session):
    return {"lease_token": "b" * 32}
