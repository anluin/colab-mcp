import asyncio
from types import SimpleNamespace

import pytest

from src import server
from src.manager import TransferError


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


def test_workspace_dry_run_returns_plan_without_transfer(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("pass", encoding="utf-8")
    plan = {"plan_id": "a" * 64, "files_to_transfer_count": 1}

    async def planned(**_kwargs):
        return plan, ["model.py"], ["model.py"], "runtime", {"lease_token": "b" * 32}

    async def should_not_upload(*_args, **_kwargs):
        raise AssertionError("dry run must not upload")

    monkeypatch.setattr(server.manager, "workspace_sync_plan", planned)
    monkeypatch.setattr(server.manager, "workspace_upload", should_not_upload)
    result = asyncio.run(
        server.colab_workspace_sync(
            "push",
            str(source),
            Progress(),
            session="runtime",
            lease_token="b" * 32,
            dry_run=True,
            include=["model.py"],
        )
    )
    assert result["dry_run"] is True
    assert result["plan"] == plan


def test_workspace_expected_plan_rejects_drift_before_transfer(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("pass", encoding="utf-8")

    async def planned(**_kwargs):
        return (
            {"plan_id": "new"},
            ["model.py"],
            ["model.py"],
            "runtime",
            {"lease_token": "b" * 32},
        )

    async def should_not_upload(*_args, **_kwargs):
        raise AssertionError("changed plan must not upload")

    monkeypatch.setattr(server.manager, "workspace_sync_plan", planned)
    monkeypatch.setattr(server.manager, "workspace_upload", should_not_upload)
    with pytest.raises(TransferError) as raised:
        asyncio.run(
            server.colab_workspace_sync(
                "push",
                str(source),
                Progress(),
                session="runtime",
                lease_token="b" * 32,
                expected_plan_id="old",
            )
        )
    assert raised.value.code == "sync_plan_changed"
