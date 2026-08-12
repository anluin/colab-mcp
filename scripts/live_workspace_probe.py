"""Minimal opt-in live probe for the public folder synchronization workflow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path


def digest_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class Progress:
    async def report_progress(self, *_args) -> None:
        pass


async def run() -> dict:
    with tempfile.TemporaryDirectory(prefix="colab-mcp-workspace-") as temporary:
        root = Path(temporary)
        os.environ["COLAB_MCP_STATE_DIR"] = str(root / "state")
        from src import server

        session = "workspace-probe-" + uuid.uuid4().hex[:8]
        source = root / "source"
        destination = root / "downloaded"
        (source / "src" / "nested").mkdir(parents=True)
        (source / ".git").mkdir()
        (source / "src" / "main.py").write_text("print('first')\n", encoding="utf-8")
        (source / "src" / "nested" / "config.json").write_text(
            json.dumps({"batch_size": 8}, sort_keys=True), encoding="utf-8"
        )
        (source / "weights.bin").write_bytes(os.urandom(2_300_000))
        (source / "destination-only.txt").write_text("preserve me", encoding="utf-8")
        (source / ".git" / "config").write_text("must not transfer", encoding="utf-8")
        started = False
        try:
            await server.manager.start(session, None)
            started = True
            lease = await server.manager.allocation_probe(session, observations=2, interval=0.1)
            first_push = await server.colab_workspace_sync(
                "push",
                str(source),
                Progress(),
                "/content/workspace",
                session,
                lease["lease_token"],
                chunk_size=256_000,
            )
            (source / "src" / "main.py").write_text("print('second')\n", encoding="utf-8")
            (source / "src" / "new.py").write_text("VALUE = 42\n", encoding="utf-8")
            (source / "destination-only.txt").unlink()
            second_push = await server.colab_workspace_sync(
                "push",
                str(source),
                Progress(),
                "/content/workspace",
                session,
                lease["lease_token"],
                chunk_size=256_000,
            )
            pulled = await server.colab_workspace_sync(
                "pull",
                str(destination),
                Progress(),
                "/content/workspace",
                session,
                lease["lease_token"],
                chunk_size=256_000,
            )
            expected = digest_tree(source)
            expected.pop(".git/config")
            expected["destination-only.txt"] = hashlib.sha256(b"preserve me").hexdigest()
            observed = digest_tree(destination)
            if expected != observed:
                raise AssertionError({"expected": expected, "observed": observed})
            if ".git/config" in observed:
                raise AssertionError("built-in VCS exclusion was transferred")
            if len(first_push["files_transferred"]) != 4:
                raise AssertionError(f"first push was not multi-file: {first_push}")
            if sorted(
                item["remote_path"].removeprefix("/content/workspace/")
                for item in second_push["files_transferred"]
            ) != [
                "src/main.py",
                "src/new.py",
            ]:
                raise AssertionError(f"second push delta was incorrect: {second_push}")
            if len(second_push["files_skipped"]) != 2:
                raise AssertionError(f"second push did not skip unchanged files: {second_push}")
            stopped = await server.manager.stop(session)
            started = False
            result = {
                "status": "passed",
                "files": len(expected),
                "first_push_files": len(first_push["files_transferred"]),
                "second_push_files": len(second_push["files_transferred"]),
                "second_push_skipped": len(second_push["files_skipped"]),
                "first_push_bytes": first_push["total_bytes"],
                "second_push_bytes": second_push["total_bytes"],
                "pull_bytes": pulled["total_bytes"],
                "destination_only_preserved": "destination-only.txt" in observed,
                "vcs_excluded": ".git/config" not in observed,
                "runtime_released": stopped["runtime_was_active"],
            }
            return result
        finally:
            if started:
                await server.manager.stop(session)


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), separators=(",", ":")))
