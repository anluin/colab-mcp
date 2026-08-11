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
        source = Path(__file__).resolve().parents[1] / "skills" / "colab-workspace-sync"
        destination = root / "downloaded"
        started = False
        try:
            await server.manager.start(session, None)
            started = True
            lease = await server.manager.allocation_probe(session, observations=2, interval=0.1)
            pushed = await server.colab_workspace_sync(
                "push",
                str(source),
                Progress(),
                "/content/workspace",
                session,
                lease["lease_token"],
            )
            pulled = await server.colab_workspace_sync(
                "pull",
                str(destination),
                Progress(),
                "/content/workspace",
                session,
                lease["lease_token"],
            )
            expected = digest_tree(source)
            observed = digest_tree(destination)
            if expected != observed:
                raise AssertionError({"expected": expected, "observed": observed})
            return {
                "status": "passed",
                "files": len(expected),
                "push_bytes": pushed["total_bytes"],
                "pull_bytes": pulled["total_bytes"],
                "runtime_released": True,
            }
        finally:
            if started:
                await server.manager.stop(session)


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), separators=(",", ":")))
