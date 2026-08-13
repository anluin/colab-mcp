"""Standalone CLI acceptance test for bidirectional WebRTC transfer.

This deliberately instantiates ``ColabManager`` directly and never starts or calls
an MCP server. It owns a dedicated runtime and releases it in ``finally``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from src.manager import ColabManager


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_payload(path: Path, size: int) -> None:
    with path.open("wb") as handle:
        remaining = size
        while remaining:
            chunk = os.urandom(min(1024 * 1024, remaining))
            handle.write(chunk)
            remaining -= len(chunk)


async def run(size_mib: int, transport: str, lanes: int) -> dict:
    if not 1 <= size_mib <= 512:
        raise ValueError("size_mib must be between 1 and 512")
    if transport not in {"webrtc", "kernel", "auto"}:
        raise ValueError("transport must be webrtc, kernel, or auto")
    if not 1 <= lanes <= 16:
        raise ValueError("lanes must be between 1 and 16")
    state_root = Path(tempfile.mkdtemp(prefix="colab-mcp-p2p-state-"))
    cleanup_confirmed = False
    print(json.dumps({"phase": "state_ready", "recovery_path": str(state_root)}), flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="colab-mcp-p2p-cli-") as temporary:
            return await _run_owned(Path(temporary), state_root, size_mib, transport, lanes)
    finally:
        # _run_owned removes its assignment before returning. If the process is
        # forcibly terminated, this owner-only state survives for explicit recovery.
        sessions = state_root / "sessions.json"
        if not sessions.is_file() or not ColabManager().store.list():
            cleanup_confirmed = True
        if cleanup_confirmed:
            shutil.rmtree(state_root, ignore_errors=True)
        else:
            print(
                json.dumps({"phase": "recovery_required", "state_root": str(state_root)}),
                flush=True,
            )


async def _run_owned(
    root: Path, state_root: Path, size_mib: int, transport: str, lanes: int
) -> dict:
    os.environ["COLAB_MCP_STATE_DIR"] = str(state_root)
    os.environ["COLAB_MCP_WEBRTC_LANES"] = str(lanes)
    manager = ColabManager()
    session = "p2p-cli-" + uuid.uuid4().hex[:8]
    remote_root = "/content/workspaces/" + session + "/source"
    source = root / "source"
    destination = root / "downloaded"
    source.mkdir()
    payload = source / "payload.bin"
    create_payload(payload, size_mib * 1024 * 1024)
    expected = sha256(payload)
    started = False
    try:
        allocation_started = time.monotonic()
        await manager.start(session, None)
        started = True
        lease = await manager.allocation_probe(session, observations=2, interval=0.1)
        allocation_seconds = time.monotonic() - allocation_started
        print(
            json.dumps({"phase": "allocated", "seconds": round(allocation_seconds, 3)}), flush=True
        )
        push = await manager.workspace_upload(
            str(source),
            remote_root,
            session,
            max_total_bytes=size_mib * 1024 * 1024 + 1024 * 1024,
            max_files=10,
            compression="none",
            lease_token=lease["lease_token"],
            transport=transport,
        )
        print(
            json.dumps(
                {
                    "phase": "push_complete",
                    "seconds": push["timings"]["data_transfer_seconds"],
                    "transport": push.get("data_transport"),
                }
            ),
            flush=True,
        )
        pull = await manager.transfer_download(
            remote_root,
            str(destination),
            session,
            overwrite=True,
            max_total_bytes=size_mib * 1024 * 1024 + 1024 * 1024,
            max_files=10,
            compression="none",
            lease_token=lease["lease_token"],
            transport=transport,
        )
        print(
            json.dumps(
                {
                    "phase": "pull_complete",
                    "seconds": pull["timings"]["data_transfer_seconds"],
                    "transport": pull.get("data_transport"),
                }
            ),
            flush=True,
        )
        downloaded = destination / "payload.bin"
        observed = sha256(downloaded)
        if observed != expected:
            raise AssertionError(f"round-trip SHA-256 mismatch: {observed} != {expected}")
        stopped = await manager.stop(session)
        started = False
        print(json.dumps({"phase": "released"}), flush=True)
        push_seconds = float(push["timings"]["data_transfer_seconds"])
        pull_seconds = float(pull["timings"]["data_transfer_seconds"])
        return {
            "status": "passed",
            "execution": "standalone_cli_no_mcp",
            "session": session,
            "size_bytes": payload.stat().st_size,
            "sha256": expected,
            "requested_transport": transport,
            "lanes": lanes,
            "allocation_seconds": round(allocation_seconds, 3),
            "push": {
                "transport": push.get("data_transport"),
                "wire_bytes": push["wire_bytes"],
                "seconds": push_seconds,
                "mib_per_second": round(push["wire_bytes"] / 1048576 / push_seconds, 3),
            },
            "pull": {
                "transport": pull.get("data_transport"),
                "wire_bytes": pull["wire_bytes"],
                "seconds": pull_seconds,
                "mib_per_second": round(pull["wire_bytes"] / 1048576 / pull_seconds, 3),
            },
            "runtime_released": stopped["runtime_was_active"],
        }
    finally:
        if started:
            cleanup = asyncio.create_task(manager.stop(session))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=int, default=32)
    parser.add_argument("--transport", choices=["webrtc", "kernel", "auto"], default="webrtc")
    parser.add_argument("--lanes", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.size_mib, args.transport, args.lanes)),
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
