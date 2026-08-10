"""Opt-in Colab T4/L4 durability acceptance workflow with deterministic cleanup."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from src.manager import (
    AutoExportRule,
    ColabManager,
    KernelConnectionError,
    OperationLeaseError,
    TransferError,
)

PHASES = (
    "allocated",
    "probed",
    "cuda_verified",
    "small_uploaded",
    "checkpoint_uploaded",
    "process_started",
    "server_stopped",
    "server_recovered",
    "process_exited",
    "auto_exported",
    "hashes_verified",
    "released",
)


class InjectedFailure(RuntimeError):
    pass


def checkpoint(phase: str, fail_after: str | None, evidence: dict) -> None:
    evidence.setdefault("phases", []).append({"phase": phase, "at": time.time()})
    print(json.dumps({"phase": phase}), flush=True)
    if fail_after == phase:
        raise InjectedFailure(f"injected failure after {phase}")


def worker_source(duration: float, count: int, size: int) -> str:
    source = f"""import hashlib
import json
import os
import pathlib
import time

root = pathlib.Path('/content/checkpoints')
root.mkdir(parents=True, exist_ok=True)
manifest = {{}}
interval = {duration!r} / {count!r}
for index in range({count!r}):
    started = time.monotonic()
    path = root / f'checkpoint-{{index:02d}}.bin'
    data = os.urandom({size!r})
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    manifest[path.name] = hashlib.sha256(data).hexdigest()
    manifest_temporary = root / 'manifest.tmp'
    manifest_temporary.write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')
    manifest_temporary.replace(root / 'manifest.json')
    print(json.dumps({{'checkpoint': index, 'sha256': manifest[path.name]}}), flush=True)
    time.sleep(max(0, interval - (time.monotonic() - started)))
"""
    target = 63 * 1024
    if len(source.encode()) > target:
        raise ValueError("worker source unexpectedly exceeds 63 KiB")
    return source + "#" + ("p" * (target - len(source.encode()) - 1))


def verify_export(root: Path, expected_count: int) -> dict[str, str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != expected_count:
        raise AssertionError(f"manifest has {len(manifest)} files, expected {expected_count}")
    for name, expected in manifest.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
    return manifest


async def resilient_upload(
    manager: ColabManager,
    local_path: str,
    remote_path: str,
    session: str,
    lease: dict,
    progress: list[dict],
    *,
    compression: str = "auto",
) -> dict:
    """Retry only an explicitly resumable upload on the same fingerprint."""
    transfer_id = uuid.uuid4().hex
    original_fingerprint = lease["runtime_fingerprint"]
    for attempt in range(1, 6):
        try:
            result = await manager.transfer_upload(
                local_path,
                remote_path,
                session,
                chunk_size=1_000_000,
                compression=compression,
                lease_token=lease["lease_token"],
                transfer_id=transfer_id,
                progress=lambda event: _record_progress(progress, event),
            )
            result["acceptance_attempts"] = attempt
            return result
        except TransferError as error:
            if not error.details.get("safe_to_resume") or attempt == 5:
                raise
            await asyncio.sleep(2)
            refreshed = await manager.allocation_probe(session, observations=2, interval=0.1)
            if refreshed["runtime_fingerprint"] != original_fingerprint:
                raise AssertionError("upload retry observed a replacement incarnation") from error
            lease.update(refreshed)
    raise AssertionError("unreachable")


async def resilient_process_start(
    manager: ColabManager,
    session: str,
    lease: dict,
    rule: AutoExportRule,
) -> dict:
    """Retry process creation only when the request is confirmed not submitted."""
    fingerprint = lease["runtime_fingerprint"]
    for attempt in range(1, 6):
        try:
            result = await manager.process_start(
                ["python", "/content/worker.py"],
                session,
                output_limit=10_000_000,
                export_on_exit=[rule],
                lease_token=lease["lease_token"],
            )
            result["acceptance_attempts"] = attempt
            return result
        except OperationLeaseError as error:
            if error.code != "assignment_lookup_timed_out" or attempt == 5:
                raise
        except KernelConnectionError as error:
            if error.details.get("request_submission") != "not_submitted" or attempt == 5:
                raise
        await asyncio.sleep(2)
        refreshed = await manager.allocation_probe(session, observations=2, interval=0.1)
        if refreshed["runtime_fingerprint"] != fingerprint:
            raise AssertionError("process-start retry observed a replacement incarnation")
        lease.update(refreshed)
    raise AssertionError("unreachable")


async def resilient_execute(manager: ColabManager, code: str, session: str, lease: dict) -> dict:
    """Retry guarded execution only when no request was submitted."""
    fingerprint = lease["runtime_fingerprint"]
    for attempt in range(1, 6):
        try:
            result = await manager.execute_python_detailed(
                code, session, timeout=120, lease_token=lease["lease_token"]
            )
            result["acceptance_attempts"] = attempt
            return result
        except OperationLeaseError as error:
            if error.code != "assignment_lookup_timed_out" or attempt == 5:
                raise
        except KernelConnectionError as error:
            if error.details.get("request_submission") != "not_submitted" or attempt == 5:
                raise
        await asyncio.sleep(2)
        refreshed = await manager.allocation_probe(session, observations=2, interval=0.1)
        if refreshed["runtime_fingerprint"] != fingerprint:
            raise AssertionError("execute retry observed a replacement incarnation")
        lease.update(refreshed)
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> dict:
    evidence: dict = {"accelerator": args.accelerator, "failure_injection": args.fail_after}
    session = f"acceptance-{args.accelerator.lower()}-{uuid.uuid4().hex[:8]}"
    state_root = Path(args.state_dir).resolve()
    artifact_root = Path(args.artifacts).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    os.environ["COLAB_MCP_STATE_DIR"] = str(state_root)
    manager = ColabManager()
    manager.export_poll_seconds = args.poll_seconds
    started = False
    try:
        await manager.start(session, args.accelerator)
        started = True
        checkpoint("allocated", args.fail_after, evidence)

        lease = await manager.allocation_probe(session, observations=3, interval=0.25)
        evidence["runtime_fingerprint"] = lease["runtime_fingerprint"]
        checkpoint("probed", args.fail_after, evidence)

        cuda = await resilient_execute(
            manager,
            "import json, torch\nprint(json.dumps({'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))",
            session,
            lease,
        )
        if not any('"cuda": true' in str(item.get("text", "")).lower() for item in cuda["outputs"]):
            raise AssertionError(f"CUDA verification failed: {cuda['outputs']}")
        evidence["cuda_timings"] = cuda["timings"]
        checkpoint("cuda_verified", args.fail_after, evidence)

        worker = artifact_root / "worker.py"
        worker.write_bytes(
            worker_source(args.duration, args.checkpoint_count, args.checkpoint_size).encode(
                "utf-8"
            )
        )
        small_progress: list[dict] = []
        await resilient_upload(
            manager,
            str(worker),
            "/content/worker.py",
            session,
            lease,
            small_progress,
        )
        evidence["small_upload_progress"] = small_progress
        checkpoint("small_uploaded", args.fail_after, evidence)

        local_checkpoint = artifact_root / "input-checkpoint.bin"
        local_checkpoint.write_bytes(os.urandom(args.checkpoint_size))
        large_progress: list[dict] = []
        upload = await resilient_upload(
            manager,
            str(local_checkpoint),
            "/content/input-checkpoint.bin",
            session,
            lease,
            large_progress,
            compression="auto",
        )
        evidence["checkpoint_upload"] = {
            "progress": large_progress,
            "timings": upload["timings"],
            "sha256": upload["files_transferred"][0]["sha256"],
        }
        checkpoint("checkpoint_uploaded", args.fail_after, evidence)

        automatic_destination = artifact_root / "automatic-export"
        process = await resilient_process_start(
            manager,
            session,
            lease,
            AutoExportRule(
                remote_path="/content/checkpoints",
                local_path=str(automatic_destination),
                max_total_bytes=args.checkpoint_count * args.checkpoint_size + 2_000_000,
                chunk_size=2_000_000,
                compression="none",
            ),
        )
        process_id = process["process_id"]
        evidence["process_id"] = process_id
        checkpoint("process_started", args.fail_after, evidence)

        await asyncio.sleep(min(2, args.duration / 4))
        await manager.shutdown_process_export_watchers()
        await manager.shutdown_keepalives()
        checkpoint("server_stopped", args.fail_after, evidence)

        manager = ColabManager()
        manager.export_poll_seconds = args.poll_seconds
        recovered_keepalives = await manager.recover_keepalives()
        recovered_exports = await manager.recover_process_export_watchers()
        evidence["recovery"] = {
            "keepalives": recovered_keepalives,
            "exports": recovered_exports,
        }
        if process_id not in recovered_exports["recovered_process_ids"]:
            raise AssertionError("process export watcher was not recovered after restart")
        checkpoint("server_recovered", args.fail_after, evidence)

        deadline = time.monotonic() + args.duration + args.timeout_margin
        while True:
            status = await manager.process_status(process_id, session)
            if status["status"] == "exited":
                if status.get("exit_code") != 0:
                    raise AssertionError(f"worker failed: {status}")
                break
            if status["status"] != "running" or time.monotonic() >= deadline:
                raise AssertionError(f"worker did not exit cleanly: {status}")
            await asyncio.sleep(args.poll_seconds)
        checkpoint("process_exited", args.fail_after, evidence)

        while True:
            record = manager.process_journal.get(session, process_id) or {}
            auto_export = record.get("auto_export") or {}
            if auto_export.get("status") == "completed":
                evidence["auto_export"] = auto_export
                break
            if auto_export.get("status") == "held" or time.monotonic() >= deadline:
                raise AssertionError(f"automatic export did not complete: {auto_export}")
            await asyncio.sleep(args.poll_seconds)
        checkpoint("auto_exported", args.fail_after, evidence)

        manifest = verify_export(automatic_destination, args.checkpoint_count)
        evidence["verified_files"] = len(manifest)
        checkpoint("hashes_verified", args.fail_after, evidence)

        release_destination = artifact_root / "release-export"
        release = await manager.process_export(
            process_id,
            "/content/checkpoints",
            str(release_destination),
            session,
            release_on_success=True,
            max_total_bytes=args.checkpoint_count * args.checkpoint_size + 2_000_000,
            chunk_size=2_000_000,
            compression="none",
        )
        if not release.get("runtime_released"):
            raise AssertionError(f"release-on-success did not release: {release}")
        started = False
        verify_export(release_destination, args.checkpoint_count)
        evidence["release"] = release
        checkpoint("released", args.fail_after, evidence)
        return evidence
    finally:
        if started:
            with contextlib.suppress(Exception):
                await manager.stop(session)
        await manager.shutdown_process_export_watchers()
        await manager.shutdown_keepalives()


async def _record_progress(target: list, event: dict) -> None:
    target.append(event)
    print(json.dumps({"progress": event}, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", choices=("T4", "L4"), required=True)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--checkpoint-count", type=int, default=20)
    parser.add_argument("--checkpoint-size", type=int, default=1_900_000)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--timeout-margin", type=float, default=600)
    parser.add_argument("--fail-after", choices=PHASES)
    parser.add_argument("--state-dir")
    parser.add_argument("--artifacts")
    args = parser.parse_args()
    temporary = Path(tempfile.mkdtemp(prefix="colab-mcp-live-"))
    args.state_dir = args.state_dir or str(temporary / "state")
    args.artifacts = args.artifacts or str(temporary / "artifacts")
    return args


if __name__ == "__main__":
    result = asyncio.run(run(parse_args()))
    print(json.dumps(result, indent=2, default=str), flush=True)
