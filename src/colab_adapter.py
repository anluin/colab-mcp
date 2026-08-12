"""Narrow isolation layer for timeout gaps in the pinned kernel client."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import jupyter_kernel_client
from jupyter_kernel_client.manager import KernelHttpManager


class BoundedKernelHttpManager(KernelHttpManager):
    """Apply the caller's connection bound to constructor-time model refresh.

    Upstream accepts a timeout in ``refresh_model`` but does not forward a constructor
    setting when reconnecting to an existing kernel. Keep this dependency-specific
    workaround here so it can be deleted when the pinned adapter exposes that option.
    """

    def __init__(self, *args: Any, refresh_timeout: float = 30, **kwargs: Any) -> None:
        self._colab_mcp_refresh_timeout = refresh_timeout
        super().__init__(*args, **kwargs)

    def refresh_model(self, timeout: float | None = None) -> dict[str, Any] | None:
        return super().refresh_model(
            timeout=self._colab_mcp_refresh_timeout if timeout is None else timeout
        )


def kernel_client(*, connection_timeout: float, **kwargs: Any) -> Any:
    """Construct the pinned Colab client with bounded existing-kernel lookup."""
    return jupyter_kernel_client.ColabKernelClient(
        kernel_manager_class=BoundedKernelHttpManager,
        refresh_timeout=connection_timeout,
        **kwargs,
    )


def binary_upload(
    kernel: Any,
    source: Path,
    remote_path: str,
    offset: int,
    expected_fingerprint: str,
    lease_token: str,
    timeout: float = 300,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Stream raw buffers over the authenticated kernel websocket."""
    setup = f"""
import datetime as _cm_datetime
import hashlib as _cm_hashlib
import json as _cm_json
import os as _cm_os
import pathlib as _cm_pathlib
_cm_expected_fingerprint = {expected_fingerprint!r}
_cm_expected_lease = {lease_token!r}
_cm_state_root = _cm_pathlib.Path('/content/.colab-mcp')
_cm_actual_fingerprint = (_cm_state_root / 'runtime-incarnation').read_text().strip()
if _cm_actual_fingerprint != _cm_expected_fingerprint:
    raise RuntimeError('runtime_replaced')
_cm_lease = _cm_json.loads((_cm_state_root / 'operation-lease.json').read_text())
_cm_candidates = _cm_lease.get('leases') or [_cm_lease]
_cm_match = next((item for item in _cm_candidates
    if item.get('token') == _cm_expected_lease), None)
if (_cm_lease.get('runtime_fingerprint') != _cm_expected_fingerprint
        or _cm_match is None or _cm_datetime.datetime.fromisoformat(
        _cm_match['expires_at']).astimezone(_cm_datetime.timezone.utc)
        <= _cm_datetime.datetime.now(_cm_datetime.timezone.utc)):
    raise RuntimeError('operation_lease_stale')

def _cm_binary_target(comm, open_msg):
    data = open_msg['content']['data']
    target = _cm_pathlib.Path(data['path']).resolve()
    root = _cm_pathlib.Path('/content').resolve()
    if (root not in target.parents or '.colab-mcp-wire-' not in target.name):
        raise ValueError('invalid binary upload staging path')
    expected_offset = int(data['offset'])
    current_size = target.stat().st_size if target.is_file() else 0
    if current_size != expected_offset:
        raise RuntimeError('transfer_offset_conflict')
    handle = target.open('ab')
    digest = _cm_hashlib.sha256()
    with target.open('rb') as existing:
        for block in iter(lambda: existing.read(1024 * 1024), b''):
            digest.update(block)
    received = current_size

    @comm.on_msg
    def _receive(msg):
        nonlocal received
        try:
            buffers = msg.get('buffers', [])
            if len(buffers) != 1:
                raise ValueError('binary upload requires exactly one buffer')
            chunk = bytes(buffers[0])
            handle.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if msg['content']['data'].get('final'):
                handle.flush()
                _cm_os.fsync(handle.fileno())
                handle.close()
                comm.send({{'offset': received, 'sha256': digest.hexdigest(), 'final': True}})
        except BaseException as error:
            try:
                handle.close()
            except BaseException:
                pass
            comm.send({{'final': True, 'error': type(error).__name__ + ': ' + str(error)}})

get_ipython().kernel.comm_manager.register_target(
    'colab_mcp_binary_upload', _cm_binary_target)
"""
    setup_reply = kernel.execute(setup, timeout=min(timeout, 60))
    setup_errors = [
        output
        for output in (setup_reply or {}).get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if setup_errors:
        raise RuntimeError("Binary upload receiver setup failed: " + str(setup_errors[0]))
    client = kernel._manager.client
    comm_id = uuid.uuid4().hex
    client.shell_channel.send(
        client.session.msg(
            "comm_open",
            {
                "comm_id": comm_id,
                "target_name": "colab_mcp_binary_upload",
                "data": {"path": remote_path, "offset": offset},
            },
        )
    )
    total = source.stat().st_size
    sent = offset
    started = time.monotonic()
    with source.open("rb") as handle:
        handle.seek(offset)
        sent_message = False
        while sent < total:
            chunk = handle.read(min(chunk_size, total - sent))
            sent += len(chunk)
            message = client.session.msg(
                "comm_msg",
                {"comm_id": comm_id, "data": {"final": sent == total}},
            )
            message["buffers"] = [chunk]
            client.shell_channel.send(message)
            sent_message = True
    if not sent_message:
        message = client.session.msg("comm_msg", {"comm_id": comm_id, "data": {"final": True}})
        message["buffers"] = [b""]
        client.shell_channel.send(message)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("binary upload acknowledgement timed out")
        reply = client.get_iopub_msg(timeout=remaining)
        content = reply.get("content", {})
        if reply.get("msg_type") != "comm_msg" or content.get("comm_id") != comm_id:
            continue
        data = content.get("data", {})
        if data.get("final"):
            if data.get("error"):
                raise RuntimeError("Binary upload failed: " + str(data["error"]))
            return {
                "offset": int(data["offset"]),
                "sha256": str(data["sha256"]),
                "seconds": time.monotonic() - started,
            }
