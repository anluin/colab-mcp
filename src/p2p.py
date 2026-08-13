"""Lease-bound WebRTC data-channel transport for bulk Colab transfers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

P2P_PROTOCOL_VERSION = 1
P2P_DEPENDENCY = "aiortc==1.15.0"
DEFAULT_ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
DEFAULT_P2P_MIN_BYTES = 4 * 1024 * 1024
P2P_CHUNK_SIZE = 32 * 1024
P2P_BUFFER_HIGH_WATER = 4 * 1024 * 1024


class P2PError(RuntimeError):
    """A direct or relayed WebRTC transfer could not complete safely."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ice_servers_from_environment() -> list[dict[str, Any]]:
    """Load bounded ICE configuration without logging TURN credentials."""
    raw = os.environ.get("COLAB_MCP_WEBRTC_ICE_SERVERS")
    if raw is None:
        return [dict(item) for item in DEFAULT_ICE_SERVERS]
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("COLAB_MCP_WEBRTC_ICE_SERVERS must be valid JSON") from error
    if not isinstance(records, list) or not 1 <= len(records) <= 8:
        raise ValueError("COLAB_MCP_WEBRTC_ICE_SERVERS must contain 1-8 ICE servers")
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each ICE server must be an object")
        urls = record.get("urls")
        urls = [urls] if isinstance(urls, str) else urls
        if not isinstance(urls, list) or not urls or len(urls) > 8:
            raise ValueError("Each ICE server needs 1-8 URLs")
        checked_urls = []
        for url in urls:
            if (
                not isinstance(url, str)
                or len(url) > 2_000
                or not url.startswith(("stun:", "stuns:", "turn:", "turns:"))
            ):
                raise ValueError("ICE URLs must use stun, stuns, turn, or turns")
            checked_urls.append(url)
        item: dict[str, Any] = {"urls": checked_urls}
        for key in ("username", "credential"):
            value = record.get(key)
            if value is not None:
                if not isinstance(value, str) or len(value) > 4_096:
                    raise ValueError(f"ICE {key} must be a string no longer than 4096 characters")
                item[key] = value
        normalized.append(item)
    return normalized


def p2p_min_bytes() -> int:
    raw = os.environ.get("COLAB_MCP_WEBRTC_MIN_BYTES", str(DEFAULT_P2P_MIN_BYTES))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("COLAB_MCP_WEBRTC_MIN_BYTES must be an integer") from error
    if not 0 <= value <= 1_000_000_000:
        raise ValueError("COLAB_MCP_WEBRTC_MIN_BYTES must be between 0 and 1000000000")
    return value


def p2p_lane_count(wire_bytes: int) -> int:
    raw = os.environ.get("COLAB_MCP_WEBRTC_LANES", "1")
    try:
        configured = int(raw)
    except ValueError as error:
        raise ValueError("COLAB_MCP_WEBRTC_LANES must be an integer") from error
    if not 1 <= configured <= 16:
        raise ValueError("COLAB_MCP_WEBRTC_LANES must be between 1 and 16")
    # Avoid paying for extra ICE/DTLS handshakes on small payloads.
    useful = max(1, (wire_bytes + 4 * 1024 * 1024 - 1) // (4 * 1024 * 1024))
    return min(configured, useful)


def should_use_p2p(transport: str, wire_bytes: int) -> bool:
    if transport not in {"auto", "webrtc", "kernel"}:
        raise ValueError("transport must be auto, webrtc, or kernel")
    return transport == "webrtc" or (transport == "auto" and wire_bytes >= p2p_min_bytes())


def _rtc_configuration(records: list[dict[str, Any]]) -> RTCConfiguration:
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(
                urls=item["urls"],
                username=item.get("username"),
                credential=item.get("credential"),
            )
            for item in records
        ]
    )


async def _wait_for_ice_gathering(peer: RTCPeerConnection, timeout: float) -> None:
    if peer.iceGatheringState == "complete":
        return
    complete = asyncio.Event()

    @peer.on("icegatheringstatechange")
    def gathering_changed() -> None:
        if peer.iceGatheringState == "complete":
            complete.set()

    await asyncio.wait_for(complete.wait(), timeout=timeout)


async def _wait_for_result(
    result: asyncio.Future[Any], failure: asyncio.Future[Any], timeout: float
) -> Any:
    result_task = asyncio.ensure_future(asyncio.shield(result))
    failure_task = asyncio.ensure_future(asyncio.shield(failure))
    done, pending = await asyncio.wait(
        {result_task, failure_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    if not done:
        raise TimeoutError("WebRTC transfer timed out")
    winner = done.pop()
    return winner.result()


async def _drain_channel(channel: Any) -> None:
    while channel.bufferedAmount > P2P_BUFFER_HIGH_WATER:
        await asyncio.sleep(0.002)


async def _create_offer(
    ice_servers: list[dict[str, Any]], connect_timeout: float
) -> tuple[RTCPeerConnection, Any, dict[str, str], asyncio.Future[Any]]:
    peer = RTCPeerConnection(_rtc_configuration(ice_servers))
    # Reliable but unordered messages avoid stream-wide head-of-line blocking when
    # a single UDP packet is lost on a long-haul path. Each binary message carries
    # its absolute file offset and final SHA-256 still covers the complete file.
    channel = peer.createDataChannel("colab-mcp-transfer", ordered=False)
    loop = asyncio.get_running_loop()
    failure: asyncio.Future[Any] = loop.create_future()

    @peer.on("connectionstatechange")
    def connection_changed() -> None:
        if peer.connectionState in {"failed", "disconnected"} and not failure.done():
            failure.set_exception(P2PError(f"WebRTC connection {peer.connectionState}"))

    await peer.setLocalDescription(await peer.createOffer())
    await _wait_for_ice_gathering(peer, connect_timeout)
    if peer.localDescription is None:
        raise P2PError("WebRTC did not produce a local offer")
    offer = {"sdp": peer.localDescription.sdp, "type": peer.localDescription.type}
    return peer, channel, offer, failure


async def transfer_file(
    *,
    direction: Literal["upload", "download"],
    local_path: Path,
    expected_size: int,
    expected_sha256: str,
    offset: int,
    local_offset: int = 0,
    secret: str,
    ice_servers: list[dict[str, Any]],
    start_remote: Callable[[dict[str, str]], Awaitable[dict[str, Any]]],
    finish_remote: Callable[[str], Awaitable[dict[str, Any]]],
    abort_remote: Callable[[str], Awaitable[None]],
    connect_timeout: float = 45,
    transfer_timeout: float = 900,
) -> dict[str, Any]:
    """Transfer one wire file and require matching acknowledgements from both peers."""
    peer: RTCPeerConnection | None = None
    request_id: str | None = None
    started = time.monotonic()
    try:
        peer, channel, offer, failure = await _create_offer(ice_servers, connect_timeout)
        opened = asyncio.Event()
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[dict[str, Any]] = loop.create_future()
        complete: asyncio.Future[dict[str, Any]] = loop.create_future()
        received = 0
        received_offsets: set[int] = set()
        pending_complete: dict[str, Any] | None = None
        output = None
        if direction == "download":
            output = local_path.open("w+b")

        @channel.on("open")
        def channel_opened() -> None:
            opened.set()

        @channel.on("message")
        def channel_message(message: Any) -> None:
            nonlocal received, pending_complete
            try:
                if isinstance(message, str):
                    payload = json.loads(message)
                    kind = payload.get("type")
                    if kind == "ready" and not ready.done():
                        ready.set_result(payload)
                    elif kind == "complete" and not complete.done():
                        if direction == "upload" or received == expected_size:
                            complete.set_result(payload)
                        else:
                            pending_complete = payload
                    elif kind == "error" and not failure.done():
                        failure.set_exception(P2PError(str(payload.get("message", "remote error"))))
                    return
                if direction != "download" or output is None:
                    raise P2PError("Unexpected binary WebRTC message")
                framed = bytes(message)
                if len(framed) < 9:
                    raise P2PError("WebRTC binary frame is too short")
                position = int.from_bytes(framed[:8], "big")
                chunk = framed[8:]
                if position in received_offsets:
                    return
                if position < 0 or position + len(chunk) > expected_size:
                    raise P2PError("WebRTC download frame exceeded declared bounds")
                received_offsets.add(position)
                received += len(chunk)
                if received > expected_size:
                    raise P2PError("WebRTC download exceeded the declared size")
                output.seek(position)
                output.write(chunk)
                if (
                    pending_complete is not None
                    and received == expected_size
                    and not complete.done()
                ):
                    complete.set_result(pending_complete)
            except BaseException as error:
                if not failure.done():
                    failure.set_exception(error)

        remote = await start_remote(offer)
        request_id = str(remote["request_id"])
        answer = remote["answer"]
        await peer.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        await asyncio.wait_for(opened.wait(), timeout=connect_timeout)
        channel.send(json.dumps({"type": "hello", "secret": secret}, separators=(",", ":")))
        ready_payload = await _wait_for_result(ready, failure, connect_timeout)
        if int(ready_payload.get("offset", -1)) != offset:
            raise P2PError("WebRTC peer reported a conflicting resume offset")
        if direction == "upload":
            sent = offset
            with local_path.open("rb") as source:
                source.seek(local_offset + offset)
                while sent < expected_size:
                    chunk = source.read(min(P2P_CHUNK_SIZE, expected_size - sent))
                    if not chunk:
                        raise P2PError("Local source ended before its declared size")
                    await _drain_channel(channel)
                    channel.send(sent.to_bytes(8, "big") + chunk)
                    sent += len(chunk)
            await _drain_channel(channel)
            channel.send(
                json.dumps(
                    {"type": "eof", "size": expected_size, "sha256": expected_sha256},
                    separators=(",", ":"),
                )
            )
        completed = await _wait_for_result(complete, failure, transfer_timeout)
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
            output.close()
            output = None
        if int(completed.get("size", -1)) != expected_size:
            raise P2PError("WebRTC completion size mismatch")
        if completed.get("sha256") != expected_sha256:
            raise P2PError("WebRTC completion checksum mismatch")
        if direction == "download":
            if (
                received != expected_size
                or await asyncio.to_thread(_file_sha256, local_path) != expected_sha256
            ):
                raise P2PError("WebRTC download checksum mismatch")
            channel.send(json.dumps({"type": "ack", "secret": secret}, separators=(",", ":")))
        remote_result = await finish_remote(request_id)
        if not remote_result.get("ok"):
            raise P2PError(str(remote_result.get("error", "remote WebRTC endpoint failed")))
        return {
            "offset": expected_size,
            "sha256": expected_sha256,
            "seconds": time.monotonic() - started,
            "transport": "webrtc",
            "protocol_version": P2P_PROTOCOL_VERSION,
            "remote": {key: value for key, value in remote_result.items() if key != "error"},
        }
    except BaseException:
        if request_id is not None:
            try:
                await abort_remote(request_id)
            except BaseException:
                pass
        raise
    finally:
        if "output" in locals() and output is not None:
            output.close()
        if peer is not None:
            await peer.close()


P2P_ENDPOINT_SOURCE = r"""\
import asyncio
import datetime
import hashlib
import json
import os
import pathlib
import sys
import time

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

CHUNK_SIZE = 32 * 1024
HIGH_WATER = 4 * 1024 * 1024


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def validate_guard(config):
    fingerprint = pathlib.Path(config["fingerprint_path"]).read_text(encoding="ascii").strip()
    if fingerprint != config["runtime_fingerprint"]:
        raise RuntimeError("runtime_replaced")
    lease = json.loads(pathlib.Path(config["lease_path"]).read_text(encoding="utf-8"))
    candidates = lease.get("leases") or [lease]
    match = next((item for item in candidates if item.get("token") == config["lease_token"]), None)
    if lease.get("runtime_fingerprint") != fingerprint or match is None:
        raise RuntimeError("operation_lease_stale")
    expires = datetime.datetime.fromisoformat(match["expires_at"])
    if expires <= datetime.datetime.now(datetime.timezone.utc):
        raise RuntimeError("operation_lease_expired")


def configuration(records):
    return RTCConfiguration(iceServers=[RTCice(record) for record in records])


def RTCice(record):
    return RTCIceServer(
        urls=record["urls"],
        username=record.get("username"),
        credential=record.get("credential"),
    )


async def wait_ice(peer, timeout):
    if peer.iceGatheringState == "complete":
        return
    event = asyncio.Event()

    @peer.on("icegatheringstatechange")
    def changed():
        if peer.iceGatheringState == "complete":
            event.set()

    await asyncio.wait_for(event.wait(), timeout=timeout)


async def drain(channel):
    while channel.bufferedAmount > HIGH_WATER:
        await asyncio.sleep(0.002)


async def run(config):
    started = time.monotonic()
    validate_guard(config)
    root = pathlib.Path(config["content_root"]).resolve()
    target = pathlib.Path(config["path"]).resolve()
    if target != root and root not in target.parents:
        raise ValueError("transfer path must remain under the content root")
    direction = config["direction"]
    if direction == "upload" and ".colab-mcp-wire-" not in target.name:
        raise ValueError("WebRTC uploads require a transfer staging path")
    if direction == "download" and not target.is_file():
        raise FileNotFoundError(str(target))
    peer = RTCPeerConnection(configuration(config["ice_servers"]))
    done = asyncio.Event()
    channel_ref = {"value": None}
    state = {
        "authenticated": False,
        "received": int(config["offset"]),
        "offsets": set(),
        "handle": None,
        "eof": False,
    }

    async def finish_upload(channel):
        handle = state["handle"]
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        state["handle"] = None
        validate_guard(config)
        digest = hashlib.sha256()
        with target.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        if state["received"] != int(config["size"]) or checksum != config["sha256"]:
            raise ValueError("received WebRTC file did not match its declared size and checksum")
        channel.send(json.dumps({"type": "complete", "size": state["received"], "sha256": checksum}, separators=(",", ":")))
        await drain(channel)
        await asyncio.sleep(0.05)
        done.set()

    async def send_download(channel):
        validate_guard(config)
        channel.send(json.dumps({"type": "ready", "offset": 0}, separators=(",", ":")))
        sent = 0
        with target.open("rb") as source:
            source.seek(int(config.get("source_offset", 0)))
            while sent < int(config["size"]):
                chunk = source.read(min(CHUNK_SIZE, int(config["size"]) - sent))
                if not chunk:
                    raise ValueError("download source ended before its declared size")
                await drain(channel)
                channel.send(sent.to_bytes(8, "big") + chunk)
                sent += len(chunk)
        await drain(channel)
        channel.send(json.dumps({"type": "complete", "size": sent, "sha256": config["sha256"]}, separators=(",", ":")))

    async def guarded(task, channel):
        try:
            await task
        except BaseException as error:
            state["error"] = type(error).__name__ + ": " + str(error)
            try:
                channel.send(json.dumps({"type": "error", "message": state["error"]}, separators=(",", ":")))
            except BaseException:
                pass
            done.set()

    @peer.on("datachannel")
    def datachannel(channel):
        channel_ref["value"] = channel

        @channel.on("message")
        def message_received(message):
            try:
                if isinstance(message, str):
                    payload = json.loads(message)
                    kind = payload.get("type")
                    if kind == "hello":
                        if payload.get("secret") != config["secret"]:
                            raise ValueError("transfer authentication failed")
                        state["authenticated"] = True
                        if direction == "upload":
                            target.parent.mkdir(parents=True, exist_ok=True)
                            current = target.stat().st_size if target.is_file() else 0
                            if current != int(config["offset"]):
                                raise ValueError("transfer offset conflict")
                            state["handle"] = target.open("r+b" if target.exists() else "w+b")
                            channel.send(json.dumps({"type": "ready", "offset": current}, separators=(",", ":")))
                        else:
                            asyncio.create_task(guarded(send_download(channel), channel))
                    elif kind == "eof":
                        if direction != "upload" or not state["authenticated"]:
                            raise ValueError("unexpected transfer EOF")
                        state["eof"] = True
                        if state["received"] == int(config["size"]):
                            asyncio.create_task(guarded(finish_upload(channel), channel))
                    elif kind == "ack":
                        if direction != "download" or payload.get("secret") != config["secret"]:
                            raise ValueError("unexpected transfer acknowledgement")
                        validate_guard(config)
                        done.set()
                    return
                if direction != "upload" or not state["authenticated"] or state["handle"] is None:
                    raise ValueError("unexpected binary transfer message")
                framed = bytes(message)
                if len(framed) < 9:
                    raise ValueError("binary transfer frame is too short")
                position = int.from_bytes(framed[:8], "big")
                block = framed[8:]
                if position in state["offsets"]:
                    return
                if (position < int(config["offset"])
                        or position + len(block) > int(config["size"])):
                    raise ValueError("upload frame exceeded its declared bounds")
                state["offsets"].add(position)
                state["received"] += len(block)
                if state["received"] > int(config["size"]):
                    raise ValueError("upload exceeded its declared size")
                state["handle"].seek(position)
                state["handle"].write(block)
                if state["eof"] and state["received"] == int(config["size"]):
                    asyncio.create_task(guarded(finish_upload(channel), channel))
            except BaseException as error:
                try:
                    channel.send(json.dumps({"type": "error", "message": type(error).__name__ + ": " + str(error)}, separators=(",", ":")))
                except BaseException:
                    pass
                state["error"] = type(error).__name__ + ": " + str(error)
                done.set()

    try:
        offer = config["offer"]
        await peer.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
        await peer.setLocalDescription(await peer.createAnswer())
        await wait_ice(peer, float(config["connect_timeout"]))
        if peer.localDescription is None:
            raise RuntimeError("WebRTC did not produce an answer")
        atomic_json(pathlib.Path(config["answer_path"]), {"sdp": peer.localDescription.sdp, "type": peer.localDescription.type})
        await asyncio.wait_for(done.wait(), timeout=float(config["transfer_timeout"]))
        if state.get("error"):
            raise RuntimeError(state["error"])
        validate_guard(config)
        atomic_json(pathlib.Path(config["result_path"]), {
            "ok": True,
            "size": int(config["size"]),
            "sha256": config["sha256"],
            "seconds": round(time.monotonic() - started, 3),
            "protocol_version": 1,
        })
    finally:
        if state.get("handle") is not None:
            state["handle"].close()
        await peer.close()


def main():
    config_path = pathlib.Path(sys.argv[1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_path.unlink(missing_ok=True)
    result_path = pathlib.Path(config["result_path"])
    try:
        asyncio.run(run(config))
    except BaseException as error:
        atomic_json(result_path, {"ok": False, "error": type(error).__name__ + ": " + str(error)})


if __name__ == "__main__":
    main()
"""

P2P_ENDPOINT_SHA256 = hashlib.sha256(P2P_ENDPOINT_SOURCE.encode()).hexdigest()
