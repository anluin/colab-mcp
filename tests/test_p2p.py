import asyncio
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.manager import ColabManager
from src.p2p import (
    P2P_ENDPOINT_SOURCE,
    P2PError,
    ice_servers_from_environment,
    should_use_p2p,
    transfer_file,
)


def test_ice_configuration_is_bounded_and_credentials_are_preserved(monkeypatch):
    monkeypatch.setenv(
        "COLAB_MCP_WEBRTC_ICE_SERVERS",
        json.dumps(
            [
                {
                    "urls": ["turns:relay.example.test:5349?transport=tcp"],
                    "username": "temporary-user",
                    "credential": "temporary-password",
                }
            ]
        ),
    )
    assert ice_servers_from_environment() == [
        {
            "urls": ["turns:relay.example.test:5349?transport=tcp"],
            "username": "temporary-user",
            "credential": "temporary-password",
        }
    ]


@pytest.mark.parametrize("value", ["{}", "[]", '[{"urls":["https://invalid"]}]'])
def test_invalid_ice_configuration_is_rejected(monkeypatch, value):
    monkeypatch.setenv("COLAB_MCP_WEBRTC_ICE_SERVERS", value)
    with pytest.raises(ValueError, match="ICE|ice|stun"):
        ice_servers_from_environment()


def test_auto_transport_has_a_bulk_threshold(monkeypatch):
    monkeypatch.setenv("COLAB_MCP_WEBRTC_MIN_BYTES", "1024")
    assert not should_use_p2p("auto", 1023)
    assert should_use_p2p("auto", 1024)
    assert should_use_p2p("webrtc", 1)
    assert not should_use_p2p("kernel", 10_000)


def test_auto_transport_uses_measured_bulk_circuit_breaker(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_MCP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COLAB_MCP_WEBRTC_MIN_BYTES", str(4 * 1024 * 1024))
    manager = ColabManager()
    size = 32 * 1024 * 1024
    manager._record_sync_speed("push", size, 8, "kernel_websocket")
    manager._record_sync_speed("push", size, 64, "webrtc")
    assert manager._preferred_transfer_transport("push", "auto", size) == "kernel"
    assert manager._preferred_transfer_transport("push", "webrtc", size) == "webrtc"


def test_auto_transport_prefers_webrtc_only_after_a_clear_measured_win(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_MCP_STATE_DIR", str(tmp_path / "state"))
    manager = ColabManager()
    size = 32 * 1024 * 1024
    manager._record_sync_speed("pull", size, 16, "authenticated_files_proxy")
    manager._record_sync_speed("pull", size, 8, "webrtc")
    assert manager._preferred_transfer_transport("pull", "auto", size) == "webrtc"


def test_auto_transport_samples_proxy_after_first_peer_result_and_cools_down_failures(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("COLAB_MCP_STATE_DIR", str(tmp_path / "state"))
    manager = ColabManager()
    size = 32 * 1024 * 1024
    manager._record_sync_speed("push", size, 8, "webrtc")
    assert manager._preferred_transfer_transport("push", "auto", size) == "kernel"
    manager._record_transport_failure("pull", size, P2PError("injected"))
    assert manager._preferred_transfer_transport("pull", "auto", size) == "kernel"


def test_auto_upload_falls_back_but_required_webrtc_does_not(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_MCP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COLAB_MCP_WEBRTC_MIN_BYTES", "0")
    manager = ColabManager()
    source = tmp_path / "wire.bin"
    source.write_bytes(b"verified")
    session = SimpleNamespace(name="runtime")
    fallback_calls = []

    async def failed_peer(**_kwargs):
        raise P2PError("injected peer failure")

    async def fallback(*_args):
        fallback_calls.append(True)
        return {"offset": 8, "sha256": hashlib.sha256(b"verified").hexdigest(), "seconds": 1}

    monkeypatch.setattr(manager, "_parallel_p2p_upload", failed_peer)
    monkeypatch.setattr(manager, "_binary_upload_file", fallback)
    result = asyncio.run(
        manager._upload_wire_file(
            source,
            "/content/file.colab-mcp-wire-test",
            0,
            8,
            hashlib.sha256(b"verified").hexdigest(),
            100,
            session,
            "b" * 32,
            "auto",
        )
    )
    assert result["transport"] == "kernel_websocket"
    assert fallback_calls == [True]
    with pytest.raises(P2PError, match="injected"):
        asyncio.run(
            manager._upload_wire_file(
                source,
                "/content/file.colab-mcp-wire-test",
                0,
                8,
                hashlib.sha256(b"verified").hexdigest(),
                100,
                session,
                "b" * 32,
                "webrtc",
            )
        )


def _run_endpoint_transfer(tmp_path: Path, direction: str, content: bytes) -> dict:
    endpoint = tmp_path / "endpoint.py"
    endpoint.write_text(P2P_ENDPOINT_SOURCE, encoding="utf-8")
    content_root = tmp_path / "content"
    state_root = content_root / ".colab-mcp"
    state_root.mkdir(parents=True)
    fingerprint = "a" * 32
    lease_token = "b" * 32
    (state_root / "runtime-incarnation").write_text(fingerprint, encoding="ascii")
    (state_root / "operation-lease.json").write_text(
        json.dumps(
            {
                "runtime_fingerprint": fingerprint,
                "leases": [
                    {
                        "token": lease_token,
                        "expires_at": (
                            datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5)
                        ).isoformat(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checksum = hashlib.sha256(content).hexdigest()
    remote_path = content_root / "payload.colab-mcp-wire-test"
    local_path = tmp_path / "local.bin"
    if direction == "download":
        remote_path.write_bytes(content)
    else:
        local_path.write_bytes(content)
    process: subprocess.Popen | None = None

    async def exercise() -> dict:
        async def start_remote(offer):
            nonlocal process
            request = state_root / "request"
            request.mkdir()
            answer_path = request / "answer.json"
            result_path = request / "result.json"
            config_path = request / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "runtime_fingerprint": fingerprint,
                        "lease_token": lease_token,
                        "fingerprint_path": str(state_root / "runtime-incarnation"),
                        "lease_path": str(state_root / "operation-lease.json"),
                        "content_root": str(content_root),
                        "direction": direction,
                        "path": str(remote_path),
                        "size": len(content),
                        "offset": 0,
                        "sha256": checksum,
                        "secret": "c" * 64,
                        "offer": offer,
                        "ice_servers": [],
                        "connect_timeout": 10,
                        "transfer_timeout": 20,
                        "answer_path": str(answer_path),
                        "result_path": str(result_path),
                    }
                ),
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [sys.executable, str(endpoint), str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
            )
            for _ in range(200):
                if answer_path.is_file():
                    return {"request_id": "d" * 32, "answer": json.loads(answer_path.read_text())}
                if result_path.is_file():
                    raise RuntimeError(json.loads(result_path.read_text()).get("error"))
                await asyncio.sleep(0.05)
            raise TimeoutError("test endpoint did not answer")

        async def finish_remote(_request_id):
            result_path = state_root / "request" / "result.json"
            for _ in range(200):
                if result_path.is_file():
                    return json.loads(result_path.read_text())
                await asyncio.sleep(0.05)
            raise TimeoutError("test endpoint did not finish")

        async def abort_remote(_request_id):
            if process is not None and process.poll() is None:
                process.terminate()

        return await transfer_file(
            direction=direction,
            local_path=local_path,
            expected_size=len(content),
            expected_sha256=checksum,
            offset=0,
            secret="c" * 64,
            ice_servers=[],
            start_remote=start_remote,
            finish_remote=finish_remote,
            abort_remote=abort_remote,
            connect_timeout=10,
            transfer_timeout=20,
        )

    try:
        result = asyncio.run(exercise())
    finally:
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    published = remote_path if direction == "upload" else local_path
    assert published.read_bytes() == content
    return result


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_bidirectional_loopback_transfer_is_verified(tmp_path, direction):
    content = os.urandom(2 * 1024 * 1024 + 17)
    result = _run_endpoint_transfer(tmp_path, direction, content)
    assert result["transport"] == "webrtc"
    assert result["offset"] == len(content)


def test_large_unordered_download_does_not_lose_frames(tmp_path):
    content = os.urandom(16 * 1024 * 1024 + 31)
    result = _run_endpoint_transfer(tmp_path, "download", content)
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
