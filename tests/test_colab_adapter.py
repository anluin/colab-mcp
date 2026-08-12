import hashlib
from types import SimpleNamespace

import pytest

from src import colab_adapter


def test_bounded_manager_forwards_constructor_refresh_timeout(monkeypatch):
    seen = []

    def refresh(_self, timeout=30):
        seen.append(timeout)
        return {"id": "kernel"}

    monkeypatch.setattr(colab_adapter.KernelHttpManager, "refresh_model", refresh)
    manager = object.__new__(colab_adapter.BoundedKernelHttpManager)
    manager._colab_mcp_refresh_timeout = 5
    assert manager.refresh_model() == {"id": "kernel"}
    assert seen == [5]


def test_kernel_client_is_the_only_dependency_specific_construction_point(monkeypatch):
    seen = {}

    def construct(**kwargs):
        seen.update(kwargs)
        return "client"

    monkeypatch.setattr(colab_adapter.jupyter_kernel_client, "ColabKernelClient", construct)
    assert colab_adapter.kernel_client(connection_timeout=4, kernel_id="owned") == "client"
    assert seen["kernel_manager_class"] is colab_adapter.BoundedKernelHttpManager
    assert seen["refresh_timeout"] == 4
    assert seen["kernel_id"] == "owned"


@pytest.mark.parametrize("content", [b"", b"binary media" * 100])
def test_binary_upload_streams_buffers_and_verifies_final_ack(tmp_path, content):
    source = tmp_path / "media.mp4"
    source.write_bytes(content)
    received = bytearray()
    replies = []

    class Session:
        @staticmethod
        def msg(message_type, data):
            return {"msg_type": message_type, "content": data}

    class Channel:
        @staticmethod
        def send(message):
            if message["msg_type"] != "comm_msg":
                return
            received.extend(message["buffers"][0])
            if message["content"]["data"]["final"]:
                replies.append(
                    {
                        "msg_type": "comm_msg",
                        "content": {
                            "comm_id": message["content"]["comm_id"],
                            "data": {
                                "final": True,
                                "offset": len(received),
                                "sha256": hashlib.sha256(received).hexdigest(),
                            },
                        },
                    }
                )

    client = SimpleNamespace(
        session=Session(),
        shell_channel=Channel(),
        get_iopub_msg=lambda timeout: replies.pop(0),
    )
    setup = []
    kernel = SimpleNamespace(
        _manager=SimpleNamespace(client=client),
        execute=lambda code, timeout: setup.append(code) or {"outputs": []},
    )
    result = colab_adapter.binary_upload(
        kernel,
        source,
        "/content/file.colab-mcp-wire-test",
        0,
        "a" * 32,
        "b" * 32,
    )
    assert bytes(received) == content
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert "operation_lease_stale" in setup[0]


def test_binary_upload_rejects_receiver_setup_error(tmp_path):
    source = tmp_path / "media.mp4"
    source.write_bytes(b"data")
    kernel = SimpleNamespace(
        execute=lambda *_args, **_kwargs: {
            "outputs": [{"output_type": "error", "ename": "RuntimeError"}]
        }
    )
    with pytest.raises(RuntimeError, match="receiver setup failed"):
        colab_adapter.binary_upload(
            kernel,
            source,
            "/content/file.colab-mcp-wire-test",
            0,
            "a" * 32,
            "b" * 32,
        )
