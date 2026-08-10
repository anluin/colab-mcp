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
