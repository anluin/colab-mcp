"""Narrow isolation layer for timeout gaps in the pinned kernel client."""

from __future__ import annotations

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
