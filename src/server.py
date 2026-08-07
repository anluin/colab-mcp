from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

from .logging_config import configure_logging
from .manager import COMPUTE_UNITS_URL, ColabManager
from .version import COLAB_CLI_VERSION, COLAB_MCP_VERSION

mcp = FastMCP(
    "Google Colab Runtime",
    instructions=(
        "Use ephemeral Google Colab compute across Windows, macOS, and Linux. "
        "Prefer T4 unless another accelerator is required. Always stop or pause "
        "sessions after use to release quota. Pause releases the runtime; resume "
        "creates a fresh runtime and does not preserve RAM or /content files."
    ),
)
manager = ColabManager()


@mcp.tool()
async def colab_health() -> dict:
    """Report authentication readiness without starting a runtime."""
    return {
        "ready": manager.authenticated,
        "platform": "cross-platform",
        "supported_operating_systems": ["Windows", "macOS", "Linux"],
        "requires_wsl": False,
        "version": COLAB_MCP_VERSION,
        "google_colab_cli_version": COLAB_CLI_VERSION,
        "transport": "stdio",
        "interactive_auth_allowed": False,
    }


@mcp.tool()
async def colab_sessions() -> list[dict]:
    """List locally tracked sessions and whether they still exist on Colab."""
    return await manager.sessions()


@mcp.tool()
async def colab_reconcile(forget_stale: bool = False, release_orphans: bool = False) -> dict:
    """Audit local versus live assignments; cleanup actions require explicit flags."""
    return await manager.reconcile(forget_stale, release_orphans)


@mcp.tool()
async def colab_inspect(
    session: str | None = None,
    tools: list[str] | None = None,
    process_limit: int = 100,
) -> dict:
    """Inspect OS, Python, CPU, RAM, disk, GPU/CUDA, tools, and bounded processes."""
    return await manager.inspect_runtime(session, tools, process_limit)


@mcp.tool()
def colab_create_notebook(path: str, code_cells: list[str] | None = None) -> dict:
    """Create a new local Colab-ready .ipynb notebook without consuming GPU quota."""
    return manager.create_notebook(path, code_cells)


@mcp.tool()
async def colab_start(
    session: str, gpu: Literal["T4", "L4", "G4", "H100", "A100"] | None = "T4"
) -> dict:
    """Allocate a Colab CPU or GPU runtime under the authenticated personal account."""
    result = await manager.start(session, gpu)
    return result.model_dump(exclude={"token"})


@mcp.tool()
async def colab_execute(
    code: str,
    session: str | None = None,
    timeout: float = 900,
    output_limit: int = 100_000,
) -> list[dict]:
    """Execute bounded Python in an existing runtime and return Jupyter outputs."""
    return await manager.execute_python(code, session, timeout, output_limit)


@mcp.tool()
async def colab_run_command(
    argv: list[str],
    session: str | None = None,
    cwd: str = "/content",
    environment: dict[str, str] | None = None,
    timeout: float = 300,
    output_limit: int = 100_000,
) -> dict:
    """Wait for a durable command; on timeout return process_id and leave it running."""
    return await manager.run_command(argv, session, cwd, environment, timeout, output_limit)


@mcp.tool()
async def colab_process_start(
    argv: list[str],
    session: str | None = None,
    cwd: str = "/content",
    environment: dict[str, str] | None = None,
    output_limit: int = 10_000_000,
) -> dict:
    """Start a command; persist at most output_limit bytes for each output stream."""
    return await manager.process_start(argv, session, cwd, environment, output_limit)


@mcp.tool()
async def colab_process_status(process_id: str, session: str | None = None) -> dict:
    """Inspect a process previously started in the selected runtime."""
    return await manager.process_status(process_id, session)


@mcp.tool()
async def colab_process_list(session: str | None = None) -> list[dict]:
    """List processes owned by colab-mcp in the selected runtime."""
    return await manager.process_list(session)


@mcp.tool()
async def colab_process_output(
    process_id: str,
    session: str | None = None,
    stream: Literal["stdout", "stderr"] = "stdout",
    offset: int = 0,
    limit: int = 65_536,
) -> dict:
    """Read a bounded output chunk; pass next_offset to continue incrementally."""
    return await manager.process_output(process_id, session, stream, offset, limit)


@mcp.tool()
async def colab_process_signal(
    process_id: str,
    session: str | None = None,
    signal: Literal["TERM", "KILL", "INT"] = "TERM",
) -> dict:
    """Signal a running process owned by colab-mcp in the selected runtime."""
    return await manager.process_signal(process_id, session, signal)


@mcp.tool()
async def colab_fs_list(
    path: str = "/content", session: str | None = None, limit: int = 1_000
) -> dict:
    """List a bounded number of entries under /content in the selected runtime."""
    return await manager.filesystem_list(path, session, limit)


@mcp.tool()
async def colab_fs_stat(path: str, session: str | None = None, checksum: bool = False) -> dict:
    """Inspect a runtime path; optionally calculate SHA-256 for a file."""
    return await manager.filesystem_stat(path, session, checksum)


@mcp.tool()
async def colab_fs_read(
    path: str,
    session: str | None = None,
    offset: int = 0,
    limit: int = 262_144,
) -> dict:
    """Read a bounded binary chunk as base64; use next_offset to continue."""
    return await manager.filesystem_read(path, session, offset, limit)


@mcp.tool()
async def colab_fs_write(
    path: str,
    data_base64: str,
    session: str | None = None,
    append: bool = False,
    create_parents: bool = False,
) -> dict:
    """Atomically write up to 1 MB of base64 data, or explicitly append a chunk."""
    return await manager.filesystem_write(path, data_base64, session, append, create_parents)


@mcp.tool()
async def colab_fs_mkdir(
    path: str,
    session: str | None = None,
    parents: bool = True,
    exist_ok: bool = True,
) -> dict:
    """Create a directory under /content."""
    return await manager.filesystem_mkdir(path, session, parents, exist_ok)


@mcp.tool()
async def colab_fs_move(
    source: str,
    destination: str,
    session: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Move a runtime file or directory; overwrite must be explicit."""
    return await manager.filesystem_move(source, destination, session, overwrite)


@mcp.tool()
async def colab_fs_remove(
    path: str,
    session: str | None = None,
    recursive: bool = False,
    missing_ok: bool = False,
) -> dict:
    """Remove a runtime path; non-empty directories require recursive=true."""
    return await manager.filesystem_remove(path, session, recursive, missing_ok)


@mcp.tool()
async def colab_transfer_upload(
    local_path: str,
    remote_path: str,
    session: str | None = None,
    overwrite: bool = False,
    sync: bool = True,
    chunk_size: int = 524_288,
    max_total_bytes: int = 100_000_000,
    max_files: int = 10_000,
) -> dict:
    """Upload a file/directory with staged chunks, SHA-256 verification, and sync skips."""
    return await manager.transfer_upload(
        local_path,
        remote_path,
        session,
        overwrite,
        sync,
        chunk_size,
        max_total_bytes,
        max_files,
    )


@mcp.tool()
async def colab_transfer_download(
    remote_path: str,
    local_path: str,
    session: str | None = None,
    overwrite: bool = False,
    sync: bool = True,
    chunk_size: int = 524_288,
    max_total_bytes: int = 100_000_000,
    max_files: int = 10_000,
) -> dict:
    """Download a file/directory with bounded chunks, SHA-256, and atomic publication."""
    return await manager.transfer_download(
        remote_path,
        local_path,
        session,
        overwrite,
        sync,
        chunk_size,
        max_total_bytes,
        max_files,
    )


@mcp.tool()
async def colab_execute_notebook(
    source: str, output: str, session: str | None = None, cell_timeout: float = 900
) -> dict:
    """Execute code cells from a local notebook and save an output notebook locally."""
    return await manager.execute_notebook(source, output, session, cell_timeout)


@mcp.tool()
async def colab_upload(local_path: str, remote_path: str, session: str | None = None) -> dict:
    """Compatibility alias for bounded, checksummed colab_transfer_upload."""
    return await manager.upload(local_path, remote_path, session)


@mcp.tool()
async def colab_download(remote_path: str, local_path: str, session: str | None = None) -> dict:
    """Compatibility alias for bounded, checksummed colab_transfer_download."""
    return await manager.download(remote_path, local_path, session)


@mcp.tool()
async def colab_stop(session: str | None = None) -> dict:
    """Release a Colab runtime and cancel its in-process keep-alive task."""
    return await manager.stop(session)


@mcp.tool()
async def colab_pause_notebook(session: str, notebook_path: str) -> dict:
    """Checkpoint a local notebook and release its Colab GPU; RAM and runtime files are not preserved."""
    return await manager.pause(session, notebook_path)


@mcp.tool()
async def colab_resume_notebook(
    session: str,
    execute_notebook: bool = False,
    output_path: str | None = None,
    cell_timeout: float = 900,
) -> dict:
    """Allocate a fresh runtime with the prior GPU preference and optionally rerun the paused notebook."""
    return await manager.resume(session, execute_notebook, output_path, cell_timeout)


@mcp.tool()
def colab_paused_notebooks() -> list[dict]:
    """List notebook checkpoints whose GPU runtimes were released."""
    return manager.suspended()


@mcp.tool()
def colab_compute_units() -> dict:
    """Explain compute-unit visibility and return Google's official account-management URL."""
    return {
        "available_via_supported_api": False,
        "balance": None,
        "management_url": COMPUTE_UNITS_URL,
        "reason": (
            "Google Colab does not publish a supported API for a personal account's "
            "compute-unit balance. The official CLI only opens this management page."
        ),
    }


def main() -> None:
    configure_logging()
    import logging

    logger = logging.getLogger("colab_mcp.server")
    logger.info("server_started transport=stdio version=%s", COLAB_MCP_VERSION)
    try:
        mcp.run(transport="stdio")
    finally:
        logger.info("server_stopped assignments_persist_for_recovery=true")


if __name__ == "__main__":
    main()
