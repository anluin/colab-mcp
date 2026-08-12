from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from .logging_config import configure_logging
from .manager import AutoExportRule, ColabManager
from .version import COLAB_CLI_VERSION, COLAB_MCP_VERSION

manager = ColabManager()

SessionName = Annotated[str, Field(description="Unique local name of a tracked Colab assignment.")]
SessionSelector = Annotated[
    str | None,
    Field(
        description="Tracked session name. Null is allowed only when exactly one session exists."
    ),
]
ProcessId = Annotated[
    str,
    Field(description="Opaque process_id returned by colab_process_start or colab_run_command."),
]
Argv = Annotated[
    list[str],
    Field(
        description="Non-empty executable and argument array; no shell parsing, expansion, or pipes."
    ),
]
RemotePath = Annotated[
    str,
    Field(description="Runtime path confined to /content; relative paths resolve under /content."),
]
LocalPath = Annotated[
    str, Field(description="Path on the MCP host, resolved under the host user's permissions.")
]
WorkingDirectory = Annotated[
    str, Field(description="Existing runtime directory under /content. Defaults to /content.")
]
Environment = Annotated[
    dict[str, str] | None,
    Field(description="Optional environment overrides. Values are never returned or journaled."),
]
Overwrite = Annotated[
    bool, Field(description="When true, explicitly permit replacement of an existing destination.")
]
ChunkSize = Annotated[
    int,
    Field(
        ge=1,
        le=2_000_000,
        description="Transfer chunk size in bytes; 1-2,000,000. Defaults to 524,288.",
    ),
]
MaxTotalBytes = Annotated[
    int,
    Field(
        ge=1,
        le=10_000_000_000,
        description="Hard total transfer limit in bytes; checked before publication.",
    ),
]
MaxFiles = Annotated[
    int, Field(ge=1, le=100_000, description="Hard file-count limit for a directory transfer.")
]
LeaseToken = Annotated[
    str | None,
    Field(
        description="Opaque operation-bound lease from colab_allocation_probe. Null performs a fresh probe."
    ),
]
CompressionMode = Annotated[
    Literal["auto", "gzip", "none"],
    Field(
        description="Wire compression: auto uses gzip only when worthwhile; gzip forces it; none disables it."
    ),
]
CompressionMinBytes = Annotated[
    int,
    Field(
        ge=0,
        le=10_000_000_000,
        description="Auto mode only considers files at least this many content bytes; defaults to 1 MiB.",
    ),
]
CompressionMinSavings = Annotated[
    float,
    Field(
        ge=0,
        lt=1,
        description="Minimum fractional wire-byte saving required by auto mode; defaults to 0.10.",
    ),
]


@asynccontextmanager
async def server_lifespan(_server: FastMCP):
    await manager.recover_keepalives()
    await manager.recover_process_export_watchers()
    try:
        yield
    finally:
        await manager.shutdown_process_export_watchers()
        await manager.shutdown_kernel_channels()
        await manager.shutdown_keepalives()


mcp = FastMCP(
    "Google Colab Runtime",
    instructions=(
        "Use ephemeral Google Colab compute across Windows, macOS, and Linux. "
        "Prefer T4 unless another accelerator is required. Always stop or pause "
        "sessions after use to release quota. Pause releases the runtime; resume "
        "creates a fresh runtime and does not preserve RAM or /content files. "
        "Recovery contract: runtime_missing means the assignment ended; start a new "
        "session and restore durable inputs. runtime_replaced means Colab recycled the "
        "backend; never reuse its lease or assume /content/process state survived. "
        "kernel_connection_failed/request_not_submitted is retried once automatically after "
        "the same lease/assignment is confirmed; if it persists, retry with that lease while valid. "
        "response_lost may have "
        "executed remotely; inspect process/file state before retrying. operation_timeout "
        "does not imply cancellation; poll durable processes or signal them explicitly. "
        "quota_or_preemption requires releasing unused sessions and waiting or changing "
        "account capacity. local_transport_failure requires restarting the MCP client; "
        "then use sessions/process_list to recover persisted ownership. Preserve returned "
        "transfer_id, staging_path, process_id, next_offset, and recoverable_export fields. "
        "If this connector is defective, use the colab-self-repair skill to patch "
        "https://github.com/anluin/colab-mcp. GitHub access must use authenticated gh only; "
        "when gh is missing, stop and ask the user whether to install it. After a worker-code "
        "patch, use colab_connector status/reload to apply it in this task; its failure keeps the "
        "previous worker active. Supervisor/plugin changes still require a client refresh."
    ),
    lifespan=server_lifespan,
)


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
        "self_repair": {
            "repository": "https://github.com/anluin/colab-mcp",
            "github_client": "gh-only",
            "missing_gh": "Ask the user whether to install GitHub CLI; do not substitute.",
        },
    }


@mcp.tool()
async def colab_sessions() -> list[dict]:
    """List tracked/live sessions. If stale, reconcile it; do not trust its runtime files."""
    return await manager.sessions()


@mcp.tool()
async def colab_keepalive(
    session: SessionSelector = None,
    refresh: Annotated[
        bool, Field(description="True sends a ping now; false only reports persisted/task state.")
    ] = True,
) -> dict:
    """Report/refresh heartbeat. It cannot prevent reclamation; on loss, start and restore."""
    return await manager.keepalive(session, refresh)


@mcp.tool()
async def colab_reconcile(
    forget_stale: Annotated[
        bool,
        Field(description="Delete local records for assignments confirmed absent; default false."),
    ] = False,
    release_orphans: Annotated[
        bool,
        Field(
            description="Release every live account assignment not owned here; destructive, default false."
        ),
    ] = False,
) -> dict:
    """Audit ownership. Forget confirmed stale records; release orphans only if intentional."""
    return await manager.reconcile(forget_stale, release_orphans)


@mcp.tool()
async def colab_inspect(
    session: SessionSelector = None,
    tools: Annotated[
        list[str] | None,
        Field(description="Executable names to locate; null uses the documented default tool set."),
    ] = None,
    process_limit: Annotated[
        int, Field(ge=1, le=1_000, description="Maximum OS process rows returned; defaults to 100.")
    ] = 100,
) -> dict:
    """Inspect runtime resources. On incarnation error, discard results and reacquire."""
    return await manager.inspect_runtime(session, tools, process_limit)


@mcp.tool()
def colab_create_notebook(
    path: Annotated[
        str, Field(description="New local .ipynb path; existing files are never replaced.")
    ],
    code_cells: Annotated[
        list[str] | None,
        Field(description="Optional ordered Python source cells; null creates none."),
    ] = None,
) -> dict:
    """Create a new local Colab-ready .ipynb notebook without consuming GPU quota."""
    return manager.create_notebook(path, code_cells)


@mcp.tool()
async def colab_start(
    session: SessionName,
    gpu: Annotated[
        Literal["T4", "L4", "G4", "H100", "A100"] | None,
        Field(
            description="Requested GPU model; null requests CPU. T4 is the default, not guaranteed."
        ),
    ] = "T4",
) -> dict:
    """Allocate compute. On quota/preemption, release unused sessions and retry later."""
    result = await manager.start(session, gpu)
    return result.model_dump(exclude={"token", "operation_lease_token"})


@mcp.tool()
async def colab_execute(
    code: Annotated[str, Field(description="Python source executed through the runtime kernel.")],
    session: SessionSelector = None,
    timeout: Annotated[
        float, Field(ge=0.1, le=21_600, description="Maximum wait in seconds; defaults to 900.")
    ] = 900,
    output_limit: Annotated[
        int,
        Field(
            ge=1, le=1_000_000, description="Maximum returned output bytes; defaults to 100,000."
        ),
    ] = 100_000,
    lease_token: LeaseToken = None,
) -> dict:
    """Execute guarded Python. Timeout may be ambiguous; use durable process_start for long work."""
    return await manager.execute_python_detailed(code, session, timeout, output_limit, lease_token)


@mcp.tool()
async def colab_run_command(
    argv: Argv,
    session: SessionSelector = None,
    cwd: WorkingDirectory = "/content",
    environment: Environment = None,
    timeout: Annotated[
        float,
        Field(
            ge=0.1,
            le=21_600,
            description="Handoff wait in seconds; timeout leaves the durable process running.",
        ),
    ] = 300,
    output_limit: Annotated[
        int,
        Field(ge=1, le=1_000_000, description="Maximum stdout/stderr bytes returned per stream."),
    ] = 100_000,
) -> dict:
    """Run durably. Timeout leaves it running: retain process_id, poll output/status, or signal."""
    return await manager.run_command(argv, session, cwd, environment, timeout, output_limit)


@mcp.tool()
async def colab_process_start(
    argv: Argv,
    session: SessionSelector = None,
    cwd: WorkingDirectory = "/content",
    environment: Environment = None,
    output_limit: Annotated[
        int,
        Field(
            ge=1,
            le=1_000_000_000,
            description="Durable byte cap for each output stream; defaults to 10,000,000.",
        ),
    ] = 10_000_000,
    export_on_exit: Annotated[
        list[AutoExportRule] | None,
        Field(
            description="Optional durable auto-export rules. They poll in the MCP background, survive server restart, and never release the runtime."
        ),
    ] = None,
    lease_token: LeaseToken = None,
) -> dict:
    """Start durably under a lease. request_not_submitted retries once automatically after lease validation; response_lost has unknown outcome, so list processes before retrying."""
    return await manager.process_start(
        argv, session, cwd, environment, output_limit, export_on_exit, lease_token
    )


@mcp.tool()
async def colab_process_status(process_id: ProcessId, session: SessionSelector = None) -> dict:
    """Inspect owned process. If runtime vanished, use preserved metadata and restore artifacts."""
    return await manager.process_status(process_id, session)


@mcp.tool()
async def colab_process_list(session: SessionSelector = None) -> list[dict]:
    """List persisted owned processes. After restart, use this to recover IDs and export state."""
    return await manager.process_list(session)


@mcp.tool()
async def colab_process_output(
    process_id: ProcessId,
    session: SessionSelector = None,
    stream: Annotated[
        Literal["stdout", "stderr"], Field(description="Output stream to read; defaults to stdout.")
    ] = "stdout",
    offset: Annotated[
        int, Field(ge=0, description="Byte offset in the retained spool; use prior next_offset.")
    ] = 0,
    limit: Annotated[
        int, Field(ge=1, le=1_000_000, description="Maximum bytes returned; defaults to 65,536.")
    ] = 65_536,
) -> dict:
    """Read retained output. Keep next_offset; on runtime loss, the local spool remains readable."""
    return await manager.process_output(process_id, session, stream, offset, limit)


@mcp.tool()
async def colab_process_signal(
    process_id: ProcessId,
    session: SessionSelector = None,
    signal: Annotated[
        Literal["TERM", "KILL", "INT"],
        Field(description="Explicit process-group signal; TERM default, KILL is immediate."),
    ] = "TERM",
) -> dict:
    """Signal an owned process. If already exited/lost, inspect status; never signal a replacement."""
    return await manager.process_signal(process_id, session, signal)


@mcp.tool()
async def colab_process_export(
    process_id: ProcessId,
    remote_path: RemotePath,
    local_path: LocalPath,
    session: SessionSelector = None,
    release_on_success: Annotated[
        bool,
        Field(
            description="True releases the runtime only after verified publication; default false."
        ),
    ] = False,
    overwrite: Overwrite = False,
    chunk_size: ChunkSize = 524_288,
    max_total_bytes: MaxTotalBytes = 100_000_000,
    max_files: MaxFiles = 10_000,
    compression: CompressionMode = "auto",
    compression_min_bytes: CompressionMinBytes = 1_048_576,
    compression_min_savings: CompressionMinSavings = 0.10,
) -> dict:
    """Export atomically. Failure holds runtime/stage; retry same call from recoverable_export."""
    return await manager.process_export(
        process_id,
        remote_path,
        local_path,
        session,
        release_on_success,
        overwrite,
        chunk_size,
        max_total_bytes,
        max_files,
        compression,
        compression_min_bytes,
        compression_min_savings,
    )


@mcp.tool()
async def colab_process_export_cleanup(
    process_id: ProcessId,
    remote_path: RemotePath,
    local_path: LocalPath,
    session: SessionSelector = None,
) -> dict:
    """Discard failed export stage. Use only after abandoning retry; remote artifacts are unchanged."""
    return await manager.process_export_cleanup(process_id, remote_path, local_path, session)


async def colab_fs_list(
    path: Annotated[
        str, Field(description="Runtime directory under /content; defaults to /content.")
    ] = "/content",
    session: SessionSelector = None,
    limit: Annotated[int, Field(ge=1, le=10_000, description="Maximum entries returned.")] = 1_000,
) -> dict:
    """List runtime paths. On runtime_replaced/missing, reacquire; old /content is unrecoverable."""
    return await manager.filesystem_list(path, session, limit)


async def colab_fs_stat(
    path: RemotePath,
    session: SessionSelector = None,
    checksum: Annotated[
        bool, Field(description="True computes SHA-256 for a file; default false.")
    ] = False,
) -> dict:
    """Stat/checksum a path. Missing may mean reclamation; verify the session fingerprint first."""
    return await manager.filesystem_stat(path, session, checksum)


async def colab_fs_read(
    path: RemotePath,
    session: SessionSelector = None,
    offset: Annotated[int, Field(ge=0, description="Byte offset; defaults to zero.")] = 0,
    limit: Annotated[
        int, Field(ge=1, le=1_000_000, description="Maximum bytes returned as base64.")
    ] = 262_144,
) -> dict:
    """Read a chunk. Keep next_offset; after incarnation change restart from restored source."""
    return await manager.filesystem_read(path, session, offset, limit)


async def colab_fs_write(
    path: RemotePath,
    data_base64: Annotated[
        str,
        Field(description="Base64-encoded bytes; decoded payload is limited to 1,000,000 bytes."),
    ],
    session: SessionSelector = None,
    append: Annotated[
        bool, Field(description="True appends; false atomically replaces the file. Default false.")
    ] = False,
    create_parents: Annotated[
        bool, Field(description="True creates missing parent directories; default false.")
    ] = False,
) -> dict:
    """Write a small chunk. Append retries are not idempotent; stat before retrying ambiguous writes."""
    return await manager.filesystem_write(path, data_base64, session, append, create_parents)


async def colab_fs_mkdir(
    path: RemotePath,
    session: SessionSelector = None,
    parents: Annotated[
        bool, Field(description="True creates missing ancestors; defaults to true.")
    ] = True,
    exist_ok: Annotated[
        bool, Field(description="True accepts an existing directory; defaults to true.")
    ] = True,
) -> dict:
    """Create a runtime directory. Use exist_ok for idempotent retry on ambiguous responses."""
    return await manager.filesystem_mkdir(path, session, parents, exist_ok)


async def colab_fs_move(
    source: Annotated[str, Field(description="Existing source path under /content.")],
    destination: Annotated[str, Field(description="Destination path under /content.")],
    session: SessionSelector = None,
    overwrite: Overwrite = False,
) -> dict:
    """Move a path. After response loss, stat source/destination before retrying."""
    return await manager.filesystem_move(source, destination, session, overwrite)


async def colab_fs_remove(
    path: RemotePath,
    session: SessionSelector = None,
    recursive: Annotated[
        bool, Field(description="Required to remove a non-empty directory; default false.")
    ] = False,
    missing_ok: Annotated[
        bool, Field(description="True treats an absent path as success; default false.")
    ] = False,
) -> dict:
    """Remove explicitly. After response loss, stat first; use missing_ok for idempotent cleanup."""
    return await manager.filesystem_remove(path, session, recursive, missing_ok)


@mcp.tool()
async def colab_transfer_cleanup(
    staging_paths: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=1_000,
            description="Explicit remote staging_path values returned by failed transfers; ordinary files are rejected.",
        ),
    ],
    session: SessionSelector = None,
    lease_token: LeaseToken = None,
) -> dict:
    """Remove returned staging paths. If the lease expired, probe and clean only the same runtime."""
    return await manager.transfer_cleanup(staging_paths, session, lease_token)


@mcp.tool()
async def colab_allocation_probe(
    session: SessionSelector = None,
    observations: Annotated[
        int, Field(ge=2, le=5, description="Assignment observations; 2-5, defaults to 2.")
    ] = 2,
    interval: Annotated[
        float, Field(ge=0, le=5, description="Seconds between observations; defaults to 0.25.")
    ] = 0.25,
) -> dict:
    """Issue an operation lease using fast assignment observations plus the existing background heartbeat. Pass it immediately; if expired/mismatched, probe again, never follow replacement."""
    return await manager.allocation_probe(session, observations, interval)


@mcp.tool()
async def colab_execute_notebook(
    source: Annotated[str, Field(description="Existing local .ipynb input path.")],
    output: Annotated[str, Field(description="Local output .ipynb path to create or replace.")],
    session: SessionSelector = None,
    cell_timeout: Annotated[
        float, Field(ge=0.1, le=21_600, description="Maximum seconds per code cell.")
    ] = 900,
) -> dict:
    """Execute notebook cells. On failure keep the input/output checkpoint; reacquire and rerun deliberately."""
    return await manager.execute_notebook(source, output, session, cell_timeout)


@mcp.tool()
async def colab_stop(session: SessionSelector = None) -> dict:
    """Release compute idempotently. Export first; stopping permanently loses RAM and /content."""
    return await manager.stop(session)


@mcp.tool()
async def colab_pause_notebook(
    session: SessionName,
    notebook_path: Annotated[str, Field(description="Existing local .ipynb checkpoint path.")],
) -> dict:
    """Checkpoint locally then release. Transfer non-notebook artifacts first; /content is lost."""
    return await manager.pause(session, notebook_path)


@mcp.tool()
async def colab_resume_notebook(
    session: SessionName,
    execute_notebook: Annotated[
        bool, Field(description="True reruns the checkpoint on the fresh runtime; default false.")
    ] = False,
    output_path: Annotated[
        str | None,
        Field(
            description="Local rerun output path; null generates <checkpoint>.resumed.ipynb when executing."
        ),
    ] = None,
    cell_timeout: Annotated[
        float, Field(ge=0.1, le=21_600, description="Maximum seconds per rerun code cell.")
    ] = 900,
) -> dict:
    """Resume onto a fresh runtime. Restore dependencies/files; prior RAM, processes, and leases are invalid."""
    return await manager.resume(session, execute_notebook, output_path, cell_timeout)


@mcp.tool()
def colab_paused_notebooks() -> list[dict]:
    """List notebook checkpoints whose GPU runtimes were released."""
    return manager.suspended()


@mcp.tool()
async def colab_workspace_sync(
    direction: Annotated[
        Literal["push", "pull"],
        Field(description="push copies local_folder to Colab; pull copies remote_folder locally."),
    ],
    local_folder: Annotated[
        str, Field(description="Local directory tree. Push requires it to exist; pull creates it.")
    ],
    ctx: Context,
    remote_folder: Annotated[
        str,
        Field(description="Remote directory tree under /content; use a dedicated workspace root."),
    ] = "/content/workspace",
    session: SessionSelector = None,
    lease_token: LeaseToken = None,
    chunk_size: ChunkSize = 2_000_000,
    max_total_bytes: MaxTotalBytes = 1_000_000_000,
    max_files: MaxFiles = 10_000,
    compression: CompressionMode = "auto",
    include: Annotated[
        list[str] | None,
        Field(
            description="Optional positive POSIX-glob selection relative to the chosen roots; built-in safety exclusions still apply.",
            max_length=100,
        ),
    ] = None,
) -> dict:
    """Incrementally sync one purposeful folder using SHA-256 and atomic publication. Use include for a positive selection. Destination extras are never deleted; retry confirmed pre-submission failures with the same lease."""
    local = Path(local_folder).expanduser().resolve()
    if direction == "push" and not local.is_dir():
        raise ValueError("local_folder must be an existing directory for push")
    if lease_token is None:
        lease_token = (await manager.allocation_probe(session))["lease_token"]
    (
        selected_paths,
        changed_paths,
        selected_name,
        accepted_lease,
    ) = await manager.workspace_sync_selection(
        direction=direction,
        local_folder=str(local),
        remote_folder=remote_folder,
        name=session,
        include=include,
        max_files=max_files,
        max_total_bytes=max_total_bytes,
        lease_token=lease_token,
    )
    lease_token = accepted_lease["lease_token"]
    if direction == "push":

        async def progress(event: dict) -> None:
            await ctx.report_progress(
                event["bytes_sent"],
                event["total_bytes"],
                json.dumps(event, separators=(",", ":")),
            )

        result = await manager.workspace_upload(
            local_folder=str(local),
            remote_folder=remote_folder,
            name=selected_name,
            chunk_size=chunk_size,
            max_total_bytes=max_total_bytes,
            max_files=max_files,
            compression=compression,
            compression_min_savings=0.10,
            lease_token=lease_token,
            progress=progress,
            selected_paths=set(selected_paths),
            changed_paths=set(changed_paths),
        )
    else:
        remote = await manager._remote_operation(
            "fs_stat",
            {"path": remote_folder, "checksum": False},
            selected_name,
            lease_token=lease_token,
        )
        if remote.get("kind") != "directory":
            raise ValueError("remote_folder must be an existing directory for pull")
        result = await manager.transfer_download(
            remote_path=remote_folder,
            local_path=str(local),
            name=selected_name,
            overwrite=True,
            sync=True,
            chunk_size=chunk_size,
            max_total_bytes=max_total_bytes,
            max_files=max_files,
            compression=compression,
            compression_min_bytes=1_048_576,
            compression_min_savings=0.10,
            lease_token=lease_token,
            selected_paths=set(selected_paths),
        )
    transfer_seconds = float(result.get("timings", {}).get("data_transfer_seconds", 0))
    manager._record_sync_speed(direction, int(result.get("wire_bytes", 0)), transfer_seconds)
    return {
        "direction": direction,
        "mode": "content_hash_incremental",
        "deletes_destination_extras": False,
        **result,
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
