# Colab MCP

Give Codex, Grok, Claude, and other MCP clients access to general-purpose Google Colab CPU/GPU runtimes from Windows, macOS, or Linux. The server uses standard MCP over stdio and does not require WSL, SSH, an open Colab browser tab, or Colab Enterprise.

## Quick setup for Codex

Prerequisites: [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/).

### macOS or Linux

```bash
git clone https://github.com/anluin/colab-mcp.git
cd colab-mcp
sh scripts/install.sh
```

### Windows PowerShell

```powershell
git clone https://github.com/anluin/colab-mcp.git
cd colab-mcp
.\scripts\install.ps1
```

The installer runs `colab-mcp setup codex`: it creates an isolated environment from the lockfile, completes the one-time Google OAuth flow, and registers the stdio server through Codex's supported `codex mcp add` command.

Restart Codex after setup, open `/mcp`, and call `colab_health`. The ChatGPT desktop app, Codex CLI, and IDE extension share MCP configuration on the same Codex host.

Then give the agent this exact smoke task:

> Use Google Colab to start a T4 session named `gpu-probe`, call `colab_inspect`, run
> `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` with
> `colab_run_command`, and call `colab_stop` in a cleanup step even if the probe fails.

Expected evidence is a non-empty `gpu` list, a zero command exit code, and a final stopped session.

## Quick setup for Grok

Prerequisites: [Git](https://git-scm.com/), [uv](https://docs.astral.sh/uv/getting-started/installation/), and the [Grok CLI](https://grok.x.ai/) (`grok` on `PATH`).

```bash
git clone https://github.com/anluin/colab-mcp.git
cd colab-mcp
uv run colab-mcp setup grok
```

This authenticates with Google Colab once (if needed) and registers the stdio server through
`grok mcp add` into `~/.grok/config.toml`. Grok uses an isolated `uv` environment so it does not
share the repository environment with a long-running Codex or other client session—important on
Windows, where a locked console-script entry point can prevent a second `uv run` from starting.

Restart Grok after setup, open `/mcps`, and call `colab_health` (tools appear as
`colab__colab_health` after namespacing). Confirm connectivity with `grok mcp doctor colab`.

The same CLI can configure multiple clients in one human-run step:

```bash
uv run colab-mcp setup codex grok claude-desktop
```

Or run the steps separately:

```bash
uv run colab-mcp auth
uv run colab-mcp install codex
uv run colab-mcp install grok
uv run colab-mcp install claude
uv run colab-mcp install claude-desktop
uv run colab-mcp doctor
uv run colab-mcp doctor --live
```

Claude Desktop and Grok are registered with an isolated `uv` environment, so they can update or
restart while Codex continues using the repository environment for a long-running session. On
Windows, Claude Desktop setup detects both the conventional `%APPDATA%` configuration and Microsoft
Store's virtualized Claude configuration, preferring the active packaged configuration when present.

Only `auth` and `setup` may prompt for Google authorization. `serve` is strictly non-interactive: if credentials expire and cannot refresh, it instructs the human to rerun `colab-mcp auth` and never launches OAuth inside an agent session.
`doctor --live` performs a read-only assignments API check and reports only the count, never
endpoint URLs or tokens.

An existing registration is left untouched. Add `--force` to an `install` or `setup` command to replace it deliberately.

## Other MCP clients

Use `mcp.example.json` as a template, or print ready-to-paste JSON with `uv run colab-mcp install json`. Run `uv run colab-mcp auth` once, then configure this server command:

```bash
uv --directory /absolute/path/to/colab-mcp run --locked colab-mcp serve
```

## Capabilities

- Create local Colab-ready notebooks.
- Allocate CPU runtimes or T4, L4, G4, H100, and A100 GPUs, subject to account entitlement and availability.
- Execute Python or complete notebooks and preserve Jupyter outputs.
- Run arbitrary programs from argument arrays with a working directory, environment overrides,
  timeouts, exit codes, separate stdout/stderr, and bounded output.
- Start long-running processes without blocking an MCP request; list, inspect, incrementally read,
  interrupt, terminate, or kill them in later requests.
- List, stat, checksum, chunk-read, atomic-write, append, create, move, and explicitly remove
  files and directories within a runtime-owned `/content` boundary.
- Inspect OS, Python, CPU, RAM, disk, GPU/VRAM, CUDA/driver, requested tools, and a bounded process
  snapshot without assuming any workload or framework.
- Transfer files or directory trees in bounded chunks with SHA-256 verification, incremental sync
  skips, staged partials, atomic publication, cleanup, and explicit overwrite limits.
- Upload datasets and download artifacts.
- Pause by checkpointing a notebook and releasing its GPU.
- Resume on a fresh runtime with the same accelerator preference and optionally rerun the notebook.
- Keep active runtimes alive while the MCP server runs.
- Release runtimes explicitly after experiments or errors.

Tools: `colab_health`, `colab_create_notebook`, `colab_start`, `colab_sessions`, `colab_keepalive`,
`colab_run_command`, `colab_process_start`, `colab_process_status`, `colab_process_list`,
`colab_process_output`, `colab_process_signal`, `colab_process_export`, `colab_execute`,
`colab_execute_notebook`, `colab_allocation_probe`,
`colab_fs_list`, `colab_fs_stat`, `colab_fs_read`, `colab_fs_write`, `colab_fs_mkdir`,
`colab_fs_move`, `colab_fs_remove`,
`colab_transfer_upload`, `colab_transfer_download`,
`colab_upload`, `colab_download`, `colab_pause_notebook`, `colab_resume_notebook`,
`colab_paused_notebooks`, `colab_reconcile`, `colab_stop`, and `colab_compute_units`.
Use `colab_inspect` after allocation to discover the actual runtime rather than assuming that a
requested accelerator, executable, or CUDA version is present.

### Idle keep-alive

Each tracked assignment gets the upstream CLI's authenticated Tunnel Frontend keep-alive every 60
seconds. The task continues across transient errors, persists its last success/error and consecutive
failure count, and is restored from session state when the MCP server restarts. Use
`colab_keepalive(refresh=true)` to send an immediate ping and inspect heartbeat health before an
agent begins a long reasoning-only interval.

This refreshes Colab's idle timer; it does not execute synthetic workloads, bypass policy, extend a
maximum VM lifetime, reserve an accelerator, or guarantee persistence. Google states that personal
Colab idle timeouts and maximum lifetimes vary and runtimes may still be terminated. See the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html) and the upstream
[CLI session-management design](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md#5-keep-alive-protocol).

### General command example

```json
{
  "argv": ["python", "-c", "import platform; print(platform.platform())"],
  "session": "compute",
  "cwd": "/content",
  "timeout": 60
}
```

No shell parses `argv`. Remote working directories must remain under `/content`. Each stdout and
stderr result is limited to 100 KB by default (1 MB maximum). For longer work, use
`colab_process_start`, then poll status and consume output using the returned `next_offset`.
Process records belong to one runtime and disappear when that ephemeral runtime is released.
Each session records a random runtime-incarnation fingerprint both locally and under `/content`.
Every process, filesystem, transfer, and introspection call verifies it before accessing remote
state. If Colab recycles an endpoint onto a fresh backend, the call fails explicitly with
`runtime_replaced` instead of reporting `Unknown process_id` or an apparently empty filesystem;
stop the stale session record and start a new runtime.
Managed-process metadata is also journaled locally. If its remote record vanishes, process tools
return `status="lost"`, the last known argv/cwd/PID/status, and a probable-cause diagnostic such as
runtime recycling or OOM. After the first fingerprint mismatch, later file and process calls fail
locally without reconnecting to the replacement backend.
Every command is runtime-owned and receives a `process_id`. The timeout is only how long the MCP
call waits: if it expires, `process_continues=true` and the command remains alive for later
status/output/signal calls. Termination is always explicit. Python/Jupyter output is bounded to
100 KB by default (1 MB maximum) and ends with an explicit truncation marker.
Detached processes retain at most 10 MB per output stream by default (configurable up to 1 GB with
`output_limit`). They continue draining excess output so the child cannot deadlock; output reads
return `truncated=true` when retained output was capped. While a process is running,
`stored_bytes` and `total_bytes` reflect the durable spool currently available to readers;
`total_bytes_final=false` marks that live lower bound. After exit, `total_bytes` is the
complete byte count, including discarded bytes, and `total_bytes_final=true`.
The handoff deadline is best-effort because each Colab kernel status/output round-trip has latency;
it is not a hard real-time deadline.

Filesystem reads and writes use base64 so both text and binary content round-trip without encoding
ambiguity. Calls are limited to 1 MB; use offsets and append mode for larger files. Non-append writes
replace atomically. Paths are resolved on the runtime and may not escape `/content`. Moving onto an
existing path requires `overwrite=true`; removing a non-empty directory requires `recursive=true`.
Use `checksum=true` with `colab_fs_stat` to verify a completed transfer.

For host/runtime transfer, prefer `colab_transfer_upload` and `colab_transfer_download`. A directory
path maps its contents beneath the destination directory; a file path maps directly to the given
destination file. Defaults cap a call at 100 MB, 10,000 files, and 512 KB per chunk. Existing files
with the same SHA-256 are skipped when `sync=true`; differing destinations require
`overwrite=true`. Uploads stage under a unique runtime path and downloads stage beside the local
destination. A failed call removes its partial file before returning an error.
The shorter `colab_upload` and `colab_download` names are compatibility aliases for the same bounded,
verified implementation; they do not bypass its limits.

Every transfer first performs a bounded allocation lease probe: it observes the owned endpoint
twice, refreshes its lease, and verifies the runtime-incarnation marker before reading or mutating
remote files. A missing assignment fails as `allocation_lease_lost`; a recycled backend continues
to fail as `runtime_replaced`.

After an owned process exits, `colab_process_export` downloads one file or directory into a hidden
sibling staging path and publishes it locally with one filesystem rename. Any status, lease,
transfer, checksum, publication, or release failure returns `disposition="held"` and leaves the
runtime tracked. `release_on_success=false` is the safe default; setting it to true releases compute
only after publication succeeds. Atomic overwrite of an existing directory is intentionally
unsupported across platforms—export to a new destination instead.

### Crash recovery and orphan cleanup

Allocated endpoints are persisted before runtime preflight, so even a double failure during startup
remains recoverable. `colab_reconcile` compares persisted sessions with the account's live Colab
assignments. Its default is read-only: it reports stale local records and live orphan endpoints.
Pass `forget_stale=true` to remove records whose runtime is already gone. Pass
`release_orphans=true` only when you intend to release every live assignment not owned by this
colab-mcp state directory. Failures are returned per endpoint for safe retry. `colab_stop` is
idempotent when a tracked runtime has already disappeared.

## Recommended agent lifecycle

1. Check `colab_health`.
2. Start with a T4 unless another accelerator is required.
3. Create or execute a notebook.
4. Download important artifacts.
5. Pause to checkpoint and release compute, or stop when finished.
6. Always stop a runtime after an error if it was not already released.

## Pause and resume semantics

Colab does not expose a supported suspended-VM or runtime-snapshot operation. Pause records the local notebook checkpoint and accelerator preference, then releases the runtime. Resume allocates a new runtime and can rerun the notebook.

RAM variables, ad-hoc package installs, and files left only in `/content` do not survive. Put installation commands in the notebook and download checkpoints before pausing.

## Compute units

Google does not publish a supported API for reading the compute-unit balance of a personal Colab account. `colab_compute_units` returns this limitation and Google's official account-management URL. The server does not scrape undocumented browser endpoints.

Free, Pro, Pro+, and Pay As You Go personal accounts are supported. Actual GPU models, runtime length, and compute usage remain controlled by Google Colab.

## Security

This MCP executes arbitrary Python and can consume the authenticated account's quota. Keep it as a local stdio server and connect only trusted clients. OAuth and runtime proxy tokens are never returned through MCP tools.

Session state defaults to `~/.config/colab-mcp`. Override it with `COLAB_MCP_STATE_DIR`. Set `COLAB_MCP_AUTH=adc` only if you deliberately configured Google Application Default Credentials with the required Colab scopes.

Operational logs are single-line JSON on stderr so MCP stdout framing remains clean. Configure the
threshold with `COLAB_MCP_LOG_LEVEL`. Graceful server shutdown cancels only local keep-alive tasks;
it deliberately preserves owned assignments/processes for restart recovery. Use `colab_stop` or
`colab_reconcile` for explicit quota release.

## Development and validation

```bash
uv sync --locked --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv build
uv run twine check dist/*
```

CI runs on Ubuntu, macOS, and Windows with Python 3.12. The live integration has also been verified against a real Tesla T4: allocation, CUDA execution, notebook execution, pause/release, fresh-runtime resume, rerun, and cleanup with zero assignments remaining.

The Google Colab integration version is pinned to the live-tested release. This project imports its portable client components; it does not invoke the platform-limited CLI executable.

Further documentation: [architecture](docs/architecture.md), [security model](docs/security.md),
[troubleshooting](docs/troubleshooting.md), [contributing](CONTRIBUTING.md), and
[release procedure](docs/releasing.md). Version history and readiness evidence live in Git tags,
commits, and GitHub release notes rather than duplicated version-specific repository files.

## Upstream projects

- https://github.com/googlecolab/google-colab-cli
- https://github.com/googlecolab/jupyter-kernel-client
- https://github.com/modelcontextprotocol/python-sdk

Codex registration follows OpenAI's documented MCP flow: https://learn.chatgpt.com/docs/extend/mcp
