# Troubleshooting

Run `uv run colab-mcp doctor` first. It reports versions, authentication presence, client
executables, and the exact server command without allocating compute.

## Authentication missing or expired

Run `uv run colab-mcp auth` in a human terminal, then restart the MCP client. Agent calls never
launch OAuth. If refresh continues to fail, revoke the old Colab authorization in the Google
account and authenticate again.

## Client does not show tools

Run `uv run colab-mcp install codex --force` (or `grok`, `claude`, `claude-desktop`) and restart that
client. Confirm `uv` and the repository path still exist. `uv run --locked colab-mcp serve` should
remain silent while it waits for MCP messages; ordinary stdout logging would corrupt stdio framing.

For Grok, also run `grok mcp doctor colab`. On Windows, if doctor reports a failed handshake with a
locked `colab-mcp.exe` under the project `.venv`, reinstall with
`uv run colab-mcp install grok --force` so Grok keeps its isolated environment and does not compete
with another client holding the project console-script entry point.

## Runtime is stale or quota appears occupied

Call `colab_reconcile` without flags. Stale sessions are local records whose assignment vanished;
orphan endpoints are live assignments not owned by this state directory. Use cleanup flags only
after checking that an orphan is not a browser or another client session. Cleanup errors are safe
to retry.

If an operation reports `runtime_replaced`, Colab has recycled the endpoint onto a different
ephemeral backend (or the session predates incarnation tracking). Do not retry old process IDs or
interpret an empty `/content` as the old runtime. Stop the stale session, start a new one, and
restore files from durable local storage.

## Commands time out or the kernel disappears

Prefer `colab_process_start` for long work. Poll status and output, and signal it when needed. A
reclaimed runtime cannot be resumed; allocate a fresh one and restore files/checkpoints. Always
download durable results before releasing compute.

Idempotent read/status/introspection calls reconnect once after a kernel-channel timeout. If Colab
then returns a terminal proxy error such as HTTP 502 while creating the replacement kernel, the
assignment is not recoverable through a supported client operation. Do not retry mutations whose
outcome is unknown. Reconcile/stop the assignment, allocate a fresh runtime, and restore previously
downloaded data. Download important results before notebook or other kernel-intensive work when
possible.

## GPU differs from the request

Availability and entitlement are controlled by Colab. Call `colab_inspect` and use the actual GPU,
VRAM, driver, and CUDA values returned. A CPU result contains an empty GPU list.
