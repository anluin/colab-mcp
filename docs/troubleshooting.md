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
On Windows, the installer prefers Codex's launchable per-user CLI over the protected Windows Store
application binary. `colab-mcp doctor` reports the selected path; set `CODEX_CLI_PATH` explicitly
only when the automatic per-user CLI discovery cannot find it.

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

Process tools return a lost-process record when local metadata exists but the remote record does
not. Inspect `last_known_process` and `diagnostic.probable_cause`; common causes are Colab runtime
recycling, runtime OOM/reset, or deleted remote process state. This is diagnostic evidence, not a
guaranteed root-cause determination.

Call `colab_keepalive(refresh=true)` when diagnosing an idle runtime. `healthy` means the upstream
idle-timer ping was accepted; `degraded` records a transient failure that the background task will
retry; `lease_lost` means the assignment was confirmed absent. A healthy ping cannot prevent
Colab-enforced maximum lifetime, quota exhaustion, reclamation, or policy termination.

## Running process has empty output

Current process runners persist each available stdout/stderr pipe write immediately. A process
started by an older server may still use the previous buffered relay and expose no spool data until
64 KiB or process exit, even when the child flushes. Updating the local server cannot replace an
already-running remote runner; let that process finish or explicitly terminate and restart it after
updating. Checkpoint files written directly by the child remain independent of the output spool.

## Commands time out or the kernel disappears

Prefer `colab_process_start` for long work. Poll status and output, and signal it when needed. A
reclaimed runtime cannot be resumed; allocate a fresh one and restore files/checkpoints. Always
download durable results before releasing compute.

The server reuses a verified kernel channel and sends a harmless preflight before each requested
operation. If connection or preflight fails, caller code was not submitted: the server closes the
channel, rechecks the same lease/assignment where applicable, and reconnects once. This is safe even
for a mutation such as process start. If Colab then returns a terminal proxy error such as HTTP 502,
the assignment is not recoverable through a supported client operation. A failure after the caller's
execute call begins has unknown outcome and is never automatically retried; reconcile through
process status or checksums first. Download important results before kernel-intensive work.

## Transfer lease probe fails

`allocation_lease_lost` means the tracked endpoint disappeared during the pre-transfer stability
window; reconcile the session before retrying. `runtime_replaced` means the endpoint no longer
contains the expected runtime incarnation. Neither condition starts a partial transfer.

`operation_lease_stale` or `operation_lease_expired` means another probe superseded the token or
its one-hour lifetime ended. Probe again. `assignment_no_longer_exists` is distinct from a
replacement fingerprint; `assignment_lookup_timed_out` means the Colab control plane did not answer
within five seconds and no critical request was submitted.

For `transfer_failed_staging_preserved`, inspect `request_submission`, `staged_bytes`, and
`staging_path`. Retry the unchanged source/destination with the returned `transfer_id` only while
the same runtime fingerprint is alive. Use `colab_transfer_cleanup` when the partial is no longer
wanted. A resume conflict is not retried automatically because it indicates different local bytes
or an unexpected remote staging mutation.

If `colab_process_export` returns `disposition="held"`, the runtime remains tracked. Correct the
reported status, destination, transfer, or release error and retry. A successfully published local
artifact is retained even when the subsequent explicit release fails.
When `recoverable_export.staging_exists=true`, retry with identical process/remote/local paths and
limits; completed files are checksum-skipped. Use `colab_process_export_cleanup` only when you intend
to discard that recovery state. Cleanup never releases the runtime.

For automatic exports, inspect `auto_export.results` through `colab_process_status` or
`colab_process_list`. `degraded` means at least one matching rule failed and will be retried with
backoff while the MCP server remains alive; restarting the server recreates unfinished watchers.
`held` means the process/runtime state was lost before export and requires human or agent recovery.
An unmatched exit-code rule is recorded as `skipped`, not as an error.

## Compression is not reducing transfer size

`compression="auto"` intentionally sends small or poorly compressible files unchanged. Inspect
each result's `compression`, `content_bytes`, `wire_bytes`, and `wire_ratio`. Lower
`compression_min_bytes` or `compression_min_savings` only when extra CPU and staging space are
worthwhile, or use `compression="gzip"` to force it. A gzip decode, size, or checksum error leaves
the destination unpublished and removes transfer staging files; retry after confirming the runtime
lease and source file are stable.

## Explicit WebRTC connects but is slow or fails

Peer throughput depends on NAT type, region, packet loss, relay choice, and runtime CPU. A successful
ICE/DTLS connection is not evidence that it is faster. Inspect `data_transport`, `wire_bytes`, and
`timings.data_transfer_seconds`. Auto mode does not attempt WebRTC: it directly uses the
authenticated binary-upload/ranged-download path. Use `transport="webrtc"` only while diagnosing
ICE/TURN behavior. More `COLAB_MCP_WEBRTC_LANES` can hurt on two-core runtimes.

`WebRTC endpoint answer timed out` means ICE gathering or endpoint startup did not complete.
`WebRTC transfer timed out` means a connection opened but did not finish within the bounded window.
Explicit WebRTC aborts peer endpoint processes after its bounded timeout, preserves the original
resumable upload stage, and reports failure. It never changes transport behind the caller's back.
Validate JSON in `COLAB_MCP_WEBRTC_ICE_SERVERS`; TURN URLs normally require a username and
credential. Do not paste long-lived relay credentials into logs or issue reports.

## GPU differs from the request

Availability and entitlement are controlled by Colab. Call `colab_inspect` and use the actual GPU,
VRAM, driver, and CUDA values returned. A CPU result contains an empty GPU list.
## Agent-visible recovery guidance

Every MCP tool description includes the recovery action for failures specific to that
operation. The server-level instructions define the common error-code playbook. MCP
clients should expose both descriptions to the agent; if a client hides server
instructions, the tool descriptions remain sufficient for the immediate next action.

## Worker reload fails

Call `colab_connector(action="status")`. If `worker_running=true`, the previous implementation is
still serving tools. `hot_reload_source_changed` means files changed after status; inspect them and
retry with the new available fingerprint. A candidate initialization, tool-list, or health failure
means the candidate was discarded; run the local quality gates and retry. A drain timeout never
cancels submitted work: wait for the call to return or manage a durable process explicitly.

`reload_available=false` means the reloadable worker source matches the active fingerprint. Changes
to `supervisor.py`, dependencies, plugin skills, or the manifest are intentionally outside this
boundary and require reinstall/restart. If `colab_connector` is absent, the client is still attached
to a pre-supervisor release and needs this one-time restart before future worker fixes can hot-reload.
## Lease-bound transfer cannot connect

All operations automatically retry one
`kernel_connection_failed_request_not_submitted` failure after confirming that the same assignment
and lease are still owned. Connection setup has a 21-second local wall-clock deadline for the
configured 20-second upstream timeout; a reused healthy channel normally reports zero connection
seconds. The lease is not invalidated by this safe pre-submission failure. If both attempts fail,
keep the lease token and retry the operation while it remains unexpired; call
`colab_allocation_probe` again only after expiry.
The remote fingerprint guard still runs inside the first successfully submitted request, so
the retry never follows a recycled backend.
