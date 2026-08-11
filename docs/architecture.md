# Architecture

Colab MCP is a local stdio MCP server. A human authenticates once; trusted agents then use a
non-interactive server to control ephemeral Colab assignments.

`colab-mcp serve` is a stable protocol supervisor; `serve-worker` is its private replaceable MCP
implementation. The supervisor owns the client stdio streams, augments the worker's tool list with
one connector-administration tool, and forwards all other JSON-RPC messages without changing request
IDs. A reload drains forwarded calls, validates a candidate through initialize, tools/list, required
lifecycle tools, and health, then atomically changes the forwarding target. Candidate failure closes
only the candidate. Worker shutdown never releases a runtime. The fixed source root and optional
expected source fingerprint prevent a reload from following a concurrently changed directory. An
explicit source-root binding is restricted to a local Git checkout whose origin is exactly the
project's GitHub repository and becomes active only after candidate validation.

## Boundaries

1. `cli.py` owns human authentication, diagnostics, and MCP-client registration.
2. `server.py` defines concise public MCP schemas and delegates to one manager.
3. `manager.py` owns local session state, assignment lifecycle, keep-alive, kernel connections,
   recovery, and convenience adapters.
4. `remote.py` contains the workload-agnostic program injected into the runtime. Payloads are
   base64 JSON, commands never use a shell, and filesystem operations stay under `/content`.
5. `google-colab-cli` and `jupyter-kernel-client` are pinned integration adapters. They are the only
   layer coupled to Colab's published client implementation and should be upgraded deliberately
   with live validation.

`colab_adapter.py` isolates one pinned-client timeout gap: reconnect construction otherwise performs
an unconfigurable 30-second HTTP model refresh before the public start timeout applies. The bounded
manager subclass forwards the critical-operation timeout to that existing-kernel lookup. No other
module imports the upstream HTTP manager internals.

Session ownership is persisted in the configured state directory. Process ownership is persisted
inside its assigned runtime. Releasing the assignment is the lifecycle boundary: `/content`,
processes, and runtime metadata are ephemeral. Notebook execution is an adapter over Python/kernel
execution, not a separate compute model.

Allocation also persists a random incarnation fingerprint in local session state and in a marker
under the assigned backend's `/content`. The common remote-operation envelope validates the marker
before process, filesystem, transfer, or introspection code runs. A missing or different marker is
reported as `runtime_replaced`; it is never interpreted as an empty filesystem or missing process.
Process observations are journaled in owner-only local state. A fingerprint mismatch is persisted
on the session, making subsequent operations fail before kernel access. Process APIs can therefore
return the last observation with `status="lost"` and a probable-cause diagnostic even when the
replacement backend no longer contains `.colab-mcp/processes`.

The server does not expose a network listener. MCP request boundaries do not bound remote process
lifetime; detached process state and output are polled in later requests. A runner continuously
drains both output pipes, retains each stream only up to its configured cap, and records truncation
and total byte counts without stopping the workload.
Server shutdown does not silently terminate remotely running work. Local keep-alive tasks and
cached kernel channels end with the event loop, while persisted assignment/process metadata enables
a new server instance to reconnect. Closing a channel never shuts down the remote kernel. Compute
release remains explicit and auditable.

The server lifespan restores background keep-alive tasks for assignments that still exist upstream.
Each task uses the pinned CLI client's Tunnel Frontend ping, records observable health in session
state, retries transient failures, and stops when repeated failures are confirmed as a lost lease.
Graceful shutdown cancels local heartbeat tasks, closes local kernel channels, and never releases
compute implicitly. A verified channel is reused across MCP requests for the same session and each
request first sends a harmless `/content` preflight. This avoids repeated model lookup and websocket
startup while detecting a dead cached connection before the caller's operation is submitted.
Operations for one runtime are serialized because the Jupyter channel is sequential; different
runtimes remain independent. A confirmed connection/preflight failure reconnects once for every
operation, including mutations, while preserving the owned kernel identity, lease, and fingerprint.
Once caller code may have been sent, ambiguous outcomes are never automatically retried.

Worker reload emits the standard tool-list-changed notification after a successful swap. Existing
tool implementations change in the same client task; visibility of newly added schemas still
depends on the client's notification support and model-turn boundary. The stable supervisor,
dependencies, plugin skills, and manifest remain outside the worker reload boundary.

Transfers begin with repeated upstream assignment observations and a remote incarnation check. The
probe relies on the session's existing background heartbeat instead of synchronously waiting on the
Tunnel Frontend heartbeat for every observation. Individual filesystem calls retain their own
incarnation guard, so stabilization does not weaken fail-fast runtime replacement detection during
a transfer.
The probe also publishes a random operation lease into that incarnation and persists only the
current token locally. Critical tools accept the opaque token. The generated remote request checks
the incarnation and lease before entering its operation branch; this closes the probe-to-mutation
gap without pretending that the Colab assignment itself can be locked. An assignment control-plane
check is bounded to five seconds before critical work.

Upload chunks use offset-checked, fsynced staging writes. The transfer ID deterministically names
each staging file, and retry verifies the complete staged prefix checksum before seeking the local
wire representation. A duplicate chunk is accepted only when its bytes match. Per-transfer
heartbeats run independently of MCP reasoning time, and every chunk revalidates the remote lease.
Failures preserve staging and return recovery metadata; publication or explicit cleanup removes it.
Wire compression is a transport concern inside those transfer primitives. Each eligible file is
gzip-compressed into a unique staging object, transferred in bounded chunks, and decompressed into
a second staging object before atomic publication. Both wire bytes and original content are
size/checksum verified. `auto` chooses compression from measured savings, so file semantics and
directory layout never change and incompressible data does not pay network expansion.
Only folder synchronization is public. Low-level filesystem and single-file transfer primitives
remain internal implementation details. Folder sync is content-hash incremental and intentionally
does not delete destination-only files; this provides rsync-like update behavior without requiring
SSH, rsync, or a platform-specific executable.

Completed-process export is a local publication transaction. Data is checksummed into a sibling
staging path, renamed into visibility only after the complete transfer succeeds, and optionally
followed by explicit runtime release. Every failure path preserves the tracked assignment for retry.
The stage name is deterministic from process ownership and both paths. Completed files survive
interruption and are checksum-skipped on retry, including after server restart. Publication removes
the stage; failure journals a recoverable export record, while explicit cleanup is a separate tool.

Optional process auto-export rules are journaled with process ownership. In-memory watchers only
schedule work; their source of truth is the owner-only journal, so server startup can recreate them.
They poll status outside MCP request lifetimes, select rules by the recorded exit code, call the same
atomic export primitive, persist each outcome, and retry failures with bounded backoff. They never
release an assignment or infer artifacts from workload type.

## Upstream constraints

Personal Colab has no supported VM suspend/snapshot operation, guaranteed accelerator inventory,
or compute-unit balance API. The server does not scrape the browser-only balance endpoint; it links
to Colab's account-management page instead. Reconnect is possible while an assignment endpoint and
kernel remain live; after Colab reclaims it, only persisted local metadata remains. The server
reports these conditions rather than emulating unsupported behavior.
