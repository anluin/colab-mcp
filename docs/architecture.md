# Architecture

Colab MCP is a local stdio MCP server. A human authenticates once; trusted agents then use a
non-interactive server to control ephemeral Colab assignments.

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
Server shutdown does not silently terminate remotely running work. Local keep-alive tasks end with
the event loop, while persisted assignment/process metadata enables a new server instance to
reconnect. Compute release remains explicit and auditable.

The server lifespan restores background keep-alive tasks for assignments that still exist upstream.
Each task uses the pinned CLI client's Tunnel Frontend ping, records observable health in session
state, retries transient failures, and stops when repeated failures are confirmed as a lost lease.
Graceful shutdown cancels only local heartbeat tasks and never releases compute implicitly.
If a kernel channel times out, idempotent status/read/list/introspection operations clear the stale
kernel identity and reconnect once. Mutations and arbitrary code are never retried automatically
because their outcome may be unknown and duplicate execution would be unsafe.
Connection setup is tracked separately from execution: if setup fails before the requested code is
sent, the client clears the stale kernel identity and retries once for any operation. Once code may
have been sent, only idempotent operations are eligible for retry.

Transfers begin with repeated upstream assignment observations, keep-alive refreshes, and a remote
incarnation check. Individual filesystem calls retain their own incarnation guard, so stabilization
does not weaken fail-fast runtime replacement detection during a transfer.

Completed-process export is a local publication transaction. Data is checksummed into a sibling
staging path, renamed into visibility only after the complete transfer succeeds, and optionally
followed by explicit runtime release. Every failure path preserves the tracked assignment for retry.

## Upstream constraints

Personal Colab has no supported VM suspend/snapshot operation, guaranteed accelerator inventory,
or compute-unit balance API. Reconnect is possible while an assignment endpoint and kernel remain
live; after Colab reclaims it, only persisted local metadata remains. The server reports these
conditions rather than emulating unsupported behavior.
