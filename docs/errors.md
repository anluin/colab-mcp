# Error contract

Errors use a stable code and structured details whenever the server can observe the failing phase.

| Code | Meaning | Submission state | Safe action |
|---|---|---|---|
| `assignment_no_longer_exists` | The tracked endpoint is absent from the account assignment list. | Not submitted | Start a new runtime. |
| `assignment_lookup_timed_out` | Colab's assignment control plane did not answer in five seconds. | Not submitted | Retry the lookup; do not assume the assignment vanished. |
| `runtime_replaced` | The endpoint answered but its incarnation marker is missing or different. | Guard rejected before operation mutation | Never reuse old process IDs or staging data on that backend. |
| `operation_lease_stale` | The token is not the latest lease for this session/incarnation. | Not submitted locally, or rejected by remote guard | Probe again. |
| `operation_lease_expired` | The one-hour token lifetime elapsed. | Not submitted locally, or rejected by remote guard | Probe again. |
| `kernel_connection_failed_request_not_submitted` | Kernel construction, channel startup, or preflight failed. | Confirmed not submitted | Retry only while the assignment and fingerprint are unchanged. |
| `operation_timed_out_submission_outcome_unknown` | The synchronous kernel adapter timed out after its execute call began. | Unknown | Inspect idempotent state; never blindly retry a mutation. |
| `request_submission_outcome_unknown_response_lost` | The kernel channel failed after its execute call began. | Unknown | Reconcile through status/checksum calls before retry. |
| `transfer_failed_staging_preserved` | Upload failed and its deterministic staging file was retained. | See `request_submission` in details | Resume with the same transfer ID and incarnation, or explicitly clean up. |
| `process_state_lost` | The runtime still answered but managed process metadata was absent. | N/A | Use the last-known record; inspect memory/system evidence before diagnosing OOM. |

Colab does not publish a supported event that distinguishes idle reclamation, quota exhaustion,
accelerator preemption, and account policy termination. The server therefore reports observed
assignment/incarnation facts and does not invent one of those causes. A local MCP stdio/host crash
occurs outside a tool invocation and is classified by the MCP client, not by this server; persisted
session/process/export records are the recovery boundary.

The pinned kernel adapter exposes request submission, remote execution, and I/O collection as one
synchronous call. Successful detailed execution reports that combined interval plus a remote guard
duration. On failure after the call begins, submission outcome is explicitly `unknown`; the server
does not falsely label it submitted or unsent.
