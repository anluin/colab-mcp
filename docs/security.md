# Security model

## Trust assumptions

This server deliberately provides arbitrary remote code execution and quota-consuming runtime
allocation. Install it only for trusted local agents and keep the default stdio transport. The
authenticated Google account and the host user are inside the trust boundary; remote Colab
runtimes are ephemeral execution environments.

## Controls

- OAuth is human-only. MCP tools never start a browser, device flow, or terminal prompt.
- OAuth credentials, runtime proxy tokens, and environment values are never returned by tools or
  stored in process metadata. Session-token fields are excluded from public responses.
- On POSIX hosts, the local state directory is mode 0700 and token-bearing session/checkpoint files
  are mode 0600. On Windows, they inherit the current user's profile ACL.
- Last-known process metadata is stored in the same protected local state directory; environment
  values remain excluded, while argv is retained because managed process arguments are inspectable.
- Detached-process environment values are consumed through a mode-0600 launch file that is removed
  before process creation returns; they are not placed in the runner command line. Put secrets in
  environment overrides, never in `argv`, because process arguments are intentionally inspectable.
- Remote commands accept argument arrays and use `shell=False`.
- Command/process output, stored detached-process logs, directory listings, process snapshots, and
  file chunks are bounded. A capped log is explicitly reported as truncated.
- Runtime paths resolve under `/content`; `/content` itself cannot be recursively removed.
- Overwrite, append, recursive deletion, stale-record removal, and orphan release are explicit.
- Transfers verify the owned allocation lease before remote access. Process export holds the
  runtime on every failure, and release after export requires `release_on_success=true`.
- Operation leases are opaque, expire after one hour, are bound to one persisted session and remote
  incarnation, and are checked inside the same remote request before critical mutation. Only the
  newest probe token is accepted. Tokens are excluded from session-list/start responses.
- Resumable upload chunks require exact offsets and matching bytes. Failed staging is never
  published and is removable only through constrained staging cleanup or explicit filesystem tools.
- Compressed transfers enforce declared original sizes, total transfer limits, exact decompressed
  lengths, and SHA-256 checks before publication; malformed or expanding gzip data is discarded.
- WebRTC transfers authenticate a per-transfer secret inside DTLS, validate the runtime fingerprint
  and operation lease at endpoint start and completion, restrict uploads to hidden transfer staging
  names, and verify every range plus the assembled wire object before publication. SDP contains no
  OAuth/runtime proxy token. TURN credentials never appear in process arguments or tool results.
- No unauthenticated TURN service is embedded. Operators supply bounded STUN/TURN URLs and
  short-lived relay credentials through `COLAB_MCP_WEBRTC_ICE_SERVERS`; relay operators still see
  network metadata, while DTLS protects file content.
- Auto-export destinations are explicit host paths persisted in owner-only process state. Rules are
  capped, destinations must be unique, publication stays atomic, and automatic release is forbidden.
- Process-export stages are deterministically scoped to an owned process and explicit paths. Failed
  stages are retained for recovery and can only be discarded through the ownership-checked cleanup
  tool or normal host filesystem access.
- New endpoint ownership is persisted before preflight so cleanup failures remain discoverable.
- Random runtime-incarnation markers prevent operations from crossing into a recycled Colab
  backend that happens to reuse an endpoint or kernel connection.
- The server listens on no TCP port and performs no interactive authentication.
- Hot reload accepts no command from MCP input. The supervisor uses a source root fixed at startup,
  optionally verifies an expected SHA-256 over worker source, drains in-flight calls, and swaps only
  after candidate initialization and a safe health check. An explicit root binding must be an
  absolute local Git checkout whose origin is exactly `anluin/colab-mcp`; it becomes active only
  after validation. Reload never downloads code or changes credentials, so source updates remain
  subject to the host's normal repository policy and the repair skill's `gh`-only rule.

## Residual risks

A trusted agent can execute arbitrary programs, exfiltrate data deliberately uploaded to the
runtime, consume account quota, and delete runtime files. Colab and its pinned client dependencies
remain upstream trust dependencies. Local upload/download convenience tools can access paths
available to the MCP host user; do not connect untrusted clients. Rotate/revoke Google credentials
if host compromise is suspected.

Report vulnerabilities privately to the repository owner; do not include credentials, tokens, or
live endpoint URLs in an issue.
