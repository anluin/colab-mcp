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
- New endpoint ownership is persisted before preflight so cleanup failures remain discoverable.
- Random runtime-incarnation markers prevent operations from crossing into a recycled Colab
  backend that happens to reuse an endpoint or kernel connection.
- The server listens on no TCP port and performs no interactive authentication.

## Residual risks

A trusted agent can execute arbitrary programs, exfiltrate data deliberately uploaded to the
runtime, consume account quota, and delete runtime files. Colab and its pinned client dependencies
remain upstream trust dependencies. Local upload/download convenience tools can access paths
available to the MCP host user; do not connect untrusted clients. Rotate/revoke Google credentials
if host compromise is suspected.

Report vulnerabilities privately to the repository owner; do not include credentials, tokens, or
live endpoint URLs in an issue.
