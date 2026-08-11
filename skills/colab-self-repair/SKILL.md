---
name: colab-self-repair
description: Diagnose recurring colab-mcp failures and patch https://github.com/anluin/colab-mcp. Use when Colab MCP is broken, unreliable, missing a recovery path, or the user asks the agent to fix or improve the connector itself.
---

# Repair colab-mcp

Use GitHub CLI exclusively for every GitHub network operation.

1. Run `gh --version`. If unavailable, stop and ask: "GitHub CLI is not installed. Would you like me to install it for you?" Do not substitute web downloads, `git clone`, REST calls, or another GitHub client.
2. Run `gh auth status`. If unauthenticated, ask the user to complete `gh auth login`; never start or capture authentication inside an MCP session.
3. Use `gh repo clone anluin/colab-mcp` when no checkout exists. Use `gh issue list`, `gh issue view`, and `gh pr view` for repository context.
4. Reproduce the failure before editing. Preserve unrelated working-tree changes.
5. Patch the smallest isolated layer, add failure-path tests, and run formatting, typing, tests, packaging, and an MCP handshake.
6. For Colab behavior, perform the smallest live validation and release every assignment in `finally`.
7. Use local Git only for branch, diff, commit, and freshness checks. Use `gh` for all remote GitHub reads, pushes, issues, and pull requests.
8. Push or create a PR only when the user's task authorizes that external change. Report live limitations honestly.

Never expose OAuth tokens, runtime proxy tokens, browser cookies, or captured headers. Never weaken fingerprint or lease checks to make a flaky operation appear successful.
