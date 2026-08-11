---
name: colab-workspace-sync
description: Synchronize whole local project or artifact directories with a Colab runtime through colab_workspace_sync. Use for source upload, checkpoint retrieval, changed-file synchronization, restart recovery, or any file transfer involving Colab.
---

# Colab workspace sync

Use only `colab_workspace_sync` for ordinary file movement. Do not read, write, or shuttle individual files through MCP.

- Use a dedicated remote root such as `/content/workspace`.
- Push the local project directory before execution.
- Pull a dedicated remote artifacts directory before release.
- Reuse the operation lease while its fingerprint remains unchanged.
- Let SHA-256 synchronization skip unchanged files and atomically replace changed files.
- Expect compression to help text/source but not already-compressed checkpoints.
- Do not expect deletion: destination-only files are deliberately preserved.

If synchronization is interrupted before submission, retry with the same lease. If a transfer ID or staging path is returned, preserve it for recovery. If the fingerprint changes, abandon that staging record and restore onto a new runtime. Never release compute until returned local hashes are verified.
