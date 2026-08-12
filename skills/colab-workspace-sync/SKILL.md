---
name: colab-workspace-sync
description: Synchronize whole local project or artifact directories with a Colab runtime through colab_workspace_sync. Use for source upload, checkpoint retrieval, changed-file synchronization, restart recovery, or any file transfer involving Colab.
---

# Colab workspace sync

Use only `colab_workspace_sync` for ordinary file movement. Do not read, write, or shuttle individual files through MCP.

- Create a temporary local sync root for each run. Populate it with the complete workload snapshot,
  not the whole development checkout: include required tracked source, configuration, and inputs;
  omit `.git`, caches, environments, old artifacts, and unrelated checkpoints.
- Use task-specific remote roots such as `/content/workspaces/<task>/source` and
  `/content/workspaces/<task>/artifacts`. Never mix source and generated artifacts.
- Push the temporary source root before execution. Pull only the dedicated artifact root afterward.
- Remove temporary local staging only after inputs are published or outputs are verified locally.
- Reuse the operation lease while its fingerprint remains unchanged.
- Let the verified bundle delta path batch changed source files, skip identical SHA-256 content,
  and atomically replace each published file.
- Expect compression to help text/source but not already-compressed checkpoints.
- Do not expect deletion: destination-only files are deliberately preserved.

If synchronization is interrupted before submission, retry with the same lease. If a transfer ID or staging path is returned, preserve it for recovery. If the fingerprint changes, abandon that staging record and restore onto a new runtime. Never release compute until returned local hashes are verified.
