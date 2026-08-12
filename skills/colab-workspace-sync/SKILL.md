---
name: colab-workspace-sync
description: Safely synchronize purposeful temporary source or artifact directories with a Colab runtime through colab_workspace_sync. Use for source upload, checkpoint retrieval, changed-file synchronization, restart recovery, or any file transfer involving Colab.
---

# Colab workspace sync

Use only `colab_workspace_sync` for ordinary file movement. Do not read, write, or shuttle individual files through MCP.

- MUST create a purpose-specific temporary local sync root containing only the files required by
  this run. MUST NOT naively sync a repository root, home directory, model cache, environment,
  historical checkpoint tree, or mixed source/output directory.
- Prefer downloading public datasets and Hugging Face models directly on Colab. Sync only private,
  modified, or otherwise unavailable inputs.
- Use task-specific remote roots such as `/content/workspaces/<task>/source` and
  `/content/workspaces/<task>/artifacts`. Never mix source and generated artifacts.
- Use `include` as a positive selection when a staging root still contains optional files. Built-in
  secret, VCS, cache, and environment exclusions cannot be overridden.
- Run `dry_run=true` before a large or uncertain transfer. Inspect changed-file reasons, selected
  and excluded counts, bytes, and MiB/s estimate. If acceptable, execute with the returned
  `plan_id` as `expected_plan_id`; re-plan if `sync_plan_changed` is returned.
- Push the temporary source root before execution. Pull only the dedicated artifact root afterward.
- Remove temporary local staging only after inputs are published or outputs are verified locally.
- Reuse the operation lease while its fingerprint remains unchanged.
- Let the verified bundle delta path batch changed source files, skip identical SHA-256 content,
  and atomically replace each published file.
- Treat `insufficient_history` as no speed estimate; never present it as measured throughput.
- Expect compression to help text/source but not already-compressed checkpoints.
- Do not expect deletion: destination-only files are deliberately preserved.

If synchronization is interrupted before submission, retry with the same lease. If a transfer ID or staging path is returned, preserve it for recovery. If the fingerprint changes, abandon that staging record and restore onto a new runtime. Never release compute until returned local hashes are verified.
