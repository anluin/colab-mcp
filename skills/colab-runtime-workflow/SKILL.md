---
name: colab-runtime-workflow
description: Operate Google Colab through colab-mcp for short commands, durable jobs, GPU/CPU allocation, recovery, artifact publication, and cleanup. Use whenever an agent runs work on a Colab runtime or must choose between synchronous and durable execution.
---

# Colab runtime workflow

Treat Colab as ephemeral compute. Keep source and final output in local durable folders.

1. Call `colab_health`, then `colab_sessions`; reuse only an owned healthy session.
2. Otherwise call `colab_start`, then `colab_inspect` and `colab_allocation_probe`.
3. Pass the probe lease immediately to critical work. Never follow a changed fingerprint.
4. Push the complete project folder with `colab_workspace_sync(direction="push")`.
5. For work expected to finish quickly, use `colab_execute` or `colab_run_command`.
6. For uncertain or long work, use `colab_process_start` with `export_on_exit`. Poll status and output with increasing intervals. A timeout does not kill the process.
7. Pull the artifact folder with `colab_workspace_sync(direction="pull")`; verify returned hashes.
8. Call `colab_stop` only after durable local publication. Always attempt cleanup in `finally`.

Use `colab_keepalive` as a health signal, not a lifetime guarantee. Durable work must checkpoint into the synchronized artifact folder. After an MCP restart, recover with `colab_sessions` and `colab_process_list`.

On `runtime_replaced` or `assignment_no_longer_exists`, do not reuse process IDs, leases, or `/content`. Allocate a new runtime and restore the workspace. Retry automatically only when the error explicitly says the request was not submitted.
