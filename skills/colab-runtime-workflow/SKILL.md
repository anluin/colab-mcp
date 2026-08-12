---
name: colab-runtime-workflow
description: Operate Google Colab through colab-mcp for short commands, durable jobs, GPU/CPU allocation, recovery, artifact publication, and cleanup. Use whenever an agent runs work on a Colab runtime or must choose between synchronous and durable execution.
---

# Colab runtime workflow

Treat Colab as ephemeral compute. Keep source and final output in local durable folders.
Stage each run in dedicated temporary source and artifact directories so synchronization never
walks caches, VCS metadata, historical checkpoints, or unrelated outputs.

1. Call `colab_health`, then `colab_sessions`; reuse only an owned healthy session.
2. Otherwise call `colab_start`, then `colab_inspect` and `colab_allocation_probe`.
3. Pass the probe lease immediately to critical work. Never follow a changed fingerprint.
4. Create a temporary local source snapshot containing every file the workload needs and nothing
   else. Prefer tracked source plus required configs/data; exclude `.git`, caches, environments,
   prior outputs, and checkpoints unless the run consumes them.
5. Push that snapshot to a task-specific root such as `/content/workspaces/<task>/source`.
   Reserve `/content/workspaces/<task>/artifacts` for outputs only.
6. For work expected to finish quickly, use `colab_execute` or `colab_run_command`.
7. For uncertain or long work, use `colab_process_start` with `export_on_exit` targeting the
   dedicated artifact directory. Poll status and output with increasing intervals. A timeout does
   not kill the process.
8. Pull only the artifact directory; verify returned hashes and local publication.
9. Call `colab_stop` only after durable local publication. Always attempt remote and temporary
   local cleanup in `finally`.

Use `colab_keepalive` as a health signal, not a lifetime guarantee. Durable work must checkpoint into the synchronized artifact folder. After an MCP restart, recover with `colab_sessions` and `colab_process_list`.

On `runtime_replaced` or `assignment_no_longer_exists`, do not reuse process IDs, leases, or `/content`. Allocate a new runtime and restore the workspace. Retry automatically only when the error explicitly says the request was not submitted.
