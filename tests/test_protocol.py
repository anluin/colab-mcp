import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_mcp_initialize_list_and_safe_calls(tmp_path):
    async def smoke():
        environment = dict(os.environ)
        environment["COLAB_MCP_STATE_DIR"] = str(tmp_path / "state")
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.cli", "serve"],
            env=environment,
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as client:
                initialized = await client.initialize()
                tools = await client.list_tools()
                names = {tool.name for tool in tools.tools}
                health = await client.call_tool("colab_health", {})
                compute_units = await client.call_tool("colab_compute_units", {})
                assert initialized.serverInfo.name == "Google Colab Runtime"
                assert names == {
                    "colab_connector",
                    "colab_health",
                    "colab_sessions",
                    "colab_keepalive",
                    "colab_reconcile",
                    "colab_inspect",
                    "colab_create_notebook",
                    "colab_start",
                    "colab_execute",
                    "colab_run_command",
                    "colab_process_start",
                    "colab_process_status",
                    "colab_process_list",
                    "colab_process_output",
                    "colab_process_signal",
                    "colab_process_export",
                    "colab_process_export_cleanup",
                    "colab_transfer_cleanup",
                    "colab_allocation_probe",
                    "colab_execute_notebook",
                    "colab_stop",
                    "colab_pause_notebook",
                    "colab_resume_notebook",
                    "colab_paused_notebooks",
                    "colab_workspace_sync",
                    "colab_compute_units",
                }
                assert not health.isError
                assert not compute_units.isError
                assert {
                    "colab_fs_read",
                    "colab_fs_write",
                    "colab_upload",
                    "colab_download",
                    "colab_transfer_upload",
                    "colab_transfer_download",
                }.isdisjoint(names)
                workspace_tool = next(
                    tool for tool in tools.tools if tool.name == "colab_workspace_sync"
                )
                workspace_properties = workspace_tool.inputSchema["properties"]
                assert {
                    "dry_run",
                    "expected_plan_id",
                    "preview_limit",
                }.isdisjoint(workspace_properties)

                status = await client.call_tool("colab_connector", {"action": "status"})
                assert not status.isError
                status_payload = status.structuredContent["result"]
                original_pid = status_payload["worker_pid"]
                assert (
                    status_payload["active_source_fingerprint"]
                    == status_payload["available_source_fingerprint"]
                )

                rejected = await client.call_tool(
                    "colab_connector",
                    {
                        "action": "reload",
                        "expected_source_fingerprint": "0" * 64,
                        "drain_timeout_seconds": 5,
                    },
                )
                assert rejected.isError
                assert rejected.structuredContent["result"]["worker_pid"] == original_pid
                assert rejected.structuredContent["result"]["rollback_used"] is True

                reloaded = await client.call_tool(
                    "colab_connector",
                    {
                        "action": "reload",
                        "expected_source_fingerprint": status_payload[
                            "available_source_fingerprint"
                        ],
                        "drain_timeout_seconds": 30,
                    },
                )
                assert not reloaded.isError
                reload_payload = reloaded.structuredContent["result"]
                assert reload_payload["reloaded"] is True
                assert reload_payload["previous_worker_pid"] == original_pid
                assert reload_payload["worker_pid"] != original_pid
                assert reload_payload["validated_tool_count"] == len(names)
                assert not (await client.call_tool("colab_health", {})).isError

                recovery_terms = {
                    "colab_sessions": "stale",
                    "colab_keepalive": "reclamation",
                    "colab_start": "quota",
                    "colab_execute": "durable",
                    "colab_run_command": "poll",
                    "colab_process_start": "request_not_submitted",
                    "colab_process_output": "spool",
                    "colab_process_export": "retry",
                    "colab_allocation_probe": "never follow replacement",
                    "colab_workspace_sync": "same lease",
                    "colab_stop": "permanently loses",
                }
                descriptions = {tool.name: tool.description or "" for tool in tools.tools}
                for tool_name, recovery_term in recovery_terms.items():
                    assert recovery_term in descriptions[tool_name], (
                        f"{tool_name} must document recovery strategy {recovery_term!r}"
                    )

                for tool in tools.tools:
                    schema = tool.inputSchema
                    for name, property_schema in schema.get("properties", {}).items():
                        assert property_schema.get("description"), (
                            f"{tool.name}.{name} must describe its semantics and default"
                        )
                    for definition_name, definition in schema.get("$defs", {}).items():
                        for name, property_schema in definition.get("properties", {}).items():
                            assert property_schema.get("description"), (
                                f"{tool.name}.{definition_name}.{name} must be described"
                            )

    asyncio.run(smoke())
