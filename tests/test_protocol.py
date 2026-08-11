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
                assert {
                    "colab_start",
                    "colab_run_command",
                    "colab_process_export",
                    "colab_process_export_cleanup",
                    "colab_transfer_cleanup",
                    "colab_workspace_sync",
                    "colab_allocation_probe",
                    "colab_keepalive",
                    "colab_stop",
                } <= names
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
