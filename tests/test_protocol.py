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
                    "colab_allocation_probe",
                    "colab_keepalive",
                    "colab_stop",
                } <= names
                assert not health.isError
                assert not compute_units.isError

    asyncio.run(smoke())
