"""Smoke test: spawns the MCP server over stdio and calls every tool.

Start the stub backend first (from mcp_server/dev):
    uvicorn stub_backend:app --port 8000
Then run:
    python smoke_test.py
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parent.parent / "server.py"


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print("TOOLS:", [t.name for t in (await s.list_tools()).tools])
            for name, args in [
                ("index_folder", {"folder": "/docs"}),
                ("index_status", {}),
                ("search_files", {"query": "budget meeting notes"}),
                ("read_file", {"path": "/docs/pasta.txt"}),
            ]:
                res = await s.call_tool(name, args)
                print(f"\n{name} ->\n{res.content[0].text}")


asyncio.run(main())
