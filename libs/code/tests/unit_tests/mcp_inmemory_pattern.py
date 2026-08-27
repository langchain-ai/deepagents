"""Drive dcode's loader against real in-memory FastMCP servers (no mocks)."""
import asyncio, json, tempfile
from pathlib import Path
from unittest.mock import patch

from fastmcp import FastMCP
from fastmcp.client.transports import FastMCPTransport

from deepagents_code.mcp_tools import get_mcp_tools


def build_server(name: str) -> FastMCP:
    server = FastMCP(name)

    @server.tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"{name}:read:{path}"

    @server.tool
    def write_file(path: str, body: str = "") -> str:
        """Write a file."""
        return f"{name}:write:{path}:{body or '<empty>'}"

    return server


async def main() -> None:
    servers = {"srv": build_server("srv")}

    def fake_stdio(**kwargs):
        # dcode passes command/args/env; route to the in-memory server instead.
        return FastMCPTransport(servers["srv"])

    cfg = {"mcpServers": {"srv": {"command": "node", "args": ["server.js"]}}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f); path = f.name

    with patch("deepagents_code.mcp_tools.StdioTransport", fake_stdio), \
         patch("deepagents_code.mcp_tools._check_stdio_server", lambda *a, **k: None):
        tools, manager, infos = await get_mcp_tools(path)

    print("INFOS:", [(i.name, i.status, [t.name for t in i.tools]) for i in infos])
    print("TOOLS:", [t.name for t in tools])
    read = next(t for t in tools if t.name.endswith("read_file"))
    print("CALL:", await read.ainvoke({"path": "/tmp/x"}))
    # optional-arg normalization, exercised against a real schema
    write = next(t for t in tools if t.name.endswith("write_file"))
    print("NORMALIZE:", await write.ainvoke({"path": "/tmp/y", "body": ""}))
    if manager: await manager.cleanup()

asyncio.run(main())
