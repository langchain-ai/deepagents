"""MCP tools loader for the whatsapp-channel example.

Mirrors the pattern used by the deepagents CLI
(libs/cli/deepagents_cli/mcp_tools.py), trimmed to what a self-hosted
example needs: load a single explicit Claude-Desktop-style config file,
open persistent stdio sessions via `langchain-mcp-adapters`, and expose
them as LangChain tools.
"""

from __future__ import annotations

import json
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import Connection, MultiServerMCPClient


_SUPPORTED_REMOTE_TYPES = {"sse", "http"}


def _resolve_server_type(server_config: dict[str, Any]) -> str:
    t = server_config.get("type")
    if t is not None:
        return t
    return server_config.get("transport", "stdio")


def _validate_server_config(server_name: str, server_config: dict[str, Any]) -> None:
    if not isinstance(server_config, dict):
        raise TypeError(f"Server '{server_name}' config must be a dictionary")

    server_type = _resolve_server_type(server_config)

    if server_type in _SUPPORTED_REMOTE_TYPES:
        if "url" not in server_config:
            raise ValueError(
                f"Server '{server_name}' with type '{server_type}' "
                "missing required 'url' field"
            )
        headers = server_config.get("headers")
        if headers is not None and not isinstance(headers, dict):
            raise TypeError(f"Server '{server_name}' 'headers' must be a dictionary")
    elif server_type == "stdio":
        if "command" not in server_config:
            raise ValueError(f"Server '{server_name}' missing required 'command' field")
        if "args" in server_config and not isinstance(server_config["args"], list):
            raise TypeError(f"Server '{server_name}' 'args' must be a list")
        if "env" in server_config and not isinstance(server_config["env"], dict):
            raise TypeError(f"Server '{server_name}' 'env' must be a dictionary")
    else:
        raise ValueError(
            f"Server '{server_name}' has unsupported transport type "
            f"'{server_type}'. Supported: stdio, sse, http"
        )


def load_mcp_config(config_path: str) -> dict[str, Any]:
    """Load and validate a Claude-Desktop-style MCP config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"MCP config file not found: {config_path}")

    with path.open(encoding="utf-8") as f:
        config = json.load(f)

    if "mcpServers" not in config:
        raise ValueError(
            "MCP config must contain 'mcpServers' field. "
            'Expected format: {"mcpServers": {"server-name": {...}}}'
        )
    if not isinstance(config["mcpServers"], dict):
        raise TypeError("'mcpServers' field must be a dictionary")
    if not config["mcpServers"]:
        raise ValueError("'mcpServers' field is empty - no servers configured")

    for server_name, server_config in config["mcpServers"].items():
        _validate_server_config(server_name, server_config)

    return config


class MCPSessionManager:
    """Owns persistent stdio sessions; call `cleanup()` on shutdown."""

    def __init__(self) -> None:
        self.client: MultiServerMCPClient | None = None
        self.exit_stack = AsyncExitStack()

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()


def _check_stdio_server(server_name: str, server_config: dict[str, Any]) -> None:
    command = server_config.get("command")
    if command is None:
        raise RuntimeError(f"MCP server '{server_name}': missing 'command' in config.")
    if shutil.which(command) is None:
        raise RuntimeError(
            f"MCP server '{server_name}': command '{command}' not found on PATH."
        )


async def get_mcp_tools(
    config_path: str,
) -> tuple[list[BaseTool], MCPSessionManager]:
    """Load MCP tools from a config file with persistent sessions."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StdioConnection,
        StreamableHttpConnection,
    )
    from langchain_mcp_adapters.tools import load_mcp_tools

    config = load_mcp_config(config_path)

    # Pre-flight: make sure stdio commands exist before spawning.
    for server_name, server_config in config["mcpServers"].items():
        if _resolve_server_type(server_config) == "stdio":
            _check_stdio_server(server_name, server_config)

    connections: dict[str, Connection] = {}
    for server_name, server_config in config["mcpServers"].items():
        server_type = _resolve_server_type(server_config)
        if server_type in _SUPPORTED_REMOTE_TYPES:
            if server_type == "http":
                conn: Connection = StreamableHttpConnection(
                    transport="streamable_http",
                    url=server_config["url"],
                )
            else:
                conn = SSEConnection(
                    transport="sse",
                    url=server_config["url"],
                )
            if "headers" in server_config:
                conn["headers"] = server_config["headers"]
            connections[server_name] = conn
        else:
            connections[server_name] = StdioConnection(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env") or None,
                transport="stdio",
            )

    manager = MCPSessionManager()
    try:
        client = MultiServerMCPClient(connections=connections)
        manager.client = client

        all_tools: list[BaseTool] = []
        for server_name in config["mcpServers"]:
            session = await manager.exit_stack.enter_async_context(
                client.session(server_name)
            )
            tools = await load_mcp_tools(
                session, server_name=server_name, tool_name_prefix=True
            )
            all_tools.extend(tools)
        all_tools.sort(key=lambda t: t.name)
    except Exception:
        await manager.cleanup()
        raise

    return all_tools, manager
