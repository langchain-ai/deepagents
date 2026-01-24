"""MCP (Model Context Protocol) tools loader for deepagents CLI.

This module provides async functions to load and manage MCP servers
using langchain-mcp-adapters, supporting Claude Desktop style JSON configs.
"""

import json
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def _validate_server_config(server_name: str, server_config: dict) -> None:
    """Validate a single server configuration.

    Args:
        server_name: Name of the server
        server_config: Server configuration dictionary

    Raises:
        TypeError: If config fields have wrong types
        ValueError: If required fields are missing
    """
    if not isinstance(server_config, dict):
        error_msg = f"Server '{server_name}' config must be a dictionary"
        raise TypeError(error_msg)

    # Determine server type
    server_type = server_config.get("type", "stdio")

    if server_type in ("sse", "http"):
        # SSE/HTTP server validation - requires url field
        if "url" not in server_config:
            error_msg = (
                f"Server '{server_name}' with type '{server_type}' missing required 'url' field"
            )
            raise ValueError(error_msg)

        # headers is optional but must be correct type if present
        if "headers" in server_config and not isinstance(server_config["headers"], dict):
            error_msg = f"Server '{server_name}' 'headers' must be a dictionary"
            raise TypeError(error_msg)
    else:
        # stdio server validation (default)
        if "command" not in server_config:
            error_msg = f"Server '{server_name}' missing required 'command' field"
            raise ValueError(error_msg)

        # args and env are optional but must be correct type if present
        if "args" in server_config and not isinstance(server_config["args"], list):
            error_msg = f"Server '{server_name}' 'args' must be a list"
            raise TypeError(error_msg)

        if "env" in server_config and not isinstance(server_config["env"], dict):
            error_msg = f"Server '{server_name}' 'env' must be a dictionary"
            raise TypeError(error_msg)


def load_mcp_config(config_path: str) -> dict:
    """Load and validate MCP configuration from JSON file.

    Supports multiple server types:
    - stdio: Process-based servers with "command", "args", "env" fields (default)
    - sse: Server-Sent Events servers with "type": "sse", "url", and optional "headers"
    - http: HTTP-based servers with "type": "http", "url", and optional "headers"

    Args:
        config_path: Path to MCP JSON configuration file (Claude Desktop format)

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file contains invalid JSON
        ValueError: If config is missing required fields
    """
    path = Path(config_path)

    if not path.exists():
        error_msg = f"MCP config file not found: {config_path}"
        raise FileNotFoundError(error_msg)

    try:
        with path.open() as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in MCP config file: {e.msg}"
        raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e

    # Validate required fields
    if "mcpServers" not in config:
        error_msg = (
            "MCP config must contain 'mcpServers' field. "
            'Expected format: {"mcpServers": {"server-name": {...}}}'
        )
        raise ValueError(error_msg)

    if not isinstance(config["mcpServers"], dict):
        error_msg = "'mcpServers' field must be a dictionary"
        raise TypeError(error_msg)

    if not config["mcpServers"]:
        error_msg = "'mcpServers' field is empty - no servers configured"
        raise ValueError(error_msg)

    # Validate each server config
    for server_name, server_config in config["mcpServers"].items():
        _validate_server_config(server_name, server_config)

    return config


class MCPSessionManager:
    """Manages persistent MCP sessions for stateful stdio servers.

    This manager creates and maintains persistent sessions for stdio MCP servers,
    preventing server restarts on every tool call. Sessions are kept alive until
    explicitly cleaned up.
    """

    def __init__(self) -> None:
        """Initialize the session manager."""
        self.client: MultiServerMCPClient | None = None
        self.exit_stack = AsyncExitStack()

    async def cleanup(self) -> None:
        """Clean up all managed sessions and close connections."""
        await self.exit_stack.aclose()


async def get_mcp_tools(config_path: str) -> tuple[list[BaseTool], MCPSessionManager]:
    """Load MCP tools from configuration file with stateful sessions.

    Supports multiple server types:
    - stdio: Spawns MCP servers as subprocesses with persistent sessions
    - sse/http: Connects to remote MCP servers via URL

    For stdio servers, this creates persistent sessions that remain active across
    tool calls, avoiding server restarts. Sessions are managed by MCPSessionManager
    and should be cleaned up with session_manager.cleanup() when done.

    Args:
        config_path: Path to MCP JSON configuration file

    Returns:
        Tuple of (tools_list, session_manager) where:
        - tools_list: List of LangChain BaseTool objects
        - session_manager: MCPSessionManager instance (call cleanup() when done)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
        RuntimeError: If MCP server fails to spawn or connect
    """
    # Load and validate config
    config = load_mcp_config(config_path)

    # Create connections dict for MultiServerMCPClient
    # Convert Claude Desktop format to langchain-mcp-adapters format
    connections = {}
    for server_name, server_config in config["mcpServers"].items():
        server_type = server_config.get("type", "stdio")

        if server_type in ("sse", "http"):
            # SSE/HTTP server connection
            # Note: langchain-mcp-adapters uses "streamable_http" for HTTP transport
            transport = "streamable_http" if server_type == "http" else "sse"
            connections[server_name] = {
                "url": server_config["url"],
                "transport": transport,
            }
            # Add optional headers for authentication
            if "headers" in server_config:
                connections[server_name]["headers"] = server_config["headers"]
        else:
            # stdio server connection (default)
            connections[server_name] = {
                "command": server_config["command"],
                "args": server_config.get("args", []),
                "env": server_config.get("env", {}),
                "transport": "stdio",
            }

    # Create session manager to track persistent sessions
    manager = MCPSessionManager()

    try:
        # Initialize MultiServerMCPClient with all servers
        client = MultiServerMCPClient(connections=connections)
        manager.client = client

        # Load tools from all servers using persistent sessions
        all_tools: list[BaseTool] = []
        for server_name in config["mcpServers"]:
            # Create persistent session using AsyncExitStack to manage lifecycle
            session = await manager.exit_stack.enter_async_context(client.session(server_name))
            tools = await load_mcp_tools(session)
            all_tools.extend(tools)

    except Exception as e:
        # Clean up on failure
        await manager.cleanup()
        error_msg = (
            f"Failed to connect to MCP servers: {e}\n"
            "For stdio servers: Check that the command and args are correct, and that "
            "the MCP server is installed (e.g., run 'npx -y <package>' manually to test).\n"
            "For sse/http servers: Check that the URL is correct and the server is running."
        )
        raise RuntimeError(error_msg) from e

    return all_tools, manager
