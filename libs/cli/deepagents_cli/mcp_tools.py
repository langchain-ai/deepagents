"""MCP (Model Context Protocol) tools loader for deepagents CLI.

This module provides async functions to load and manage MCP servers
using langchain-mcp-adapters, supporting Claude Desktop style JSON configs.
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def load_mcp_config(config_path: str) -> dict:
    """Load and validate MCP configuration from JSON file.

    Supports multiple server types:
    - stdio: Process-based servers with "command" field (default)
    - sse: Server-Sent Events servers with "type": "sse" and "url" field
    - http: HTTP-based servers with "type": "http" and "url" field

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
        raise FileNotFoundError(f"MCP config file not found: {config_path}")

    try:
        with open(path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in MCP config file: {e.msg}",
            e.doc,
            e.pos
        ) from e

    # Validate required fields
    if "mcpServers" not in config:
        raise ValueError(
            "MCP config must contain 'mcpServers' field. "
            "Expected format: {\"mcpServers\": {\"server-name\": {...}}}"
        )

    if not isinstance(config["mcpServers"], dict):
        raise ValueError("'mcpServers' field must be a dictionary")

    if not config["mcpServers"]:
        raise ValueError("'mcpServers' field is empty - no servers configured")

    # Validate each server config
    for server_name, server_config in config["mcpServers"].items():
        if not isinstance(server_config, dict):
            raise ValueError(f"Server '{server_name}' config must be a dictionary")

        # Determine server type
        server_type = server_config.get("type", "stdio")

        if server_type in ("sse", "http"):
            # SSE/HTTP server validation - requires url field
            if "url" not in server_config:
                raise ValueError(f"Server '{server_name}' with type '{server_type}' missing required 'url' field")
        else:
            # stdio server validation (default)
            if "command" not in server_config:
                raise ValueError(f"Server '{server_name}' missing required 'command' field")

            # args and env are optional but must be correct type if present
            if "args" in server_config and not isinstance(server_config["args"], list):
                raise ValueError(f"Server '{server_name}' 'args' must be a list")

            if "env" in server_config and not isinstance(server_config["env"], dict):
                raise ValueError(f"Server '{server_name}' 'env' must be a dictionary")

    return config


async def get_mcp_tools(config_path: str) -> tuple[list[BaseTool], MultiServerMCPClient]:
    """Load MCP tools from configuration file.

    Supports multiple server types:
    - stdio: Spawns MCP servers as subprocesses
    - sse/http: Connects to remote MCP servers via URL

    Args:
        config_path: Path to MCP JSON configuration file

    Returns:
        Tuple of (tools_list, mcp_client) where:
        - tools_list: List of LangChain BaseTool objects
        - mcp_client: MultiServerMCPClient instance (no cleanup needed for get_tools API)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
        RuntimeError: If MCP server fails to spawn or connect
    """
    # Load and validate config
    config = await load_mcp_config(config_path)

    # Create connections dict for MultiServerMCPClient
    # Convert Claude Desktop format to langchain-mcp-adapters format
    connections = {}
    for server_name, server_config in config["mcpServers"].items():
        server_type = server_config.get("type", "stdio")

        if server_type in ("sse", "http"):
            # SSE/HTTP server connection
            connections[server_name] = {
                "url": server_config["url"],
                "transport": server_type,
            }
        else:
            # stdio server connection (default)
            connections[server_name] = {
                "command": server_config["command"],
                "args": server_config.get("args", []),
                "env": server_config.get("env", {}),
                "transport": "stdio",
            }

    try:
        # Initialize MultiServerMCPClient with all servers
        client = MultiServerMCPClient(connections=connections)

        # Get all tools from all connected servers
        tools = await client.get_tools()

        return tools, client

    except Exception as e:
        # Provide helpful error message if server fails to spawn or connect
        raise RuntimeError(
            f"Failed to connect to MCP servers: {e}\n"
            "For stdio servers: Check that the command and args are correct, and that "
            "the MCP server is installed (e.g., run 'npx -y <package>' manually to test).\n"
            "For sse/http servers: Check that the URL is correct and the server is running."
        ) from e
