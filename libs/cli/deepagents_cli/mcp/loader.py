"""MCP loading and configuration utilities."""

import json
import logging
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from deepagents_cli.config import settings

logger = logging.getLogger(__name__)

# Default timeout settings for MCP HTTP connections
# These are more generous than the langchain-mcp-adapters defaults
# to handle slow network or heavy tool operations
DEFAULT_HTTP_TIMEOUT = timedelta(seconds=60)  # 60s for initial HTTP connection
DEFAULT_SSE_READ_TIMEOUT = timedelta(seconds=60 * 10)  # 10 minutes for SSE events


def load_mcp_config() -> dict[str, Any]:
    """Load MCP configuration from project and user files.

    Merges configuration from:
    1. ~/.deepagents/mcp.json
    2. {project_root}/.deepagents/mcp.json (overrides user config)

    Returns:
        Dictionary of MCP server configurations.
    """
    config: dict[str, Any] = {}

    # Load user config
    user_config_path = settings.get_user_mcp_config_path()
    if user_config_path.exists():
        try:
            with user_config_path.open(encoding="utf-8") as f:
                user_config = json.load(f)
                if "mcpServers" in user_config:
                    config.update(user_config["mcpServers"])
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load user MCP config from %s: %s", user_config_path, e)

    # Load project config
    project_config_path = settings.get_project_mcp_config_path()
    if project_config_path and project_config_path.exists():
        try:
            with project_config_path.open(encoding="utf-8") as f:
                project_config = json.load(f)
                if "mcpServers" in project_config:
                    # Update/Overwrite with project specific servers
                    config.update(project_config["mcpServers"])
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load project MCP config from %s: %s", project_config_path, e)

    return config


def _normalize_server_config(name: str, server_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize server configuration for the MCP client.

    Handles:
    - Mapping 'type' to 'transport'
    - Inferring transport from config content
    - Adding default timeouts for HTTP-based transports
    """
    # First, handle "type" field
    if "type" in server_config and "transport" not in server_config:
        type_val = server_config["type"]
        type_to_transport = {
            "stdio": "stdio",
            "sse": "sse",
            "http": "streamable_http",
            "streamable_http": "streamable_http",
            "streamable-http": "streamable_http",
            "websocket": "websocket",
        }
        server_config["transport"] = type_to_transport.get(type_val, type_val)

    # If still no transport, infer from config content
    if "transport" not in server_config:
        if "command" in server_config:
            server_config["transport"] = "stdio"
        elif "url" in server_config:
            server_config["transport"] = "streamable_http"

    # Ensure it's a dict we can modify and filter out "type" key
    config_to_use = dict(server_config)
    config_to_use.pop("type", None)

    # Add timeout settings for HTTP-based transports
    transport = config_to_use.get("transport")
    if transport in ("streamable_http", "http", "streamable-http"):
        if "timeout" not in config_to_use:
            config_to_use["timeout"] = DEFAULT_HTTP_TIMEOUT
        if "sse_read_timeout" not in config_to_use:
            config_to_use["sse_read_timeout"] = DEFAULT_SSE_READ_TIMEOUT
        logger.debug(
            "[MCP] Configured %s with timeout=%s, sse_read_timeout=%s",
            name,
            config_to_use["timeout"],
            config_to_use["sse_read_timeout"],
        )
    elif transport == "sse":
        if "timeout" not in config_to_use:
            config_to_use["timeout"] = DEFAULT_HTTP_TIMEOUT.total_seconds()
        if "sse_read_timeout" not in config_to_use:
            config_to_use["sse_read_timeout"] = DEFAULT_SSE_READ_TIMEOUT.total_seconds()
        logger.debug(
            "[MCP] Configured %s (SSE) with timeout=%ss, sse_read_timeout=%ss",
            name,
            config_to_use["timeout"],
            config_to_use["sse_read_timeout"],
        )

    logger.debug(
        "[MCP] Server '%s': transport=%s, url=%s",
        name,
        transport,
        config_to_use.get("url", "N/A"),
    )
    return config_to_use


async def load_mcp_tools_from_config() -> list[BaseTool]:
    """Load tools from all configured MCP servers.

    Returns:
        List of LangChain-compatible tools.
    """
    connections = load_mcp_config()
    if not connections:
        return []

    normalized_connections = {
        name: _normalize_server_config(name, server_config)
        for name, server_config in connections.items()
    }

    try:
        logger.info("[MCP] Connecting to %d MCP server(s)...", len(normalized_connections))
        client = MultiServerMCPClient(connections=normalized_connections)
        logger.debug("[MCP] Loading tools from MCP servers...")
        tools = await client.get_tools()
    except BaseException:
        logger.exception("Error loading MCP tools")
        return []
    else:
        logger.info("[MCP] Successfully loaded %d tool(s) from MCP servers", len(tools))
        for tool in tools:
            logger.debug("[MCP] - Tool: %s", tool.name)
        return tools
