"""MCP loading and configuration utilities."""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from deepagents_cli.config import settings

logger = logging.getLogger(__name__)


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
            with open(user_config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                if "mcpServers" in user_config:
                    config.update(user_config["mcpServers"])
                else:
                    # Assume flat structure if no mcpServers key, or maybe just ignore?
                    # Let's support the standard "mcpServers" key for compatibility with Claude
                    pass
        except Exception as e:
            logger.warning(f"Failed to load user MCP config from {user_config_path}: {e}")

    # Load project config
    project_config_path = settings.get_project_mcp_config_path()
    if project_config_path and project_config_path.exists():
        try:
            with open(project_config_path, "r", encoding="utf-8") as f:
                project_config = json.load(f)
                if "mcpServers" in project_config:
                    # Update/Overwrite with project specific servers
                    config.update(project_config["mcpServers"])
        except Exception as e:
            logger.warning(f"Failed to load project MCP config from {project_config_path}: {e}")

    return config


async def load_mcp_tools_from_config() -> list[BaseTool]:
    """Load tools from all configured MCP servers.
    
    Returns:
        List of LangChain-compatible tools.
    """
    connections = load_mcp_config()
    
    if not connections:
        return []

    # Prepare connections for MultiServerMCPClient
    # LangChain adapter expects specific keys.
    # Claude config: "command", "args", "env"
    # Adapter: "command", "args", "env", "transport"="stdio" (defaulting to stdio if not specified?)
    
    # We need to normalize the config for the adapter
    normalized_connections = {}
    for name, server_config in connections.items():
        # First, handle "type" field - Claude config may use "type" instead of "transport"
        # Map type -> transport if transport is not already set
        if "type" in server_config and "transport" not in server_config:
            type_val = server_config["type"]
            # Map known type values to transport values
            # langchain-mcp-adapters supports: stdio, sse, streamable_http (or http), websocket
            type_to_transport = {
                "stdio": "stdio",
                "sse": "sse",
                "http": "streamable_http",  # Modern HTTP-based MCP protocol
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
                # Default to streamable_http for URL-based connections
                # This is the modern MCP protocol (2025) that supersedes SSE
                # It uses standard HTTP POST/GET and can optionally use SSE for streaming
                server_config["transport"] = "streamable_http"
        
        # Ensure it's a dict we can modify and filter out "type" key
        # (langchain-mcp-adapters doesn't expect "type", only "transport")
        config_to_use = dict(server_config)
        config_to_use.pop("type", None) 
        
        normalized_connections[name] = config_to_use

    try:
        client = MultiServerMCPClient(connections=normalized_connections)
        # We need to keep the client alive or managing sessions?
        # get_tools() creates new sessions.
        
        # Note: MultiServerMCPClient manager context was removed in 0.1.0 (as seen in the file view)
        # "new session will be created for each tool call"
        tools = await client.get_tools()
        return tools
    except Exception:
        import traceback
        logger.error(f"Error loading MCP tools:\n{traceback.format_exc()}")
        return []
