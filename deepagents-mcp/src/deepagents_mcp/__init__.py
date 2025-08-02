"""DeepAgents MCP Integration Package.

This package provides MCP client integration for DeepAgents,
allowing agents to use tools from external MCP servers.
"""

from deepagents_mcp.mcp_client import MCPToolProvider, load_mcp_tools, create_mcp_config_from_file

__all__ = ["MCPToolProvider", "load_mcp_tools", "create_mcp_config_from_file"]
__version__ = "0.1.0"