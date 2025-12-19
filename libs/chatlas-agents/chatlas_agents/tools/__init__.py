"""Tools for ChATLAS agents."""

import logging
from typing import Any, List

from chatlas_agents.config import MCPServerConfig

logger = logging.getLogger(__name__)


async def load_mcp_tools(mcp_config: MCPServerConfig) -> List[Any]:
    """Load tools from MCP server using LangChain adapters with native interceptors.

    This function uses the native langchain-mcp-adapters tool_interceptors mechanism
    for handling retries, error handling, and argument sanitization. This is more
    efficient and maintainable than wrapping tools manually.

    Args:
        mcp_config: MCP server configuration

    Returns:
        List of LangChain tools loaded from MCP server with retry/sanitize interceptor
    """
    # Import here to avoid circular dependency
    from chatlas_agents.mcp import create_mcp_client_and_load_tools
    
    # Delegate to the MCP client which now uses native interceptors
    return await create_mcp_client_and_load_tools(mcp_config)
