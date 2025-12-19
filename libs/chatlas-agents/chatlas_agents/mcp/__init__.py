"""MCP (Model Context Protocol) client for ChATLAS server using LangChain adapters."""

import logging
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from chatlas_agents.config import MCPServerConfig
from chatlas_agents.tools.interceptors import RetryAndSanitizeInterceptor

logger = logging.getLogger(__name__)


async def create_mcp_client_and_load_tools(config: MCPServerConfig) -> List[Any]:
    """Create MCP client and load tools from the server.

    Args:
        config: MCP server configuration

    Returns:
        List of LangChain tools loaded from MCP server

    Raises:
        Exception: If connection to MCP server fails
    """
    # Create HTTP connection configuration
    # The server uses HTTP/HTTPS with JSON-RPC, not SSE
    connection = {
        "url": config.url,
        "timeout": config.timeout,
        "transport": "http",  # Use HTTP transport for JSON-RPC over HTTP
    }

    if config.headers:
        connection["headers"] = config.headers

    logger.debug(f"Connecting to MCP server at {config.url} using HTTP transport")
    logger.debug(f"Connection config: {connection}")

    # Create MCP client
    mcp_client = MultiServerMCPClient(
        connections={
            "chatlas": connection,
        }
    )

    # Load tools from the server using explicit session
    # Use a small retry/backoff loop to handle transient gateway errors (504)
    max_attempts = 3
    backoff_base = 1.0
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug("Loading tools from MCP server via session (attempt %d)", attempt)
            
            # Create the interceptor for retry and sanitization
            interceptor = RetryAndSanitizeInterceptor(
                max_attempts=3,  # Retry 3 times for transient errors
                backoff_base=1.0  # 1 second base for exponential backoff
            )
            
            # Load tools with the interceptor
            tools = await load_mcp_tools(
                session=None,  # Don't provide a session
                connection=connection,  # Let tools create sessions on demand
                server_name="chatlas",
                tool_interceptors=[interceptor],  # Use native interceptor support
            )
            logger.debug(f"Loaded {len(tools)} tools from MCP server")
            logger.info(f"Successfully loaded {len(tools)} tools from MCP server with retry/sanitize interceptor")
            return tools
        except httpx.HTTPStatusError as e:
            last_exc = e
            status = getattr(e.response, "status_code", None)
            logger.warning(
                "MCP server returned HTTP %s on attempt %d/%d while loading tools",
                status,
                attempt,
                max_attempts,
            )
            # Retry on 5xx errors (transient)
            if status is not None and 500 <= status < 600 and attempt < max_attempts:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                logger.debug("Retrying after %.1fs", sleep_time)
                await asyncio.sleep(sleep_time)
                continue
            # Non-retryable or last attempt
            logger.error("Failed to load tools from MCP server: %s", str(e), exc_info=True)
            raise
        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout, asyncio.TimeoutError) as e:
            # Explicitly handle timeouts: retry up to max_attempts
            last_exc = e
            logger.warning("Timeout when connecting to MCP server on attempt %d/%d: %s", attempt, max_attempts, e)
            if attempt < max_attempts:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                logger.debug("Retrying after %.1fs due to timeout", sleep_time)
                await asyncio.sleep(sleep_time)
                continue
            logger.error("Timeout while loading tools from MCP server (final attempt): %s", e, exc_info=True)
            raise
        except Exception as e:
            last_exc = e
            logger.error("Failed to load tools from MCP server on attempt %d/%d: %s", attempt, max_attempts, e, exc_info=True)
            if attempt < max_attempts:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_time)
                continue
            raise last_exc


def create_mcp_client(config: MCPServerConfig) -> MultiServerMCPClient:
    """Create an MCP client instance (for compatibility).

    Args:
        config: MCP server configuration

    Returns:
        MCP client instance
    """
    connection = {
        "url": config.url,
        "timeout": config.timeout,
        "transport": "http",  # Use HTTP transport for JSON-RPC over HTTP
    }
    
    if config.headers:
        connection["headers"] = config.headers
    
    return MultiServerMCPClient(
        connections={
            "chatlas": connection,
        }
    )


async def cleanup_mcp_session():
    """Cleanup MCP session when agent is shutting down.
    
    With on-demand session creation, this is a no-op but kept for API compatibility.
    """
    pass

