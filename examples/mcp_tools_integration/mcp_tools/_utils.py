"""Utility functions shared across MCP tools.

This module contains helper functions for:
- JSON parsing and payload extraction
- Async/sync context handling
- Configuration management
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Any


# =============================================================================
# Configuration - MCP Server Paths
# =============================================================================

def get_config(key: str, default: str = "") -> str:
    """Get configuration value from environment variable."""
    return os.getenv(key, default)


# Server paths
GOOGLE_SEARCH_SERVER = get_config(
    "MCP_GOOGLE_SEARCH_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/google_search_server.py"
)
WEB_FETCH_SERVER = get_config(
    "MCP_WEB_FETCH_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/web_fetch_server.py"
)
RAG_SERVER = get_config(
    "MCP_RAG_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/spaceops_rag.py"
)
WEATHER_SERVER = get_config(
    "MCP_WEATHER_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/weather.py"
)
SENTINEL_SERVER = get_config(
    "MCP_SENTINEL_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/sentinel_server.py"
)
ARXIV_STORAGE_PATH = get_config(
    "ARXIV_STORAGE_PATH",
    os.path.expanduser("~/.arxiv-mcp-storage")
)


# =============================================================================
# JSON Parsing Helpers
# =============================================================================

def try_parse_json(s: str) -> dict | None:
    """Try to parse a JSON string, handling BOM and wrapped content.

    Args:
        s: String to parse as JSON

    Returns:
        Parsed dict or None if parsing fails
    """
    s = (s or "").strip().lstrip("\ufeff")

    # Direct parse attempt
    try:
        return json.loads(s)
    except Exception:
        pass

    # Extract first JSON object from text
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def extract_payload(mcp_output: Any) -> dict[str, Any]:
    """Extract payload from MCP output wrapper.

    MCP outputs can be:
    - list[{"text": "<json-string>"}]
    - dict{"text": "<json-string>"}
    - dict (direct payload)

    Args:
        mcp_output: Raw MCP output

    Returns:
        Extracted payload as dict
    """
    if mcp_output is None:
        return {}

    # Handle list - recurse on first element
    if isinstance(mcp_output, list):
        if not mcp_output:
            return {}
        return extract_payload(mcp_output[0])

    # Handle dict
    if isinstance(mcp_output, dict):
        # Already a payload with items
        if "items" in mcp_output and isinstance(mcp_output.get("items"), list):
            return mcp_output

        # Wrapper with text field - parse and recurse
        if "text" in mcp_output:
            t = mcp_output.get("text", "")
            if isinstance(t, str):
                parsed = try_parse_json(t)
                if parsed is None:
                    return {"text": t}
                return extract_payload(parsed)
            return mcp_output

        return mcp_output

    return {}


# =============================================================================
# Async/Sync Context Helpers
# =============================================================================

def run_in_new_loop(coro):
    """Run async coroutine in a new event loop in a separate thread.

    This is necessary because MCP clients require their own clean event loop
    and cannot be nested inside another running loop.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine

    Raises:
        TimeoutError: If operation times out after 120 seconds
        Exception: Any exception raised by the coroutine
    """
    result = None
    exception = None

    def run():
        nonlocal result, exception
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()
        except Exception as e:
            exception = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=120)

    if thread.is_alive():
        raise TimeoutError("MCP operation timed out after 120 seconds")

    if exception is not None:
        raise exception

    return result


def run_async(coro):
    """Run async coroutine safely from sync context.

    Handles the case where we might already be in an async context
    by running the coroutine in a separate thread with its own event loop.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    try:
        asyncio.get_running_loop()
        return run_in_new_loop(coro)
    except RuntimeError:
        return asyncio.run(coro)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "get_config",
    "GOOGLE_SEARCH_SERVER",
    "WEB_FETCH_SERVER",
    "RAG_SERVER",
    "WEATHER_SERVER",
    "SENTINEL_SERVER",
    "ARXIV_STORAGE_PATH",
    # JSON helpers
    "try_parse_json",
    "extract_payload",
    # Async helpers
    "run_in_new_loop",
    "run_async",
]
