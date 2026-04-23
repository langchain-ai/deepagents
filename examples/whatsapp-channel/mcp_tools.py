"""MCP tools loader for the whatsapp-channel example.

Mirrors the pattern used by the deepagents CLI
(libs/cli/deepagents_cli/mcp_tools.py), trimmed to what a self-hosted
example needs: load a single explicit Claude-Desktop-style config file,
open persistent stdio sessions via `langchain-mcp-adapters`, and expose
them as LangChain tools.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import Connection, MultiServerMCPClient


logger = logging.getLogger(__name__)

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

    _validate_tool_filter_fields(server_name, server_config)


def _validate_tool_filter_fields(
    server_name: str, server_config: dict[str, Any]
) -> None:
    """Validate optional `allowedTools` / `disabledTools` fields."""
    has_allowed = "allowedTools" in server_config
    has_disabled = "disabledTools" in server_config
    if has_allowed and has_disabled:
        raise ValueError(
            f"Server '{server_name}' cannot set both 'allowedTools' and"
            " 'disabledTools' — pick one."
        )

    for field_name in ("allowedTools", "disabledTools"):
        if field_name not in server_config:
            continue
        value = server_config[field_name]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise TypeError(
                f"Server '{server_name}' '{field_name}' must be a list of strings"
            )


_GLOB_METACHARS = frozenset("*?[")


def _entry_matches_tool(entry: str, tool_name: str, prefix: str) -> bool:
    """Return True if a single filter entry matches a tool name.

    Entries with `*`, `?`, or `[` are treated as fnmatch globs; others are
    matched literally. Each entry is tried against both the full prefixed
    tool name and the bare (post-prefix) form.
    """
    is_glob = any(ch in _GLOB_METACHARS for ch in entry)
    if is_glob:
        if fnmatch.fnmatchcase(tool_name, entry):
            return True
        if tool_name.startswith(prefix):
            return fnmatch.fnmatchcase(tool_name[len(prefix) :], entry)
        return False
    if tool_name == entry:
        return True
    return tool_name.startswith(prefix) and tool_name[len(prefix) :] == entry


def _apply_tool_filter(
    tools: list[BaseTool],
    server_name: str,
    server_config: dict[str, Any],
) -> list[BaseTool]:
    """Filter a server's tools by its `allowedTools` / `disabledTools` list."""
    allowed = server_config.get("allowedTools")
    disabled = server_config.get("disabledTools")
    if allowed is None and disabled is None:
        return tools

    prefix = f"{server_name}_"

    def _any_entry_matches(tool_name: str, entries: list[str]) -> bool:
        return any(_entry_matches_tool(e, tool_name, prefix) for e in entries)

    if allowed is not None:
        filtered = [t for t in tools if _any_entry_matches(t.name, allowed)]
        missing = [
            e
            for e in allowed
            if not any(_entry_matches_tool(e, t.name, prefix) for t in tools)
        ]
        if missing:
            logger.warning(
                "MCP server '%s' allowedTools entries matched no tools: %s",
                server_name,
                ", ".join(missing),
            )
        return filtered

    return [t for t in tools if not _any_entry_matches(t.name, disabled or [])]


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
        for server_name, server_config in config["mcpServers"].items():
            session = await manager.exit_stack.enter_async_context(
                client.session(server_name)
            )
            tools = await load_mcp_tools(
                session, server_name=server_name, tool_name_prefix=True
            )
            tools = _apply_tool_filter(tools, server_name, server_config)
            all_tools.extend(tools)
        all_tools.sort(key=lambda t: t.name)
    except Exception:
        await manager.cleanup()
        raise

    return all_tools, manager
