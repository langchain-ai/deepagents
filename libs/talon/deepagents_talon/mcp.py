"""MCP configuration and tool loading for Talon.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.mcp_tools import (
    MCPConfigError,
    MCPServerInfo,
    get_mcp_tools,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

_MCP_CONFIG_ENV_KEYS = ("DEEPAGENTS_TALON_MCP_CONFIG", "MCP_CONFIG")


@dataclass(frozen=True, slots=True)
class MCPTools:
    """Loaded MCP tools and per-server load statuses.

    Args:
        tools: LangChain tools exposed to the agent.
        servers: Per-server load results.
    """

    tools: Sequence[BaseTool]
    servers: Sequence[MCPServerInfo]


def discover_mcp_config_paths(config: TalonConfig) -> list[Path]:
    """Return existing MCP config files in Talon fallback order.

    Args:
        config: Talon runtime configuration.

    Returns:
        Existing files, ordered from lowest to highest precedence.
    """
    paths = [
        Path.home() / ".deepagents" / ".mcp.json",
        config.home / ".mcp.json",
    ]
    return [path for path in paths if _is_file(path)]


async def load_mcp_tools(
    config: TalonConfig,
    *,
    allow_empty: bool = False,
) -> MCPTools:
    """Load configured MCP tools for a Talon runtime.

    Args:
        config: Talon runtime configuration.
        allow_empty: Whether an existing config with an empty `mcpServers` object
            should be treated as no local MCP config.

    Returns:
        Loaded tools and status for each configured server.

    Raises:
        MCPConfigError: If a selected config source is malformed.
    """
    path = _select_mcp_config_path(config)
    if path is None:
        return MCPTools(tools=(), servers=())
    if allow_empty and _mcp_servers_empty(path):
        return MCPTools(tools=(), servers=())
    try:
        tools, manager, infos = await get_mcp_tools(str(path))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        msg = str(exc)
        raise MCPConfigError(msg) from exc
    if manager is not None:
        logger.debug("Loaded MCP tools with a persistent session manager: %r", manager)
    return MCPTools(tools=tuple(tools), servers=tuple(infos))


def print_mcp_config_paths(config: TalonConfig) -> None:
    """Print Talon MCP config discovery paths.

    Args:
        config: Talon runtime configuration.
    """
    rows = [
        ("~/.deepagents/.mcp.json", Path.home() / ".deepagents" / ".mcp.json"),
        ("<assistant-home>/.mcp.json", config.home / ".mcp.json"),
    ]
    width = max(len(label) for label, _ in rows)
    print("MCP config discovery paths (lowest to highest precedence):")  # noqa: T201
    for label, path in rows:
        marker = "found" if _is_file(path) else "missing"
        print(f"  [{marker:>7}]  {label:<{width}}  {path}")  # noqa: T201
    print(  # noqa: T201
        "Override with DEEPAGENTS_TALON_MCP_CONFIG or MCP_CONFIG as a config file path.",
    )
    print("The highest-precedence existing path is loaded; configs are not merged.")  # noqa: T201
    print("Edit <assistant-home>/.mcp.json directly to add Talon MCP servers.")  # noqa: T201


def _select_mcp_config_path(config: TalonConfig) -> Path | None:
    env_value = _first_env_value(config.env)
    if env_value:
        return Path(env_value).expanduser()

    paths = discover_mcp_config_paths(config)
    if paths:
        return paths[-1]
    return None


def _first_env_value(env: Mapping[str, str]) -> str | None:
    for key in _MCP_CONFIG_ENV_KEYS:
        value = env.get(key) or os.environ.get(key)
        if value:
            return value
    return None


def _is_file(path: Path) -> bool:
    try:
        return path.expanduser().is_file()
    except OSError:
        logger.warning("Could not inspect MCP config path %s", path, exc_info=True)
        return False


def _mcp_servers_empty(path: Path) -> bool:
    try:
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, Mapping) and data.get("mcpServers") == {}
