"""MCP configuration and tool loading for Talon.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.mcp_tools import (
    MCP_CONFIG_DISCOVERY_PATHS,
    MCPConfigError,
    MCPServerInfo,
    discover_mcp_configs,
    resolve_and_load_mcp_tools,
)
from deepagents_code.project_utils import ProjectContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

_MCP_CONFIG_ENV_KEYS = ("DEEPAGENTS_TALON_MCP_CONFIG", "MCP_CONFIG")
_WORKSPACE_ENV = "DEEPAGENTS_TALON_WORKSPACE"


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
    """Return existing MCP config files in Deep Agents Code discovery order.

    Args:
        config: Talon runtime configuration.

    Returns:
        Existing files, ordered from lowest to highest precedence.
    """
    return discover_mcp_configs(project_context=_project_context(config))


async def load_mcp_tools(config: TalonConfig) -> MCPTools:
    """Load configured MCP tools for a Talon runtime.

    Args:
        config: Talon runtime configuration.

    Returns:
        Loaded tools and status for each configured server.

    Raises:
        MCPConfigError: If a selected config source is malformed.
    """
    try:
        tools, manager, infos = await resolve_and_load_mcp_tools(
            explicit_config_path=_first_env_value(config.env),
            trust_project_mcp=None,
            project_context=_project_context(config),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        msg = str(exc)
        raise MCPConfigError(msg) from exc
    if manager is not None:
        logger.debug("Loaded MCP tools with a persistent session manager: %r", manager)
    return MCPTools(tools=tuple(tools), servers=tuple(infos))


def print_mcp_config_paths(config: TalonConfig) -> None:
    """Print Deep Agents Code MCP config discovery paths.

    Args:
        config: Talon runtime configuration.
    """
    project_context = _project_context(config)
    found = {str(path.resolve()) for path in discover_mcp_config_paths(config)}
    project_root = (
        project_context.project_root
        if project_context is not None and project_context.project_root is not None
        else (project_context.user_cwd if project_context is not None else Path.cwd())
    )
    rows = [
        (
            display,
            label,
            _resolve_discovery_display_path(display, project_root),
        )
        for display, label in MCP_CONFIG_DISCOVERY_PATHS
    ]
    width = max(len(path) for path, _, _ in rows)
    print("MCP config discovery paths (lowest to highest precedence):")  # noqa: T201
    for display, label, path in rows:
        marker = "found" if str(path.resolve()) in found or _is_file(path) else "missing"
        print(f"  [{marker:>7}]  {display:<{width}}  ({label})")  # noqa: T201
    print()  # noqa: T201
    print(  # noqa: T201
        "<project-root> = nearest ancestor with `.git`, else current directory.",
    )


def _first_env_value(env: Mapping[str, str]) -> str | None:
    for key in _MCP_CONFIG_ENV_KEYS:
        value = env.get(key) or os.environ.get(key)
        if value:
            return value
    return None


def _project_context(config: TalonConfig) -> ProjectContext | None:
    raw = config.env.get(_WORKSPACE_ENV)
    try:
        return ProjectContext.from_user_cwd(raw or Path.cwd())
    except OSError:
        logger.warning("Could not determine working directory for MCP discovery")
        return None


def _resolve_discovery_display_path(display: str, project_root: Path) -> Path:
    if display == "~/.deepagents/.mcp.json":
        return Path.home() / ".deepagents" / ".mcp.json"
    if display == "<project-root>/.deepagents/.mcp.json":
        return project_root / ".deepagents" / ".mcp.json"
    if display == "<project-root>/.mcp.json":
        return project_root / ".mcp.json"
    return Path(display).expanduser()


def _is_file(path: Path) -> bool:
    try:
        return path.expanduser().is_file()
    except OSError:
        logger.warning("Could not inspect MCP config path %s", path, exc_info=True)
        return False
