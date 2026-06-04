"""MCP configuration and tool loading for Talon."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents_code.mcp_tools import (
    MCPConfigError,
    MCPServerInfo,
    get_mcp_tools_from_config,
    load_mcp_config,
    load_mcp_config_from_dict,
    merge_mcp_configs,
    write_mcp_server_config as write_code_mcp_server_config,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

_MCP_CONFIG_ENV_KEYS = ("DEEPAGENTS_TALON_MCP_CONFIG", "MCP_CONFIG")

JsonObject = dict[str, object]


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
    """Return existing MCP config files in Talon precedence order.

    Args:
        config: Talon runtime configuration.

    Returns:
        Existing files, ordered from lowest to highest precedence.
    """
    paths = [
        Path.home() / ".deepagents" / ".mcp.json",
        config.manifest_dir / "tools.json",
    ]
    return [path for path in paths if _is_file(path)]


def write_mcp_server_config(
    *,
    path: Path,
    name: str,
    server: JsonObject,
    overwrite: bool = False,
) -> None:
    """Add one MCP server to a JSON config file.

    Args:
        path: Config file to create or update.
        name: Server name under `mcpServers`.
        server: Server configuration.
        overwrite: Whether an existing server entry may be replaced.

    Raises:
        MCPConfigError: If the file cannot be read or parsed for update.
        FileExistsError: If `name` already exists and `overwrite` is `False`.
        TypeError: If config fields have wrong types.
        ValueError: If the resulting config is missing required fields.
    """
    write_code_mcp_server_config(
        path=path,
        name=name,
        server=server,
        overwrite=overwrite,
    )


async def load_mcp_tools(config: TalonConfig) -> MCPTools:
    """Load configured MCP tools for a Talon runtime.

    Args:
        config: Talon runtime configuration.

    Returns:
        Loaded tools and status for each configured server.

    Raises:
        MCPConfigError: If a selected config source is malformed.
    """
    data = _load_config(config)
    if data is None:
        return MCPTools(tools=(), servers=())
    return await load_mcp_tools_from_config(data)


async def load_mcp_tools_from_config(data: Mapping[str, Any]) -> MCPTools:
    """Load MCP tools from a parsed config object.

    Args:
        data: MCP configuration with an `mcpServers` object.

    Returns:
        Loaded tools and status for each configured server.
    """
    tools, manager, infos = await get_mcp_tools_from_config(data)
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
        ("<assistant-home>/agent/tools.json", config.manifest_dir / "tools.json"),
    ]
    width = max(len(label) for label, _ in rows)
    print("MCP config discovery paths (lowest to highest precedence):")  # noqa: T201
    for label, path in rows:
        marker = "found" if _is_file(path) else "missing"
        print(f"  [{marker:>7}]  {label:<{width}}  {path}")  # noqa: T201
    print(  # noqa: T201
        "Override with DEEPAGENTS_TALON_MCP_CONFIG or MCP_CONFIG as a path or JSON object.",
    )


def _load_config(config: TalonConfig) -> dict[str, Any] | None:
    env_value = _first_env_value(config.env)
    if env_value:
        return _load_env_config(env_value)

    configs = [load_mcp_config(str(path)) for path in discover_mcp_config_paths(config)]
    if not configs:
        return None
    try:
        return load_mcp_config_from_dict(merge_mcp_configs(configs))
    except (TypeError, ValueError) as exc:
        msg = str(exc)
        raise MCPConfigError(msg) from exc


def _first_env_value(env: Mapping[str, str]) -> str | None:
    for key in _MCP_CONFIG_ENV_KEYS:
        value = env.get(key) or os.environ.get(key)
        if value:
            return value
    return None


def _load_env_config(value: str) -> dict[str, Any]:
    stripped = value.strip()
    try:
        if stripped.startswith("{"):
            data = json.loads(stripped)
            if not isinstance(data, dict):
                msg = "MCP_CONFIG must contain a JSON object"
                raise MCPConfigError(msg)
            return load_mcp_config_from_dict(data, source="MCP_CONFIG")
        return load_mcp_config(str(Path(stripped).expanduser()))
    except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
        msg = str(exc)
        raise MCPConfigError(msg) from exc


def _is_file(path: Path) -> bool:
    try:
        return path.expanduser().is_file()
    except OSError:
        logger.warning("Could not inspect MCP config path %s", path, exc_info=True)
        return False
