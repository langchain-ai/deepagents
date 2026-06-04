"""MCP configuration and tool loading for Talon."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents_code.mcp_tools import (
    MCPConfigError as CodeMCPConfigError,
    MCPServerInfo as CodeMCPServerInfo,
    get_mcp_tools_from_config,
    load_mcp_config,
    load_mcp_config_from_dict,
    merge_mcp_configs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

_MCP_CONFIG_ENV_KEYS = ("DEEPAGENTS_TALON_MCP_CONFIG", "MCP_CONFIG")

JsonObject = dict[str, object]


class MCPConfigError(CodeMCPConfigError):
    """Raised when an MCP configuration is invalid."""


@dataclass(frozen=True, slots=True)
class MCPServerInfo:
    """Load status for one MCP server.

    Args:
        name: Server name from `mcpServers`.
        transport: Canonical transport name.
        tool_count: Number of tools loaded after filtering.
        error: Error message when loading failed.
    """

    name: str
    transport: str
    tool_count: int = 0
    error: str | None = None


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
        MCPConfigError: If the server name or config is invalid.
        FileExistsError: If `name` already exists and `overwrite` is `False`.
    """
    data = _load_config_for_update(path) if path.exists() else {"mcpServers": {}}
    servers = data["mcpServers"]
    if name in servers and not overwrite:
        msg = f"MCP server {name!r} already exists in {path}"
        raise FileExistsError(msg)

    servers[name] = dict(server)
    _validate_config_for_talon(data, source=str(path))

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


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
    try:
        tools, manager, infos = await get_mcp_tools_from_config(data)
    except (TypeError, ValueError, RuntimeError) as exc:
        msg = str(exc)
        raise MCPConfigError(msg) from exc
    if manager is not None:
        logger.debug("Loaded MCP tools with a persistent session manager: %r", manager)
    return MCPTools(tools=tuple(tools), servers=tuple(_server_info(info) for info in infos))


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


def _load_config_for_update(path: Path) -> dict[str, Any]:
    try:
        with path.expanduser().open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        msg = f"Invalid MCP config JSON in {path}: {exc.msg}"
        raise MCPConfigError(msg) from exc
    except OSError as exc:
        msg = f"Could not read MCP config {path}: {exc}"
        raise MCPConfigError(msg) from exc

    if not isinstance(data, dict):
        msg = f"{path} must contain a JSON object"
        raise MCPConfigError(msg)

    servers = data.get("mcpServers")
    if servers is None:
        data["mcpServers"] = {}
    elif not isinstance(servers, dict):
        msg = f"{path} 'mcpServers' field must be a dictionary"
        raise MCPConfigError(msg)
    return data


def _validate_config_for_talon(data: Mapping[str, Any], *, source: str) -> None:
    try:
        load_mcp_config_from_dict(data, source=source)
    except (TypeError, ValueError) as exc:
        msg = str(exc)
        raise MCPConfigError(msg) from exc


def _server_info(info: CodeMCPServerInfo) -> MCPServerInfo:
    error = info.error if info.status != "ok" else None
    return MCPServerInfo(
        name=info.name,
        transport=info.transport,
        tool_count=len(info.tools),
        error=error,
    )


def _is_file(path: Path) -> bool:
    try:
        return path.expanduser().is_file()
    except OSError:
        logger.warning("Could not inspect MCP config path %s", path, exc_info=True)
        return False
