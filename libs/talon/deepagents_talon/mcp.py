"""MCP configuration and tool loading for Talon."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

_SERVER_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)}")
_REMOTE_TRANSPORTS = {"sse", "http"}
_TRANSPORT_ALIASES = {"streamable_http": "http", "streamable-http": "http"}
_GLOB_METACHARS = frozenset("*?[")
_MCP_CONFIG_ENV_KEYS = ("DEEPAGENTS_TALON_MCP_CONFIG", "MCP_CONFIG")

JsonObject = dict[str, object]


class MCPConfigError(ValueError):
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
    _validate_server_name(name)
    _validate_server_config(name, server)

    data: JsonObject = _load_config_path(path) if path.exists() else {"mcpServers": {}}
    servers = _servers(data)
    if name in servers and not overwrite:
        msg = f"MCP server {name!r} already exists in {path}"
        raise FileExistsError(msg)

    servers[name] = server
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


async def load_mcp_tools_from_config(data: JsonObject) -> MCPTools:
    """Load MCP tools from a parsed config object.

    Args:
        data: MCP configuration with an `mcpServers` object.

    Returns:
        Loaded tools and status for each configured server.
    """
    _validate_config(data)
    servers = _servers(data)
    tools: list[BaseTool] = []
    infos: list[MCPServerInfo] = []

    for name, raw in servers.items():
        server = _as_object(raw, f"mcpServers.{name}")
        transport = _resolve_transport(server)
        try:
            server_tools = await _load_server_tools(name, server)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP server %s failed to load: %s", name, exc)
            infos.append(MCPServerInfo(name=name, transport=transport, error=str(exc)))
            continue

        filtered = _apply_tool_filter(server_tools, name, server)
        tools.extend(filtered)
        infos.append(MCPServerInfo(name=name, transport=transport, tool_count=len(filtered)))

    return MCPTools(tools=tools, servers=infos)


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


def _load_config(config: TalonConfig) -> JsonObject | None:
    env_value = _first_env_value(config.env)
    if env_value:
        return _load_env_config(env_value)

    merged: JsonObject | None = None
    for path in discover_mcp_config_paths(config):
        current = _load_config_path(path)
        merged = current if merged is None else _merge_configs(merged, current)
    return merged


def _first_env_value(env: Mapping[str, str]) -> str | None:
    for key in _MCP_CONFIG_ENV_KEYS:
        value = env.get(key) or os.environ.get(key)
        if value:
            return value
    return None


def _load_env_config(value: str) -> JsonObject:
    stripped = value.strip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
        return _as_object(data, "MCP_CONFIG")
    return _load_config_path(Path(stripped))


def _load_config_path(path: Path) -> JsonObject:
    try:
        with path.expanduser().open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        msg = f"Invalid MCP config JSON in {path}: {exc.msg}"
        raise MCPConfigError(msg) from exc
    return _as_object(data, str(path))


def _merge_configs(low: JsonObject, high: JsonObject) -> JsonObject:
    _validate_config(low)
    _validate_config(high)
    return {"mcpServers": {**_servers(low), **_servers(high)}}


async def _load_server_tools(name: str, server: JsonObject) -> list[BaseTool]:
    client = MultiServerMCPClient(connections={name: _connection(server)})
    return list(await client.get_tools())


def _connection(server: JsonObject) -> Connection:
    transport = _resolve_transport(server)
    if transport in _REMOTE_TRANSPORTS:
        connection: JsonObject = {
            "transport": "streamable_http" if transport == "http" else "sse",
            "url": _required_str(server, "url"),
        }
        headers = server.get("headers")
        if headers is not None:
            connection["headers"] = _resolve_headers(_as_str_mapping(headers, "headers"))
        return cast("Connection", connection)

    connection: JsonObject = {
        "transport": "stdio",
        "command": _required_str(server, "command"),
    }
    args = server.get("args")
    if args is not None:
        connection["args"] = _as_str_list(args, "args")
    env = server.get("env")
    if env is not None:
        connection["env"] = _resolve_headers(_as_str_mapping(env, "env"))
    return cast("Connection", connection)


def _validate_config(data: JsonObject) -> None:
    servers = _servers(data)
    if not servers:
        msg = "MCP config must contain at least one server in `mcpServers`"
        raise MCPConfigError(msg)
    for name, raw in servers.items():
        server = _as_object(raw, f"mcpServers.{name}")
        _validate_server_name(name)
        _validate_server_config(name, server)


def _validate_server_name(name: str) -> None:
    if not _SERVER_NAME_RE.fullmatch(name):
        msg = "MCP server names must contain only letters, numbers, underscores, or hyphens"
        raise MCPConfigError(msg)


def _validate_server_config(name: str, server: JsonObject) -> None:
    transport = _resolve_transport(server)
    if transport in _REMOTE_TRANSPORTS:
        _required_str(server, "url")
    elif transport == "stdio":
        _required_str(server, "command")
    else:
        msg = f"MCP server {name!r} uses unsupported transport {transport!r}"
        raise MCPConfigError(msg)

    if "allowedTools" in server and "disabledTools" in server:
        msg = f"MCP server {name!r} cannot set both `allowedTools` and `disabledTools`"
        raise MCPConfigError(msg)
    for field in ("allowedTools", "disabledTools"):
        if field in server and not _as_str_list(server[field], field):
            msg = f"MCP server {name!r} `{field}` must be non-empty"
            raise MCPConfigError(msg)


def _resolve_transport(server: JsonObject) -> str:
    raw = server.get("type") or server.get("transport")
    if raw is not None:
        if not isinstance(raw, str):
            msg = "MCP server transport must be a string"
            raise MCPConfigError(msg)
        return _TRANSPORT_ALIASES.get(raw, raw)
    return "http" if "url" in server else "stdio"


def _apply_tool_filter(
    tools: Sequence[BaseTool],
    server_name: str,
    server: JsonObject,
) -> list[BaseTool]:
    allowed = cast("list[str] | None", server.get("allowedTools"))
    disabled = cast("list[str] | None", server.get("disabledTools"))
    entries = allowed if allowed is not None else disabled
    if entries is None:
        return list(tools)

    prefix = f"{server_name}_"

    def matches(tool: BaseTool) -> bool:
        return any(_entry_matches_tool(entry, tool.name, prefix) for entry in entries)

    if allowed is not None:
        return [tool for tool in tools if matches(tool)]
    return [tool for tool in tools if not matches(tool)]


def _entry_matches_tool(entry: str, tool: str, prefix: str) -> bool:
    if any(char in _GLOB_METACHARS for char in entry):
        return fnmatch.fnmatchcase(tool, entry) or (
            tool.startswith(prefix) and fnmatch.fnmatchcase(tool[len(prefix) :], entry)
        )
    return tool == entry or (tool.startswith(prefix) and tool[len(prefix) :] == entry)


def _resolve_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: _expand_env(value, key) for key, value in headers.items()}


def _expand_env(value: str, where: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        found = os.environ.get(name)
        if found is None:
            msg = f"{where} references unset environment variable {name}"
            raise MCPConfigError(msg)
        return found

    return _ENV_VAR_RE.sub(replace, value)


def _servers(data: JsonObject) -> JsonObject:
    servers = data.get("mcpServers")
    return _as_object(servers, "mcpServers")


def _as_object(value: object, where: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = f"{where} must be an object"
        raise MCPConfigError(msg)
    return cast("JsonObject", value)


def _as_str_mapping(value: object, where: str) -> dict[str, str]:
    data = _as_object(value, where)
    if not all(isinstance(item, str) for item in data.values()):
        msg = f"{where} must contain only string values"
        raise MCPConfigError(msg)
    return cast("dict[str, str]", data)


def _as_str_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"{where} must be a list of strings"
        raise MCPConfigError(msg)
    return cast("list[str]", value)


def _required_str(data: JsonObject, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        msg = f"MCP server field `{key}` is required"
        raise MCPConfigError(msg)
    return value


def _is_file(path: Path) -> bool:
    try:
        return path.expanduser().is_file()
    except OSError:
        logger.warning("Could not inspect MCP config path %s", path, exc_info=True)
        return False
