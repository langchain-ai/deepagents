"""MCP configuration and tool loading for Talon.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import copy
import fnmatch
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from httpx import HTTPError
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError

from deepagents_talon.mcp_auth import (
    FileTokenStorage,
    build_oauth_provider,
    format_login_error,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import Connection

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

_MCP_CONFIG_ENV = "DEEPAGENTS_TALON_MCP_CONFIG"
_MCP_LOAD_TIMEOUT_SECONDS = 30
_DEFAULT_MCP_CONFIG = Path(".deepagents/.mcp.json")
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^{}]*))?\}")
_SERVER_NAME = re.compile(r"[A-Za-z0-9_-]+")
_DANGEROUS_STDIO_ENV = frozenset(
    {
        "BASH_ENV",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "ENV",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "ZDOTDIR",
    }
)


class MCPConfigError(ValueError):
    """An MCP configuration is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class MCPToolInfo:
    """Metadata for one MCP tool."""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class MCPServerInfo:
    """Load status and tool metadata for one MCP server."""

    name: str
    transport: str
    tools: tuple[MCPToolInfo, ...] = ()
    status: Literal["ok", "error"] = "ok"
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MCPTools:
    """Loaded MCP tools and per-server load statuses."""

    tools: Sequence[BaseTool]
    servers: Sequence[MCPServerInfo]


def mcp_config_path(config: TalonConfig) -> Path:
    """Return the configured MCP path or Talon's standard user path."""
    configured = config.env.get(_MCP_CONFIG_ENV) or os.environ.get(_MCP_CONFIG_ENV)
    return Path(configured).expanduser() if configured else Path.home() / _DEFAULT_MCP_CONFIG


async def load_mcp_tools(config: TalonConfig) -> MCPTools:
    """Load configured MCP tools for a Talon runtime."""
    path = mcp_config_path(config)
    if not _is_file(path):
        return MCPTools(tools=(), servers=())
    servers = _load_config(path, config.env)

    tools: list[BaseTool] = []
    infos: list[MCPServerInfo] = []
    for name, server in servers.items():
        transport = _transport_label(server)
        try:
            connection, transport = await _connection(name, server)
            client = MultiServerMCPClient(
                {name: connection}, tool_name_prefix=True, handle_tool_errors=True
            )
            loaded = await asyncio.wait_for(
                client.get_tools(server_name=name),
                timeout=_MCP_LOAD_TIMEOUT_SECONDS,
            )
            loaded = _filter_tools(name, server, loaded)
        except (HTTPError, McpError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            logger.warning("MCP server %s failed to load: %s", name, exc)
            infos.append(
                MCPServerInfo(
                    name=name,
                    transport=transport,
                    status="error",
                    error=str(exc),
                )
            )
            continue
        tools.extend(loaded)
        infos.append(
            MCPServerInfo(
                name=name,
                transport=transport,
                tools=tuple(
                    MCPToolInfo(name=tool.name, description=tool.description or "")
                    for tool in loaded
                ),
            )
        )
    tools.sort(key=lambda tool: tool.name)
    return MCPTools(tools=tuple(tools), servers=tuple(infos))


def print_mcp_config_paths(config: TalonConfig) -> None:
    """Print the MCP config path used by Talon."""
    path = mcp_config_path(config)
    marker = "found" if _is_file(path) else "missing"
    print(f"MCP config: [{marker}] {path}")  # noqa: T201
    print(f"Override with {_MCP_CONFIG_ENV}.")  # noqa: T201


async def login_mcp_server(
    config: TalonConfig, server_name: str, config_path: str | None = None
) -> int:
    """Authenticate one configured remote MCP server."""
    path = Path(config_path) if config_path else mcp_config_path(config)
    try:
        path = await asyncio.to_thread(path.expanduser)
        servers = _load_config(path, config.env)
        server = servers.get(server_name)
        if server is None:
            msg = f"MCP server {server_name!r} was not found in {path}"
            raise MCPConfigError(msg)
        connection, transport = await _connection(server_name, server, interactive=True)
        if transport not in {"sse", "streamable_http"}:
            msg = f"MCP server {server_name!r} does not use a remote transport"
            raise MCPConfigError(msg)
        client = MultiServerMCPClient({server_name: connection})
        await asyncio.wait_for(
            _open_mcp_session(client, server_name), timeout=_MCP_LOAD_TIMEOUT_SECONDS
        )
    except (HTTPError, McpError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        print(f"MCP login failed: {format_login_error(exc)}", file=sys.stderr)  # noqa: T201
        return 1
    print(f"Logged in to MCP server {server_name!r}.")  # noqa: T201
    return 0


async def _open_mcp_session(client: MultiServerMCPClient, server_name: str) -> None:
    async with client.session(server_name):
        pass


def _load_config(path: Path, env: Mapping[str, str]) -> dict[str, dict[str, object]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Could not load MCP config {path}: {exc}"
        raise MCPConfigError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"MCP config {path} must contain a JSON object"
        raise MCPConfigError(msg)
    raw_servers = raw.get("mcpServers")
    if not isinstance(raw_servers, dict):
        msg = f"MCP config {path} must contain an mcpServers object"
        raise MCPConfigError(msg)

    servers: dict[str, dict[str, object]] = {}
    for name, value in raw_servers.items():
        if not isinstance(name, str) or _SERVER_NAME.fullmatch(name) is None:
            msg = f"MCP server name {name!r} must contain only letters, numbers, _ or -"
            raise MCPConfigError(msg)
        if not isinstance(value, dict):
            msg = f"MCP server {name!r} must be an object"
            raise MCPConfigError(msg)
        servers[name] = _resolve_server_env(name, value, env)
    return servers


def _resolve_server_env(
    name: str, server: Mapping[str, object], env: Mapping[str, str]
) -> dict[str, object]:
    resolved = copy.deepcopy(dict(server))
    for field in ("command", "url"):
        if field in resolved:
            resolved[field] = _resolve_string(resolved[field], f"{name}.{field}", env)
    if "args" in resolved:
        args = resolved["args"]
        if not isinstance(args, list):
            msg = f"MCP server {name!r} args must be a list"
            raise MCPConfigError(msg)
        resolved["args"] = [
            _resolve_string(value, f"{name}.args[{index}]", env) for index, value in enumerate(args)
        ]
    for field in ("env", "headers"):
        if field not in resolved:
            continue
        values = resolved[field]
        if not isinstance(values, dict):
            msg = f"MCP server {name!r} {field} must be an object"
            raise MCPConfigError(msg)
        if not all(isinstance(key, str) for key in values):
            msg = f"MCP server {name!r} {field} keys must be strings"
            raise MCPConfigError(msg)
        resolved[field] = {
            key: _resolve_string(value, f"{name}.{field}.{key}", env)
            for key, value in values.items()
        }
    return resolved


def _resolve_string(value: object, field: str, env: Mapping[str, str]) -> str:
    if not isinstance(value, str):
        msg = f"MCP config field {field} must be a string"
        raise MCPConfigError(msg)

    def replace(match: re.Match[str]) -> str:
        key, default = match.group(1), match.group(2)
        selected = env.get(key, os.environ.get(key))
        if selected:
            return selected
        if default is not None:
            return default
        if selected is not None:
            return selected
        msg = f"MCP config field {field} references unset env var {key}"
        raise MCPConfigError(msg)

    result = _ENV_REF.sub(replace, value)
    if "${" in result:
        msg = f"MCP config field {field} contains a malformed environment reference"
        raise MCPConfigError(msg)
    return result


async def _connection(
    name: str, server: Mapping[str, object], *, interactive: bool = False
) -> tuple[Connection, str]:
    raw_transport = server.get("transport", server.get("type"))
    if raw_transport is None:
        raw_transport = "stdio" if "command" in server else "http"
    if not isinstance(raw_transport, str):
        msg = f"MCP server {name!r} transport must be a string"
        raise MCPConfigError(msg)
    transport = raw_transport.replace("-", "_")
    if transport == "http":
        transport = "streamable_http"
    if transport == "stdio":
        return _stdio_connection(name, server), transport
    if transport in {"sse", "streamable_http"}:
        return await _remote_connection(name, server, transport, interactive=interactive), transport
    msg = f"MCP server {name!r} uses unsupported transport {raw_transport!r}"
    raise MCPConfigError(msg)


def _transport_label(server: Mapping[str, object]) -> str:
    raw = server.get("transport", server.get("type"))
    if raw is None:
        raw = "stdio" if "command" in server else "http"
    return raw.replace("-", "_") if isinstance(raw, str) else "unknown"


def _stdio_connection(name: str, server: Mapping[str, object]) -> Connection:
    command = server.get("command")
    args = server.get("args", [])
    values = server.get("env")
    if not isinstance(command, str) or not command:
        msg = f"MCP stdio server {name!r} requires command"
        raise MCPConfigError(msg)
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        msg = f"MCP stdio server {name!r} args must contain strings"
        raise MCPConfigError(msg)
    if values is not None:
        _validate_stdio_env(name, values)
    connection: dict[str, object] = {
        "transport": "stdio",
        "command": command,
        "args": args,
    }
    if values is not None:
        connection["env"] = values
    return cast("Connection", connection)


def _validate_stdio_env(name: str, values: object) -> None:
    if not isinstance(values, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in values.items()
    ):
        msg = f"MCP stdio server {name!r} env must contain strings"
        raise MCPConfigError(msg)
    blocked = _DANGEROUS_STDIO_ENV.intersection(values)
    if blocked:
        msg = f"MCP stdio server {name!r} cannot set {', '.join(sorted(blocked))}"
        raise MCPConfigError(msg)


async def _remote_connection(
    name: str, server: Mapping[str, object], transport: str, *, interactive: bool = False
) -> Connection:
    url = server.get("url")
    headers = server.get("headers")
    if not isinstance(url, str) or not url:
        msg = f"MCP remote server {name!r} requires url"
        raise MCPConfigError(msg)
    if headers is not None and (
        not isinstance(headers, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
        )
    ):
        msg = f"MCP remote server {name!r} headers must contain strings"
        raise MCPConfigError(msg)
    connection: dict[str, object] = {"transport": transport, "url": url, "timeout": 30.0}
    if headers is not None:
        connection["headers"] = headers
    if server.get("auth") == "oauth" or interactive:
        if isinstance(headers, dict) and any(
            isinstance(key, str) and key.lower() == "authorization" for key in headers
        ):
            msg = f"MCP server {name!r} cannot combine OAuth with an Authorization header"
            raise MCPConfigError(msg)
        storage = FileTokenStorage(name, server_url=url)
        if not interactive and await storage.get_tokens() is None:
            msg = f"MCP server {name!r} needs authentication; run deepagents-talon mcp login {name}"
            raise MCPConfigError(msg)
        connection["auth"] = build_oauth_provider(
            server_name=name,
            server_url=url,
            storage=storage,
            interactive=interactive,
        )
    elif server.get("auth") is not None:
        msg = f"MCP server {name!r} uses unsupported auth {server['auth']!r}"
        raise MCPConfigError(msg)
    return cast("Connection", connection)


def _filter_tools(
    server_name: str, server: Mapping[str, object], tools: Sequence[BaseTool]
) -> Sequence[BaseTool]:
    allowed = _tool_filter(server_name, server, "allowedTools")
    disabled = _tool_filter(server_name, server, "disabledTools")
    if allowed is not None and disabled is not None:
        msg = f"MCP server {server_name!r} cannot set both allowedTools and disabledTools"
        raise MCPConfigError(msg)
    entries = allowed if allowed is not None else disabled
    if entries is None:
        return tools

    prefix = f"{server_name}_"

    def matches(tool: BaseTool) -> bool:
        names = (tool.name, tool.name.removeprefix(prefix))
        return any(fnmatch.fnmatchcase(name, entry) for entry in entries for name in names)

    if allowed is not None:
        return [tool for tool in tools if matches(tool)]
    return [tool for tool in tools if not matches(tool)]


def _tool_filter(server_name: str, server: Mapping[str, object], field: str) -> list[str] | None:
    value = server.get(field)
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        msg = f"MCP server {server_name!r} {field} must be a non-empty list of strings"
        raise MCPConfigError(msg)
    return cast("list[str]", value)


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        logger.warning("Could not inspect MCP config path %s", path, exc_info=True)
        return False
