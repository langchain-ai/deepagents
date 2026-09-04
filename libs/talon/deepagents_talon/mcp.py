"""MCP configuration and tool loading for Talon.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from httpx import HTTPError
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import Connection

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

_MCP_CONFIG_ENV_KEYS = ("DEEPAGENTS_TALON_MCP_CONFIG", "MCP_CONFIG")
_TRUST_PROJECT_MCP_ENV = "DEEPAGENTS_TALON_TRUST_PROJECT_MCP"
_WORKSPACE_ENV = "DEEPAGENTS_TALON_WORKSPACE"
_MCP_LOAD_TIMEOUT_SECONDS = 30
_MCP_CONFIG_DISCOVERY_PATHS = (
    ("~/.deepagents/.mcp.json", "user-level"),
    ("<project-root>/.deepagents/.mcp.json", "project subdir"),
    ("<project-root>/.mcp.json", "project root"),
)
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


def discover_mcp_config_paths(config: TalonConfig) -> list[Path]:
    """Return existing MCP config files from lowest to highest precedence."""
    return [path for path in _mcp_config_paths(config) if _is_file(path)]


async def load_mcp_tools(config: TalonConfig) -> MCPTools:
    """Load configured MCP tools for a Talon runtime."""
    explicit = _first_env_value(config.env)
    if explicit is not None:
        paths = [_explicit_path(explicit)]
    else:
        discovered = discover_mcp_config_paths(config)
        paths = [path for path in discovered if _is_user_config(path)]
        project = [path for path in discovered if not _is_user_config(path)]
        if project and _env_enabled(config.env, _TRUST_PROJECT_MCP_ENV):
            paths.extend(project)
        elif project:
            logger.warning(
                "Ignoring project MCP config; set %s=true to trust it",
                _TRUST_PROJECT_MCP_ENV,
            )
    if not paths:
        return MCPTools(tools=(), servers=())

    servers: dict[str, dict[str, object]] = {}
    for path in paths:
        servers.update(_load_config(path, config.env))

    tools: list[BaseTool] = []
    infos: list[MCPServerInfo] = []
    for name, server in servers.items():
        connection, transport = _connection(name, server)
        try:
            client = MultiServerMCPClient(
                {name: connection}, tool_name_prefix=True, handle_tool_errors=True
            )
            loaded = await asyncio.wait_for(
                client.get_tools(server_name=name),
                timeout=_MCP_LOAD_TIMEOUT_SECONDS,
            )
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
    """Print Talon MCP config discovery paths."""
    found = {path.resolve() for path in discover_mcp_config_paths(config)}
    rows = [
        (display, label, path)
        for (display, label), path in zip(
            _MCP_CONFIG_DISCOVERY_PATHS, _mcp_config_paths(config), strict=True
        )
    ]
    width = max(len(display) for display, _, _ in rows)
    print("MCP config discovery paths (lowest to highest precedence):")  # noqa: T201
    for display, label, path in rows:
        marker = "found" if path.resolve() in found else "missing"
        print(f"  [{marker:>7}]  {display:<{width}}  ({label})")  # noqa: T201
    print()  # noqa: T201
    print("<project-root> = nearest ancestor with `.git`, else the workspace.")  # noqa: T201
    print(  # noqa: T201
        f"Project configs require {_TRUST_PROJECT_MCP_ENV}=true unless selected explicitly."
    )


def _explicit_path(value: str) -> Path:
    return Path(value).expanduser()


def _mcp_config_paths(config: TalonConfig) -> tuple[Path, Path, Path]:
    workspace = Path(config.env.get(_WORKSPACE_ENV, Path.cwd())).expanduser().resolve()
    project_root = next(
        (path for path in (workspace, *workspace.parents) if (path / ".git").is_dir()),
        workspace,
    )
    return (
        Path.home() / ".deepagents" / ".mcp.json",
        project_root / ".deepagents" / ".mcp.json",
        project_root / ".mcp.json",
    )


def _is_user_config(path: Path) -> bool:
    return path.resolve() == (Path.home() / ".deepagents" / ".mcp.json").resolve()


def _first_env_value(env: Mapping[str, str]) -> str | None:
    for key in _MCP_CONFIG_ENV_KEYS:
        value = env.get(key) or os.environ.get(key)
        if value:
            return value
    return None


def _env_enabled(env: Mapping[str, str], key: str) -> bool:
    return (env.get(key) or os.environ.get(key, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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


def _connection(name: str, server: Mapping[str, object]) -> tuple[Connection, str]:
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
        return _remote_connection(name, server, transport), transport
    msg = f"MCP server {name!r} uses unsupported transport {raw_transport!r}"
    raise MCPConfigError(msg)


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


def _remote_connection(name: str, server: Mapping[str, object], transport: str) -> Connection:
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
    return cast("Connection", connection)


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        logger.warning("Could not inspect MCP config path %s", path, exc_info=True)
        return False
