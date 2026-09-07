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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

from httpx import HTTPError
from langchain_core.tools import InjectedToolCallId, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.auth import OAuthFlowError
from mcp.shared.exceptions import McpError

from deepagents_talon.authorization import (
    AuthorizationAttempt,
    AuthorizationCompleted,
    AuthorizationFailed,
    AuthorizationFailureReason,
    current_authorization_handler,
    reset_authorization_attempt,
    reset_authorization_invocation,
    set_authorization_attempt,
    set_authorization_invocation,
)
from deepagents_talon.mcp_auth import (
    DeviceAuthorizationCompletedError,
    FileTokenStorage,
    MCPAuthorizationError,
    build_oauth_provider,
    format_login_error,
    prepare_oauth_login,
)
from deepagents_talon.mcp_config import (
    MCPConfigStore,
    agent_workspace_root,
    auto_approve_enabled,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence

    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import Connection
    from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult

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


class _MCPLoginRequiredError(MCPConfigError):
    """An MCP server requires OAuth login before loading tools."""


def _exception_leaves(exc: BaseException) -> Iterator[BaseException]:
    """Yield `exc`, or every leaf of it when it is a possibly nested group.

    Anyio task groups wrap even a single exception, so any failure raised inside
    an MCP `ClientSession` reaches callers as a group and must be matched on its
    leaves rather than on the group type.

    Args:
        exc: Exception to flatten.

    Yields:
        Each non-group exception, in the order the group holds them.
    """
    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            yield from _exception_leaves(nested)
    else:
        yield exc


def _authentication_required(exc: BaseException) -> bool:
    return any(
        isinstance(leaf, (_MCPLoginRequiredError, MCPAuthorizationError))
        for leaf in _exception_leaves(exc)
    )


def _device_authorization_completed(exc: BaseException) -> bool:
    return any(
        isinstance(leaf, DeviceAuthorizationCompletedError) for leaf in _exception_leaves(exc)
    )


@dataclass(frozen=True, slots=True)
class MCPToolInfo:
    """Metadata for one MCP tool."""

    name: str
    description: str
    input_schema: dict[str, object] | None = None


MCPServerStatus = Literal[
    "ok",
    "unauthenticated",
    "awaiting_reconnect",
    "error",
    "disabled",
]


@dataclass(frozen=True, slots=True)
class MCPServerInfo:
    """Load status and tool metadata for one MCP server."""

    name: str
    transport: str
    tools: tuple[MCPToolInfo, ...] = ()
    status: MCPServerStatus = "ok"
    error: str | None = None
    pending_reconnect: bool = False
    uses_oauth: bool = False

    def __post_init__(self) -> None:
        """Enforce status, error, tool, and reconnect consistency."""
        if self.status == "ok":
            if self.error is not None:
                msg = f"MCPServerInfo {self.name!r}: status='ok' cannot carry an error"
                raise ValueError(msg)
        else:
            if self.error is None:
                msg = (
                    f"MCPServerInfo {self.name!r}: status={self.status!r} requires an error message"
                )
                raise ValueError(msg)
            if self.tools:
                msg = f"MCPServerInfo {self.name!r}: status={self.status!r} cannot carry tools"
                raise ValueError(msg)
        if self.pending_reconnect and self.status != "disabled":
            msg = f"MCPServerInfo {self.name!r}: pending_reconnect requires status='disabled'"
            raise ValueError(msg)

    def needs_attention(self) -> bool:
        """Return whether this server is blocked on user login."""
        return self.status == "unauthenticated"


@dataclass(frozen=True, slots=True)
class MCPTools:
    """Loaded MCP tools and per-server load statuses."""

    tools: Sequence[BaseTool]
    servers: Sequence[MCPServerInfo]


class MCPToolProvider:
    """Load MCP tools and refresh them when their availability changes."""

    def __init__(self, config: TalonConfig) -> None:
        """Bind the provider to one Talon configuration."""
        self._config = config
        self._oauth_servers: frozenset[str] = frozenset()
        self._refresh_revision = 0
        self._applied_revision = 0
        self._lock = asyncio.Lock()
        self._config_store = MCPConfigStore(
            mcp_config_path(config),
            self.request_refresh,
            agent_root=agent_workspace_root(config.env),
            auto_approve=auto_approve_enabled(config.env),
        )

    async def load(self) -> MCPTools:
        """Load tools and include the narrow proactive authorization capability."""
        loaded = await load_mcp_tools(self._config)
        self._oauth_servers = frozenset(
            server.name for server in loaded.servers if server.uses_oauth
        )
        tools = tuple(loaded.tools)
        if loaded.servers:
            tools = (*tools, self._status_tool(loaded.servers))
        if self._oauth_servers:
            tools = (*tools, self._authorization_tool())
        tools = (*tools, self._reload_tool(), *self._config_store.tools())
        if len({tool.name for tool in tools}) != len(tools):
            msg = "MCP tool names conflict with Talon management tools"
            raise MCPConfigError(msg)
        return MCPTools(tools=tools, servers=loaded.servers)

    async def refresh_if_needed(self) -> Sequence[BaseTool] | None:
        """Reload MCP schemas once after their availability changes."""
        return await self._reload(force=False)

    async def reload(self) -> Sequence[BaseTool]:
        """Reload MCP tools from the configured path."""
        refreshed = await self._reload(force=True)
        if refreshed is None:
            msg = "forced MCP reload did not produce tools"
            raise RuntimeError(msg)
        return refreshed

    def request_refresh(self) -> None:
        """Schedule an MCP configuration reload before the next agent turn."""
        self._refresh_revision += 1

    async def _reload(self, *, force: bool) -> Sequence[BaseTool] | None:
        async with self._lock:
            revision = self._refresh_revision
            if not force and revision == self._applied_revision:
                return None
            try:
                tools = (await self.load()).tools
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._applied_revision = revision
                raise
            self._applied_revision = revision
            return tools

    def _reload_tool(self) -> BaseTool:
        @tool(
            "reload_mcp_configuration",
            description=(
                "Reload Talon's configured MCP servers before the next agent turn. "
                "Use after the operator changes the MCP configuration. Verify activation "
                "with get_agent_tools; running tasks retain their original capabilities."
            ),
        )
        def reload_mcp_configuration() -> dict[str, str]:
            self.request_refresh()
            return {"status": "scheduled", "available": "after_successful_reload"}

        return reload_mcp_configuration

    def _status_tool(self, servers: Sequence[MCPServerInfo]) -> BaseTool:
        statuses: tuple[dict[str, object], ...] = tuple(
            {
                "server_name": server.name,
                "status": server.status,
                "can_authenticate": server.uses_oauth,
            }
            for server in servers
        )
        summary = (
            ", ".join(f"{status['server_name']} ({status['status']})" for status in statuses)
            or "none configured"
        )

        @tool(
            "get_mcp_server_status",
            description=f"Report configured MCP server availability. Current servers: {summary}.",
        )
        def get_mcp_server_status() -> tuple[dict[str, object], ...]:
            return statuses

        return get_mcp_server_status

    def _authorization_tool(self) -> BaseTool:
        @tool(
            "authenticate_mcp_server",
            description=(
                "Authenticate a configured OAuth MCP server through the current Talon channel. "
                "Use only the configured server name; authorization links are handled by Talon. "
                "If current credentials work, returns already_authenticated without starting "
                "a new authorization flow. Set reauthenticate only when the user explicitly "
                "asks to log in again or switch accounts."
            ),
        )
        async def authenticate_mcp_server(
            server_name: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
            *,
            reauthenticate: Annotated[
                bool,
                "Set true only when the user explicitly asks to log in again or switch accounts.",
            ] = False,
        ) -> dict[str, str]:
            return await self._authenticate(
                server_name,
                tool_call_id,
                reauthenticate=reauthenticate,
            )

        return authenticate_mcp_server

    async def _authenticate(
        self,
        server_name: str,
        invocation_id: str,
        *,
        reauthenticate: bool = False,
    ) -> dict[str, str]:
        if server_name not in self._oauth_servers:
            return {"status": "failed"}
        attempt = AuthorizationAttempt(terminal=True)
        try:
            path = mcp_config_path(self._config)
            servers = _load_config(path, self._config.env)
            server = servers.get(server_name)
            if server is None or server.get("auth") != "oauth":
                return {"status": "failed", "server_name": server_name}

            opened = await _run_authorized(
                invocation_id,
                lambda: _open_authenticated_session(
                    server_name,
                    server,
                    force_authorization=reauthenticate,
                ),
                attempt=attempt,
            )
            if not opened:
                return {"status": "failed", "server_name": server_name}
        except asyncio.CancelledError:
            if attempt.completed:
                self._refresh_revision += 1
            raise
        except (
            ExceptionGroup,
            HTTPError,
            McpError,
            OAuthFlowError,
            OSError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValueError,
        ):
            if not attempt.completed:
                return {"status": "failed", "server_name": server_name}
            logger.debug(
                "MCP authorization session failed after credentials persisted",
                exc_info=True,
            )
        except Exception:
            if not attempt.completed:
                raise
            logger.debug(
                "MCP authorization session failed after credentials persisted",
                exc_info=True,
            )
        status = (
            "completed"
            if attempt.completed
            else "already_authenticated"
            if attempt.binding is None
            else "failed"
        )
        self._refresh_revision += int(attempt.completed)
        return {"status": status, "server_name": server_name}


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
        input_schemas: dict[str, dict[str, object]] = {}
        try:
            connection, transport = await _connection(name, server)
            client = MultiServerMCPClient(
                {name: connection},
                tool_interceptors=[
                    _authorization_interceptor,
                    partial(
                        _argument_normalization_interceptor,
                        input_schemas=input_schemas,
                    ),
                ],
                tool_name_prefix=True,
                handle_tool_errors=True,
            )
            loaded = await asyncio.wait_for(
                client.get_tools(server_name=name),
                timeout=_MCP_LOAD_TIMEOUT_SECONDS,
            )
            loaded = _filter_tools(name, server, loaded)
            prefix = f"{name}_"
            input_schemas.update(
                {
                    tool.name.removeprefix(prefix): tool.args_schema
                    for tool in loaded
                    if isinstance(tool.args_schema, dict)
                }
            )
        except (
            ExceptionGroup,
            HTTPError,
            McpError,
            OAuthFlowError,
            OSError,
            RuntimeError,
            TimeoutError,
            ValueError,
        ) as exc:
            authentication_required = _authentication_required(exc)
            error = (
                f"MCP server {name!r} needs authentication"
                if authentication_required
                else format_login_error(exc)
                if isinstance(exc, ExceptionGroup)
                else str(exc)
            )
            logger.warning("MCP server %s failed to load: %s", name, error)
            infos.append(
                MCPServerInfo(
                    name=name,
                    transport=transport,
                    status="unauthenticated" if authentication_required else "error",
                    error=error,
                    uses_oauth=server.get("auth") == "oauth",
                )
            )
            continue
        tools.extend(loaded)
        infos.append(
            MCPServerInfo(
                name=name,
                transport=transport,
                tools=tuple(
                    MCPToolInfo(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=copy.deepcopy(tool.args_schema)
                        if isinstance(tool.args_schema, dict)
                        else None,
                    )
                    for tool in loaded
                ),
                uses_oauth=server.get("auth") == "oauth",
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
        connection, transport = await _connection(
            server_name,
            server,
            interactive=True,
            force_authorization=True,
        )
        if transport not in {"sse", "streamable_http"}:
            msg = f"MCP server {server_name!r} does not use a remote transport"
            raise MCPConfigError(msg)
        client = MultiServerMCPClient({server_name: connection})
        await _open_mcp_session(client, server_name)
    except DeviceAuthorizationCompletedError:
        pass
    except ExceptionGroup as group:
        # A completed device flow raises from inside the session's task group,
        # which wraps it; only a group holding some other failure is an error.
        if not _device_authorization_completed(group):
            print(f"MCP login failed: {format_login_error(group)}", file=sys.stderr)  # noqa: T201
            return 1
    except (
        HTTPError,
        McpError,
        OAuthFlowError,
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"MCP login failed: {format_login_error(exc)}", file=sys.stderr)  # noqa: T201
        return 1
    print(f"Logged in to MCP server {server_name!r}.")  # noqa: T201
    return 0


async def _open_mcp_session(client: MultiServerMCPClient, server_name: str) -> None:
    async with client.session(server_name):
        pass


async def _open_authenticated_session(
    server_name: str,
    server: Mapping[str, object],
    *,
    force_authorization: bool,
) -> bool:
    connection, transport = await _connection(
        server_name,
        server,
        channel_authorization=True,
        force_authorization=force_authorization,
    )
    if transport not in {"sse", "streamable_http"}:
        return False
    client = MultiServerMCPClient({server_name: connection})
    await _open_mcp_session(client, server_name)
    return True


async def _authorization_interceptor(
    request: MCPToolCallRequest,
    handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
) -> MCPToolCallResult:
    """Bind OAuth prompts to the exact LangGraph MCP tool invocation."""
    invocation_id = getattr(request.runtime, "tool_call_id", None)
    normalized_id = invocation_id if isinstance(invocation_id, str) and invocation_id else None
    return await _run_authorized(normalized_id, lambda: handler(request))


async def _argument_normalization_interceptor(
    request: MCPToolCallRequest,
    handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    *,
    input_schemas: Mapping[str, dict[str, object]],
) -> MCPToolCallResult:
    schema = input_schemas.get(request.name)
    arguments = _normalize_mcp_arguments(request.args, schema)
    normalized = request if arguments == request.args else request.override(args=arguments)
    return await handler(normalized)


def _normalize_mcp_arguments(
    arguments: Mapping[str, object], input_schema: object
) -> dict[str, object]:
    """Omit empty strings for optional string-like MCP arguments."""
    if not isinstance(input_schema, dict):
        return dict(arguments)
    raw_required = input_schema.get("required")
    required = (
        {item for item in raw_required if isinstance(item, str)}
        if isinstance(raw_required, list)
        else set()
    )
    raw_properties = input_schema.get("properties")
    properties = raw_properties if isinstance(raw_properties, dict) else {}
    return {
        name: value
        for name, value in arguments.items()
        if value != "" or name in required or _is_explicitly_non_string(properties.get(name))
    }


def _is_explicitly_non_string(property_schema: object) -> bool:
    if not isinstance(property_schema, dict):
        return False
    property_type = property_schema.get("type")
    if property_type is None or property_type == "string":
        return False
    return not (isinstance(property_type, list) and "string" in property_type)


async def _run_authorized[AuthorizedResult](
    invocation_id: str | None,
    operation: Callable[[], Awaitable[AuthorizedResult]],
    *,
    attempt: AuthorizationAttempt | None = None,
) -> AuthorizedResult:
    attempt = attempt or AuthorizationAttempt()
    invocation_token = set_authorization_invocation(invocation_id)
    attempt_token = set_authorization_attempt(attempt)
    try:
        result = await operation()
    except asyncio.CancelledError:
        await _finish_authorization(
            attempt,
            reason=None if attempt.completed else "cancelled",
        )
        raise
    except Exception as exc:
        if attempt.completed:
            await _finish_authorization(attempt)
            raise
        if attempt.binding is not None:
            await _finish_authorization(attempt, reason=_authorization_failure_reason(exc))
            msg = "MCP authorization failed"
            raise MCPAuthorizationError(msg) from None
        raise
    else:
        await _finish_authorization(attempt)
        return result
    finally:
        reset_authorization_attempt(attempt_token)
        reset_authorization_invocation(invocation_token)


async def _finish_authorization(
    attempt: AuthorizationAttempt,
    *,
    reason: AuthorizationFailureReason | None = None,
) -> None:
    binding = attempt.binding
    handler = current_authorization_handler()
    if binding is None or handler is None:
        return
    event = (
        AuthorizationCompleted(binding=binding, terminal=attempt.terminal)
        if attempt.completed and reason is None
        else AuthorizationFailed(binding=binding, reason=reason or "error")
    )
    try:
        await handler(event)
    except Exception:  # noqa: BLE001  # status delivery cannot expose or undo OAuth state.
        return


def _authorization_failure_reason(exc: Exception) -> AuthorizationFailureReason:
    leaves = tuple(_exception_leaves(exc))
    if any(isinstance(leaf, TimeoutError) for leaf in leaves):
        return "expired"
    if any(isinstance(leaf, MCPAuthorizationError) for leaf in leaves):
        return "invalid_callback"
    return "error"


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
    name: str,
    server: Mapping[str, object],
    *,
    interactive: bool = False,
    channel_authorization: bool = False,
    force_authorization: bool = False,
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
        return (
            await _remote_connection(
                name,
                server,
                transport,
                interactive=interactive,
                channel_authorization=channel_authorization,
                force_authorization=force_authorization,
            ),
            transport,
        )
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


async def _remote_connection(  # noqa: PLR0913  # keeps distinct OAuth modes explicit.
    name: str,
    server: Mapping[str, object],
    transport: str,
    *,
    interactive: bool = False,
    channel_authorization: bool = False,
    force_authorization: bool = False,
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
        storage = (
            FileTokenStorage(name, server_url=url, force_authorization=True)
            if force_authorization
            else FileTokenStorage(name, server_url=url)
        )
        if not interactive and not channel_authorization and await storage.get_tokens() is None:
            msg = f"MCP server {name!r} needs authentication; run deepagents-talon mcp login {name}"
            raise _MCPLoginRequiredError(msg)
        await prepare_oauth_login(server_url=url, storage=storage)
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
