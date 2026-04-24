"""MCP (Model Context Protocol) tools loader for deepagents CLI.

This module provides async functions to load and manage MCP servers using
`langchain-mcp-adapters`, supporting Claude Desktop style JSON configs.
It also supports automatic discovery of `.mcp.json` files from user-level
and project-level locations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import Connection
    from mcp import ClientSession

    from deepagents_cli.project_utils import ProjectContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MCPToolInfo:
    """Metadata for a single MCP tool."""

    name: str
    """Tool name (may include server name prefix)."""

    description: str
    """Human-readable description of what the tool does."""


MCPServerStatus = Literal["ok", "unauthenticated", "error"]
"""Load states a configured MCP server can end up in."""


@dataclass(frozen=True, slots=True)
class MCPServerInfo:
    """Metadata for a configured MCP server and its tools."""

    name: str
    """Server name from the MCP configuration."""

    transport: str
    """Transport identifier — `stdio`, `sse`, `http`, or the synthetic
    `config` value used for entries surfacing a bad config file."""

    tools: tuple[MCPToolInfo, ...] = ()
    """Tools exposed by this server (empty when `status != "ok"`)."""

    status: MCPServerStatus = "ok"
    """Load status — `ok`, `unauthenticated`, or `error`."""

    error: str | None = None
    """Human-readable reason when `status != "ok"`."""

    def __post_init__(self) -> None:
        """Enforce the status/error/tools consistency invariant.

        Raises:
            ValueError: If any of: `status='ok'` with a non-`None` error;
                non-`ok` status without an error message; non-`ok` status
                carrying tools.
        """
        if self.status == "ok":
            if self.error is not None:
                msg = (
                    f"MCPServerInfo {self.name!r}: status='ok' cannot carry "
                    f"an error (got {self.error!r})"
                )
                raise ValueError(msg)
        else:
            if self.error is None:
                msg = (
                    f"MCPServerInfo {self.name!r}: status={self.status!r} "
                    "requires an error message"
                )
                raise ValueError(msg)
            if self.tools:
                msg = (
                    f"MCPServerInfo {self.name!r}: status={self.status!r} "
                    "cannot carry tools"
                )
                raise ValueError(msg)


_SUPPORTED_REMOTE_TYPES = {"sse", "http"}
"""Supported transport types for remote MCP servers (SSE and HTTP)."""

_TRANSPORT_ALIASES = {"streamable_http": "http", "streamable-http": "http"}
"""Aliases that normalize to canonical transport names.

The MCP spec and `langchain_mcp_adapters` use `streamable_http` for what the
CLI calls `http`. Accept both so users copy-pasting from upstream docs don't
hit a validation error.
"""


_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
"""Server names become token-file basenames and must remain path-safe."""


class MCPConfigError(ValueError):
    """An MCP configuration file is malformed or structurally invalid.

    Subclasses `ValueError` so existing `except ValueError` handlers
    keep working; new code can catch this specifically to render a
    user-actionable message (typically with a file path and hint).
    """


def _is_transient_session_error(exc: BaseException) -> bool:
    """Return `True` when `exc` signals the MCP session transport is dead.

    The anyio import is guarded so an anyio rename or removal surfaces as
    an `ImportError` at module import rather than silent mis-classification
    at runtime. Standard-library socket/pipe/EOF errors are covered as a
    fallback regardless of anyio's presence.
    """
    try:
        import anyio
    except ImportError:  # pragma: no cover - anyio is a transitive MCP dep
        anyio_excs: tuple[type[BaseException], ...] = ()
    else:
        anyio_excs = (
            anyio.ClosedResourceError,
            anyio.BrokenResourceError,
            anyio.EndOfStream,
        )
    return isinstance(
        exc,
        (
            *anyio_excs,
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
            EOFError,
            asyncio.IncompleteReadError,
        ),
    )


@dataclass(frozen=True, slots=True)
class _MCPSessionEntry:
    """Cached MCP session and its close stack."""

    session: ClientSession
    exit_stack: AsyncExitStack


def _connection_signature(value: Any) -> Any:  # noqa: ANN401
    """Return a stable comparison signature for MCP connection configs."""
    from mcp.client.auth import OAuthClientProvider

    if isinstance(value, dict):
        return tuple(
            sorted((key, _connection_signature(item)) for key, item in value.items())
        )
    if isinstance(value, list | tuple):
        return tuple(_connection_signature(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, OAuthClientProvider):
        context = value.context
        storage_path = getattr(getattr(context, "storage", None), "path", None)
        return (
            "oauth",
            _connection_signature(context.server_url),
            _connection_signature(
                context.client_metadata.model_dump(mode="json", exclude_none=True)
            ),
            _connection_signature(storage_path),
            _connection_signature(context.timeout),
            _connection_signature(context.client_metadata_url),
            _connection_signature(context.auth_server_url),
            _connection_signature(context.protocol_version),
        )

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _connection_signature(model_dump(mode="json", exclude_none=True))
    return value


def _connections_signature(
    connections: dict[str, Connection],
) -> tuple[tuple[str, Any], ...]:
    """Return a stable signature for a full MCP connections mapping."""
    return tuple(
        sorted(
            (name, _connection_signature(connection))
            for name, connection in connections.items()
        )
    )


class MCPSessionManager:
    """Lazy, per-server cache of persistent MCP sessions.

    Discovery always happens through throwaway sessions. Live sessions are
    only created on the first real tool call inside the runtime event loop
    so sessions stay bound to the loop that owns their subprocess/transport
    handles, and so stdio servers are not restarted on every invocation.
    """

    def __init__(self, *, connections: dict[str, Connection] | None = None) -> None:
        """Initialize the session manager.

        Args:
            connections: Optional initial server connection configs.
        """
        self._connections: dict[str, Connection] = dict(connections or {})
        self._entries: dict[str, _MCPSessionEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed = False

    def configure(self, connections: dict[str, Connection]) -> None:
        """Set or validate the connection configs used by this manager.

        When no sessions exist yet, `connections` overwrites the stored
        configs unconditionally. Once any session has been created, the
        new `connections` must produce the same signature as the stored
        ones — otherwise this raises to prevent rebinding live sessions
        to different transports or auth providers.

        Args:
            connections: Connection configs keyed by server name.

        Raises:
            RuntimeError: If the manager is closed or reconfigured
                incompatibly after sessions already exist.
        """
        if self._closed:
            msg = "Cannot configure a closed MCP session manager"
            raise RuntimeError(msg)

        if not self._entries:
            self._connections = dict(connections)
            return

        if _connections_signature(self._connections) != _connections_signature(
            connections
        ):
            msg = "Cannot reconfigure MCP session manager after sessions are active"
            raise RuntimeError(msg)
        self._connections = dict(connections)

    async def get_session(self, server_name: str) -> ClientSession:
        """Return a cached session for `server_name`, creating it lazily."""
        entry = self._entries.get(server_name)
        if entry is not None:
            return entry.session

        lock = self._get_lock(server_name)
        async with lock:
            entry = self._entries.get(server_name)
            if entry is not None:
                return entry.session

            entry = await self._create_entry(server_name)
            self._entries[server_name] = entry
            return entry.session

    async def invalidate(
        self,
        server_name: str,
        *,
        expected_session: ClientSession | None = None,
    ) -> None:
        """Evict and close a cached session if it still matches `expected_session`.

        Args:
            server_name: MCP server name.
            expected_session: Optional identity check for race-safe eviction.
        """
        lock = self._get_lock(server_name)
        async with lock:
            entry = self._entries.get(server_name)
            if entry is None:
                return
            if expected_session is not None and entry.session is not expected_session:
                return
            self._entries.pop(server_name, None)
            exit_stack = entry.exit_stack

        await exit_stack.aclose()

    async def cleanup(self) -> None:
        """Close all cached sessions concurrently and reject future creation.

        Each server's `exit_stack.aclose()` runs with a 5 second timeout so
        one slow stdio server cannot stall shutdown. Per-server failures
        are logged — teardown is best-effort — but `CancelledError` is
        re-raised so the enclosing `asyncio.gather` still cancels peers.
        """
        if self._closed and not self._entries:
            return

        self._closed = True
        names = list(self._entries)

        async def _close(server_name: str) -> None:
            try:
                await asyncio.wait_for(self.invalidate(server_name), timeout=5.0)
            except TimeoutError:
                logger.warning(
                    "MCP session cleanup for %r timed out after 5s", server_name
                )
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except Exception:
                logger.warning(
                    "MCP session cleanup for %r failed",
                    server_name,
                    exc_info=True,
                )

        await asyncio.gather(*[_close(name) for name in names])

    def _get_lock(self, server_name: str) -> asyncio.Lock:
        """Return the per-server creation/eviction lock."""
        lock = self._locks.get(server_name)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[server_name] = lock
        return lock

    async def _create_entry(self, server_name: str) -> _MCPSessionEntry:
        """Create and initialize a new cached session entry.

        Args:
            server_name: MCP server name.

        Returns:
            A cached session entry containing the live session and close stack.

        Raises:
            RuntimeError: If the manager has already been cleaned up.
            ValueError: If `server_name` is not configured in the manager.
        """
        if self._closed:
            msg = "Cannot create an MCP session after cleanup"
            raise RuntimeError(msg)

        try:
            connection = self._connections[server_name]
        except KeyError as exc:
            msg = (
                f"Couldn't find an MCP server named '{server_name}', "
                f"expected one of {sorted(self._connections)}"
            )
            raise ValueError(msg) from exc

        from langchain_mcp_adapters.sessions import create_session

        exit_stack = AsyncExitStack()
        try:
            session = await exit_stack.enter_async_context(create_session(connection))
            await session.initialize()
        except Exception:
            await exit_stack.aclose()
            raise

        return _MCPSessionEntry(session=session, exit_stack=exit_stack)


def _resolve_server_type(server_config: dict[str, Any]) -> str:
    """Determine the transport type for a server config.

    Accepts `type` or `transport` interchangeably. When neither is set, a
    `url` field implies a remote server (defaulting to `http`) and the
    absence of `url` implies stdio. This matches Claude Code's `.mcp.json`
    convention where remote entries are commonly written as `{"url": "..."}`
    alone.

    Args:
        server_config: Server configuration dictionary.

    Returns:
        Transport type string (`stdio`, `sse`, or `http`).
    """
    transport = server_config.get("type") or server_config.get("transport")
    if transport is not None:
        return _TRANSPORT_ALIASES.get(transport, transport)
    if "url" in server_config:
        return "http"
    return "stdio"


def _validate_server_config(server_name: str, server_config: dict[str, Any]) -> None:
    """Validate a single server configuration.

    Performs only shape checks — `${VAR}` header interpolation is deferred
    to activation time so one unset env var only fails its own server
    rather than hiding every other MCP entry in the same file.

    Args:
        server_name: Name of the server.
        server_config: Server configuration dictionary.

    Raises:
        TypeError: If config fields have wrong types.
        ValueError: If required fields are missing or server type is unsupported.
    """
    if not _SERVER_NAME_RE.fullmatch(server_name):
        error_msg = (
            f"Invalid server name {server_name!r}: server names must contain "
            "only alphanumerics, hyphens, and underscores."
        )
        raise ValueError(error_msg)

    if not isinstance(server_config, dict):
        error_msg = f"Server '{server_name}' config must be a dictionary"
        raise TypeError(error_msg)

    server_type = _resolve_server_type(server_config)

    if server_type in _SUPPORTED_REMOTE_TYPES:
        if "url" not in server_config:
            error_msg = (
                f"Server '{server_name}' with type '{server_type}' "
                "missing required 'url' field"
            )
            raise ValueError(error_msg)

        if "command" in server_config:
            error_msg = (
                f"Server '{server_name}' has type '{server_type}' (remote) "
                "but also declares a 'command' field. Remove 'command' or "
                'set `"type": "stdio"`.'
            )
            raise ValueError(error_msg)

        headers = server_config.get("headers")
        if headers is not None and not isinstance(headers, dict):
            error_msg = f"Server '{server_name}' 'headers' must be a dictionary"
            raise TypeError(error_msg)

        if isinstance(headers, dict):
            for name, value in headers.items():
                if not isinstance(value, str):
                    error_msg = (
                        f"Server '{server_name}' header {name!r} must be "
                        f"a string, got {type(value).__name__}"
                    )
                    raise TypeError(error_msg)
    elif server_type == "stdio":
        if "command" not in server_config:
            error_msg = f"Server '{server_name}' missing required 'command' field"
            raise ValueError(error_msg)

        if "url" in server_config:
            error_msg = (
                f"Server '{server_name}' has type 'stdio' but also declares "
                "a 'url' field. Remove 'url' or set "
                '`"type": "http"` (or `"sse"`) for a remote server.'
            )
            raise ValueError(error_msg)

        if "args" in server_config and not isinstance(server_config["args"], list):
            error_msg = f"Server '{server_name}' 'args' must be a list"
            raise TypeError(error_msg)

        if "env" in server_config and not isinstance(server_config["env"], dict):
            error_msg = f"Server '{server_name}' 'env' must be a dictionary"
            raise TypeError(error_msg)
    else:
        error_msg = (
            f"Server '{server_name}' has unsupported transport type '{server_type}'. "
            "Supported types: stdio, sse, http"
        )
        raise ValueError(error_msg)

    auth = server_config.get("auth")
    if auth is not None:
        if auth != "oauth":
            msg = (
                f"Server '{server_name}' has unsupported auth value "
                f"{auth!r}. Only 'oauth' is supported."
            )
            raise ValueError(msg)
        if server_type == "stdio":
            msg = (
                f"Server '{server_name}' uses stdio transport; "
                "'auth: oauth' is only valid for http/sse transports."
            )
            raise ValueError(msg)
        header_names = {name.lower() for name in (server_config.get("headers") or {})}
        if "authorization" in header_names:
            msg = (
                f"Server '{server_name}' cannot combine 'auth: oauth' "
                "with an 'Authorization' header."
            )
            raise ValueError(msg)


def load_mcp_config(config_path: str) -> dict[str, Any]:
    """Load and validate MCP configuration from a JSON file.

    Supports multiple server types:

    - stdio: Process-based servers with `command`, `args`, `env` fields (default)
    - sse: Server-Sent Events servers with `type: "sse"`, `url`, and optional `headers`
    - http: HTTP-based servers with `type: "http"`, `url`, and optional `headers`

    Args:
        config_path: Path to the MCP JSON configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file contains invalid JSON.
        TypeError: If config fields have wrong types.
        ValueError: If config is missing required fields.
        RuntimeError: If header env-var interpolation references an unset var.
    """  # noqa: DOC502 - `_validate_server_config()` raises `RuntimeError` indirectly
    path = Path(config_path)

    if not path.exists():
        error_msg = f"MCP config file not found: {config_path}"
        raise FileNotFoundError(error_msg)

    try:
        with path.open(encoding="utf-8") as file_obj:
            config = json.load(file_obj)
    except json.JSONDecodeError as exc:
        error_msg = f"Invalid JSON in MCP config file: {exc.msg}"
        raise json.JSONDecodeError(error_msg, exc.doc, exc.pos) from exc

    if "mcpServers" not in config:
        error_msg = (
            "MCP config must contain 'mcpServers' field. "
            'Expected format: {"mcpServers": {"server-name": {...}}}'
        )
        raise ValueError(error_msg)

    if not isinstance(config["mcpServers"], dict):
        error_msg = "'mcpServers' field must be a dictionary"
        raise TypeError(error_msg)

    if not config["mcpServers"]:
        error_msg = "'mcpServers' field is empty - no servers configured"
        raise ValueError(error_msg)

    for server_name, server_config in config["mcpServers"].items():
        _validate_server_config(server_name, server_config)

    return config


def _resolve_project_config_base(project_context: ProjectContext | None) -> Path:
    """Resolve the base directory for project-level MCP configuration lookup.

    Args:
        project_context: Explicit project path context, if available.

    Returns:
        Project root when one exists, otherwise the user working directory.
    """
    if project_context is not None:
        return project_context.project_root or project_context.user_cwd

    from deepagents_cli.project_utils import find_project_root

    return find_project_root() or Path.cwd()


def discover_mcp_configs(
    *, project_context: ProjectContext | None = None
) -> list[Path]:
    """Find MCP config files from standard locations.

    Checks three paths in precedence order (lowest to highest):

    1. `~/.deepagents/.mcp.json` (user-level global)
    2. `<project-root>/.deepagents/.mcp.json` (project subdir)
    3. `<project-root>/.mcp.json` (project root, Claude Code compat)

    Args:
        project_context: Explicit project path context, if available.

    Returns:
        Existing config file paths, ordered from lowest to highest precedence.
    """
    user_dir = Path.home() / ".deepagents"
    project_root = _resolve_project_config_base(project_context)

    candidates = [
        user_dir / ".mcp.json",
        project_root / ".deepagents" / ".mcp.json",
        project_root / ".mcp.json",
    ]

    found: list[Path] = []
    for path in candidates:
        try:
            if path.is_file():
                found.append(path)
        except OSError:
            logger.warning("Could not check MCP config %s", path, exc_info=True)
    return found


def classify_discovered_configs(
    config_paths: list[Path],
) -> tuple[list[Path], list[Path]]:
    """Split discovered config paths into user-level and project-level configs.

    Args:
        config_paths: Candidate config paths from discovery.

    Returns:
        Tuple of `(user_configs, project_configs)`.
    """
    user_dir = Path.home() / ".deepagents"
    user: list[Path] = []
    project: list[Path] = []
    for path in config_paths:
        try:
            if path.resolve().is_relative_to(user_dir.resolve()):
                user.append(path)
            else:
                project.append(path)
        except (OSError, ValueError):
            project.append(path)
    return user, project


def extract_stdio_server_commands(
    config: dict[str, Any],
) -> list[tuple[str, str, list[str]]]:
    """Extract stdio server entries from a parsed MCP config.

    Args:
        config: Parsed MCP config dictionary.

    Returns:
        List of `(server_name, command, args)` tuples for stdio servers.
    """
    results: list[tuple[str, str, list[str]]] = []
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        return results
    for name, server in servers.items():
        if not isinstance(server, dict):
            continue
        if _resolve_server_type(server) == "stdio":
            results.append((name, server.get("command", ""), server.get("args", [])))
    return results


def _filter_project_stdio_servers(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `config` with stdio servers removed."""
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        return config
    filtered = {
        name: server
        for name, server in servers.items()
        if isinstance(server, dict) and _resolve_server_type(server) != "stdio"
    }
    return {"mcpServers": filtered}


def merge_mcp_configs(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple MCP config dicts by server name.

    Args:
        configs: Config dictionaries in ascending precedence order.

    Returns:
        A single config dict with later server definitions overriding earlier ones.
    """
    merged: dict[str, Any] = {}
    for config in configs:
        servers = config.get("mcpServers")
        if isinstance(servers, dict):
            merged.update(servers)
    return {"mcpServers": merged}


def load_mcp_config_lenient(config_path: Path) -> dict[str, Any] | None:
    """Load an MCP config file, returning `None` on any error.

    Args:
        config_path: Config path to load.

    Returns:
        The parsed config, or `None` if loading or validation fails.
    """
    config, _ = load_mcp_config_with_error(config_path)
    return config


def load_mcp_config_with_error(
    config_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load an MCP config file, returning `(config, error)`.

    Missing files yield `(None, None)` — not an error. Malformed files
    yield `(None, error_text)` so callers can surface the reason to users.

    Args:
        config_path: Config path to load.

    Returns:
        `(parsed_config, None)` on success, `(None, None)` when the file
        doesn't exist, or `(None, error_message)` on load/validate failure.
    """
    try:
        return load_mcp_config(str(config_path)), None
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        logger.warning("Skipping unreadable MCP config %s: %s", config_path, exc)
        return None, f"Unreadable: {exc}"
    except (json.JSONDecodeError, ValueError, TypeError, RuntimeError) as exc:
        logger.warning("Skipping invalid MCP config %s: %s", config_path, exc)
        return None, str(exc)


def _check_stdio_server(server_name: str, server_config: dict[str, Any]) -> None:
    """Verify that a stdio server's command exists on PATH.

    Args:
        server_name: Server name for error messages.
        server_config: Validated server config.

    Raises:
        RuntimeError: If the command is missing or not found on PATH.
    """
    command = server_config.get("command")
    if command is None:
        msg = f"MCP server '{server_name}': missing 'command' in config."
        raise RuntimeError(msg)
    if shutil.which(command) is None:
        msg = (
            f"MCP server '{server_name}': command '{command}' not found on PATH. "
            "Install it or check your MCP config."
        )
        raise RuntimeError(msg)


async def _check_remote_server(server_name: str, server_config: dict[str, Any]) -> None:
    """Check network connectivity to a remote MCP server URL.

    Args:
        server_name: Server name for error messages.
        server_config: Validated remote server config.

    Raises:
        RuntimeError: If the URL is missing, unreachable, or returns 5xx.
    """
    import httpx

    url = server_config.get("url")
    if url is None:
        msg = f"MCP server '{server_name}': missing 'url' in config."
        raise RuntimeError(msg)
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.head(url)
    except (httpx.HTTPError, httpx.InvalidURL, OSError) as exc:
        msg = (
            f"MCP server '{server_name}': URL '{url}' is unreachable: {exc}. "
            "Check that the URL is correct and the server is running."
        )
        raise RuntimeError(msg) from exc
    if response.status_code >= 500:  # noqa: PLR2004  # HTTP server-error band
        msg = (
            f"MCP server '{server_name}': {url} returned HTTP "
            f"{response.status_code}. Server may be down; retry later."
        )
        raise RuntimeError(msg)


async def _discover_tools(session: ClientSession) -> list[Any]:
    """Enumerate MCP tools from `session`, paginating until exhausted.

    Args:
        session: Initialized MCP client session.

    Returns:
        Discovered MCP tool definitions.

    Raises:
        RuntimeError: If pagination never terminates within the hard safety bound.
    """
    cursor: str | None = None
    tools: list[Any] = []
    for _ in range(1000):
        page = await session.list_tools(cursor=cursor)
        if page.tools:
            tools.extend(page.tools)
        if not page.nextCursor:
            return tools
        cursor = page.nextCursor
    msg = (
        "Reached max of 1000 iterations while listing MCP tools; "
        "server may be returning a non-terminating cursor."
    )
    raise RuntimeError(msg)


def _build_cached_mcp_tool(
    *,
    mcp_tool: Any,  # noqa: ANN401
    server_name: str,
    session_manager: MCPSessionManager,
    tool_name_prefix: bool,
) -> BaseTool:
    """Build a `StructuredTool` backed by the cached session manager.

    Args:
        mcp_tool: MCP tool metadata object.
        server_name: Owning MCP server name.
        session_manager: Runtime session cache used for tool calls.
        tool_name_prefix: Whether to prefix the LangChain tool name with the
            server name.

    Returns:
        A LangChain `BaseTool` wrapper around the MCP tool.
    """
    from langchain_core.tools import StructuredTool, ToolException
    from langchain_mcp_adapters.tools import (
        _convert_call_tool_result,  # noqa: PLC2701
    )

    original_tool_name = mcp_tool.name
    lc_tool_name = (
        f"{server_name}_{original_tool_name}"
        if tool_name_prefix and server_name
        else original_tool_name
    )

    meta = getattr(mcp_tool, "meta", None)
    base_meta = (
        mcp_tool.annotations.model_dump() if mcp_tool.annotations is not None else {}
    )
    wrapped_meta = {"_meta": meta} if meta is not None else {}
    metadata = {**base_meta, **wrapped_meta} or None

    async def coroutine(
        # `runtime` is injected by LangChain's tool-calling plumbing.
        # MCP tools don't use it but the kwarg must still be accepted.
        runtime: Any = None,  # noqa: ANN401, ARG001
        **arguments: Any,
    ) -> Any:  # noqa: ANN401
        from deepagents_cli.mcp_auth import find_reauth_required

        session = await session_manager.get_session(server_name)
        try:
            result = await session.call_tool(original_tool_name, arguments)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, ToolException):
            raise
        except Exception as exc:
            reauth = find_reauth_required(exc)
            if reauth is not None:
                await session_manager.invalidate(
                    server_name,
                    expected_session=session,
                )
                raise ToolException(str(reauth)) from exc
            if not _is_transient_session_error(exc):
                msg = (
                    f"MCP tool {lc_tool_name!r} failed on server "
                    f"{server_name!r}: {type(exc).__name__}: {exc}"
                )
                raise ToolException(msg) from exc
            logger.info(
                "MCP session for %r appears dead (%s: %s); "
                "invalidating and retrying once",
                server_name,
                type(exc).__name__,
                exc,
            )
            await session_manager.invalidate(
                server_name,
                expected_session=session,
            )

            retry_session = await session_manager.get_session(server_name)
            try:
                result = await retry_session.call_tool(original_tool_name, arguments)
            except (
                asyncio.CancelledError,
                KeyboardInterrupt,
                SystemExit,
                ToolException,
            ):
                raise
            except Exception as retry_exc:  # noqa: BLE001 - wrapped into ToolException below so the agent sees it
                try:
                    retry_reauth = find_reauth_required(retry_exc)
                    if retry_reauth is not None:
                        raise ToolException(str(retry_reauth)) from retry_exc
                    msg = (
                        f"MCP tool {lc_tool_name!r} failed after one retry on "
                        f"server {server_name!r}: {type(retry_exc).__name__}: "
                        f"{retry_exc}"
                    )
                    raise ToolException(msg) from retry_exc
                finally:
                    # Invalidate the retry session last; log cleanup failure
                    # so resource leaks are observable.
                    try:
                        await session_manager.invalidate(
                            server_name,
                            expected_session=retry_session,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to invalidate retry session for %r after "
                            "tool failure",
                            server_name,
                            exc_info=True,
                        )

        return _convert_call_tool_result(result)

    return StructuredTool(
        name=lc_tool_name,
        description=mcp_tool.description or "",
        args_schema=mcp_tool.inputSchema,
        coroutine=coroutine,
        response_format="content_and_artifact",
        metadata=metadata,
    )


async def _load_tools_from_config(
    config: dict[str, Any],
    *,
    stateless: bool = False,
    session_manager: MCPSessionManager | None = None,
) -> tuple[list[BaseTool], MCPSessionManager | None, list[MCPServerInfo]]:
    """Build MCP connections from a validated config and load tools.

    Discovery always opens throwaway sessions to capture tool metadata only.
    Runtime tools either:

    - bind to a caller-managed `session_manager` (server mode),
    - bind to a new local `session_manager` returned to the caller, or
    - stay fully stateless and open a fresh session per tool call.

    Per-server config/auth/setup failures are captured in the returned
    `server_infos` list rather than propagated — one bad server never
    hides the others.

    Args:
        config: Validated MCP configuration dict with `mcpServers` key.
        stateless: When `True`, tools avoid returning an owned session manager.
        session_manager: Optional externally owned runtime session manager.

    Returns:
        Tuple of `(tools_list, session_manager, server_infos)`.

    Raises:
        RuntimeError: If `session_manager` is reconfigured incompatibly with
            sessions already active on it.
    """  # noqa: DOC501, DOC502 - `RuntimeError` surfaces via `MCPSessionManager.configure`; `KeyboardInterrupt` / `SystemExit` / `CancelledError` are re-raised pass-throughs
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StdioConnection,
        StreamableHttpConnection,
        create_session,
    )
    from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

    skipped: dict[str, tuple[MCPServerStatus, str]] = {}

    for server_name, server_config in config["mcpServers"].items():
        server_type = _resolve_server_type(server_config)
        try:
            if server_type in _SUPPORTED_REMOTE_TYPES:
                await _check_remote_server(server_name, server_config)
            elif server_type == "stdio":
                _check_stdio_server(server_name, server_config)
        except RuntimeError as exc:
            logger.warning(
                "MCP server '%s' skipped: pre-flight failed: %s",
                server_name,
                exc,
            )
            skipped[server_name] = ("error", str(exc))

    connections: dict[str, Connection] = {}
    for server_name, server_config in config["mcpServers"].items():
        if server_name in skipped:
            continue
        server_type = _resolve_server_type(server_config)
        try:
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
                    from deepagents_cli.mcp_auth import resolve_headers

                    conn["headers"] = resolve_headers(
                        server_config["headers"],
                        server_name=server_name,
                    )

                if server_config.get("auth") == "oauth":
                    from deepagents_cli.mcp_auth import (
                        FileTokenStorage,
                        build_oauth_provider,
                    )

                    storage = FileTokenStorage(
                        server_name,
                        server_url=server_config["url"],
                    )
                    if await storage.get_tokens() is None:
                        auth_msg = f"Run: deepagents mcp login {server_name}"
                        logger.warning(
                            "MCP server '%s' skipped: not authenticated. %s",
                            server_name,
                            auth_msg,
                        )
                        skipped[server_name] = ("unauthenticated", auth_msg)
                        continue
                    conn["auth"] = build_oauth_provider(
                        server_name=server_name,
                        server_url=server_config["url"],
                        storage=storage,
                        interactive=False,
                    )

                connections[server_name] = conn
            else:
                connections[server_name] = StdioConnection(
                    command=server_config["command"],
                    args=server_config.get("args", []),
                    env=server_config.get("env") or None,
                    transport="stdio",
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "MCP server '%s' skipped: config/setup failed: %s",
                server_name,
                exc,
            )
            skipped[server_name] = ("error", str(exc))

    runtime_manager: MCPSessionManager | None = session_manager
    if runtime_manager is not None:
        runtime_manager.configure(connections)
    elif not stateless:
        runtime_manager = MCPSessionManager(connections=connections)

    all_tools: list[BaseTool] = []
    server_infos: list[MCPServerInfo] = []

    for server_name, server_config in config["mcpServers"].items():
        transport = _resolve_server_type(server_config)
        if server_name in skipped:
            status, error = skipped[server_name]
            server_infos.append(
                MCPServerInfo(
                    name=server_name,
                    transport=transport,
                    status=status,
                    error=error,
                )
            )
            continue

        try:
            async with create_session(connections[server_name]) as discover_session:
                await discover_session.initialize()
                mcp_tools = await _discover_tools(discover_session)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            from deepagents_cli.mcp_auth import find_reauth_required

            reauth = find_reauth_required(exc)
            status: MCPServerStatus
            if reauth is not None:
                # Tokens existed (we checked above) but the OAuth provider
                # fell back to interactive reauth — the refresh attempt
                # failed. Flag unauthenticated so the user is prompted to
                # re-login, and keep the original exception in the log so
                # debugging a real provider outage is possible.
                status = "unauthenticated"
                error = (
                    f"{reauth} "
                    "(token refresh failed; the original error is in debug logs)"
                )
            else:
                status = "error"
                error = str(exc)
            logger.warning(
                "MCP server '%s' skipped: tool discovery failed",
                server_name,
                exc_info=True,
            )
            server_infos.append(
                MCPServerInfo(
                    name=server_name,
                    transport=transport,
                    status=status,
                    error=error,
                )
            )
            continue

        if runtime_manager is None:
            server_tools = [
                convert_mcp_tool_to_langchain_tool(
                    None,
                    mcp_tool,
                    connection=connections[server_name],
                    server_name=server_name,
                    tool_name_prefix=True,
                )
                for mcp_tool in mcp_tools
            ]
        else:
            server_tools = [
                _build_cached_mcp_tool(
                    mcp_tool=mcp_tool,
                    server_name=server_name,
                    session_manager=runtime_manager,
                    tool_name_prefix=True,
                )
                for mcp_tool in mcp_tools
            ]

        all_tools.extend(server_tools)
        server_infos.append(
            MCPServerInfo(
                name=server_name,
                transport=transport,
                tools=tuple(
                    MCPToolInfo(name=tool.name, description=tool.description or "")
                    for tool in server_tools
                ),
            )
        )

    all_tools.sort(key=lambda tool: tool.name)
    return all_tools, None if stateless else runtime_manager, server_infos


async def get_mcp_tools(
    config_path: str,
) -> tuple[list[BaseTool], MCPSessionManager | None, list[MCPServerInfo]]:
    """Load MCP tools from a configuration file.

    Args:
        config_path: Path to an MCP config file.

    Returns:
        Tuple of `(tools_list, runtime_session_manager, server_infos)`.

    Raises:
        FileNotFoundError: If `config_path` doesn't exist.
        json.JSONDecodeError: If the config file contains invalid JSON.
        TypeError: If config fields have wrong types.
        ValueError: If the config is missing required fields.
    """  # noqa: DOC502 - surfaced via `load_mcp_config`
    config = load_mcp_config(config_path)
    return await _load_tools_from_config(config)


async def resolve_and_load_mcp_tools(
    *,
    explicit_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    project_context: ProjectContext | None = None,
    stateless: bool = False,
    session_manager: MCPSessionManager | None = None,
) -> tuple[list[BaseTool], MCPSessionManager | None, list[MCPServerInfo]]:
    """Resolve MCP config and load tools.

    Auto-discovers configs from standard locations and merges them. When
    `explicit_config_path` is provided it is added as the highest-precedence
    source and errors in that file are fatal.

    Args:
        explicit_config_path: Extra config file to layer on top of
            auto-discovered configs.
        no_mcp: If `True`, disable all MCP loading.
        trust_project_mcp: Controls project-level stdio server trust.

            - `True`: always trust project configs, including stdio servers.
            - `False`: drop stdio entries from project configs.
            - `None`: consult the persistent trust store — trusted configs
              load fully, untrusted project stdio servers are dropped.
        project_context: Explicit project path context for config discovery
            and trust resolution.
        stateless: When `True`, do not return an owned runtime session manager.
        session_manager: Optional externally owned runtime session manager.

    Returns:
        Tuple of `(tools_list, session_manager, server_infos)`.

    Raises:
        FileNotFoundError: If `explicit_config_path` was provided and points
            at a missing file.
        json.JSONDecodeError: If `explicit_config_path` contains invalid
            JSON.
        TypeError: If `explicit_config_path` contents have wrong field
            types.
        ValueError: If `explicit_config_path` is missing required fields
            or declares an unsupported transport.
        RuntimeError: If the merged MCP config is malformed, or header
            env-var interpolation in `explicit_config_path` references an
            unset variable.
    """  # noqa: DOC502 - FileNotFoundError / JSONDecodeError / TypeError / ValueError surface via `load_mcp_config`
    if no_mcp:
        return [], None, []

    config_load_errors: list[tuple[Path, str]] = []

    try:
        config_paths = discover_mcp_configs(project_context=project_context)
    except (OSError, RuntimeError) as exc:
        logger.warning("MCP config auto-discovery failed", exc_info=True)
        config_paths = []
        config_load_errors.append((Path("<discovery>"), str(exc)))

    user_configs, project_configs = classify_discovered_configs(config_paths)
    configs: list[dict[str, Any]] = []

    for path in user_configs:
        config, error = load_mcp_config_with_error(path)
        if error is not None:
            config_load_errors.append((path, error))
        if config is not None:
            configs.append(config)

    for path in project_configs:
        config, error = load_mcp_config_with_error(path)
        if error is not None:
            config_load_errors.append((path, error))
        if config is None:
            continue

        stdio_servers = extract_stdio_server_commands(config)
        if not stdio_servers:
            configs.append(config)
            continue

        if trust_project_mcp is True:
            configs.append(config)
        elif trust_project_mcp is False:
            filtered = _filter_project_stdio_servers(config)
            if filtered.get("mcpServers"):
                configs.append(filtered)
            skipped = [
                f"{name}: {cmd} {' '.join(args)}" for name, cmd, args in stdio_servers
            ]
            logger.warning(
                "Skipped untrusted project stdio MCP servers: %s",
                "; ".join(skipped),
            )
        else:
            from deepagents_cli.mcp_trust import (
                compute_config_fingerprint,
                is_project_mcp_trusted,
            )

            project_root = str(_resolve_project_config_base(project_context).resolve())
            fingerprint = compute_config_fingerprint(project_configs)
            if is_project_mcp_trusted(project_root, fingerprint):
                configs.append(config)
            else:
                filtered = _filter_project_stdio_servers(config)
                if filtered.get("mcpServers"):
                    configs.append(filtered)
                skipped = [
                    f"{name}: {cmd} {' '.join(args)}"
                    for name, cmd, args in stdio_servers
                ]
                logger.warning(
                    "Skipped untrusted project stdio MCP servers "
                    "(config changed or not yet approved): %s",
                    "; ".join(skipped),
                )

    if explicit_config_path:
        config_path = (
            str(project_context.resolve_user_path(explicit_config_path))
            if project_context is not None
            else explicit_config_path
        )
        configs.append(load_mcp_config(config_path))

    def _bad_config_infos() -> list[MCPServerInfo]:
        return [
            MCPServerInfo(
                name=f"<config:{path.name}>",
                transport="config",
                status="error",
                error=f"{path}: {error}",
            )
            for path, error in config_load_errors
        ]

    if not configs:
        return [], None, _bad_config_infos()

    merged = merge_mcp_configs(configs)
    if not merged.get("mcpServers"):
        return [], None, _bad_config_infos()

    try:
        for server_name, server_config in merged["mcpServers"].items():
            _validate_server_config(server_name, server_config)
    except (TypeError, ValueError, RuntimeError) as exc:
        msg = f"Invalid MCP server configuration: {exc}"
        raise RuntimeError(msg) from exc

    tools, manager, server_infos = await _load_tools_from_config(
        merged,
        stateless=stateless,
        session_manager=session_manager,
    )
    server_infos.extend(_bad_config_infos())
    return tools, manager, server_infos
