"""MCP (Model Context Protocol) tools loader for deepagents CLI.

This module provides async functions to load and manage MCP servers using
`langchain-mcp-adapters`, supporting Claude Desktop style JSON configs.
It also supports automatic discovery of `.mcp.json` files from user-level
and project-level locations.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import re
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import Connection
    from mcp import ClientSession

    from deepagents_cli.project_utils import ProjectContext

logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """Metadata for a single MCP tool."""

    name: str
    """Tool name (may include server name prefix)."""

    description: str
    """Human-readable description of what the tool does."""


@dataclass
class MCPServerInfo:
    """Metadata for a configured MCP server and its tools."""

    name: str
    """Server name from the MCP configuration."""

    transport: str
    """Transport type (`stdio`, `sse`, or `http`)."""

    tools: list[MCPToolInfo] = field(default_factory=list)
    """Tools exposed by this server (empty when `status != "ok"`)."""

    status: str = "ok"
    """Load status — `"ok"`, `"unauthenticated"`, or `"error"`. A non-`"ok"`
    server is reported in the viewer but its tools are not mounted on the
    agent, so a single failing server never blocks startup."""

    error: str | None = None
    """Human-readable reason when `status != "ok"`, suitable for display in
    the TUI (e.g. `"Run: deepagents mcp login gh"`)."""


_SUPPORTED_REMOTE_TYPES = {"sse", "http"}
"""Supported transport types for remote MCP servers (SSE and HTTP)."""


_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
"""Server names become token-file basenames under ~/.deepagents/mcp-tokens/;
restricting to alphanumerics, hyphens, and underscores keeps them path-safe."""


def _resolve_server_type(server_config: dict[str, Any]) -> str:
    """Determine the transport type for a server config.

    Supports both `type` and `transport` field names, defaulting to `stdio`.

    Args:
        server_config: Server configuration dictionary.

    Returns:
        Transport type string (`stdio`, `sse`, or `http`).
    """
    t = server_config.get("type")
    if t is not None:
        return t
    return server_config.get("transport", "stdio")


def _validate_server_config(server_name: str, server_config: dict[str, Any]) -> None:
    """Validate a single server configuration.

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
        # SSE/HTTP server validation - requires url field
        if "url" not in server_config:
            error_msg = (
                f"Server '{server_name}' with type '{server_type}'"
                " missing required 'url' field"
            )
            raise ValueError(error_msg)

        # headers is optional but must be correct type if present
        headers = server_config.get("headers")
        if headers is not None and not isinstance(headers, dict):
            error_msg = f"Server '{server_name}' 'headers' must be a dictionary"
            raise TypeError(error_msg)

        # Fail fast on unset env-var references in header values.
        # We discard the resolved result; real resolution happens at
        # connection time (see _load_tools_from_config).
        if headers is not None:
            from deepagents_cli.mcp_auth import resolve_headers

            resolve_headers(headers, server_name=server_name)
    elif server_type == "stdio":
        # stdio server validation
        if "command" not in server_config:
            error_msg = f"Server '{server_name}' missing required 'command' field"
            raise ValueError(error_msg)

        # args and env are optional but must be correct type if present
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

    # Validate the optional `auth` field.
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
        # Can't drive both OAuth and a static Authorization header.
        header_names = {k.lower() for k in (server_config.get("headers") or {})}
        if "authorization" in header_names:
            msg = (
                f"Server '{server_name}' cannot combine 'auth: oauth' "
                "with an 'Authorization' header."
            )
            raise ValueError(msg)

    _validate_tool_filter_fields(server_name, server_config)


def _validate_tool_filter_fields(
    server_name: str, server_config: dict[str, Any]
) -> None:
    """Validate optional `allowedTools` / `disabledTools` fields.

    Both fields, when present, must be lists of strings. Setting both on the
    same server is rejected to keep the filter semantics unambiguous.

    Args:
        server_name: Name of the server (for error messages).
        server_config: Server configuration dictionary.

    Raises:
        TypeError: If a field is not a list of strings.
        ValueError: If both fields are set on the same server.
    """
    has_allowed = "allowedTools" in server_config
    has_disabled = "disabledTools" in server_config
    if has_allowed and has_disabled:
        error_msg = (
            f"Server '{server_name}' cannot set both 'allowedTools' and"
            " 'disabledTools' — pick one."
        )
        raise ValueError(error_msg)

    for field_name in ("allowedTools", "disabledTools"):
        if field_name not in server_config:
            continue
        value = server_config[field_name]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            error_msg = (
                f"Server '{server_name}' '{field_name}' must be a list of strings"
            )
            raise TypeError(error_msg)


def load_mcp_config(config_path: str) -> dict[str, Any]:
    """Load and validate MCP configuration from JSON file.

    Supports multiple server types:

    - stdio: Process-based servers with `command`, `args`, `env` fields (default)
    - sse: Server-Sent Events servers with `type: "sse"`, `url`, and optional `headers`
    - http: HTTP-based servers with `type: "http"`, `url`, and optional `headers`

    Any server type may also set an optional tool filter:

    - `allowedTools`: list of tool names or patterns to keep (all others dropped)
    - `disabledTools`: list of tool names or patterns to drop (all others kept)

    Entries are either literal tool names or `fnmatch`-style glob patterns
    (entries containing `*`, `?`, or `[`). Each entry is matched against both
    the bare MCP tool name and the server-prefixed form
    (`f"{server_name}_{tool}"`), so either `read_*` or `fs_read_*` works.
    Setting both fields on a single server is an error.

    Args:
        config_path: Path to MCP JSON configuration file (Claude Desktop format).

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file contains invalid JSON.
        TypeError: If config fields have wrong types.
        ValueError: If config is missing required fields.
    """
    path = Path(config_path)

    if not path.exists():
        error_msg = f"MCP config file not found: {config_path}"
        raise FileNotFoundError(error_msg)

    try:
        with path.open(encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in MCP config file: {e.msg}"
        raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e

    # Validate required fields
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

    # Validate each server config
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

    Project root is determined from `project_context` when provided, otherwise
    by `find_project_root()`, falling back to CWD.

    Returns:
        List of existing config file paths, ordered lowest-to-highest precedence.
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
    """Split discovered config paths into user-level and project-level.

    User-level configs live under `~/.deepagents/`. Everything else is
    considered project-level.

    Args:
        config_paths: Paths returned by `discover_mcp_configs`.

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
        config: Parsed MCP config dict with `mcpServers` key.

    Returns:
        List of `(server_name, command, args)` for each stdio server.
    """
    results: list[tuple[str, str, list[str]]] = []
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        return results
    for name, srv in servers.items():
        if not isinstance(srv, dict):
            continue
        if _resolve_server_type(srv) == "stdio":
            results.append((name, srv.get("command", ""), srv.get("args", [])))
    return results


def _filter_project_stdio_servers(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *config* with stdio servers removed.

    Remote (SSE/HTTP) servers are kept because they don't execute local code.

    Args:
        config: Parsed MCP config dict.

    Returns:
        Filtered config dict.
    """
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        return config
    filtered = {
        name: srv
        for name, srv in servers.items()
        if isinstance(srv, dict) and _resolve_server_type(srv) != "stdio"
    }
    return {"mcpServers": filtered}


def merge_mcp_configs(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple MCP config dicts by server name.

    Later entries override earlier ones for the same server name
    (simple `dict.update` on `mcpServers`).

    Args:
        configs: Ordered list of parsed config dicts (each with `mcpServers` key).

    Returns:
        Merged config with combined `mcpServers`.
    """
    merged: dict[str, Any] = {}
    for cfg in configs:
        servers = cfg.get("mcpServers")
        if isinstance(servers, dict):
            merged.update(servers)
    return {"mcpServers": merged}


def load_mcp_config_lenient(config_path: Path) -> dict[str, Any] | None:
    """Load an MCP config file, returning None on any error.

    Wraps `load_mcp_config` with lenient error handling suitable for
    auto-discovery. Missing files are skipped silently; parse and validation
    errors are logged as warnings.

    Args:
        config_path: Path to the MCP config file.

    Returns:
        Parsed config dict, or None if the file is missing or invalid.
    """
    try:
        return load_mcp_config(str(config_path))
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.warning("Skipping unreadable MCP config %s: %s", config_path, e)
        return None
    except (json.JSONDecodeError, ValueError, TypeError, RuntimeError) as e:
        logger.warning("Skipping invalid MCP config %s: %s", config_path, e)
        return None


# Exception-type names that indicate the MCP session's transport is dead and
# should be torn down + reopened. We match by class name to avoid importing
# anyio at module top level and to keep the check loop-agnostic.
_TRANSIENT_EXC_NAMES = frozenset(
    {
        "ClosedResourceError",
        "BrokenResourceError",
        "EndOfStream",
        "WouldBlock",
    }
)


def _is_transient_session_error(exc: BaseException) -> bool:
    """Return True when *exc* signals the MCP session is no longer usable.

    Triggers invalidate-and-retry. Narrow on purpose — we don't want to retry
    logical tool errors (`ToolException`), only transport/connection failures
    and auth-layer errors where a fresh session will pick up refreshed tokens.
    """
    name = type(exc).__name__
    if name in _TRANSIENT_EXC_NAMES:
        return True
    # OSError covers BrokenPipeError, ConnectionResetError, ConnectionError, etc.
    # ToolException is NOT an OSError subclass, so tool-logic errors still fall
    # through the retry gate.
    return isinstance(exc, (OSError, asyncio.IncompleteReadError))


class MCPSessionManager:
    """Process-wide cache for persistent MCP client sessions.

    Sessions are created lazily on first use in whatever asyncio event loop
    calls `get_or_create_session`. This is critical for the CLI's server-
    subprocess path: module import does metadata discovery in a throwaway
    `asyncio.run()` loop, then the LangGraph server's uvicorn loop calls tools
    — and a session created in the throwaway loop would have dead anyio memory
    streams by the time the uvicorn loop uses it.

    Each cached server gets its own `AsyncExitStack` so a single bad session
    can be invalidated (and its subprocess reaped) without disturbing peers.

    This class is the singleton returned by
    :func:`get_default_session_manager`. Directly instantiating
    :class:`MCPSessionManager` produces an isolated instance, which is useful
    in tests but does not participate in the process-wide session cache.
    """

    def __init__(self) -> None:
        """Initialize an empty session cache."""
        self._lock = asyncio.Lock()
        self._stacks: dict[str, AsyncExitStack] = {}
        self._sessions: dict[str, ClientSession] = {}

    async def get_or_create_session(
        self,
        server_name: str,
        connection: Connection,
    ) -> ClientSession:
        """Return a live `ClientSession` for *server_name*, opening one if needed.

        Uses double-checked locking so concurrent first-callers do not spawn
        duplicate subprocesses. Session + subprocess lifecycle is tied to a
        per-server `AsyncExitStack` so individual entries can be invalidated
        without closing unrelated sessions.

        Any error from `create_session` / `initialize` (including
        `CancelledError`) propagates after the partial stack is rolled back,
        so the cache is left in a clean state on failure.

        Args:
            server_name: Identifier used as the cache key.
            connection: Adapter connection config (stdio / http / sse).

        Returns:
            An initialized `ClientSession` bound to the current event loop.
        """
        existing = self._sessions.get(server_name)
        if existing is not None:
            return existing

        async with self._lock:
            existing = self._sessions.get(server_name)
            if existing is not None:
                return existing

            from langchain_mcp_adapters.sessions import create_session

            stack = AsyncExitStack()
            await stack.__aenter__()  # noqa: PLC2801 — manual enter: we hold stack lifetime across method boundaries
            try:
                session = await stack.enter_async_context(create_session(connection))
                await session.initialize()
            except BaseException:
                # Roll back so a subsequent call can retry cleanly.
                try:
                    await stack.aclose()
                except Exception:
                    logger.debug(
                        "Error closing partially-opened MCP session for %s",
                        server_name,
                        exc_info=True,
                    )
                raise

            self._stacks[server_name] = stack
            self._sessions[server_name] = session
            return session

    async def invalidate(self, server_name: str) -> None:
        """Close and forget a single server's cached session.

        Safe to call when no entry exists (no-op). After invalidation the next
        `get_or_create_session` call will open a fresh session, which re-reads
        OAuth tokens from storage and re-establishes the transport.
        """
        async with self._lock:
            self._sessions.pop(server_name, None)
            stack = self._stacks.pop(server_name, None)
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                logger.debug(
                    "Error closing MCP session for %s during invalidate",
                    server_name,
                    exc_info=True,
                )

    async def aclose_all(self) -> None:
        """Close every cached session. Idempotent — safe to call repeatedly."""
        async with self._lock:
            names = list(self._stacks.keys())
            stacks = [self._stacks.pop(name) for name in names]
            for name in names:
                self._sessions.pop(name, None)
        for stack in stacks:
            try:
                await stack.aclose()
            except Exception:
                logger.debug("Error closing MCP session on shutdown", exc_info=True)

    async def cleanup(self) -> None:
        """Back-compat alias for `aclose_all`."""
        await self.aclose_all()


_DEFAULT_MANAGER: MCPSessionManager | None = None


def get_default_session_manager() -> MCPSessionManager:
    """Return the process-wide `MCPSessionManager` singleton.

    The singleton is created lazily on first access. Proxy tools built by
    `_load_tools_from_config` resolve the manager at call time (not
    construction time) via this accessor so tests can swap the manager via
    `reset_default_session_manager_for_testing`.
    """
    global _DEFAULT_MANAGER  # noqa: PLW0603 — module-level singleton cache
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = MCPSessionManager()
    return _DEFAULT_MANAGER


def reset_default_session_manager_for_testing() -> None:
    """Drop the cached singleton. Call from test fixtures only."""
    global _DEFAULT_MANAGER  # noqa: PLW0603 — test-only reset
    _DEFAULT_MANAGER = None


def _check_stdio_server(server_name: str, server_config: dict[str, Any]) -> None:
    """Verify that a stdio server's command exists on PATH.

    Args:
        server_name: Name of the server (for error messages).
        server_config: Server configuration dictionary with `command` key.

    Raises:
        RuntimeError: If the command is missing from config or not found on PATH.
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

    Sends a lightweight HEAD request with a 2-second timeout to detect DNS
    failures, refused connections, and network timeouts early, before the MCP
    session handshake. HTTP error responses (4xx, 5xx) are not treated as
    failures — only transport errors, invalid URLs, and OS-level socket
    errors raise.

    Args:
        server_name: Name of the server (for error messages).
        server_config: Server configuration dictionary with `url` key.

    Raises:
        RuntimeError: If the server URL is unreachable or invalid.
    """
    import httpx

    url = server_config.get("url")
    if url is None:
        msg = f"MCP server '{server_name}': missing 'url' in config."
        raise RuntimeError(msg)
    try:
        async with httpx.AsyncClient() as client:
            await client.head(url, timeout=2)
    except (httpx.TransportError, httpx.InvalidURL, OSError) as exc:
        msg = (
            f"MCP server '{server_name}': URL '{url}' is unreachable: {exc}. "
            "Check that the URL is correct and the server is running."
        )
        raise RuntimeError(msg) from exc


async def _discover_tools(session: ClientSession) -> list[Any]:
    """Enumerate MCP tools from *session*, paginating until exhausted.

    Mirrors `langchain_mcp_adapters.tools._list_all_tools` but avoids importing
    that private helper so this module does not silently break on an adapter
    refactor. Caps pagination at 1000 pages to match the adapter's guard
    against a misbehaving server returning infinite cursors.

    Returns:
        The full flat list of `mcp.types.Tool` objects returned across all
        pages.

    Raises:
        RuntimeError: If pagination does not terminate within 1000 pages,
            indicating a misbehaving server that keeps returning cursors.
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
    msg = "Reached max of 1000 iterations while listing MCP tools."
    raise RuntimeError(msg)


def _build_cached_mcp_tool(
    *,
    mcp_tool: Any,  # noqa: ANN401 — mcp.types.Tool; avoid importing at module top
    server_name: str,
    connection: Connection,
    tool_name_prefix: bool,
) -> BaseTool:
    """Build a StructuredTool that routes through the cached session manager.

    The returned tool looks up the `MCPSessionManager` singleton at call time
    (not closure time) so tests can swap the manager. On transient session
    errors (transport dead, OAuth token expired mid-stream) the tool
    invalidates the cache entry and retries once; a second transient failure
    surfaces as a `ToolException` rather than crashing the agent loop.

    Args:
        mcp_tool: MCP `Tool` object as returned by `session.list_tools()`.
        server_name: Server identifier used as the session-cache key and, if
            *tool_name_prefix* is true, as a prefix on the LangChain tool name.
        connection: Adapter connection config passed to `create_session`.
        tool_name_prefix: When true, prefix the LangChain tool name with
            ``<server_name>_`` to avoid name collisions across servers.

    Returns:
        A `StructuredTool` wrapping the remote MCP tool.
    """
    from langchain_core.tools import StructuredTool, ToolException
    from langchain_mcp_adapters.tools import (
        _convert_call_tool_result,  # noqa: PLC2701 — adapter's private helper is the only path to materialize a CallToolResult as a LangChain ToolMessage
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
        runtime: Any = None,  # noqa: ANN401, ARG001 — LangGraph may pass runtime; unused here
        **arguments: Any,
    ) -> Any:  # noqa: ANN401 — StructuredTool response_format="content_and_artifact" returns a tuple
        async def _call_once() -> Any:  # noqa: ANN401 — MCP CallToolResult; dynamic to avoid top-level import
            manager = get_default_session_manager()
            session = await manager.get_or_create_session(server_name, connection)
            return await session.call_tool(original_tool_name, arguments)

        from deepagents_cli.mcp_auth import find_reauth_required

        try:
            result = await _call_once()
        except ToolException:
            # Logical error from the MCP server — not transient. Let it bubble.
            raise
        except BaseException as exc:
            # If the SDK's OAuth refresh path tripped our non-interactive
            # re-auth signal, surface it as a ToolException so the LLM sees
            # the "run deepagents mcp login" hint instead of the agent
            # crashing or (worse) blocking on input().
            reauth = find_reauth_required(exc)
            if reauth is not None:
                await get_default_session_manager().invalidate(server_name)
                raise ToolException(str(reauth)) from exc
            if not _is_transient_session_error(exc):
                raise
            logger.info(
                "MCP session for %r appears dead (%s: %s); "
                "invalidating and retrying once",
                server_name,
                type(exc).__name__,
                exc,
            )
            await get_default_session_manager().invalidate(server_name)
            try:
                result = await _call_once()
            except ToolException:
                raise
            except BaseException as retry_exc:
                # Second failure: clear the cache again so the next turn can
                # try once more from scratch, but surface this call as a
                # ToolException rather than crashing the agent loop.
                await get_default_session_manager().invalidate(server_name)
                retry_reauth = find_reauth_required(retry_exc)
                if retry_reauth is not None:
                    raise ToolException(str(retry_reauth)) from retry_exc
                msg = (
                    f"MCP tool {lc_tool_name!r} failed after one retry on "
                    f"server {server_name!r}: {type(retry_exc).__name__}: "
                    f"{retry_exc}"
                )
                raise ToolException(msg) from retry_exc

        return _convert_call_tool_result(result)

    return StructuredTool(
        name=lc_tool_name,
        description=mcp_tool.description or "",
        args_schema=mcp_tool.inputSchema,
        coroutine=coroutine,
        response_format="content_and_artifact",
        metadata=metadata,
    )


_GLOB_METACHARS = frozenset("*?[")


def _entry_matches_tool(entry: str, tool_name: str, prefix: str) -> bool:
    """Return True if a single filter entry matches a tool name.

    An entry containing `*`, `?`, or `[` is treated as an `fnmatch`-style glob;
    otherwise it is matched literally. Each entry is tried against both the
    bare MCP tool name and the server-prefixed form (`f"{prefix}{tool}"`), so
    users can write either `read_*` or `fs_read_*`.

    Args:
        entry: Filter list entry from `allowedTools` / `disabledTools`.
        tool_name: Adapter-supplied tool name (already server-prefixed).
        prefix: Server prefix (`f"{server_name}_"`).

    Returns:
        True if the entry matches this tool under either match mode.
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
    """Filter a server's loaded tools by its `allowedTools` / `disabledTools`.

    Entries may be literal tool names or `fnmatch`-style glob patterns
    (entries containing `*`, `?`, or `[`). Each entry is tried against both
    the bare MCP tool name and the server-prefixed name produced by
    `tool_name_prefix=True` (`f"{server_name}_{tool}"`). Entries that match
    no loaded tool are logged but not an error — the underlying MCP server
    may expose different tools across versions.

    Args:
        tools: Tools returned by `load_mcp_tools` for a single server.
        server_name: Server name used by the adapter to build the prefix.
        server_config: Server config dict (read for filter fields).

    Returns:
        Filtered tool list preserving input order.
    """
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


async def _load_tools_from_config(
    config: dict[str, Any],
) -> tuple[list[BaseTool], MCPSessionManager, list[MCPServerInfo]]:
    """Build MCP connections from a validated config and load tools.

    Discovery opens a throwaway session per server (torn down before this
    function returns) purely to capture tool metadata — names, descriptions,
    input schemas. The returned tools are proxies that lazily bind real
    sessions through the `MCPSessionManager` singleton on first invocation,
    inside whatever event loop the agent runs on. This avoids the cross-loop
    `ClosedResourceError` that would occur if we cached sessions created in
    a temporary loop (e.g., `asyncio.run` at module import).

    Shared implementation used by both `get_mcp_tools` (explicit path) and
    `resolve_and_load_mcp_tools` (auto-discovery).

    Args:
        config: Validated MCP configuration dict with `mcpServers` key.

    Returns:
        Tuple of `(tools_list, session_manager, server_infos)` — the manager
        is the process-wide singleton; callers need not manage its lifetime.
        Servers that fail pre-flight, lack OAuth tokens, or can't complete
        a discovery session are skipped (tools omitted) and reported in
        `server_infos` with `status != "ok"`.
    """
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StdioConnection,
        StreamableHttpConnection,
        create_session,
    )

    # Per-server status tracking. A failure at any stage (pre-flight,
    # OAuth-token lookup, session init) marks the server with a non-"ok"
    # status and skips it instead of aborting the whole startup — so a
    # single dark server never blocks the agent. The TUI's MCP viewer and
    # the welcome surface read `status` / `error` from the returned
    # `MCPServerInfo`s to signal which servers need attention.
    skipped: dict[str, tuple[str, str]] = {}  # server_name -> (status, error)

    # Pre-flight health checks (best-effort early detection; the session setup
    # below has its own error handling for TOCTOU races).
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

    # Create connections dict for MultiServerMCPClient
    # Convert Claude Desktop format to langchain-mcp-adapters format
    connections: dict[str, Connection] = {}
    for server_name, server_config in config["mcpServers"].items():
        if server_name in skipped:
            continue
        server_type = _resolve_server_type(server_config)

        if server_type in _SUPPORTED_REMOTE_TYPES:
            # langchain-mcp-adapters uses "streamable_http" for HTTP transport
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
                    server_config["headers"], server_name=server_name
                )
            if server_config.get("auth") == "oauth":
                from deepagents_cli.mcp_auth import (
                    FileTokenStorage,
                    build_oauth_provider,
                )

                storage = FileTokenStorage(server_name)
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
            # stdio server connection (default)
            connections[server_name] = StdioConnection(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env") or None,
                transport="stdio",
            )

    # Discovery: open one throwaway session per server just to enumerate
    # tools, then close it. No session state survives this loop — the proxy
    # tools built below bind fresh, cached sessions lazily in the event loop
    # that actually invokes them (see _build_cached_mcp_tool).
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
            async with create_session(connections[server_name]) as session:
                await session.initialize()
                mcp_tools = await _discover_tools(session)
        except Exception as e:  # noqa: BLE001 — per-server skip
            logger.warning(
                "MCP server '%s' skipped: tool discovery failed: %s",
                server_name,
                e,
            )
            server_infos.append(
                MCPServerInfo(
                    name=server_name,
                    transport=transport,
                    status="error",
                    error=str(e),
                )
            )
            continue

        server_tools: list[BaseTool] = [
            _build_cached_mcp_tool(
                mcp_tool=mcp_tool,
                server_name=server_name,
                connection=connections[server_name],
                tool_name_prefix=True,
            )
            for mcp_tool in mcp_tools
        ]
        server_tools = _apply_tool_filter(server_tools, server_name, server_config)
        all_tools.extend(server_tools)
        server_infos.append(
            MCPServerInfo(
                name=server_name,
                transport=transport,
                tools=[
                    MCPToolInfo(name=t.name, description=t.description or "")
                    for t in server_tools
                ],
            )
        )

    # Sort tools deterministically by name so the tools block in API requests
    # is stable across turns. MCP's list_tools() does not guarantee order, and
    # any change in the tools array busts the prompt cache at the tools block.
    all_tools.sort(key=lambda t: t.name)

    return all_tools, get_default_session_manager(), server_infos


async def get_mcp_tools(
    config_path: str,
) -> tuple[list[BaseTool], MCPSessionManager, list[MCPServerInfo]]:
    """Load MCP tools from configuration file with stateful sessions.

    Supports multiple server types:
    - stdio: Spawns MCP servers as subprocesses with persistent sessions
    - sse/http: Connects to remote MCP servers via URL

    For stdio servers, this creates persistent sessions that remain active
    across tool calls, avoiding server restarts. Sessions are managed by
    `MCPSessionManager` and should be cleaned up with
    `session_manager.cleanup()` when done.

    Args:
        config_path: Path to MCP JSON configuration file.

    Returns:
        Tuple of `(tools_list, session_manager, server_infos)` where:
            - tools_list: List of LangChain `BaseTool` objects
            - session_manager: `MCPSessionManager` instance
                (call `cleanup()` when done)
            - server_infos: List of `MCPServerInfo` with per-server metadata
    """
    config = load_mcp_config(config_path)
    return await _load_tools_from_config(config)


async def resolve_and_load_mcp_tools(
    *,
    explicit_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    project_context: ProjectContext | None = None,
) -> tuple[list[BaseTool], MCPSessionManager | None, list[MCPServerInfo]]:
    """Resolve MCP config and load tools.

    Auto-discovers configs from standard locations and merges them.
    When `explicit_config_path` is provided it is added as the
    highest-precedence source (errors in that file are fatal).

    Args:
        explicit_config_path: Extra config file to layer on top of
            auto-discovered configs (highest precedence). Errors are
            fatal.
        no_mcp: If True, disable all MCP loading.
        trust_project_mcp: Controls project-level stdio server trust:

            - `True`: allow all project stdio servers (flag/prompt approved).
            - `False`: filter out project stdio servers, log warning.
            - `None` (default): check the persistent trust store; if the
                fingerprint matches, allow; otherwise filter + warn.
        project_context: Explicit project path context for config discovery
            and trust resolution.

    Returns:
        Tuple of `(tools_list, session_manager, server_infos)`.

            When no tools are loaded, returns `([], None, [])`.

    Raises:
        RuntimeError: If the MCP config itself is malformed. Per-server
            spawn/connect failures are reported via `server_infos` rather
            than raising.
    """
    if no_mcp:
        return [], None, []

    # Auto-discovery
    try:
        config_paths = discover_mcp_configs(project_context=project_context)
    except (OSError, RuntimeError):
        logger.warning("MCP config auto-discovery failed", exc_info=True)
        config_paths = []

    # Classify discovered configs and apply trust filtering
    user_configs, project_configs = classify_discovered_configs(config_paths)

    configs: list[dict[str, Any]] = []

    # User-level configs are always trusted
    for path in user_configs:
        cfg = load_mcp_config_lenient(path)
        if cfg is not None:
            configs.append(cfg)

    # Project-level configs need trust gating for stdio servers
    for path in project_configs:
        cfg = load_mcp_config_lenient(path)
        if cfg is None:
            continue

        stdio_servers = extract_stdio_server_commands(cfg)
        if not stdio_servers:
            # No stdio servers — safe to load (remote only)
            configs.append(cfg)
            continue

        if trust_project_mcp is True:
            configs.append(cfg)
        elif trust_project_mcp is False:
            filtered = _filter_project_stdio_servers(cfg)
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
            # None — check trust store
            from deepagents_cli.mcp_trust import (
                compute_config_fingerprint,
                is_project_mcp_trusted,
            )

            project_root = str(_resolve_project_config_base(project_context).resolve())
            fingerprint = compute_config_fingerprint(project_configs)
            if is_project_mcp_trusted(project_root, fingerprint):
                configs.append(cfg)
            else:
                filtered = _filter_project_stdio_servers(cfg)
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

    # Explicit path is highest precedence — errors are fatal
    if explicit_config_path:
        config_path = (
            str(project_context.resolve_user_path(explicit_config_path))
            if project_context is not None
            else explicit_config_path
        )
        configs.append(load_mcp_config(config_path))

    if not configs:
        return [], None, []

    merged = merge_mcp_configs(configs)
    if not merged.get("mcpServers"):
        return [], None, []

    # Validate each server in the merged config
    try:
        for server_name, server_config in merged["mcpServers"].items():
            _validate_server_config(server_name, server_config)
    except (TypeError, ValueError) as e:
        msg = f"Invalid MCP server configuration: {e}"
        raise RuntimeError(msg) from e

    return await _load_tools_from_config(merged)
