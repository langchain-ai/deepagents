"""MCP (Model Context Protocol) tools loader.

This module provides async functions to load and manage MCP servers,
supporting Claude Desktop style JSON configs. It also supports automatic
discovery of `.mcp.json` files from user-level and project-level locations.

Connection management is FastMCP's: one `fastmcp.Client` per configured
server, holding its transport, its auth, and — for stdio — a subprocess kept
alive across tool calls. Turning the tools that client discovers into
LangChain tools is `langchain.mcp`'s. What is left here is the part neither
owns: finding config files, merging them, resolving `${VAR}` references,
deciding which project-local servers may run at all, and reporting per-server
status to the TUI.
"""

from __future__ import annotations

import asyncio
import copy
import fnmatch
import functools
import json
import logging
import os
import re
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast, overload

from deepagents_code import _env_vars
from deepagents_code._paths import PATHS, project_paths
from deepagents_code.mcp_config import resolve_mcp_server_env

if TYPE_CHECKING:
    from collections.abc import (
        Awaitable,
        Callable,
        Collection,
        Mapping,
        Sequence,
    )
    from typing import TextIO

    import httpx
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.transports import ClientTransport
    from langchain_core.tools import BaseTool

    from deepagents_code.model_config import McpServerTrustLists
    from deepagents_code.project_utils import ProjectContext

logger = logging.getLogger(__name__)

# Maintainer note: `deepagents-talon` imports `MCPConfigError`,
# `MCPServerInfo`, and `get_mcp_tools` from this module, and its tests construct
# `MCPToolInfo`. Keep those symbols' names, signatures, and return/dataclass
# shapes stable unless `deepagents-talon` is migrated in the same change.


@dataclass(frozen=True, slots=True)
class MCPToolInfo:
    """Metadata for a single MCP tool."""

    name: str
    """Tool name (may include server name prefix)."""

    description: str
    """Human-readable description of what the tool does."""

    input_schema: dict[str, Any] | None = None
    """Raw MCP `inputSchema` dict (JSON Schema), or `None` when unavailable.

    Supplied directly from `mcp_tool.inputSchema` at tool-load time. The viewer
    reads `properties` and `required` from this dict for parameter display;
    `None` is rendered as "no parameters".
    """


MCPServerStatus = Literal[
    "ok",
    "unauthenticated",
    "awaiting_reconnect",
    "error",
    "disabled",
]
"""Load states a configured MCP server can end up in.

`ok` means the server loaded successfully and has an authoritative tool list.

`unauthenticated` means the server requires OAuth login before tools can load.

`error` means the server failed to load after a connection or configuration
failure.

`disabled` is set when the user has turned the server off via the TUI
(`/mcp` -> F2). No connection is attempted and no tools are loaded, but
the entry is still surfaced in the viewer so the user can re-enable it.

`awaiting_reconnect` is a transient UI-only state used after OAuth login
has succeeded but before the LangGraph server has restarted and loaded
the newly available MCP tools.
"""


@dataclass(frozen=True, slots=True)
class MCPServerInfo:
    """Metadata for a configured MCP server and its tools."""

    name: str
    """Server name from the MCP configuration."""

    transport: str
    """Transport identifier — `stdio`, `sse`, `http`, the synthetic
    `config` value used for entries surfacing a bad config file, or
    `unknown` for a disabled server whose original config could not be
    classified."""

    tools: tuple[MCPToolInfo, ...] = ()
    """Tools exposed by this server (empty when `status != "ok"`)."""

    status: MCPServerStatus = "ok"
    """Load status.

    One of `ok`, `unauthenticated`, `awaiting_reconnect`, `error`, or
    `disabled`.
    """

    error: str | None = None
    """Human-readable reason when `status != "ok"`."""

    pending_reconnect: bool = False
    """`True` for a disabled entry that was just re-enabled in the TUI and is
    awaiting a reconnect to load its tools.

    Lets `/tools` (`tool_catalog.split_mcp_server_info`) preserve the reconnect
    guidance held in `error` instead of collapsing it to the generic "disabled
    by user" label — an explicit flag rather than a fragile match on the
    guidance text. Only meaningful while `status == "disabled"`.
    """

    uses_oauth: bool = False
    """`True` when this server's connection carries an OAuth provider.

    Mirrors the condition that governs whether OAuth is actually used for the
    connection: the config opted in with `auth: oauth`, or a prior login stored
    tokens and no static `Authorization` header overrides them. Lets the TUI
    offer re-authentication only where it would mean something — a server
    authenticated by a static header ignores stored OAuth tokens, and a public
    server has no OAuth flow to run.

    Only meaningful while `status == "ok"`; `unauthenticated` servers are
    already covered by `needs_attention()`.
    """

    def __post_init__(self) -> None:
        """Enforce the status/error/tools consistency invariant.

        Raises:
            ValueError: If any of: `status='ok'` with a non-`None` error;
                non-`ok` status without an error message; non-`ok` status
                carrying tools; or `pending_reconnect` set without
                `status='disabled'`.
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
        if self.pending_reconnect and self.status != "disabled":
            msg = (
                f"MCPServerInfo {self.name!r}: pending_reconnect requires "
                f"status='disabled' (got {self.status!r})"
            )
            raise ValueError(msg)

    def needs_attention(self) -> bool:
        """Return whether this server is blocked on user login."""
        return self.status == "unauthenticated"


_SUPPORTED_REMOTE_TYPES = {"sse", "http"}
"""Supported transport types for remote MCP servers (SSE and HTTP)."""

_TRANSPORT_ALIASES = {"streamable_http": "http", "streamable-http": "http"}
"""Aliases that normalize to canonical transport names.

The MCP spec and `langchain_mcp_adapters` use `streamable_http` for what the
app calls `http`. Accept both so users copy-pasting from upstream docs don't
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


_MCP_STDERR_LOG_LIMIT = 5 * 1024 * 1024
"""Size past which a server's stderr log is discarded when a session opens.

Nothing reads these files but a human debugging a misbehaving server, so a
chatty server is capped by dropping the old log rather than rotating it.
"""


def _server_stderr_log(server_name: str) -> Path | TextIO:
    """Return the sink FastMCP writes this server's stderr to.

    FastMCP's stdio transport sends server stderr to `sys.stderr` when given no
    `log_file`. The TUI owns the terminal, so an unredirected server corrupts
    the display — every stdio server is pointed at a file under the state
    directory instead, and at the null device when that file cannot be opened.

    Args:
        server_name: MCP server name, already validated as path-safe.

    Returns:
        The log path, or an open handle on the null device.

    Raises:
        MCPConfigError: If `server_name` is not path-safe.
    """
    if not _SERVER_NAME_RE.match(server_name):
        # Unreachable via `_validate_server_config`; guards the file name here
        # too, since a server name that escaped validation would otherwise
        # choose the path this writes to.
        msg = f"Refusing to open a stderr log for unsafe server name {server_name!r}"
        raise MCPConfigError(msg)
    try:
        log_dir = PATHS.profile.state_dir / "mcp-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{server_name}.log"
        if path.is_file() and path.stat().st_size > _MCP_STDERR_LOG_LIMIT:
            path.unlink()
    except OSError:
        logger.warning(
            "MCP server %r: stderr log unavailable; discarding server stderr",
            server_name,
            exc_info=True,
        )
        return cast("TextIO", Path(os.devnull).open("a", encoding="utf-8"))
    return path


def _server_log_handler(server_name: str) -> Callable[[Any], Awaitable[None]]:
    """Return a handler for the log messages `server_name` sends over MCP.

    MCP servers report their own diagnostics in-band, as `notifications/message`.
    FastMCP installs a callback either way; what this adds over its default
    handler is the server's identity — which matters once several backends are
    mounted behind one router — and routing to this module's logger at the level
    the server chose, so a server explaining *why* a tool failed reaches
    `--debug` output as a structured record rather than being scraped out of its
    stderr.

    Args:
        server_name: MCP server name used in log records.

    Returns:
        An async handler suitable for `fastmcp.Client(log_handler=...)`.
    """
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "notice": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
        "alert": logging.CRITICAL,
        "emergency": logging.CRITICAL,
    }

    async def handle(message: Any) -> None:  # noqa: ANN401, RUF029 - FastMCP requires an async handler
        level = levels.get(str(message.level).lower(), logging.INFO)
        origin = f"{server_name}:{message.logger}" if message.logger else server_name
        logger.log(level, "MCP server %s: %s", origin, message.data)

    return handle


class MCPSessionManager:
    """Owns the router that fronts every configured MCP server.

    All backends are mounted on one FastMCP router, and the tools handed to the
    agent call through a single client in front of it. That client and the stack
    holding the backend connections open live here, so a caller that outlives a
    tool load can shut every server down at once.
    """

    def __init__(self) -> None:
        """Initialize an empty manager."""
        self._client: FastMCPClient[Any] | None = None
        self._stack: AsyncExitStack | None = None
        self._closed = False

    @property
    def client(self) -> FastMCPClient[Any] | None:
        """Give back the router client, or `None` before any load."""
        return self._client

    def adopt(self, client: FastMCPClient[Any], stack: AsyncExitStack) -> None:
        """Take ownership of a router client and its backend connections.

        A previously adopted pair is *not* closed here — closing it would tear
        down sessions that tools from an earlier load still hold. Callers
        reloading MCP config call `cleanup` first.

        Args:
            client: Client in front of the mounted router.
            stack: Stack holding every backend connection open.

        Raises:
            RuntimeError: If the manager has already been cleaned up.
        """
        if self._closed:
            msg = "Cannot configure a closed MCP session manager"
            raise RuntimeError(msg)
        self._client = client
        self._stack = stack

    async def cleanup(self) -> None:
        """Close the router client and every backend, rejecting later adoption.

        Teardown is bounded at 5 seconds so one unresponsive stdio server cannot
        stall shutdown, and failures are logged rather than raised — but
        `CancelledError` propagates so an enclosing gather still cancels peers.
        """  # noqa: DOC501 - CancelledError/KeyboardInterrupt/SystemExit are re-raised pass-throughs
        self._closed = True
        client, stack = self._client, self._stack
        self._client = self._stack = None

        for label, close in (
            ("router client", getattr(client, "close", None)),
            ("MCP backends", getattr(stack, "aclose", None)),
        ):
            if close is None:
                continue
            try:
                await asyncio.wait_for(close(), timeout=5.0)
            except TimeoutError:
                logger.warning("MCP %s cleanup timed out after 5s", label)
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except Exception:
                logger.warning("MCP %s cleanup failed", label, exc_info=True)


def _resolve_server_type(server_config: Mapping[str, Any]) -> str:
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

    Performs only shape checks — `${VAR}` config interpolation is deferred
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

    _validate_tool_filter_fields(server_name, server_config)


def _validate_tool_filter_fields(
    server_name: str,
    server_config: dict[str, Any],
) -> None:
    """Validate optional `allowedTools` / `disabledTools` fields.

    Both fields, when present, must be non-empty lists of strings. Setting
    both on the same server is rejected to keep the filter semantics
    unambiguous. An empty list is rejected because it would silently strip
    every tool from the server (`allowedTools`) or be a no-op
    (`disabledTools`) — both are almost certainly user errors; omit the field
    instead.

    Args:
        server_name: Name of the server (for error messages).
        server_config: Server configuration dictionary.

    Raises:
        TypeError: If a field is not a list of strings.
        ValueError: If both fields are set, or either field is empty.
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
        if not value:
            error_msg = (
                f"Server '{server_name}' '{field_name}' must be non-empty;"
                " omit the field to disable filtering."
            )
            raise ValueError(error_msg)


def _looks_like_comment(doc: str, lineno: int) -> bool:
    """Return `True` if the offending line *begins* with `//` or `/*`.

    Only the failing line is checked, and only its leading characters (after
    stripping indentation). A `url` value such as `"url": "https://..."`
    begins with a quote, not `//`, so a URL scheme inside a quoted string
    never triggers a false comment hint.

    Args:
        doc: Full source text that failed to parse.
        lineno: 1-based line number of the error; out-of-range values
            return `False`.

    Returns:
        `True` when the stripped failing line starts with `//` or `/*`.
    """
    lines = doc.splitlines()
    if lineno < 1 or lineno > len(lines):
        return False
    stripped = lines[lineno - 1].lstrip()
    return stripped.startswith(("//", "/*"))


def _json_error_hint(exc: json.JSONDecodeError) -> str | None:
    """Return an actionable hint for a common JSON mistake, or `None`.

    Checks are ordered most-specific-first (trailing comma, then comment,
    then generic decoder-message keywords) so a more precise hint wins when
    several could apply.

    Args:
        exc: The decode error to classify.

    Returns:
        A hint string for a recognized mistake, or `None` when no specific
            guidance applies.
    """
    msg = exc.msg.lower()
    if "trailing comma" in msg:
        return (
            "Hint: JSON does not allow trailing commas. Remove the comma "
            "before the closing '}' or ']'."
        )
    if _looks_like_comment(exc.doc, exc.lineno):
        return "Hint: JSON does not allow comments (// or /* */). Remove them."
    if "expecting property name" in msg:
        return (
            "Hint: check for trailing commas, a missing key, or an unquoted "
            "property name near this position."
        )
    if "expecting value" in msg:
        return (
            "Hint: check for a missing value, an extra comma, or unquoted text "
            "near this position."
        )
    if "delimiter" in msg:
        return (
            "Hint: check for a missing comma, ':', or closing bracket near "
            "this position."
        )
    return None


def _trailing_comma_pos(doc: str, pos: int) -> int | None:
    """Return the comma position for decoder errors at a trailing comma."""
    if pos < 0 or pos >= len(doc) or doc[pos] not in "}]":
        return None
    idx = pos - 1
    while idx >= 0 and doc[idx].isspace():
        idx -= 1
    if idx >= 0 and doc[idx] == ",":
        return idx
    return None


def _json_error_snippet(
    doc: str, lineno: int, colno: int, *, pos: int | None = None
) -> str | None:
    """Build a caret snippet pointing at a JSON error location.

    Args:
        doc: Full source text that failed to parse.
        lineno: 1-based line number of the error.
        colno: 1-based column number of the error.
        pos: 0-based absolute error offset, if available.

    Returns:
        A two-line `<source line>` + caret string, or `None` when the line
        is out of range or blank.
    """
    if pos is not None:
        trailing_pos = _trailing_comma_pos(doc, pos)
        if trailing_pos is not None:
            lineno = doc.count("\n", 0, trailing_pos) + 1
            line_start = doc.rfind("\n", 0, trailing_pos) + 1
            colno = trailing_pos - line_start + 1
    lines = doc.splitlines()
    if lineno < 1 or lineno > len(lines):
        return None
    source = lines[lineno - 1].rstrip()
    if not source:
        return None
    caret_col = max(0, min(colno - 1, len(source)))
    return f"    {source}\n    {' ' * caret_col}^"


def _load_mcp_config_json(config_path: str) -> dict[str, Any]:
    """Load MCP configuration JSON with parser diagnostics.

    Args:
        config_path: Path to the MCP JSON configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file contains invalid JSON.
    """
    path = Path(config_path)

    if not path.exists():
        error_msg = f"MCP config file not found: {config_path}"
        raise FileNotFoundError(error_msg)

    try:
        with path.open(encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except json.JSONDecodeError as exc:
        # Build a layered message: core reason, an actionable hint for common
        # mistakes, then a caret snippet last so the auto-appended
        # "line X column Y" suffix reads as the location of the caret.
        parts = [f"Invalid JSON in MCP config file: {exc.msg}"]
        hint = _json_error_hint(exc)
        if hint is not None:
            parts.append(hint)
        snippet = _json_error_snippet(exc.doc, exc.lineno, exc.colno, pos=exc.pos)
        if snippet is not None:
            parts.append(snippet)
        error_msg = "\n".join(parts)
        raise json.JSONDecodeError(error_msg, exc.doc, exc.pos) from exc


def _validate_mcp_config_top_level(config: dict[str, Any]) -> None:
    """Validate top-level MCP configuration fields.

    Args:
        config: Parsed MCP config dictionary.

    Raises:
        TypeError: If top-level fields have wrong types.
        ValueError: If required top-level fields are missing.
    """
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


def _validate_mcp_config_servers(config: dict[str, Any]) -> None:
    """Validate every server in an MCP configuration.

    Args:
        config: Parsed MCP config dictionary.
    """
    for server_name, server_config in config["mcpServers"].items():
        _validate_server_config(server_name, server_config)


def _drop_invalid_mcp_config_servers(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Remove invalid server entries without rejecting valid siblings.

    Callers use this only after config precedence has been resolved, so an
    invalid winning definition is dropped instead of revealing a shadowed
    lower-precedence server with the same name.

    Args:
        config: Parsed MCP config with a top-level `mcpServers` mapping.

    Returns:
        A tuple containing the config with only valid servers and a mapping of
            dropped server names to validation errors.
    """
    valid: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, server in config["mcpServers"].items():
        try:
            _validate_server_config(name, server)
        except (ValueError, TypeError, RuntimeError) as exc:
            errors[name] = str(exc)
        else:
            valid[name] = server
    return {**config, "mcpServers": valid}, errors


def _load_mcp_config_top_level(config_path: Path) -> dict[str, Any]:
    """Load an MCP config file and validate only its top-level shape.

    Args:
        config_path: Config path to load.

    Returns:
        Parsed configuration dictionary with a valid `mcpServers` mapping.
    """
    config = _load_mcp_config_json(str(config_path))
    _validate_mcp_config_top_level(config)
    return config


def load_mcp_config(config_path: str) -> dict[str, Any]:
    """Load and validate MCP configuration from a JSON file.

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
        config_path: Path to the MCP JSON configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file contains invalid JSON.
        TypeError: If config fields have wrong types.
        ValueError: If config is missing required fields.
    """  # noqa: DOC502 - raised indirectly by `_load_mcp_config_json` / `_validate_server_config` (which does shape-only checks; `${VAR}` config interpolation is deferred to activation time, so no RuntimeError here)
    config = _load_mcp_config_top_level(Path(config_path))
    _validate_mcp_config_servers(config)

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

    from deepagents_code.project_utils import find_project_root

    return find_project_root() or Path.cwd()


def filter_trusted_project_servers(
    servers: Mapping[str, Any],
    trust_lists: McpServerTrustLists,
    *,
    project_root: Path,
    config_trusted: bool = False,
) -> dict[str, Any]:
    """Return only the project servers that survive the user's trust policy.

    The single place the per-server trust rule lives, shared by the runtime
    tool loader and the `mcp login` resolver so reject-precedence cannot drift
    between them: a disabled name is dropped even from a `config_trusted`
    config; otherwise a server is kept when the whole config is trusted or the
    user's scoped approvals / env allowlist enable it (`is_enabled`).

    Args:
        servers: `mcpServers`-shaped mapping of name to definition.
        trust_lists: The user's allow/deny policy.
        project_root: Resolved project root owning `servers`, for scoped
            fingerprint matching.
        config_trusted: Whether the config as a whole is trusted (e.g.
            `--trust-project-mcp`). Defaults to `False`.

    Returns:
        The kept subset of `servers`, in input order.
    """
    kept: dict[str, Any] = {}
    for name, server in servers.items():
        if name in trust_lists.disabled:
            # Explicit reject always wins, even for a trusted config.
            continue
        if config_trusted or trust_lists.is_enabled(
            name, project_root=project_root, server=server
        ):
            kept[name] = server
    return kept


MCP_CONFIG_DISCOVERY_PATHS: tuple[tuple[str, str], ...] = (
    (PATHS.display(PATHS.profile.mcp_config_file), "user-level"),
    ("<project-root>/.deepagents/.mcp.json", "project subdir"),
    ("<project-root>/.mcp.json", "project root"),
)
"""Display strings for the auto-discovered MCP config paths.

Ordered from lowest to highest precedence. Each entry is `(path, label)`
suitable for rendering in help screens and error messages. The runtime
discovery in `discover_mcp_config_sources` builds the same paths from the
immutable profile snapshot and `_resolve_project_config_base()`.

The two kinds of entry are not interchangeable. The user-level path is already
resolved and abbreviated for display, evaluated once at import. The two project
entries are templates that still carry a `<project-root>` placeholder, because
the project root is not known until a command runs. A renderer must not
substitute into the first or assume the other two are real paths.
"""


class MCPConfigScope(StrEnum):
    """Trust provenance assigned when an MCP config candidate is discovered."""

    USER = "user"
    """The exact `.mcp.json` selected by the immutable user profile root."""

    PROJECT = "project"
    """A repository-controlled `.mcp.json` that requires project trust."""


@dataclass(frozen=True, slots=True)
class DiscoveredMCPConfig:
    """An MCP config path together with its discovery trust provenance."""

    path: Path
    scope: MCPConfigScope
    project_root: Path | None = None

    def __post_init__(self) -> None:
        """Tie `project_root` to the scope that requires it.

        `project_root` is the key that project-trust approvals are recorded
        against, so a `PROJECT` config without one would be checked against a
        re-derived fallback root instead of failing. Rejecting the combination
        here keeps that from degrading into "trusted against the wrong root".

        Raises:
            ValueError: If the scope and `project_root` disagree.
        """
        if (self.scope is MCPConfigScope.PROJECT) != (self.project_root is not None):
            need = (
                "requires" if self.scope is MCPConfigScope.PROJECT else "must not carry"
            )
            msg = f"{self.scope} MCP config {need} a project root: {self.path}"
            raise ValueError(msg)


class MCPConfigIdentity(StrEnum):
    """Whether two discovered config paths are the same underlying file."""

    SAME = "same"
    """The paths are lexically equal or resolve to one file."""

    DIFFERENT = "different"
    """The paths resolve to distinct files."""

    UNKNOWN = "unknown"
    """Resolution failed, so identity could not be determined."""


def _same_config_location(first: Path, second: Path) -> MCPConfigIdentity:
    """Return whether two discovered paths identify the same config.

    Lexically identical paths match without filesystem access. `samefile`
    compares filesystem identity, collapsing symlink and case aliases that
    path resolution can preserve on case-insensitive filesystems. `UNKNOWN`
    preserves an indeterminate identity error so the caller can fail closed
    when scopes differ rather than retaining a possibly aliased user-trusted
    path.

    Returns:
        The identity relationship between the two paths.
    """
    if first == second:
        return MCPConfigIdentity.SAME
    try:
        identity_match = first.samefile(second)
    except (OSError, RuntimeError):
        logger.warning(
            "Could not determine whether %s and %s are the same MCP config; "
            "the pair cannot be told apart",
            first,
            second,
            exc_info=True,
        )
        return MCPConfigIdentity.UNKNOWN
    return MCPConfigIdentity.SAME if identity_match else MCPConfigIdentity.DIFFERENT


def _append_discovered_config(
    found: list[DiscoveredMCPConfig], candidate: DiscoveredMCPConfig
) -> None:
    """Append a candidate, resolving collisions toward project provenance.

    A collision never grants user trust and never drops a config: the same file
    discovered twice keeps one record at project scope, and two files that
    cannot be told apart both load at project scope.

    Only project candidates can collide, because `discover_mcp_config_sources`
    probes the single user candidate first and `found` is therefore empty when
    it arrives. The collision branch below has no user-scope arm. A user candidate that
    reached it would fall through and be dropped, so reject that ordering
    instead.

    Raises:
        AssertionError: If a user candidate arrives after another entry, which
            would mean the candidate ordering changed.
    """
    if candidate.scope is not MCPConfigScope.PROJECT:
        if found:
            msg = (
                "The user MCP config must be discovered first; collision "
                "handling has no user-scope branch."
            )
            raise AssertionError(msg)
        found.append(candidate)
        return
    for index, existing in enumerate(found):
        identity = _same_config_location(existing.path, candidate.path)
        if identity is MCPConfigIdentity.DIFFERENT:
            continue
        if identity is MCPConfigIdentity.UNKNOWN and existing.scope is candidate.scope:
            # Identity is unknown and the trust scope is the same either way,
            # so keeping both entries changes nothing.
            continue
        # A configured home can equal a project root or its `.deepagents`
        # directory. The standard project discovery provenance wins that
        # collision so relocating the profile never self-trusts the repo's own
        # MCP file.
        if identity is MCPConfigIdentity.SAME:
            # One file: move it to this discovery position so a profile
            # collision cannot give an earlier path higher precedence.
            if existing.scope is not candidate.scope:
                # The user config and a project config are one file, so its
                # servers stop being user-trusted and start asking for project
                # approval. The user did not change anything to cause that, so
                # say which collision did.
                logger.warning(
                    "MCP config %s is both the user profile config and a "
                    "project config, so it now loads at project scope. Its "
                    "servers require project trust approval. Set "
                    "DEEPAGENTS_HOME to a directory outside %s to keep them "
                    "user-trusted.",
                    existing.path,
                    candidate.project_root,
                )
            found.pop(index)
            found.append(candidate)
            return
        # Identity is unknown, so these may be two distinct files. Demote the
        # existing entry to project scope instead of dropping it: the user's
        # servers must still load, just without user-level trust.
        logger.warning(
            "Demoting MCP config %s to project scope because it could not "
            "be distinguished from %s",
            existing.path,
            candidate.path,
        )
        found[index] = DiscoveredMCPConfig(
            existing.path,
            MCPConfigScope.PROJECT,
            candidate.project_root,
        )
        found.append(candidate)
        return
    found.append(candidate)


def discover_mcp_config_sources(
    *,
    project_context: ProjectContext | None = None,
) -> list[DiscoveredMCPConfig]:
    """Discover MCP configs while preserving their trust scope.

    Args:
        project_context: Explicit project path context, if available.

    Returns:
        Existing configs in precedence order with immutable provenance.
    """
    project_root = _resolve_project_config_base(project_context)
    project = project_paths(project_root)
    candidates = (
        DiscoveredMCPConfig(PATHS.profile.mcp_config_file, MCPConfigScope.USER),
        DiscoveredMCPConfig(
            project.config_mcp_config_file,
            MCPConfigScope.PROJECT,
            project_root,
        ),
        DiscoveredMCPConfig(
            project.root_mcp_config_file,
            MCPConfigScope.PROJECT,
            project_root,
        ),
    )

    found: list[DiscoveredMCPConfig] = []
    for candidate in candidates:
        try:
            if candidate.path.is_file():
                _append_discovered_config(found, candidate)
        except OSError:
            logger.warning(
                "Could not check MCP config %s", candidate.path, exc_info=True
            )
    return found


@dataclass(frozen=True, slots=True)
class MCPConfigSources:
    """Discovered MCP configs partitioned by trust scope.

    Consumers take the views they need. Centralizing the split keeps
    provenance from being re-derived from path shape at each call site, which
    is how trust used to be inferred.
    """

    user_paths: tuple[Path, ...]
    project_paths: tuple[Path, ...]
    project_roots: Mapping[Path, Path]
    """Trust root for each project path. Total over `project_paths`.

    Read it with `[]`, never `.get(..., fallback)`. This mapping is the key
    that project trust approvals are recorded and checked against, so a
    re-derived fallback root would check trust against something the approval
    was never granted for. A miss is a broken invariant and must be loud.
    """

    @classmethod
    def from_sources(cls, found: Sequence[DiscoveredMCPConfig]) -> MCPConfigSources:
        """Partition discovered configs by scope.

        Returns:
            The partitioned view of `found`.
        """
        project = [s for s in found if s.scope is MCPConfigScope.PROJECT]
        return cls(
            user_paths=tuple(s.path for s in found if s.scope is MCPConfigScope.USER),
            project_paths=tuple(s.path for s in project),
            # `DiscoveredMCPConfig` guarantees a project-scoped entry carries a
            # root, so this mapping is total over `project_paths`. Wrapped so
            # the frozen dataclass does not hand out a mutable dict.
            project_roots=MappingProxyType(
                {s.path: s.project_root for s in project if s.project_root is not None}
            ),
        )


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


class ProjectServerSummary(NamedTuple):
    """A project MCP server row shown to the user and gated for trust.

    A `NamedTuple` (not a bare 3-tuple) so the three same-typed `str` slots get
    field names — a `name`/`kind` swap can't type-check silently — while staying
    tuple-compatible with existing unpacking and indexing.
    """

    name: str
    """MCP server name."""

    kind: str
    """Transport kind from `_resolve_server_type`: `"stdio"`, `"http"`, or
    `"sse"` for a well-formed entry. Typed `str`, not a `Literal`, because these
    summaries are built from *unvalidated* configs (the trust prompt inspects
    raw merged servers before validation), so a malformed `type`/`transport`
    passes through verbatim (e.g. `{"type": "banana"}` yields `"banana"`)."""

    summary: str
    """`"<command> <args>"` for stdio entries, the URL for remote entries."""


def extract_project_server_summaries(
    config: dict[str, Any],
) -> list[ProjectServerSummary]:
    """Return a `ProjectServerSummary` for every server in a project config.

    Used by the trust prompt and the untrusted-config skip warning so that
    both stdio servers (which spawn local commands) and remote servers
    (which can SSRF or exfiltrate environment variables via interpolated
    headers when an attacker controls `.mcp.json`) are gated identically.

    Args:
        config: Parsed MCP config dictionary.

    Returns:
        One `ProjectServerSummary` per server, in config order.
    """
    results: list[ProjectServerSummary] = []
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        return results
    for name, server in servers.items():
        if not isinstance(server, dict):
            logger.debug(
                "Skipping malformed MCP server entry %r: expected a table, got %s",
                name,
                type(server).__name__,
            )
            continue
        kind = _resolve_server_type(server)
        if kind == "stdio":
            args = server.get("args") or []
            summary = f"{server.get('command', '')} {' '.join(args)}".strip()
        elif kind in _SUPPORTED_REMOTE_TYPES:
            summary = str(server.get("url", ""))
        else:
            summary = ""
        results.append(ProjectServerSummary(name, kind, summary))
    return results


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


def _merge_mcp_configs_with_sources(
    configs: list[tuple[Path, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Merge MCP configs and retain the winning source for each server.

    Args:
        configs: `(path, config)` pairs in ascending precedence order.

    Returns:
        The merged config and a mapping from each server name to the path that
        supplied its highest-precedence definition.
    """
    servers: dict[str, Any] = {}
    sources: dict[str, Path] = {}
    for path, config in configs:
        config_servers = config.get("mcpServers")
        if isinstance(config_servers, dict):
            servers.update(config_servers)
            for name in cast("dict[str, Any]", config_servers):
                sources[name] = path
    return {"mcpServers": servers}, sources


def load_mcp_config_lenient(
    config_path: Path, *, disabled_servers: Collection[str] = ()
) -> dict[str, Any] | None:
    """Load a single MCP config file, returning `None` on any error.

    Disabled servers are removed before per-server validation, so explicitly
    denied entries can neither block loading nor surface to a caller inspecting
    the config. The single-file counterpart to `load_merged_mcp_configs_lenient`
    (which the trust prompt uses); this one has no production caller today and is
    retained as the standalone lenient loader.

    Args:
        config_path: Config path to load.
        disabled_servers: Server names to remove before validation.

    Returns:
        The parsed config, or `None` if loading or validation fails.
    """
    config, _ = _load_mcp_config_top_level_with_error(config_path)
    if config is None:
        return None

    servers = config["mcpServers"]
    filtered = {
        **config,
        "mcpServers": {
            name: server
            for name, server in servers.items()
            if name not in disabled_servers
        },
    }
    try:
        _validate_mcp_config_servers(filtered)
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.warning("Skipping invalid MCP config %s: %s", config_path, exc)
        return None
    return filtered


def load_merged_mcp_configs_lenient(
    config_paths: Collection[Path], *, disabled_servers: Collection[str] = ()
) -> dict[str, Any] | None:
    """Load and validate project configs after resolving precedence.

    The trust prompt must inspect the exact merged server definitions that a
    whole-config approval can activate. Parsing each file with per-server
    validation first can discard valid lower-precedence siblings when a bad
    entry in that file is replaced by a valid higher-precedence definition.

    Args:
        config_paths: Project config paths in ascending precedence order.
        disabled_servers: Server names to remove before validation.

    Returns:
        The merged, filtered config, or `None` when no config is usable. Invalid
            winning server definitions are dropped without hiding valid siblings.
    """
    configs: list[dict[str, Any]] = []
    for path in config_paths:
        config, _ = _load_mcp_config_top_level_with_error(path)
        if config is not None:
            configs.append(config)
    if not configs:
        return None

    merged = merge_mcp_configs(configs)
    servers = merged["mcpServers"]
    filtered = {
        **merged,
        "mcpServers": {
            name: server
            for name, server in servers.items()
            if name not in disabled_servers
        },
    }
    valid, errors = _drop_invalid_mcp_config_servers(filtered)
    for name, error in errors.items():
        logger.warning("Skipping invalid merged MCP server %r: %s", name, error)
    if errors and not valid["mcpServers"]:
        return None
    return valid


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


def _load_mcp_config_top_level_with_error(
    config_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load an MCP config file, validating only its top-level structure.

    Args:
        config_path: Config path to load.

    Returns:
        `(parsed_config, None)` on success, `(None, None)` when the file
        doesn't exist, or `(None, error_message)` on load/top-level validate
        failure.
    """
    try:
        return _load_mcp_config_top_level(config_path), None
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        logger.warning("Skipping unreadable MCP config %s: %s", config_path, exc)
        return None, f"Unreadable: {exc}"
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
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
            f"MCP server '{server_name}': configured command not found on PATH. "
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
        # Name the failure *class* (e.g. `ConnectTimeout`, `InvalidURL`) so the
        # failure mode stays diagnosable, but keep the URL redacted: `str(exc)`
        # echoes the URL (which may carry `${VAR}`-injected credentials), while
        # the class name never does.
        msg = (
            f"MCP server '{server_name}': configured URL is unreachable "
            f"({type(exc).__name__}). "
            "Check that the URL is correct and the server is running."
        )
        raise RuntimeError(msg) from exc
    if response.status_code >= 500:  # noqa: PLR2004  # HTTP server-error band
        msg = (
            f"MCP server '{server_name}': configured URL returned HTTP "
            f"{response.status_code}. Server may be down; retry later."
        )
        raise RuntimeError(msg)


def _config_uses_env_interpolation(server_config: dict[str, Any]) -> bool:
    """Return whether a supported config value contains an env reference.

    Exceptions raised after interpolation may include resolved connection
    values in their messages or traceback. Treat every environment-derived
    value as potentially sensitive so those failures can be reported without
    exposing the resolved value.

    Args:
        server_config: Raw, unresolved MCP server configuration.

    Returns:
        Whether a supported value contains a `${...}` reference.
    """
    scalar_values = [server_config.get("command"), server_config.get("url")]
    sequence_values = server_config.get("args")
    if isinstance(sequence_values, list):
        scalar_values.extend(sequence_values)
    for field in ("env", "headers"):
        mapping = server_config.get(field)
        if isinstance(mapping, dict):
            scalar_values.extend(mapping.values())
    return any(isinstance(value, str) and "${" in value for value in scalar_values)


def _build_mcp_tool(
    *,
    mcp_tool: Any,  # noqa: ANN401
    server_name: str,
    client: FastMCPClient[Any],
) -> BaseTool:
    """Adapt one mounted MCP tool, then badge it as this app's.

    `langchain.mcp.convert_mcp_tool_to_langchain_tool` owns everything
    protocol-facing: schema conversion, calling through `client`, and turning an
    MCP `isError` result into a failed `ToolMessage` carrying the server's own
    content. The tool already arrives namespaced by its mount, so nothing here
    renames it.

    Args:
        mcp_tool: MCP tool metadata, as returned by `Client.list_tools`.
        server_name: Owning MCP server name, recorded in metadata.
        client: The router client the returned tool calls through.

    Returns:
        A LangChain `BaseTool` wrapper around the MCP tool.
    """
    from langchain.mcp import convert_mcp_tool_to_langchain_tool

    tool = convert_mcp_tool_to_langchain_tool(mcp_tool, client)
    tool.metadata = {
        **(tool.metadata or {}),
        "_deepagents_code_mcp": True,
        "_deepagents_code_mcp_server": server_name,
    }
    return tool


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


@overload
def _apply_tool_filter(
    tools: list[BaseTool],
    server_name: str,
    server_config: dict[str, Any],
) -> list[BaseTool]: ...


@overload
def _apply_tool_filter(
    tools: Sequence[BaseTool],
    server_name: str,
    server_config: dict[str, Any],
) -> Sequence[BaseTool]: ...


def _apply_tool_filter(
    tools: Sequence[BaseTool],
    server_name: str,
    server_config: dict[str, Any],
) -> Sequence[BaseTool]:
    """Filter a server's loaded tools by its `allowedTools` / `disabledTools`.

    Entries may be literal tool names or `fnmatch`-style glob patterns
    (entries containing `*`, `?`, or `[`). Each entry is tried against both
    the bare MCP tool name and the server-prefixed name produced by
    `tool_name_prefix=True` (`f"{server_name}_{tool}"`). Entries that match
    no loaded tool are logged but not an error — the underlying MCP server
    may expose different tools across versions, so a stale entry should not
    fail startup. The same warning is emitted symmetrically for both fields
    so a typo in `disabledTools` is visible (otherwise a tool the user
    intended to disable would silently remain enabled).

    Args:
        tools: Tools returned by `load_mcp_tools` for a single server.
        server_name: Server name used by the adapter to build the prefix.
        server_config: Server config dict (read for filter fields).

    Returns:
        Filtered tool list preserving input order.
    """
    allowed: list[str] | None = server_config.get("allowedTools")
    disabled: list[str] | None = server_config.get("disabledTools")
    entries: list[str] | None = allowed if allowed is not None else disabled
    if entries is None:
        return tools

    prefix = f"{server_name}_"
    field_name = "allowedTools" if allowed is not None else "disabledTools"

    def _any_entry_matches(tool_name: str, entry_list: list[str]) -> bool:
        return any(_entry_matches_tool(e, tool_name, prefix) for e in entry_list)

    missing = [
        e
        for e in entries
        if not any(_entry_matches_tool(e, t.name, prefix) for t in tools)
    ]
    if missing:
        logger.warning(
            "MCP server '%s' %s entries matched no tools: %s",
            server_name,
            field_name,
            ", ".join(missing),
        )

    if allowed is not None:
        return [t for t in tools if _any_entry_matches(t.name, entries)]
    return [t for t in tools if not _any_entry_matches(t.name, entries)]


_MCP_LOAD_CONCURRENCY = 8
"""Upper bound on MCP servers preflighted/discovered concurrently.

Independent servers are probed in parallel so graph load no longer scales
linearly with server count, but the fan-out is capped so a large config cannot
spawn an unbounded number of simultaneous socket/subprocess handshakes (or
`asyncio.to_thread` `shutil.which` workers).
"""


def _warm_mcp_adapter_imports() -> None:
    """Eagerly import MCP modules whose first import may block.

    Run via `asyncio.to_thread` before adapter/auth symbols are used, so any
    blocking side effect of a first import happens off the server event loop
    rather than where Blockbuster would reject it. The known offender is
    `mcp_auth`, which imports `httpx`, which transitively imports `rich`;
    `rich` calls `os.getcwd()` in its module body (verified against the pinned
    versions — the exact culprit may shift as dependencies change, but the
    general risk of import-time I/O in this subtree does not).

    Warming `mcp_auth` is best-effort: it is only *used* on per-server paths
    (remote-server preflight and the per-tool call path), where an import
    failure is captured and reported per server. A failure to warm it must not
    abort loading for every server — notably stdio-only configs, which never
    import `mcp_auth` otherwise — so it is swallowed here and left to re-raise
    at the real use site. Runs only when at least one active MCP server exists.
    """
    from langchain import mcp as _langchain_mcp  # noqa: F401

    try:
        from deepagents_code import mcp_auth as _mcp_auth  # noqa: F401
    except Exception:  # warmup is a best-effort optimization; never abort load
        logger.warning(
            "Failed to warm mcp_auth import off the event loop; "
            "deferring to per-server use",
            exc_info=True,
        )


async def _gather_bounded[T](
    factories: Sequence[Callable[[], Awaitable[T]]],
    *,
    limit: int,
) -> list[T]:
    """Await coroutine factories with bounded concurrency, preserving order.

    Results are returned in submission order (not completion order), so callers
    can zip them back against their inputs to keep deterministic ordering. If a
    factory raises (including a cancellation/shutdown signal), the remaining
    tasks are cancelled and awaited before the exception propagates, so no
    background work is left running.

    `asyncio.gather` propagates only the *first* task to finish with an
    exception; when several tasks fail concurrently the rest are cancelled
    during teardown and their exceptions would otherwise be discarded silently.
    To keep concurrent failures debuggable, each dropped (non-cancellation)
    sibling exception is logged at debug level before the first one propagates.

    Args:
        factories: Zero-arg callables each returning an awaitable to run.
        limit: Maximum number of awaitables in flight at once. Values below 1
            are clamped to 1.

    Returns:
        The awaited results in the same order as `factories`.
    """
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _run(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    tasks = [asyncio.create_task(_run(factory)) for factory in factories]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                logger.debug(
                    "MCP concurrent load: a sibling task failed while another "
                    "failure was already propagating; logging the dropped "
                    "exception for debugging",
                    exc_info=result,
                )
        raise


def _build_transport(
    server_name: str,
    server_type: str,
    server_config: Mapping[str, Any],
    *,
    auth: httpx.Auth | None,
    keep_alive: bool,
) -> ClientTransport:
    """Build the FastMCP transport for one configured server.

    The config entry is handed to FastMCP's own server models, so their
    validation is what rejects a malformed entry and their `to_transport` is
    what picks the transport class — including resolving a bare `url` to
    streamable-HTTP or SSE.

    The one thing the models do not carry is `log_file`, so a stdio server's is
    attached afterwards: FastMCP writes a server's stderr to `sys.stderr`
    otherwise, and the TUI owns that terminal.

    Args:
        server_name: MCP server name, used for the stderr log file.
        server_type: Resolved transport type (`stdio`, `http`, or `sse`).
        server_config: That server's config, with `${VAR}` refs already resolved.
        auth: OAuth provider to attach, for a remote server that uses one.
        keep_alive: Whether a stdio server's subprocess outlives one connection.
            A live client reuses it across tool calls; a stateless load does not.

    Returns:
        A transport ready to mount on the router.
    """
    from fastmcp.mcp_config import RemoteMCPServer, StdioMCPServer

    if server_type in _SUPPORTED_REMOTE_TYPES:
        # Only pin the transport when the config actually named one. Left unset,
        # FastMCP infers it from the URL, which is what makes a bare
        # `{"url": ".../sse"}` entry connect over SSE instead of streamable HTTP.
        declared = server_config.get("type") or server_config.get("transport")
        remote = RemoteMCPServer.model_validate(dict(server_config))
        remote.transport = (
            ("sse" if server_type == "sse" else "http") if declared else None
        )
        # The config's `auth` is a mode name (`"oauth"`), not a credential; the
        # resolved provider replaces it.
        remote.auth = auth
        return remote.to_transport()

    stdio = StdioMCPServer.model_validate(dict(server_config))
    stdio.keep_alive = keep_alive
    transport = stdio.to_transport()
    transport.log_file = _server_stderr_log(server_name)
    return transport


async def _mount_backends(
    backends: Mapping[str, ClientTransport],
    *,
    redact: Mapping[str, bool],
) -> tuple[FastMCPClient[Any], AsyncExitStack, dict[str, tuple[MCPServerStatus, str]]]:
    """Connect every backend and mount it on one router.

    FastMCP can build this composite itself from an `MCPConfig`, but it reports
    a backend that fails to connect as a log line and moves on. The TUI has to
    tell a user *which* server is down and whether the fix is a login, so the
    mount loop is owned here instead, and each failure is classified the way
    `MCPServerInfo` needs.

    Mounting namespaces each backend's tools with its config key, which is where
    the `server_tool` names come from.

    Args:
        backends: Ready transports keyed by server name.
        redact: Per-server flag for whether error detail may quote a config
            that interpolates `${VAR}` references.

    Returns:
        The router client, the stack holding every backend open, and a
            `(status, error)` entry for each server that failed to connect.
    """  # noqa: DOC501 - CancelledError/KeyboardInterrupt/SystemExit are re-raised pass-throughs
    from fastmcp import FastMCP
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.server.providers.proxy import StatefulProxyClient
    from fastmcp.server.server import create_proxy

    router: Any = FastMCP(name="deepagents-code")
    stack = AsyncExitStack()
    failures: dict[str, tuple[MCPServerStatus, str]] = {}

    for server_name, transport in backends.items():
        try:
            backend = StatefulProxyClient(
                transport=transport,
                log_handler=_server_log_handler(server_name),
            )
            await backend.__aenter__()  # noqa: PLC2801 - paired with explicit callbacks below
            # `StatefulProxyClient.__aexit__` is deliberately a no-op — it only
            # decrements a nesting count — so `async with` would not itself stop
            # a `keep_alive` stdio subprocess. Closing the router client happens
            # to cascade to the mounted backends today, but that is an
            # implementation detail of the proxy; tear down explicitly so
            # teardown does not depend on it, mirroring
            # `MCPConfigTransport._create_proxy`. Callbacks run LIFO, so
            # `transport.close()` is pushed first in order to run last.
            stack.push_async_callback(transport.close)
            stack.push_async_callback(backend._disconnect, force=True)
            router.mount(create_proxy(backend), namespace=server_name)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - one server must not sink the rest
            failures[server_name] = _classify_connect_failure(
                server_name,
                exc,
                redact=redact.get(server_name, False),
            )

    return FastMCPClient(router), stack, failures


def _classify_connect_failure(
    server_name: str,
    exc: BaseException,
    *,
    redact: bool,
) -> tuple[MCPServerStatus, str]:
    """Describe why a server would not connect, in the terms the TUI offers.

    Two failures are worth telling apart from a generic error because the user
    can act on them: tokens that exist but no longer refresh, and a server
    answering an unauthenticated request with an RFC 9728 challenge. Both mean
    "log in", so both become `unauthenticated`.

    Classification is itself best-effort — a classifier that raises degrades the
    server to a plain error rather than taking down the rest of the load.

    Args:
        server_name: MCP server name.
        exc: The failure raised while connecting.
        redact: Whether this server's config interpolates `${VAR}`, in which
            case the message must not quote resolved (secret-bearing) values.

    Returns:
        The `(status, error)` pair for this server's `MCPServerInfo`.
    """
    from deepagents_code.mcp_auth import (
        find_oauth_challenge,
        find_reauth_required,
        format_login_failure,
    )

    try:
        reauth = find_reauth_required(exc)
        challenge = None if reauth is not None else find_oauth_challenge(exc)
    except Exception:
        logger.debug(
            "MCP server %r: failed to classify connect error",
            server_name,
            exc_info=True,
        )
        reauth = challenge = None

    if reauth is not None or challenge is not None:
        error = (
            f"{reauth} (token refresh failed)"
            if reauth is not None
            else (
                f"MCP server {server_name!r} requires authentication; "
                f"run `dcode mcp login {server_name}`."
            )
        )
        logger.warning("MCP server '%s' skipped: %s", server_name, error)
        # An expected, already-classified outcome: the actionable WARNING says
        # everything useful, so the DEBUG line stays a concise, token-safe
        # breadcrumb. `format_login_failure` names the culprit nested inside the
        # anyio `ExceptionGroup` these usually arrive wrapped in.
        logger.debug(
            "MCP server '%s' skipped: %s",
            server_name,
            format_login_failure(exc),
        )
        return ("unauthenticated", error)

    if redact:
        logger.warning(
            "MCP server '%s' skipped: connection failed (%s; details redacted "
            "because config uses environment interpolation)",
            server_name,
            exc.__class__.__name__,
        )
        return (
            "error",
            (
                f"MCP server {server_name!r}: connection failed after "
                "resolving environment variables."
            ),
        )

    logger.warning(
        "MCP server '%s' skipped: connection failed", server_name, exc_info=exc
    )
    return ("error", str(exc))


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
    """  # noqa: DOC502 - `RuntimeError` surfaces via `MCPSessionManager.configure`
    # Warm the adapter imports off the event loop *here* (rather than in the
    # caller) so a config with no active MCP servers — which returns before
    # ever reaching this function — never pays the adapter-import cost.
    await asyncio.to_thread(_warm_mcp_adapter_imports)

    server_items = list(config["mcpServers"].items())
    # Resolve each server's transport once, up front. `_resolve_server_type` is
    # pure, so this is a readability/DRY win over recomputing it in preflight,
    # discovery, and the final fold-in loop below.
    transports = {name: _resolve_server_type(cfg) for name, cfg in server_items}
    # Names whose connection got an OAuth provider, recorded during preflight
    # and folded into `MCPServerInfo.uses_oauth` at discovery. Preflight fully
    # completes before discovery starts, so this is populated by then. Computed
    # here rather than in `_discover_server` because the decision needs the
    # *resolved* config — the token file stem is derived from the expanded URL.
    oauth_servers: set[str] = set()
    # Whether each server's config interpolates `${VAR}`. Captured from the
    # *raw* config: once refs are expanded, an error message could echo a
    # resolved secret, so those servers' failures are reported without detail.
    redacts = {
        name: _config_uses_env_interpolation(cfg) for name, cfg in server_items
    }

    async def _preflight_and_connect(
        server_name: str,
        server_config: dict[str, Any],
    ) -> tuple[MCPServerStatus, str] | ClientTransport:
        """Preflight one server and build its transport.

        Per-server preflight/config failures are captured here so one bad
        server never aborts loading the others. Nothing is dialed yet — the
        transports returned here are mounted on the router below, which is
        where connection failures are classified.

        Returns:
            A `(status, error)` tuple when the server must be skipped, or a
            ready transport otherwise.
        """
        server_type = transports[server_name]
        # Capture this from the *raw* config, before resolution below rebinds
        # `server_config` to the expanded copy. Once `${...}` refs are expanded,
        # a downstream setup error may echo the resolved (secret-bearing) value,
        # so those messages are redacted; plain configs keep full detail.
        redact_failure_details = _config_uses_env_interpolation(server_config)
        # Config env-var resolution is the only step that raises `TypeError`
        # (non-string field). Keep it in its own `try` so an unexpected
        # `TypeError` from the connectivity checks below — whose contract is
        # `RuntimeError` only — surfaces as a real bug instead of being
        # relabeled as a per-server config skip.
        try:
            server_config = resolve_mcp_server_env(server_name, server_config)
        except (RuntimeError, TypeError) as exc:
            logger.warning(
                "MCP server '%s' skipped: config error: %s",
                server_name,
                exc,
            )
            return ("error", str(exc))
        try:
            if server_type in _SUPPORTED_REMOTE_TYPES:
                await _check_remote_server(server_name, server_config)
            elif server_type == "stdio":
                # `shutil.which` makes blocking `os.access` calls; run it
                # off the event loop so blockbuster doesn't reject it.
                await asyncio.to_thread(_check_stdio_server, server_name, server_config)
        except RuntimeError as exc:
            logger.warning(
                "MCP server '%s' skipped: pre-flight failed: %s",
                server_name,
                exc,
            )
            return ("error", str(exc))

        try:
            if server_type in _SUPPORTED_REMOTE_TYPES:
                from deepagents_code.mcp_auth import (
                    FileTokenStorage,
                    build_oauth_provider,
                )

                explicit_oauth = server_config.get("auth") == "oauth"
                header_names = {
                    name.lower() for name in (server_config.get("headers") or {})
                }
                has_authorization_header = "authorization" in header_names
                storage = FileTokenStorage(
                    server_name,
                    server_url=server_config["url"],
                )
                stored_tokens = await storage.get_tokens()

                if explicit_oauth and stored_tokens is None:
                    # Config opted into OAuth but no tokens are stored yet —
                    # require an upfront login before connecting.
                    auth_msg = f"MCP server {server_name!r} needs re-authentication."
                    logger.warning(
                        "MCP server '%s' skipped: not authenticated.",
                        server_name,
                    )
                    return ("unauthenticated", auth_msg)

                auth: httpx.Auth | None = None
                if explicit_oauth or (
                    stored_tokens is not None and not has_authorization_header
                ):
                    # Attach the provider when the user opted in, or when a
                    # prior login (possibly triggered by 401 auto-detection)
                    # already stored tokens for this server. Static
                    # Authorization headers take precedence over stored OAuth.
                    # `build_oauth_provider` returns an `httpx.Auth`, which is
                    # what FastMCP's remote transports take directly.
                    oauth_servers.add(server_name)
                    auth = build_oauth_provider(
                        server_name=server_name,
                        server_url=server_config["url"],
                        storage=storage,
                        interactive=False,
                    )

                return _build_transport(
                    server_name, server_type, server_config,
                    auth=auth, keep_alive=not stateless,
                )
            return _build_transport(
                server_name, server_type, server_config,
                auth=None, keep_alive=not stateless,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if redact_failure_details:
                error = (
                    f"MCP server {server_name!r}: setup failed after "
                    "resolving environment variables."
                )
                logger.warning(
                    "MCP server '%s' skipped: config/setup failed (%s; details "
                    "redacted because config uses environment interpolation)",
                    server_name,
                    exc.__class__.__name__,
                )
            else:
                error = str(exc)
                logger.warning(
                    "MCP server '%s' skipped: config/setup failed",
                    server_name,
                    exc_info=exc,
                )
            return ("error", error)

    # Preflight + connection build runs concurrently across servers (bounded).
    # Results come back in submission order, so `skipped`/`connections` are
    # assembled in config order and stay deterministic regardless of which
    # server's probe finished first.
    preflight_results = await _gather_bounded(
        [
            functools.partial(_preflight_and_connect, name, cfg)
            for name, cfg in server_items
        ],
        limit=_MCP_LOAD_CONCURRENCY,
    )

    skipped: dict[str, tuple[MCPServerStatus, str]] = {}
    backends: dict[str, ClientTransport] = {}
    for (server_name, _server_config), result in zip(
        server_items, preflight_results, strict=True
    ):
        if isinstance(result, tuple):
            skipped[server_name] = result
        else:
            backends[server_name] = result

    runtime_manager = (
        session_manager if session_manager is not None else MCPSessionManager()
    )
    client, stack, mount_failures = await _mount_backends(backends, redact=redacts)
    runtime_manager.adopt(client, stack)
    skipped.update(mount_failures)

    # One `list_tools` covers every mounted backend; FastMCP paginates
    # internally and bounds itself, so a server returning a non-terminating
    # cursor cannot hang the load.
    async with client:
        mounted_tools = await client.list_tools()

    def _owner(tool_name: str) -> str | None:
        """Return the server a mounted tool belongs to.

        Mounting prefixes each tool with `server_`, and a server name may itself
        contain an underscore — so `a_b_read` is ambiguous between server `a`
        and server `a_b`. The longest configured name that matches wins, which
        is the mount FastMCP actually resolved it against.
        """
        candidates = [
            name
            for name in backends
            if tool_name.startswith(f"{name}_") and name not in skipped
        ]
        return max(candidates, key=len) if candidates else None

    by_server: dict[str, list[Any]] = {name: [] for name in backends}
    for mcp_tool in mounted_tools:
        owner = _owner(mcp_tool.name)
        if owner is None:
            logger.debug(
                "MCP tool %r matched no configured server; ignoring", mcp_tool.name
            )
            continue
        by_server[owner].append(mcp_tool)

    def _build_server(
        server_name: str,
        server_config: dict[str, Any],
    ) -> tuple[list[BaseTool], MCPServerInfo]:
        """Adapt one server's tools and build its `MCPServerInfo`.

        Tool construction can still fail after a healthy connection — a schema
        that will not convert, or a bad tool filter — so it is isolated per
        server here, the same way connection failures are isolated at mount.

        Args:
            server_name: MCP server name.
            server_config: That server's resolved config entry.

        Returns:
            The server's LangChain tools plus its `MCPServerInfo` entry.
        """  # noqa: DOC501 - CancelledError/KeyboardInterrupt/SystemExit are re-raised pass-throughs
        redact_failure_details = redacts[server_name]
        try:
            server_tools: list[BaseTool] = [
                _build_mcp_tool(
                    mcp_tool=mcp_tool, server_name=server_name, client=client
                )
                for mcp_tool in by_server[server_name]
            ]
            server_tools = _apply_tool_filter(server_tools, server_name, server_config)

            # Pair each tool's schema by its mounted name, so the lookup needs no
            # string surgery. Deep-copy the raw dict: `MCPToolInfo` is frozen, but
            # `frozen=True` does not freeze nested mutables, and a shared reference
            # would let one holder mutate every other's view.
            schemas = {
                mcp_tool.name: copy.deepcopy(mcp_tool.input_schema)
                for mcp_tool in by_server[server_name]
            }
            tool_infos = [
                MCPToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=schemas.get(tool.name),
                )
                for tool in server_tools
            ]
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            error = (
                f"MCP server {server_name!r}: tool construction failed "
                "after resolving environment variables."
                if redact_failure_details
                else str(exc)
            )
            logger.warning(
                "MCP server '%s' skipped: tool construction failed",
                server_name,
                exc_info=None if redact_failure_details else exc,
            )
            return [], MCPServerInfo(
                name=server_name,
                transport=transports[server_name],
                status="error",
                error=error,
            )

        return server_tools, MCPServerInfo(
            name=server_name,
            transport=transports[server_name],
            tools=tuple(tool_infos),
            uses_oauth=server_name in oauth_servers,
        )

    all_tools: list[BaseTool] = []
    server_infos: list[MCPServerInfo] = []
    for server_name, server_config in server_items:
        if server_name in skipped:
            status, error = skipped[server_name]
            server_infos.append(
                MCPServerInfo(
                    name=server_name,
                    transport=transports[server_name],
                    status=status,
                    error=error,
                ),
            )
            continue
        server_tools, server_info = _build_server(server_name, server_config)
        all_tools.extend(server_tools)
        server_infos.append(server_info)

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


def _log_skipped_project_servers(
    dropped: list[ProjectServerSummary],
    *,
    trust_project_mcp: bool | None,
    config_trusted: bool,
) -> None:
    """Log project MCP servers that were dropped, explaining why.

    Split out so the trust/drop loop stays readable. The message distinguishes an
    explicit reject on an otherwise-trusted config from the untrusted-drop cases,
    which themselves differ by whether trust was declined outright
    (`--trust-project-mcp` off) or merely not yet granted.

    Args:
        dropped: `ProjectServerSummary` rows for each skipped server.
        trust_project_mcp: The caller's tri-state trust flag.
        config_trusted: Whether the project config was otherwise trusted (so the
            only reason to drop is an explicit user-level deny entry).
    """
    skipped_list = "\n".join(
        f"- {name} [{kind}]: {summary}" for name, kind, summary in dropped
    )
    if config_trusted:
        logger.warning(
            "Skipped project MCP servers rejected by user config "
            "(disabled_project_servers):\n%s",
            skipped_list,
        )
    elif trust_project_mcp is False:
        logger.warning(
            "Skipped untrusted project MCP servers:\n%s",
            skipped_list,
        )
    else:
        logger.warning(
            "Skipped untrusted project MCP servers "
            "(config changed or not yet approved):\n%s",
            skipped_list,
        )


def _mcp_trust_list_notices(
    trust_lists: McpServerTrustLists,
) -> list[tuple[Path, str]]:
    """Config-error entries surfacing a trust-list's read/migration problems.

    The loader runs in non-interactive paths where a bare `logger.warning` has
    no handler, so these must-see notices are rendered as visible config errors
    via `_bad_config_infos`. Returned (rather than appended in place) so a
    single trust-list load can surface them once for both the plugin and
    project config paths instead of duplicating them per path.

    Args:
        trust_lists: The user's loaded allow/deny policy.

    Returns:
        `(path, message)` tuples for each detected problem, empty when clean.
    """
    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    notices: list[tuple[Path, str]] = []
    if trust_lists.read_error is not None:
        # Surface the read failure as a visible config error (a bare
        # logger.warning has no handler outside debug mode).
        notices.append((DEFAULT_CONFIG_PATH, trust_lists.read_error))
    if trust_lists.legacy_ignored:
        # The removed flat allowlist stops loading these silently; make it
        # visible since the loader runs in non-interactive paths where the
        # migration warning would otherwise be unseen.
        ignored = ", ".join(sorted(trust_lists.legacy_ignored))
        notices.append(
            (
                DEFAULT_CONFIG_PATH,
                (
                    "[mcp].enabled_project_servers is no longer used; "
                    "re-approve via the project MCP prompt to keep loading: "
                    f"{ignored}"
                ),
            )
        )
    if trust_lists.legacy_env_ignored:
        # The env var was renamed; make the set-but-ignored old name visible
        # so its servers don't silently stop pre-approving.
        notices.append(
            (
                Path("<env>"),
                (
                    f"{_env_vars.LEGACY_ENABLED_PROJECT_MCP_SERVERS} is no "
                    "longer used; it was renamed to "
                    f"{_env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS}"
                ),
            )
        )
    if trust_lists.malformed_approvals:
        # A corrupt saved approval would otherwise just silently re-prompt;
        # surface it here (the loader runs in non-interactive paths where a
        # bare logger.warning is unseen), mirroring the legacy notices above.
        count = trust_lists.malformed_approvals
        entry_word = "entry" if count == 1 else "entries"
        notices.append(
            (
                DEFAULT_CONFIG_PATH,
                (
                    f"{count} [mcp].enabled_project_server_approvals {entry_word} "
                    "could not be read and were ignored; re-approve via the "
                    "project MCP prompt to keep loading affected servers"
                ),
            )
        )
    return notices


async def resolve_and_load_mcp_tools(
    *,
    explicit_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    project_context: ProjectContext | None = None,
    additional_configs: tuple[dict[str, Any], ...] = (),
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
        trust_project_mcp: Controls project-level server trust.

            Applies to stdio and remote (http/sse) servers alike — remote entries
            are gated too because an attacker-controlled `.mcp.json` can SSRF or
            exfiltrate `${VAR}` headers during the discovery preflight.

            - `True`: grant whole-config trust (all servers load).
            - `False` / `None`: no whole-config trust. `None` is treated
                identically to `False` — the persistent trust store this once
                consulted was removed, so project servers load only via the
                user's scoped approvals / env allowlist described below.

            Regardless of this flag, the user-level allow/deny policy
            (`[mcp].enabled_project_server_approvals`,
            `[mcp].disabled_project_servers`, and env equivalents via
            `load_mcp_server_trust_lists`) is applied: scoped approvals load
            from an otherwise-untrusted config only when the project root and
            server fingerprint match, and explicitly denied servers are dropped
            even from a trusted one.
        project_context: Explicit project path context for config discovery
            and trust resolution.
        additional_configs: Config layers injected by higher-level composition,
            such as plugin-provided MCP servers. Installing a plugin is treated
            as the user's trust decision for its bundled servers, so these load
            without per-server approval — but the user-level deny policy still
            applies (an explicitly disabled server stays disabled), and if that
            policy cannot be read the servers fail closed rather than bypass a
            saved rejection. A malformed layer (non-dict, or a non-mapping
            `mcpServers`) is skipped and surfaced as a config error.
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
        RuntimeError: If the merged MCP config is malformed. (`${VAR}`
            config interpolation is deferred to activation inside
            `_load_tools_from_config`, which captures such failures into the
            returned `server_infos` rather than raising here.)
    """  # noqa: DOC502 - FileNotFoundError / JSONDecodeError / TypeError / ValueError surface via `load_mcp_config`
    if no_mcp:
        return [], None, []

    config_load_errors: list[tuple[Path, str]] = []

    try:
        config_sources = discover_mcp_config_sources(project_context=project_context)
    except (OSError, RuntimeError) as exc:
        logger.warning("MCP config auto-discovery failed", exc_info=True)
        config_sources = []
        config_load_errors.append((Path("<discovery>"), str(exc)))

    sources = MCPConfigSources.from_sources(config_sources)
    user_configs = sources.user_paths
    project_configs = sources.project_paths
    project_roots = sources.project_roots
    configs: list[dict[str, Any]] = []

    for path in user_configs:
        config, error = load_mcp_config_with_error(path)
        if error is not None:
            config_load_errors.append((path, error))
        if config is not None:
            configs.append(config)

    # The user-level allow/deny policy (home config.toml + env) gates both
    # plugin-provided and project `.mcp.json` servers. Load it once — and
    # surface its read/migration notices once — so a plugin-only session and a
    # project session behave identically and a read error is not reported
    # twice. Sourced only from the user's own config (never the repo), so a
    # committed `.mcp.json` cannot self-approve. Loaded lazily: skipped when
    # there is neither a plugin layer nor a discovered project config to gate.
    trust_lists: McpServerTrustLists | None = None
    if additional_configs or project_configs:
        from deepagents_code.model_config import load_mcp_server_trust_lists

        trust_lists = load_mcp_server_trust_lists()
        config_load_errors.extend(_mcp_trust_list_notices(trust_lists))

    # Installing a plugin is the user's trust decision for every bundled
    # component, including MCP servers. Still apply the user-level deny policy
    # so an explicitly disabled server stays disabled. If that policy cannot be
    # read, fail closed rather than potentially bypass a saved rejection. The
    # `trust_lists is not None` guard holds whenever `additional_configs` is
    # non-empty (the load above ran); it only narrows the type.
    if additional_configs and trust_lists is not None:
        plugin_project_root = _resolve_project_config_base(project_context)
        for plugin_config in additional_configs:
            if not isinstance(plugin_config, dict):
                continue
            plugin_servers = plugin_config.get("mcpServers")
            if plugin_servers is None or (
                isinstance(plugin_servers, dict) and not plugin_servers
            ):
                # No servers to contribute; nothing to trust-filter.
                continue
            if not isinstance(plugin_servers, dict):
                # A present-but-malformed `mcpServers` (e.g. a list or string)
                # is a plugin authoring mistake; surface it instead of dropping
                # it silently, mirroring how project configs report bad shapes.
                config_load_errors.append(
                    (
                        Path("<plugin>"),
                        (
                            "plugin 'mcpServers' must be a mapping of name to "
                            "server definition, got "
                            f"{type(plugin_servers).__name__}"
                        ),
                    )
                )
                continue
            plugin_kept = filter_trusted_project_servers(
                plugin_servers,
                trust_lists,
                project_root=plugin_project_root,
                config_trusted=not trust_lists.load_failed,
            )
            plugin_dropped = [
                name for name in plugin_servers if name not in plugin_kept
            ]
            if plugin_dropped:
                logger.warning(
                    "Skipped plugin MCP servers denied by an explicit disable or "
                    "an unreadable trust policy: %s",
                    ", ".join(sorted(plugin_dropped)),
                )
            if plugin_kept:
                configs.append({**plugin_config, "mcpServers": plugin_kept})

    loaded_project_configs: list[tuple[Path, dict[str, Any]]] = []

    for path in project_configs:
        config, error = _load_mcp_config_top_level_with_error(path)
        if error is not None:
            config_load_errors.append((path, error))
        if config is not None:
            loaded_project_configs.append((path, config))

    if loaded_project_configs and trust_lists is not None:
        # `trust_lists` was loaded above because `project_configs` is non-empty
        # here; the `is not None` guard only narrows the type. Its read/migration
        # notices were already surfaced once at the shared load site.
        project_config, server_sources = _merge_mcp_configs_with_sources(
            loaded_project_configs
        )
        project_servers = extract_project_server_summaries(project_config)

        # Whole-config trust comes only from the flag (`--trust-project-mcp`
        # or the interactive approval prompt's decision). Without it, servers
        # load solely via the user's scoped approvals below.
        config_trusted = trust_project_mcp is True

        if trust_lists.load_failed:
            # Fail closed: the user's allow/deny policy could not be read,
            # so do not honor whole-config trust. Env-enabled names still
            # survive because the trust-list loader discards scoped
            # approvals when it records a read error.
            config_trusted = False

        # Resolve precedence before trust. If a higher-precedence file changes
        # an approved server, rejecting that winning definition must not reveal
        # the stale approved definition beneath it. Every server — even a
        # malformed one — passes through the trust filter, so no entry can reach
        # `configs` without a trust decision (defense in depth against a future
        # validator that accepts a shape `extract_project_server_summaries`
        # currently skips).
        kept: dict[str, Any] = {}
        for name, server in project_config["mcpServers"].items():
            source = server_sources[name]
            # Indexed, not `.get`: see `MCPConfigSources.project_roots`.
            project_root = project_roots[source]
            kept.update(
                filter_trusted_project_servers(
                    {name: server},
                    trust_lists,
                    project_root=project_root,
                    config_trusted=config_trusted,
                )
            )

        if kept:
            filtered = {**project_config, "mcpServers": kept}
            valid, errors = _drop_invalid_mcp_config_servers(filtered)
            for name, error in errors.items():
                logger.warning(
                    "Skipping invalid trusted project MCP server %r: %s",
                    name,
                    error,
                )
                config_load_errors.append((server_sources[name], error))
            if valid["mcpServers"]:
                configs.append(valid)
        elif not project_servers:
            # Nothing was trusted and no dict server produced a summary, so
            # every entry is malformed. Re-validate the merged config (no second
            # file read) to surface a precise per-server error instead of
            # dropping the file silently.
            try:
                _validate_mcp_config_servers(project_config)
            except (ValueError, TypeError, RuntimeError) as exc:
                config_load_errors.append((loaded_project_configs[-1][0], str(exc)))

        # Servers dropped by the trust decision are logged only after
        # precedence resolution, so shadowed definitions cannot be reported
        # or loaded as if they were still active.
        dropped = [summary for summary in project_servers if summary.name not in kept]
        if dropped:
            _log_skipped_project_servers(
                dropped,
                trust_project_mcp=trust_project_mcp,
                config_trusted=config_trusted,
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

    from deepagents_code.configuration.service import ManagedConfigError
    from deepagents_code.mcp_disabled import get_disabled_servers

    try:
        disabled_names = get_disabled_servers()
    except ManagedConfigError:
        # The managed deny list is unreadable, so no server can be shown to be
        # permitted. Deny every one rather than start the servers an
        # administrator may have blocked.
        logger.error(  # noqa: TRY400
            "Managed MCP policy is unreadable; disabling all MCP servers."
        )
        disabled_names = set(merged["mcpServers"])
    disabled_infos: list[MCPServerInfo] = []
    if disabled_names:
        active: dict[str, Any] = {}
        for server_name, server_config in merged["mcpServers"].items():
            if server_name in disabled_names:
                disabled_infos.append(
                    MCPServerInfo(
                        name=server_name,
                        transport=_resolve_server_type(server_config)
                        if isinstance(server_config, dict)
                        else "unknown",
                        status="disabled",
                        error="Disabled by user (F2 to re-enable).",
                    ),
                )
            else:
                active[server_name] = server_config
        merged = {"mcpServers": active}

    if not merged.get("mcpServers"):
        return [], None, disabled_infos + _bad_config_infos()

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
    server_infos.extend(disabled_infos)
    server_infos.extend(_bad_config_infos())
    return tools, manager, server_infos
