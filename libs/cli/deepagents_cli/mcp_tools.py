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
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

    from deepagents_cli.project_utils import ProjectContext

logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """Metadata for a single MCP tool."""

    name: str
    description: str


@dataclass
class MCPServerInfo:
    """Metadata for a connected MCP server and its tools."""

    name: str
    transport: str
    tools: list[MCPToolInfo] = field(default_factory=list)


_SUPPORTED_REMOTE_TYPES = {"sse", "http"}
"""Supported transport types for remote MCP servers (SSE and HTTP)."""


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


def load_mcp_config(config_path: str) -> dict[str, Any]:
    """Load and validate MCP configuration from JSON file.

    Supports multiple server types:

    - stdio: Process-based servers with `command`, `args`, `env` fields (default)
    - sse: Server-Sent Events servers with `type: "sse"`, `url`, and optional `headers`
    - http: HTTP-based servers with `type: "http"`, `url`, and optional `headers`

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
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Skipping invalid MCP config %s: %s", config_path, e)
        return None


class MCPSessionManager:
    """Manages persistent MCP sessions for stateful stdio servers.

    This manager creates and maintains persistent sessions for stdio MCP
    servers, preventing server restarts on every tool call. Sessions are kept
    alive until explicitly cleaned up.
    """

    def __init__(self) -> None:
        """Initialize the session manager."""
        self.client: MultiServerMCPClient | None = None
        self.exit_stack = AsyncExitStack()

    async def cleanup(self) -> None:
        """Clean up all managed sessions and close connections."""
        await self.exit_stack.aclose()


async def _load_tools_from_config(
    config: dict[str, Any],
) -> tuple[list[BaseTool], MCPSessionManager, list[MCPServerInfo]]:
    """Build MCP connections from a validated config and load tools.

    This is the shared implementation used by both `get_mcp_tools` (explicit
    path) and `resolve_and_load_mcp_tools` (auto-discovery).

    Args:
        config: Validated MCP configuration dict with `mcpServers` key.

    Returns:
        Tuple of `(tools_list, session_manager, server_infos)`.

    Raises:
        RuntimeError: If MCP server fails to spawn or connect.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StdioConnection,
        StreamableHttpConnection,
    )
    from langchain_mcp_adapters.tools import load_mcp_tools

    # Create connections dict for MultiServerMCPClient
    # Convert Claude Desktop format to langchain-mcp-adapters format
    connections: dict[str, Connection] = {}
    for server_name, server_config in config["mcpServers"].items():
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
                conn["headers"] = server_config["headers"]
            connections[server_name] = conn
        else:
            # stdio server connection (default)
            connections[server_name] = StdioConnection(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env") or None,
                transport="stdio",
            )

    # Create session manager to track persistent sessions
    manager = MCPSessionManager()

    try:
        client = MultiServerMCPClient(connections=connections)
        manager.client = client
    except Exception as e:
        await manager.cleanup()
        error_msg = f"Failed to initialize MCP client: {e}"
        raise RuntimeError(error_msg) from e

    try:
        all_tools: list[BaseTool] = []
        server_infos: list[MCPServerInfo] = []
        for server_name, server_config in config["mcpServers"].items():
            try:
                session = await asyncio.wait_for(
                    manager.exit_stack.enter_async_context(client.session(server_name)),
                    timeout=10.0,
                )
                tools = await asyncio.wait_for(
                    load_mcp_tools(
                        session, server_name=server_name, tool_name_prefix=True
                    ),
                    timeout=10.0,
                )
                all_tools.extend(tools)
                server_infos.append(
                    MCPServerInfo(
                        name=server_name,
                        transport=_resolve_server_type(server_config),
                        tools=[
                            MCPToolInfo(name=t.name, description=t.description)
                            for t in tools
                        ],
                    )
                )
            except asyncio.TimeoutError as e:
                logger.error("Timeout connecting to MCP server '%s'", server_name)
                error_msg = f"Timeout connecting to MCP server '{server_name}' (10s limit)"
                raise RuntimeError(error_msg) from e
            except Exception as e:
                logger.error("Failed to connect to MCP server '%s': %s", server_name, e)
                error_msg = f"Failed to connect to MCP server '{server_name}': {e}"
                raise RuntimeError(error_msg) from e
    except Exception as e:
        await manager.cleanup()
        error_msg = (
            f"Failed to load tools from MCP server '{server_name}': {e}\n"
            "For stdio servers: Check that the command and args are correct,"
            " and that the MCP server is installed"
            " (e.g., run 'npx -y <package>' manually to test).\n"
            "For sse/http servers: Check that the URL is correct"
            " and the server is running."
        )
        raise RuntimeError(error_msg) from e

    return all_tools, manager, server_infos


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
        RuntimeError: If an MCP server config is invalid or fails to
            spawn/connect.
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
