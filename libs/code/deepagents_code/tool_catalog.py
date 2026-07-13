"""Enumerate the tools available to the agent for `dcode tools list`.

The tool set is read from the *real* tool objects the agent binds rather than a
hand-maintained catalog, so names and descriptions never drift from what the
model actually sees. Built-in tools are collected by compiling the agent with a
throwaway offline chat model (no credentials, no network) and reading the bound
tool node; MCP tools are discovered via the same path the app and server use.

The collection functions here lazily import the heavy agent stack (agent
compilation, MCP discovery) inside their bodies. Only the fake-model base is
imported at module top, so importing this module is cheap relative to the agent
stack — and this module is itself imported lazily inside `_run_tools_list`,
never on the startup hot path. Those functions must only run on the
`dcode tools list` command path.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from deepagents_code._fake_models import _ToolBindingFakeModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langgraph.prebuilt.tool_node import ToolNode

    from deepagents_code.mcp_tools import MCPServerInfo, MCPServerStatus

logger = logging.getLogger(__name__)

ToolSource = Literal["built-in", "mcp"]
"""Stable source token identifying where a tool group comes from.

Emitted verbatim in the `--json` output, so it is a public contract; keep it a
`Literal` of stable tokens (not a bare `str`), following the same convention as
`mcp_tools.MCPServerStatus`.
"""

BUILT_IN_GROUP = "Built-in"
"""Display label for the group of tools bundled with `deepagents-code`."""


@dataclass(frozen=True, slots=True)
class ToolEntry:
    """A single tool's display metadata."""

    name: str
    """Tool name as bound on the agent (e.g. `read_file`)."""

    description: str
    """First non-empty line of the tool's description, whitespace-collapsed."""


@dataclass(frozen=True, slots=True)
class ToolGroup:
    """A named group of tools sharing a source."""

    label: str
    """Group heading (`Built-in`, or the MCP server name)."""

    source: ToolSource
    """Stable source token: `built-in` or `mcp`."""

    tools: tuple[ToolEntry, ...]
    """Tools in this group, in bind order."""


@dataclass(frozen=True, slots=True)
class UnavailableServer:
    """An MCP server that was discovered but currently exposes no tools."""

    name: str
    """Server name from the MCP configuration."""

    status: MCPServerStatus
    """Load status token — any non-`ok` `mcp_tools.MCPServerStatus`.

    Reuses `MCPServerStatus` (rather than a bare `str`) so the same closed value
    set governs this field and its `--json` output, and so the token list has a
    single source of truth in `mcp_tools`.
    """

    detail: str
    """Human-readable reason from discovery, or `""` when none was given.

    For config-load failures this is discovery's own reason string, which may
    include the local config file path (e.g. `~/.deepagents/mcp.json: ...`) —
    the same text the interactive `/mcp` viewer shows. See `collect_mcp_catalog`.
    """


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    """Everything `dcode tools list` needs to render, in display order."""

    groups: tuple[ToolGroup, ...]
    """Built-in group first, then one group per MCP server that exposes tools."""

    unavailable: tuple[UnavailableServer, ...] = ()
    """MCP servers discovered with no tools (errored, needing login, or disabled).

    Surfaced rather than dropped so a user debugging a missing tool can see why
    it is absent.
    """

    mcp_error: str | None = None
    """Generic notice set when MCP discovery itself failed; `None` on success.

    Raw exception detail is logged at debug level, never embedded here, so no
    file paths or stack traces leak into CLI/JSON output.
    """


class _CatalogModel(_ToolBindingFakeModel):
    """Offline placeholder model used only to compile the agent for enumeration.

    Compiling the agent binds every tool but never calls the model, so this
    never issues a request — enumeration only reads the bound tool node. It
    exists so tool enumeration works without credentials or network access.
    Inherits the `bind_tools` passthrough and minimal `profile` the agent
    runtime reads during setup from `_ToolBindingFakeModel`.
    """

    model: str = "catalog"


def _first_line(text: str | None) -> str:
    """Return the first non-empty line of `text`, whitespace-collapsed."""
    if not text:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return " ".join(stripped.split())
    return ""


def collect_built_in_tools(
    *, assistant_id: str = "agent", enable_interpreter: bool = False
) -> list[ToolEntry]:
    """Enumerate the built-in tools the agent binds by default.

    Compiles the agent with an offline placeholder model and reads the bound
    tool node. Memory and skills are disabled because they contribute no tools
    (they only augment the system prompt). The selected assistant id is still
    forwarded so agent-specific subagents are loaded from the same directory the
    normal launch path uses. The custom CLI tools are included the same way
    `server_graph._build_tools` adds them, so `web_search` appears only when
    Tavily is configured.

    Args:
        assistant_id: Resolved dcode agent identifier to compile.
        enable_interpreter: Wire the JS interpreter middleware so `js_eval`
            appears when the default agent would bind it. Callers should pass
            the resolved runtime setting (see `_resolve_enable_interpreter`) so
            the list matches the tools the agent actually binds.

    Returns:
        Built-in tools in bind order.

    Raises:
        RuntimeError: If the compiled graph does not expose its bound tools.
    """
    from deepagents_code.agent import create_cli_agent
    from deepagents_code.config import settings
    from deepagents_code.tools import fetch_url, get_current_thread_id, web_search

    # Keep in sync with `server_graph._build_tools`: web_search is bound only
    # when Tavily is configured, so it appears here only under the same gate.
    custom_tools: list[Any] = [fetch_url, get_current_thread_id]
    if settings.has_tavily:
        custom_tools.append(web_search)

    agent, _backend = create_cli_agent(
        _CatalogModel(),
        assistant_id=assistant_id,
        tools=custom_tools,
        enable_memory=False,
        enable_skills=False,
        enable_shell=True,
        enable_interpreter=enable_interpreter,
    )
    tools = collect_tools_from_agent(agent)
    if tools is None:
        msg = "Compiled agent does not expose a LangGraph tool node"
        raise RuntimeError(msg)
    return tools


def collect_tools_from_agent(agent: object) -> list[ToolEntry] | None:
    """Read tools from a local compiled agent when its graph is inspectable.

    LangGraph does not expose a public tool-enumeration API, so this reaches
    through the compiled graph's conventional `nodes["tools"].bound` shape.
    Returning `None` distinguishes an unsupported graph (including a remote
    agent) from a local graph that validly binds zero tools.

    Args:
        agent: Active local or remote agent object.

    Returns:
        Bound tools in graph order, or `None` when the agent cannot be
        inspected locally.
    """
    nodes = getattr(agent, "nodes", None)
    if not isinstance(nodes, Mapping):
        return None
    node = nodes.get("tools")
    tool_node = cast("ToolNode | None", getattr(node, "bound", None))
    tools_by_name = getattr(tool_node, "tools_by_name", None)
    if not isinstance(tools_by_name, Mapping):
        return None
    tools: list[ToolEntry] = []
    for name, tool in tools_by_name.items():
        if not isinstance(name, str):
            continue
        description = getattr(tool, "description", None)
        tools.append(
            ToolEntry(
                name=name,
                description=_first_line(
                    description if isinstance(description, str) else None
                ),
            )
        )
    return tools


def collect_mcp_catalog(
    *,
    mcp_config_path: str | None = None,
    trust_project_mcp: bool | None = None,
) -> tuple[list[ToolGroup], list[UnavailableServer], str | None]:
    """Discover MCP servers, split into tool groups and unavailable servers.

    Best-effort: if discovery itself raises (no config, offline, load error),
    the technical detail is logged and a generic `mcp_error` message is
    returned so `dcode tools list` still renders the built-in tools while
    telling the user discovery failed. Servers that loaded but expose no tools
    are reported as `UnavailableServer`s (errored, needing login, or disabled)
    rather than silently dropped — surfacing exactly what a user running this
    command to debug a missing tool needs to see.

    Args:
        mcp_config_path: Explicit MCP config path (`--mcp-config`), or `None`
            to rely on auto-discovery.
        trust_project_mcp: Project-level stdio trust decision
            (`--trust-project-mcp`), forwarded to discovery unchanged.

    Returns:
        `(groups, unavailable, mcp_error)`: per-server tool groups, discovered
        servers exposing no tools, and a generic discovery-failure message
        (`None` when discovery succeeded).
    """
    try:
        server_info = asyncio.run(
            _load_mcp_server_info(
                mcp_config_path=mcp_config_path,
                trust_project_mcp=trust_project_mcp,
            )
        )
    except Exception:
        # Log the real cause for debugging, but return a generic message so no
        # file path or stack trace leaks into CLI/JSON output.
        logger.warning("MCP tool discovery failed for `tools list`", exc_info=True)
        return [], [], "MCP discovery failed; showing built-in tools only."

    groups, unavailable = split_mcp_server_info(server_info)
    return groups, unavailable, None


def split_mcp_server_info(
    server_info: Sequence[MCPServerInfo],
) -> tuple[list[ToolGroup], list[UnavailableServer]]:
    """Split loaded MCP server metadata into tool groups and unavailable servers.

    Pure function shared by the CLI discovery path (`collect_mcp_catalog`) and
    the interactive `/tools` command, which passes the app's already-loaded
    `MCPServerInfo` list rather than re-discovering (Textual's running event
    loop forbids the `asyncio.run` discovery path).

    Servers that loaded but expose no tools are reported as `UnavailableServer`s
    (errored, needing login, or disabled) rather than silently dropped —
    surfacing exactly what a user debugging a missing tool needs to see.

    Args:
        server_info: Loaded MCP server metadata.

    Returns:
        `(groups, unavailable)`: per-server tool groups (only servers exposing
        tools) and servers discovered with no tools and a non-`ok` status.
    """
    groups: list[ToolGroup] = []
    unavailable: list[UnavailableServer] = []
    for server in server_info:
        if server.tools:
            entries = tuple(
                ToolEntry(name=tool.name, description=_first_line(tool.description))
                for tool in server.tools
            )
            groups.append(ToolGroup(label=server.name, source="mcp", tools=entries))
        elif server.status != "ok":
            # A server that loaded but has no tools *and* is not "ok" is broken,
            # unauthenticated, or disabled — report it so the omission is
            # explained. `server.error` is discovery's own reason string (the
            # same text the interactive `/mcp` viewer shows); it is not a stack
            # trace, but for config-load failures it can include the local
            # config file path — see `UnavailableServer.detail`.
            unavailable.append(
                UnavailableServer(
                    name=server.name,
                    status=server.status,
                    detail=server.error or "",
                )
            )
    return groups, unavailable


def build_catalog_from_server_info(
    built_in: Sequence[ToolEntry],
    server_info: Sequence[MCPServerInfo],
) -> ToolCatalog:
    """Assemble a `ToolCatalog` from pre-collected built-in tools and live MCP info.

    The interactive `/tools` command entry point: it avoids the `asyncio.run`
    MCP discovery in `collect_catalog` (unusable inside Textual's running event
    loop) by reusing the MCP metadata the app already loaded. `mcp_error` is
    always `None` here because discovery is not attempted — any load failures
    are already reflected per-server in `server_info` as `UnavailableServer`s.

    Args:
        built_in: Built-in tools in bind order (from `collect_built_in_tools`).
        server_info: The app's already-loaded MCP server metadata.

    Returns:
        A `ToolCatalog` with the built-in group first, then any MCP groups, plus
        unavailable servers.
    """
    groups: list[ToolGroup] = [
        ToolGroup(label=BUILT_IN_GROUP, source="built-in", tools=tuple(built_in))
    ]
    mcp_groups, unavailable = split_mcp_server_info(server_info)
    groups.extend(mcp_groups)
    return ToolCatalog(groups=tuple(groups), unavailable=tuple(unavailable))


async def _load_mcp_server_info(
    *,
    mcp_config_path: str | None,
    trust_project_mcp: bool | None,
) -> list[Any]:
    """Load MCP server metadata, cleaning up any temporary sessions.

    Args:
        mcp_config_path: Explicit MCP config path, or `None` for auto-discovery.
        trust_project_mcp: Project-level stdio trust decision.

    Returns:
        Discovered MCP server metadata, or an empty list when none load.
    """
    from deepagents_code.mcp_tools import resolve_and_load_mcp_tools
    from deepagents_code.project_utils import ProjectContext

    try:
        project_context = ProjectContext.from_user_cwd(Path.cwd())
    except (OSError, RuntimeError):
        # `Path.cwd()`/`.resolve()` raise OSError for a missing cwd and
        # RuntimeError on a symlink loop (3.11-3.12); match the codebase's own
        # convention in `project_utils` and fall back to no project context.
        logger.warning("Could not determine working directory for MCP discovery")
        project_context = None

    session_manager = None
    try:
        _tools, session_manager, server_info = await resolve_and_load_mcp_tools(
            explicit_config_path=mcp_config_path,
            no_mcp=False,
            trust_project_mcp=trust_project_mcp,
            project_context=project_context,
        )
        return server_info or []
    finally:
        if session_manager is not None:
            try:
                await session_manager.cleanup()
            except Exception:
                logger.warning("MCP discovery cleanup failed", exc_info=True)


def collect_catalog(
    *,
    assistant_id: str = "agent",
    enable_interpreter: bool = False,
    include_mcp: bool = True,
    mcp_config_path: str | None = None,
    trust_project_mcp: bool | None = None,
) -> ToolCatalog:
    """Collect everything `dcode tools list` renders.

    Args:
        assistant_id: Resolved dcode agent identifier to compile for built-in
            tools, including any agent-specific subagents.
        enable_interpreter: Whether the default agent binds `js_eval`; forwarded
            to `collect_built_in_tools`.
        include_mcp: When `True`, discover MCP servers and append their groups
            after the built-in group (best-effort). Pass `False` to mirror
            `--no-mcp`.
        mcp_config_path: Explicit MCP config path (`--mcp-config`).
        trust_project_mcp: Project-level stdio trust decision
            (`--trust-project-mcp`).

    Returns:
        A `ToolCatalog` with the built-in group first, then any MCP groups,
        plus unavailable servers and any discovery-failure notice.
    """
    groups: list[ToolGroup] = [
        ToolGroup(
            label=BUILT_IN_GROUP,
            source="built-in",
            tools=tuple(
                collect_built_in_tools(
                    assistant_id=assistant_id,
                    enable_interpreter=enable_interpreter,
                )
            ),
        )
    ]
    unavailable: list[UnavailableServer] = []
    mcp_error: str | None = None
    if include_mcp:
        mcp_groups, unavailable, mcp_error = collect_mcp_catalog(
            mcp_config_path=mcp_config_path,
            trust_project_mcp=trust_project_mcp,
        )
        groups.extend(mcp_groups)
    return ToolCatalog(
        groups=tuple(groups),
        unavailable=tuple(unavailable),
        mcp_error=mcp_error,
    )
