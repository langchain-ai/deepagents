"""Enumerate the tools available to the agent.

Backs two entry points: the `dcode tools list` CLI command (`_run_tools_list`)
and the interactive `/tools` slash command (`app._handle_tools_command`).

The tool set is read from the *real* tool objects the agent binds rather than a
hand-maintained catalog, so names and descriptions never drift from what the
model actually sees. Built-in tools are collected by compiling the agent with a
throwaway offline chat model (no credentials, no network) and reading the bound
tool node; MCP tools are discovered via the same path the app and server use.

The collection functions here lazily import the heavy agent stack (agent
compilation, MCP discovery) inside their bodies. Only the fake-model base is
imported at module top, so importing this module is cheap relative to the agent
stack — and this module is itself imported lazily by both entry points
(`_run_tools_list` and `_handle_tools_command`), never on the startup hot path.
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

    from deepagents import FsToolName
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

# Mirror of the SDK's `FsToolName` literal members, used to identify which
# enumerated tools the `fs_tools` allowlist governs. Kept as a literal set (the
# `get_args(FsToolName)` drift guard in `test_tool_catalog` pins it) so a new or
# renamed SDK filesystem tool fails the test instead of silently escaping the
# post-filter below.
_FILESYSTEM_TOOL_NAMES = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
)


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

    def __post_init__(self) -> None:
        """Enforce that an unavailable server is never `ok`.

        An `ok` server exposes tools and belongs in a `ToolGroup`, never here;
        rejecting it at construction keeps the documented non-`ok` invariant
        from being silently violated by a future producer.

        Raises:
            ValueError: If `status` is `"ok"`.
        """
        if self.status == "ok":
            msg = "UnavailableServer.status must be a non-'ok' MCPServerStatus"
            raise ValueError(msg)


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


def unavailable_server_display(server: UnavailableServer) -> tuple[str, str]:
    """Return the `(status_label, detail)` display pair for an unavailable server.

    Shared by the CLI (`client.commands.tools._print_unavailable_servers`) and
    TUI (`app._render_tool_catalog`) renderers so both describe a server the same
    way. A disabled server shows its reconnect guidance if present, else the
    generic "disabled by user", with no separate detail; other statuses show the
    status token plus discovery's reason string when present.

    Args:
        server: A server that loaded with no usable tools.

    Returns:
        `(status_label, detail)`: the primary status text and any secondary
        detail (`""` when none). Each renderer lays these out itself, e.g. as
        `status_label: detail`.
    """
    if server.status == "disabled":
        return (server.detail or "disabled by user", "")
    return (server.status, server.detail)


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
    *,
    assistant_id: str = "agent",
    enable_interpreter: bool = False,
    fs_tools: Literal["all"] | list[FsToolName] | None = None,
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
        fs_tools: Filesystem tool allowlist. Forwarded to the catalog agent so
            it is built exactly like the runtime session, then applied as a
            defensive post-filter below. The SDK's `FilesystemMiddleware` omits
            disallowed tools from the node entirely, so forwarding alone already
            narrows the enumeration; the post-filter is a backstop that keeps
            the listing correct if that ever stops holding (see the comment on
            the filter below).

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
        fs_tools=fs_tools,
    )
    tools = collect_tools_from_agent(agent)
    if tools is None:
        msg = "Compiled agent does not expose a LangGraph tool node"
        raise RuntimeError(msg)
    # Defensive backstop, normally a no-op: the SDK's `FilesystemMiddleware`
    # omits disallowed filesystem tools from the node entirely (its own source
    # comment: "Excluded tools are omitted here entirely, not just hidden from
    # the model's schema"), so `collect_tools_from_agent` already returns only
    # the allowlisted filesystem tools. This filter is kept as belt-and-braces
    # so `/tools` / `dcode tools list` still reflects an explicit allowlist if
    # that SDK behavior changes, or if the by-name middleware replacement ever
    # left a second, unrestricted `FilesystemMiddleware` bound. Since it only
    # ever removes already-absent tools, it is safe to keep and safe to drop.
    # (`"all"` and `None` intentionally skip filtering.)
    if isinstance(fs_tools, list):
        enabled = frozenset(fs_tools)
        return [
            tool
            for tool in tools
            if tool.name not in _FILESYSTEM_TOOL_NAMES or tool.name in enabled
        ]
    return tools


def collect_tools_from_agent(agent: object) -> list[ToolEntry] | None:
    """Read tools from a local compiled agent when its graph is inspectable.

    LangGraph does not expose a public tool-enumeration API, so this reaches
    through the compiled graph's conventional `nodes["tools"].bound` shape.
    Returning `None` distinguishes an uninspectable graph (a remote agent, or a
    local graph whose internals no longer match that convention) from a local
    graph that validly binds zero tools (`[]`).

    Args:
        agent: Active local or remote agent object.

    Returns:
        Bound tools in graph order; `[]` for an inspectable local graph with no
        tools; or `None` when the agent cannot be inspected locally.
    """
    nodes = getattr(agent, "nodes", None)
    if not isinstance(nodes, Mapping):
        # No conventional node map: a remote agent or a non-graph object. Expected
        # for remote agents, so debug rather than warning.
        logger.debug("Agent %r has no inspectable node map", type(agent))
        return None
    if "tools" not in nodes:
        # LangChain omits the tool node when an otherwise valid local agent
        # binds no tools. The graph is still inspectable; its tool set is empty.
        return []
    node = nodes.get("tools")
    tool_node = cast("ToolNode | None", getattr(node, "bound", None))
    tools_by_name = getattr(tool_node, "tools_by_name", None)
    if not isinstance(tools_by_name, Mapping):
        # A "tools" node exists but does not expose the expected
        # `bound.tools_by_name` mapping — a LangGraph internal-shape change, not
        # a remote agent. Warn so this drift is visible in logs even though the
        # user-facing notice attributes it to an uninspectable agent.
        logger.warning(
            "Agent 'tools' node is not introspectable (bound=%r); "
            "LangGraph internals may have changed",
            type(tool_node),
        )
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
            # explained. A plainly-disabled server drops discovery's reason so
            # the renderers show the generic "disabled by user" label; a
            # just-re-enabled one (`pending_reconnect`) keeps its reconnect
            # guidance so the renderer can distinguish it from a server the user
            # left disabled. Other statuses retain discovery's reason string —
            # not a stack trace, but config-load failures can include the local
            # config file path — see `UnavailableServer.detail`.
            detail = server.error or ""
            if server.status == "disabled" and not server.pending_reconnect:
                detail = ""
            unavailable.append(
                UnavailableServer(
                    name=server.name,
                    status=server.status,
                    detail=detail,
                )
            )
    return groups, unavailable


def build_catalog_from_server_info(
    built_in: Sequence[ToolEntry],
    server_info: Sequence[MCPServerInfo],
) -> ToolCatalog:
    """Assemble a `ToolCatalog` from pre-collected built-in tools and live MCP info.

    The interactive `/tools` command entry point: it avoids the `asyncio.run`
    MCP discovery reached via `collect_catalog` (the `asyncio.run` call itself
    lives in `collect_mcp_catalog`), which cannot run inside Textual's running
    event loop, by reusing the MCP metadata the app already loaded. `mcp_error`
    is always `None` here because discovery is not attempted — any load failures
    are already reflected per-server in `server_info` as non-`ok` `MCPServerInfo`
    entries, which `split_mcp_server_info` surfaces as `UnavailableServer`s.

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
    from deepagents_code.plugins.adapters.mcp import discover_plugin_mcp_configs
    from deepagents_code.project_utils import ProjectContext

    try:
        project_context = ProjectContext.from_user_cwd(Path.cwd())
    except (OSError, RuntimeError):
        # `Path.cwd()`/`.resolve()` raise OSError for a missing cwd and
        # RuntimeError on a symlink loop (3.11-3.12); match the codebase's own
        # convention in `project_utils` and fall back to no project context.
        logger.warning("Could not determine working directory for MCP discovery")
        project_context = None
    project_dir = (
        project_context.project_root or project_context.user_cwd
        if project_context is not None
        else None
    )

    session_manager = None
    try:
        _tools, session_manager, server_info = await resolve_and_load_mcp_tools(
            explicit_config_path=mcp_config_path,
            no_mcp=False,
            trust_project_mcp=trust_project_mcp,
            project_context=project_context,
            additional_configs=discover_plugin_mcp_configs(project_dir=project_dir),
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
    fs_tools: Literal["all"] | list[FsToolName] | None = None,
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
        fs_tools: Filesystem tool allowlist; forwarded to
            `collect_built_in_tools`, which filters the built-in enumeration so
            it matches the configured session.
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
                    fs_tools=fs_tools,
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
