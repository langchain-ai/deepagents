"""Enumerate the tools available to the agent for `dcode tools list`.

The tool set is read from the *real* tool objects the agent binds rather than a
hand-maintained catalog, so names and descriptions never drift from what the
model actually sees. Built-in tools are collected by compiling the agent with a
throwaway offline chat model (no credentials, no network) and reading the bound
tool node; MCP tools are discovered via the same path the app and server use.

This module imports the heavy agent stack, so it must only be imported from the
`dcode tools list` command path — never on the startup hot path.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt.tool_node import ToolNode

logger = logging.getLogger(__name__)

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

    source: str
    """Stable source token: `built-in` or `mcp`."""

    tools: tuple[ToolEntry, ...]
    """Tools in this group, in bind order."""


class _CatalogModel(GenericFakeChatModel):
    """Offline placeholder model used only to compile the agent for enumeration.

    Compiling the agent binds every tool but never calls the model, so this
    never issues a request. It exists so tool enumeration works without
    credentials or network access. Subclasses `GenericFakeChatModel` to stay
    aligned with LangChain's fake-model surface and provides the `bind_tools`
    passthrough and `profile` the agent runtime reads during setup.
    """

    model: str = "catalog"
    messages: object = Field(default_factory=lambda: iter(()))
    profile: dict[str, Any] | None = Field(
        default_factory=lambda: {"tool_calling": True, "max_input_tokens": 8000}
    )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],  # noqa: ARG002
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Return self so the agent can bind tool schemas during enumeration."""
        return self


def _first_line(text: str | None) -> str:
    """Return the first non-empty line of `text`, whitespace-collapsed."""
    if not text:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return " ".join(stripped.split())
    return ""


def collect_built_in_tools() -> list[ToolEntry]:
    """Enumerate the built-in tools the agent binds by default.

    Compiles the agent with an offline placeholder model and reads the bound
    tool node. Memory and skills are disabled because they contribute no tools
    (they only augment the system prompt), which keeps enumeration free of
    on-disk agent-directory side effects. The custom CLI tools are included the
    same way `server_graph._build_tools` adds them, so `web_search` appears only
    when Tavily is configured.

    Returns:
        Built-in tools in bind order.
    """
    from deepagents_code.agent import create_cli_agent
    from deepagents_code.config import settings
    from deepagents_code.tools import fetch_url, get_current_thread_id, web_search

    custom_tools: list[Any] = [fetch_url, get_current_thread_id]
    if settings.has_tavily:
        custom_tools.append(web_search)

    agent, _backend = create_cli_agent(
        _CatalogModel(),
        assistant_id="agent",
        tools=custom_tools,
        enable_memory=False,
        enable_skills=False,
        enable_shell=True,
    )
    tool_node = cast("ToolNode", agent.nodes["tools"].bound)
    tools_by_name = tool_node.tools_by_name
    return [
        ToolEntry(name=name, description=_first_line(tool.description))
        for name, tool in tools_by_name.items()
    ]


def collect_mcp_tool_groups() -> list[ToolGroup]:
    """Discover MCP servers and their tools, grouped per server.

    Best-effort: any discovery failure (no config, offline, load error) is
    logged and yields no groups rather than raising, so `dcode tools list`
    always renders the built-in tools. Servers that expose no tools (errored,
    unauthenticated, or disabled) are omitted.

    Returns:
        One group per MCP server that exposes tools.
    """
    try:
        server_info = asyncio.run(_load_mcp_server_info())
    except Exception:
        logger.warning("MCP tool discovery failed for `tools list`", exc_info=True)
        return []

    groups: list[ToolGroup] = []
    for server in server_info or []:
        if not server.tools:
            continue
        entries = tuple(
            ToolEntry(name=tool.name, description=_first_line(tool.description))
            for tool in server.tools
        )
        groups.append(ToolGroup(label=server.name, source="mcp", tools=entries))
    return groups


async def _load_mcp_server_info() -> list[Any]:
    """Load MCP server metadata, cleaning up any temporary sessions.

    Returns:
        Discovered MCP server metadata, or an empty list when none load.
    """
    from deepagents_code.mcp_tools import resolve_and_load_mcp_tools
    from deepagents_code.project_utils import ProjectContext

    try:
        project_context = ProjectContext.from_user_cwd(Path.cwd())
    except OSError:
        logger.warning("Could not determine working directory for MCP discovery")
        project_context = None

    session_manager = None
    try:
        _tools, session_manager, server_info = await resolve_and_load_mcp_tools(
            explicit_config_path=None,
            no_mcp=False,
            trust_project_mcp=None,
            project_context=project_context,
        )
        return server_info or []
    finally:
        if session_manager is not None:
            try:
                await session_manager.cleanup()
            except Exception:
                logger.warning("MCP discovery cleanup failed", exc_info=True)


def collect_tool_groups(*, include_mcp: bool = True) -> list[ToolGroup]:
    """Collect all tool groups for `dcode tools list`.

    Args:
        include_mcp: When `True`, append per-server MCP groups after the
            built-in group. MCP discovery is best-effort.

    Returns:
        The built-in group followed by any MCP groups.
    """
    groups = [
        ToolGroup(
            label=BUILT_IN_GROUP,
            source="built-in",
            tools=tuple(collect_built_in_tools()),
        )
    ]
    if include_mcp:
        groups.extend(collect_mcp_tool_groups())
    return groups
