"""Read-only estimates of the context configured for a Talon conversation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from deepagents.middleware.memory import MEMORY_SYSTEM_PROMPT, MemoryState
from deepagents.middleware.skills import SkillsMiddleware, SkillsState
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

from deepagents_code.context_doctor import (
    build_context_doctor_report,
    format_memory_prompt,
    format_skills_prompt,
    render_context_doctor_report,
)
from deepagents_code.tool_catalog import ToolEntry, collect_tools_from_agent
from deepagents_talon.mcp_middleware import (
    MCP_SKILL_ONLY_METADATA_KEY,
    MCP_TOOL_METADATA_KEY,
)
from deepagents_talon.memory import TalonMemoryMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.language_models import BaseChatModel

    from deepagents_code.provider_tool_search import MCPProviderToolSearchMiddleware
    from deepagents_code.skills.load import ExtendedSkillMetadata

_MAX_MESSAGES = 10_000


def _initial_tools(
    graph: object,
    resolve_model: Callable[[], BaseChatModel],
    tool_search: MCPProviderToolSearchMiddleware,
) -> tuple[list[ToolEntry] | None, str]:
    entries = collect_tools_from_agent(graph)
    if not entries:
        return entries, ""
    node = getattr(graph, "nodes", {})["tools"].bound
    mcp = [
        tool
        for tool in node.tools_by_name.values()
        if isinstance(tool, BaseTool) and (tool.metadata or {}).get(MCP_TOOL_METADATA_KEY)
    ]
    hidden = {tool.name for tool in mcp if (tool.metadata or {}).get(MCP_SKILL_ONLY_METADATA_KEY)}
    ordinary = [tool for tool in mcp if tool.name not in hidden]
    deferred: set[str] = set()
    search = []
    if ordinary:
        request = ModelRequest(model=resolve_model(), messages=[], tools=[*ordinary])
        prepared = tool_search._prepare(request)  # noqa: SLF001  # Apply the graph's real loader without invoking the model.
        for definition in prepared.tools:
            if isinstance(definition, dict):
                search.append(ToolEntry(definition["type"], "", definition))
            elif (definition.extras or {}).get("defer_loading"):
                deferred.add(definition.name)
    excluded = hidden | deferred
    initial = [entry for entry in entries if entry.name not in excluded]
    details = []
    if deferred:
        details.append(f"deferred MCP tools: {len(deferred)}")
    if hidden:
        details.append(f"skill-only tools: {len(hidden)}")
    return [*initial, *search], ", ".join(details)


@dataclass(frozen=True, slots=True)
class ContextDoctor:
    """Configuration snapshot paired with a successfully compiled graph.

    Args:
        backend: The graph's backend for reading configured memory and skills.
        system_prompt: Resolved authored system instructions.
        skills: Configured skill sources, in precedence order.
        memory: Configured memory paths.
        tool_search: Native search middleware installed on the graph.
        default_model: Resolve the graph's default model for tool discovery.
    """

    backend: BackendProtocol
    system_prompt: str | None
    skills: tuple[str, ...]
    memory: tuple[str, ...]
    default_model: Callable[[], BaseChatModel]
    tool_search: MCPProviderToolSearchMiddleware

    async def render(
        self, graph: object, conversation_id: str, *, model: BaseChatModel | None = None
    ) -> str:
        """Inspect a checkpoint and format a bounded report without changing it.

        Args:
            graph: Active compiled graph belonging to this configuration.
            conversation_id: Host-resolved thread identifier.
            model: Host-selected model override.

        Returns:
            Estimated token counts without source contents or paths.
        """
        get_state = getattr(graph, "aget_state", None)
        if not callable(get_state):
            msg = "Graph does not expose async state inspection"
            raise TypeError(msg)
        snapshot = await get_state({"configurable": {"thread_id": conversation_id}})
        values = cast("Mapping[str, object]", snapshot.values)
        memory = await self._memory(values)
        skills = await self._skills(values)
        tools, tool_detail = await asyncio.to_thread(
            _initial_tools,
            graph,
            self.default_model if model is None else lambda: model,
            self.tool_search,
        )
        conversation, provider = await asyncio.to_thread(_usage, values)
        report = build_context_doctor_report(
            system_prompt=self.system_prompt or "",
            memory_prompt=format_memory_prompt(list(memory.items()), MEMORY_SYSTEM_PROMPT)
            if self.memory
            else "",
            memory_files=len(memory),
            skills_prompt=format_skills_prompt(skills, self.skills) if self.skills else "",
            skills=skills,
            built_in_tools=tools,
            mcp_servers=(),
            conversation_tokens=conversation,
            provider_tokens=provider,
        )
        rows = list(report.rows)
        rows[0] = replace(rows[0], label="System prompt (configured)")
        tool_row = rows[3]
        detail = "" if tool_row.detail == "sent with every request" else tool_row.detail
        rows[3] = replace(
            tool_row,
            label=tool_row.label.replace("Built-in", "Initial"),
            detail=", ".join(filter(None, (detail, tool_detail))),
        )
        rows = [replace(row, detail=" ".join(row.detail.split())[:240]) for row in rows]
        text = render_context_doctor_report(replace(report, rows=tuple(rows)))
        return (
            f"{text}\nMiddleware additions and provider overhead are not included.\n"
            "Provider count is the last model input; conversation uses the current checkpoint."
        )

    async def _memory(self, values: Mapping[str, object]) -> dict[str, str]:
        if not self.memory:
            return {}
        state = cast("MemoryState", dict(values))
        middleware = TalonMemoryMiddleware(backend=self.backend, sources=list(self.memory))
        loaded = await middleware.abefore_agent(state, Runtime(), {})
        return (loaded or state).get("memory_contents", {})

    async def _skills(self, values: Mapping[str, object]) -> list[ExtendedSkillMetadata]:
        if not self.skills:
            return []
        state = cast("SkillsState", dict(values))
        middleware = SkillsMiddleware(backend=self.backend, sources=list(self.skills))
        loaded = await middleware.abefore_agent(state, Runtime(), {})
        return [
            {**skill, "source": "user"} for skill in (loaded or state).get("skills_metadata") or []
        ]


def _usage(values: Mapping[str, object]) -> tuple[int | None, int | None]:
    messages = values.get("messages", [])
    if not isinstance(messages, list) or len(messages) > _MAX_MESSAGES:
        return None, None
    messages = [message for message in messages if isinstance(message, BaseMessage)]
    provider = next(
        (
            message.usage_metadata["input_tokens"]
            for message in reversed(messages)
            if isinstance(message, AIMessage) and message.usage_metadata
        ),
        None,
    )
    event = values.get("_summarization_event")
    if isinstance(event, dict):
        cutoff, summary = event.get("cutoff_index"), event.get("summary_message")
        if (
            isinstance(cutoff, int)
            and 0 <= cutoff <= len(messages)
            and isinstance(summary, BaseMessage)
        ):
            messages = [summary, *messages[cutoff:]]
    return count_tokens_approximately(messages), provider
