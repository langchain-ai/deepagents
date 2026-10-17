"""Read-only estimates of the context configured for a Talon conversation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from deepagents.middleware.memory import MEMORY_SYSTEM_PROMPT, MemoryMiddleware, MemoryState
from deepagents.middleware.skills import SkillsMiddleware, SkillsState
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime

from deepagents_code.context_doctor import (
    build_context_doctor_report,
    format_memory_prompt,
    format_skills_prompt,
    render_context_doctor_report,
)
from deepagents_code.tool_catalog import collect_tools_from_agent

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents.backends.protocol import BackendProtocol

    from deepagents_code.skills.load import ExtendedSkillMetadata

_MAX_MESSAGES = 10_000


@dataclass(frozen=True, slots=True)
class ContextDoctor:
    """Configuration snapshot paired with a successfully compiled graph.

    Args:
        backend: The graph's backend for reading configured memory and skills.
        system_prompt: Resolved authored system instructions.
        skills: Configured skill sources, in precedence order.
        memory: Configured memory paths.
    """

    backend: BackendProtocol
    system_prompt: str | None
    skills: tuple[str, ...]
    memory: tuple[str, ...]

    async def render(self, graph: object, conversation_id: str) -> str:
        """Inspect a checkpoint and format a bounded report without changing it.

        Args:
            graph: Active compiled graph belonging to this configuration.
            conversation_id: Host-resolved thread identifier.

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
        tools = await asyncio.to_thread(collect_tools_from_agent, graph)
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
        rows[3] = replace(rows[3], label=rows[3].label.replace("Built-in", "All (including MCP)"))
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
        middleware = MemoryMiddleware(backend=self.backend, sources=list(self.memory))
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
