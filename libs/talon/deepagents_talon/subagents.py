"""Explicit local tool attachments and task-only research graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from deepagents.middleware.subagents import SubAgent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from deepagents.middleware.async_subagents import AsyncSubAgent
    from deepagents.middleware.subagents import CompiledSubAgent
    from langchain.agents.middleware import InterruptOnConfig
    from langchain_core.language_models import BaseChatModel


class LocalSubAgent(SubAgent):
    """Local frontmatter additions resolved before SDK graph construction."""

    fresh: NotRequired[bool]
    tool_names: NotRequired[list[str]]


class Attachment(TypedDict):
    """Credential-free capability inventory for one graph."""

    name: str
    mode: str
    tools: list[str] | None


def _task_only(state: dict[str, object]) -> dict[str, object]:
    return {"messages": state["messages"]}


def _compile_fresh(
    spec: LocalSubAgent,
    model: str | BaseChatModel,
    interrupt_on: Mapping[str, bool | InterruptOnConfig] | None,
) -> CompiledSubAgent:
    approvals = {key: value for key, value in (interrupt_on or {}).items() if value}
    graph = create_agent(
        model=spec.get("model", model),
        tools=spec.get("tools", []),
        system_prompt=spec.get("system_prompt", ""),
        middleware=[HumanInTheLoopMiddleware(interrupt_on=approvals)] if approvals else [],
        checkpointer=False,
    )
    return {
        "name": spec["name"],
        "description": spec["description"],
        "runnable": RunnableLambda(_task_only) | graph,
    }


def prepare_subagents(
    specs: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent],
    tools: Sequence[BaseTool | Callable[..., object]],
    model: str | BaseChatModel,
    interrupt_on: Mapping[str, bool | InterruptOnConfig] | None,
) -> tuple[list[SubAgent | CompiledSubAgent | AsyncSubAgent], list[Attachment]]:
    """Resolve exact attachments, compiling fresh roles without inherited middleware.

    Args:
        specs: Loaded local, compiled, or remote definitions.
        tools: Currently available tools, including loaded MCP tools.
        model: Default model for fresh agents.
        interrupt_on: Operator approval policy retained by fresh agents.

    Returns:
        SDK definitions and a safe inventory; opaque agents have unknown tools.

    Raises:
        ValueError: An attachment is unavailable or the fallback is privileged.
    """
    available = _tool_map(tools)
    candidates = list(specs)
    if any(spec.get("fresh") is True for spec in candidates):
        _add_fresh_fallback(candidates)
    prepared: list[SubAgent | CompiledSubAgent | AsyncSubAgent] = []
    inventory: list[Attachment] = []
    for original in candidates:
        spec = cast("LocalSubAgent", original.copy())
        if "tool_names" in spec:
            names = spec.pop("tool_names")
            if any(name not in available for name in names):
                msg = "Subagent attachment is unavailable; previous configuration retained"
                raise ValueError(msg)
            spec["tools"] = [available[name] for name in names]
        fresh = spec.pop("fresh", False)
        inventory.append(
            {
                "name": spec["name"],
                "mode": "fresh"
                if fresh
                else str(spec.get("mode", "remote" if "graph_id" in spec else "isolated")),
                "tools": sorted(_tool_map(spec.get("tools", []))) if fresh else None,
            }
        )
        prepared.append(_compile_fresh(spec, model, interrupt_on) if fresh else spec)
    if not any(spec["name"] == "general-purpose" for spec in candidates):
        inventory.append({"name": "general-purpose", "mode": "isolated", "tools": None})
    return prepared, inventory


def _tool_map(tools: Sequence[BaseTool | Callable[..., object]]) -> dict[str, BaseTool]:
    result: dict[str, BaseTool] = {}
    for item in tools:
        resolved = item if isinstance(item, BaseTool) else tool(item)
        if resolved.name in result:
            msg = "Ambiguous tool names in subagent attachments"
            raise ValueError(msg)
        result[resolved.name] = resolved
    return result


def _add_fresh_fallback(specs: list[SubAgent | CompiledSubAgent | AsyncSubAgent]) -> None:
    existing = next((spec for spec in specs if spec["name"] == "general-purpose"), None)
    if existing is not None:
        if existing.get("fresh") is not True:
            msg = "Fresh subagents require a fresh general-purpose fallback"
            raise ValueError(msg)
        return
    fallback: LocalSubAgent = {
        "name": "general-purpose",
        "description": "Answer a delegated question without tools or private context.",
        "system_prompt": "Answer only the delegated question using the supplied information.",
        "fresh": True,
        "tool_names": [],
    }
    specs.append(fallback)
