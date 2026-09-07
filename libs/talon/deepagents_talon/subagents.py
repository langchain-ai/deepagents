"""Explicit local tool attachments and task-only research graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from deepagents.middleware.subagents import SubAgent
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain.tools import ToolRuntime  # noqa: TC002  # tool schemas inspect injected annotations
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command  # noqa: TC002  # tool schemas resolve return annotations

from deepagents_talon.background import _IN_SUBAGENT

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from deepagents.middleware.async_subagents import AsyncSubAgent
    from deepagents.middleware.subagents import CompiledSubAgent
    from langchain.agents.middleware import InterruptOnConfig
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.language_models import BaseChatModel


class LocalSubAgent(SubAgent):
    """Local frontmatter additions resolved before SDK graph construction."""

    tool_names: NotRequired[list[str]]


class Attachment(TypedDict):
    """Credential-free capability inventory for one graph."""

    name: str
    mode: str
    tools: list[str] | None
    selectable_tools: NotRequired[list[str]]


_DELEGATION_TOOLS = frozenset(
    {
        "task",
        "start_async_task",
        "update_async_task",
        "cancel_async_task",
        "check_async_task",
        "list_async_tasks",
        "list_subagents",
        "cancel_subagent",
    }
)


class TaskTools(AgentMiddleware):
    """Let the main agent select general-purpose capabilities for each task."""

    def __init__(
        self,
        model: str | BaseChatModel,
        interrupt_on: Mapping[str, bool | InterruptOnConfig] | None,
        general: SubAgent | None = None,
    ) -> None:
        """Retain this graph's model and operator approval policy."""
        self._model = model
        self._interrupt_on = dict(interrupt_on or {})
        self._general = general or {}
        self._task: BaseTool | None = None

    def bind(self, catalog: Mapping[str, BaseTool]) -> list[str]:
        """Bind the compiled graph's tools and return selectable names.

        Args:
            catalog: Actual tools exposed by the parent graph.

        Returns:
            Names available for explicit attachment, excluding delegation.
        """
        available = {name: item for name, item in catalog.items() if name not in _DELEGATION_TOOLS}
        original = catalog["task"]

        @tool(
            "task",
            description=original.description
            + (
                " For general-purpose, tools defaults to none: choose exact tool names "
                "from get_agent_tools, including execute for shell access. Supply task context "
                "and skill instructions in description, or select read_file to read the skill. "
                "No parent history or skills are inherited. Named agents use configured tools."
            ),
        )
        async def task(
            description: str,
            subagent_type: str,
            runtime: ToolRuntime,
            tools: list[str] | None = None,
        ) -> str | Command:
            if subagent_type != "general-purpose":
                if tools is not None:
                    return "Named subagent tools are fixed by configuration."
                return await original.ainvoke(
                    {
                        "description": description,
                        "subagent_type": subagent_type,
                        "runtime": runtime,
                    }
                )
            tools = tools or []
            if len(tools) != len(set(tools)) or any(name not in available for name in tools):
                return "Specify tools as a list of unique names from get_agent_tools."
            spec: LocalSubAgent = {
                "name": "general-purpose",
                "description": "Complete the delegated task.",
                "system_prompt": "Complete only the delegated task using the selected tools.",
                "tools": [available[name] for name in tools],
            }
            if "model" in self._general:
                spec["model"] = self._general["model"]
            if "system_prompt" in self._general:
                spec["system_prompt"] = self._general["system_prompt"]
            agent = _compile_fresh(spec, self._model, self._interrupt_on)["runnable"]
            result = await agent.ainvoke({"messages": [HumanMessage(description)]})
            if result.get("__interrupt__"):
                return "Subagent needs tool approval; the protected action has not run."
            return str(result["messages"][-1].content)

        self._task = task
        return sorted(available)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Expose the extended task schema to the main agent."""
        return await handler(
            request.override(
                tools=[
                    self._task if getattr(item, "name", None) == "task" and self._task else item
                    for item in request.tools
                ]
            )
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Select the wrapper before background dispatch snapshots the tool."""
        if request.tool_call["name"] == "task" and self._task:
            if _IN_SUBAGENT.get():
                return ToolMessage(
                    "Delegate from the main agent.", tool_call_id=request.tool_call["id"]
                )
            request = request.override(tool=self._task)
        return await handler(request)


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
        ValueError: An attachment is unavailable or a configuration is unsupported.
    """
    available = _tool_map(tools)
    candidates = list(specs)
    _add_general_subagent(candidates)
    prepared: list[SubAgent | CompiledSubAgent | AsyncSubAgent] = []
    inventory: list[Attachment] = []
    for original in candidates:
        if original["name"] == "general-purpose" and (
            "runnable" in original or "graph_id" in original
        ):
            msg = "Compiled and remote subagents must use a name other than 'general-purpose'"
            raise ValueError(msg)
        if original.get("mode") == "fork":
            msg = "Talon subagents use fresh context; fork mode is unsupported"
            raise ValueError(msg)
        spec = cast("LocalSubAgent", original.copy())
        if "tool_names" in spec:
            names = spec.pop("tool_names")
            if any(name not in available for name in names):
                msg = "Subagent attachment is unavailable; previous configuration retained"
                raise ValueError(msg)
            spec["tools"] = [available[name] for name in names]
        opaque = "graph_id" in spec or "runnable" in spec
        inventory.append(
            {
                "name": spec["name"],
                "mode": "remote" if "graph_id" in spec else "fresh",
                "tools": None if opaque else sorted(_tool_map(spec.get("tools", []))),
            }
        )
        if "runnable" in original:
            compiled = cast("CompiledSubAgent", original.copy())
            compiled.pop("mode", None)
            compiled["runnable"] = RunnableLambda(_task_only) | compiled["runnable"]
            prepared.append(compiled)
        elif spec["name"] == "general-purpose":
            prepared.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "runnable": RunnableLambda(_task_only),
                }
            )
        else:
            prepared.append(original if opaque else _compile_fresh(spec, model, interrupt_on))
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


def _add_general_subagent(specs: list[SubAgent | CompiledSubAgent | AsyncSubAgent]) -> None:
    existing = next((spec for spec in specs if spec["name"] == "general-purpose"), None)
    if existing is not None:
        return
    fallback: LocalSubAgent = {
        "name": "general-purpose",
        "description": "Complete a task with tools explicitly selected by the main agent.",
        "system_prompt": "Answer only the delegated question using the supplied information.",
        "tool_names": [],
    }
    specs.append(fallback)
