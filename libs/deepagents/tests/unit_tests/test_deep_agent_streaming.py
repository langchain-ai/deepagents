"""Integration tests for create_deep_agent streaming via stream_v2/astream_v2.

Drives a real `create_deep_agent` graph end-to-end through the
streaming pipeline and asserts that subagents are surfaced as typed
child streams with the right projections. Runs in both sync and
async paths.
"""

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from deepagents import (
    SubagentRunStream,
    create_deep_agent,
)


class _Scripted(BaseChatModel):
    """Returns a scripted sequence of `AIMessage`s, clamping at the last."""

    responses: list[AIMessage] = Field(default_factory=list)
    tools: Sequence[dict[str, Any] | type | Callable | BaseTool] = ()
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = min(self._idx, len(self.responses) - 1)
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[i])])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tools = tools
        return self


@tool
def _inner(x: str) -> str:
    """Echo the input with a prefix."""
    return f"inner-{x}"


def _task_call(description: str, subagent_type: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="task",
                args={"description": description, "subagent_type": subagent_type},
            )
        ],
    )


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, args=args)],
    )


def _build_agent_with_one_subagent() -> CompiledStateGraph:
    parent = _Scripted(
        responses=[
            _task_call("look something up", "researcher", "call-parent-1"),
            AIMessage(content="parent done"),
        ]
    )
    subagent = _Scripted(
        responses=[
            _tool_call("_inner", {"x": "foo"}, "call-sub-1"),
            AIMessage(content="subagent done"),
        ]
    )
    return create_deep_agent(
        model=parent,
        tools=[],
        subagents=[
            {
                "name": "researcher",
                "description": "research subagent",
                "system_prompt": "you are a researcher",
                "model": subagent,
                "tools": [_inner],
            }
        ],
    )


class TestCreateDeepAgentAstreamV2:
    async def test_run_exposes_native_projections(self) -> None:
        """`run.subagents` / `.messages` / `.tool_calls` / `.middleware` all bound."""
        agent = _build_agent_with_one_subagent()
        run = await agent.astream_v2({"messages": [HumanMessage(content="go")]})
        for attr in ("subagents", "subgraphs", "messages", "tool_calls", "middleware", "values"):
            assert hasattr(run, attr), f"missing projection {attr!r}"

        # Drain to drive the pump to completion.
        async for _ in run.messages:
            pass

    async def test_subagents_yields_one_typed_handle(self) -> None:
        agent = _build_agent_with_one_subagent()
        run = await agent.astream_v2({"messages": [HumanMessage(content="go")]})

        handles: list[SubagentRunStream] = []
        async for sub in run.subagents:
            assert isinstance(sub, SubagentRunStream)
            handles.append(sub)
            async for _ in sub.messages:
                pass

        async for _ in run.messages:
            pass

        assert len(handles) == 1
        (sub,) = handles
        assert sub.name == "researcher"
        assert sub.status == "completed"
        # path is ("tools:<pregel_task_id>",) — the tool node hosting the subagent.
        assert len(sub.path) == 1
        assert sub.path[0].startswith("tools:")
        assert isinstance(sub.output, dict)
        assert "messages" in sub.output

    async def test_subagent_tool_calls_surface(self) -> None:
        agent = _build_agent_with_one_subagent()
        run = await agent.astream_v2({"messages": [HumanMessage(content="go")]})

        tool_names: list[str] = []
        async for sub in run.subagents:
            tool_names.extend([tc.tool_name async for tc in sub.tool_calls])
            async for _ in sub.messages:
                pass
        async for _ in run.messages:
            pass

        assert "_inner" in tool_names

    async def test_output_when_no_subagent_invoked(self) -> None:
        """A run that doesn't dispatch to a subagent still completes cleanly."""
        model = _Scripted(responses=[AIMessage(content="no tools today")])
        agent = create_deep_agent(
            model=model,
            tools=[],
            subagents=[
                {
                    "name": "researcher",
                    "description": "r",
                    "system_prompt": "p",
                    "model": _Scripted(responses=[AIMessage(content="unused")]),
                    "tools": [],
                }
            ],
        )
        run = await agent.astream_v2({"messages": [HumanMessage(content="hi")]})

        # No subagents should surface.
        handles = [sub async for sub in run.subagents]
        async for _ in run.messages:
            pass

        assert handles == []
        output = await run.output()
        assert isinstance(output, dict)
        assert output["messages"][-1].content == "no tools today"

    async def test_concurrent_iteration_does_not_drop_events(self) -> None:
        """Iterating `subagents` and `messages` in parallel drops no events."""
        agent = _build_agent_with_one_subagent()
        run = await agent.astream_v2({"messages": [HumanMessage(content="go")]})

        subagent_names: list[str] = []
        parent_message_count = 0

        async def drain_subagents() -> None:
            async for sub in run.subagents:
                subagent_names.append(sub.name)
                # Drain sub projections to keep the sub mini-mux healthy.
                async for _ in sub.messages:
                    pass

        async def drain_messages() -> None:
            nonlocal parent_message_count
            async for _ in run.messages:
                parent_message_count += 1

        await asyncio.gather(drain_subagents(), drain_messages())

        assert subagent_names == ["researcher"]
        assert parent_message_count > 0


class TestCreateDeepAgentStreamV2:
    def test_run_exposes_native_projections_sync(self) -> None:
        agent = _build_agent_with_one_subagent()
        run = agent.stream_v2({"messages": [HumanMessage(content="go")]})
        for attr in ("subagents", "subgraphs", "messages", "tool_calls", "middleware", "values"):
            assert hasattr(run, attr)

        # Drain to completion.
        for _ in run.messages:
            pass

    def test_subagents_yields_one_typed_handle_sync(self) -> None:
        agent = _build_agent_with_one_subagent()
        run = agent.stream_v2({"messages": [HumanMessage(content="go")]})

        handles: list[SubagentRunStream] = []
        for sub in run.subagents:
            handles.append(sub)
            for _ in sub.messages:
                pass

        for _ in run.messages:
            pass

        assert len(handles) == 1
        assert handles[0].name == "researcher"
        assert handles[0].status == "completed"
