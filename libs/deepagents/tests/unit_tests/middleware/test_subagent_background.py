"""Unit tests for subagents the `task` tool starts with `run_in_background`."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from deepagents.backends.state import StateBackend
from deepagents.middleware.subagents import BackgroundTask, CompiledSubAgent, SubAgentMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel

pytestmark = pytest.mark.filterwarnings("ignore:The feature `background subagents` is in beta:langchain_core._api.LangChainBetaWarning")

_THREAD: RunnableConfig = {"configurable": {"thread_id": "thread-1"}}


class _Reports:
    """Collects background reports and signals the first one."""

    def __init__(self) -> None:
        self.tasks: list[BackgroundTask] = []
        self.arrived = asyncio.Event()

    async def collect(self, task: BackgroundTask) -> None:
        self.tasks.append(task)
        self.arrived.set()

    async def first(self) -> BackgroundTask:
        await asyncio.wait_for(self.arrived.wait(), timeout=5)
        return self.tasks[0]


def _parent(work: Callable[..., Awaitable[dict[str, Any]]], reports: _Reports) -> CompiledStateGraph:
    """A parent agent whose model starts `work` in the background, then ends its turn."""
    subagent: CompiledSubAgent = {"name": "builder", "description": "Builds one act.", "runnable": RunnableLambda(work)}
    call = {"name": "task", "args": {"description": "Build act 3", "subagent_type": "builder", "run_in_background": True}, "id": "call_1"}
    model = GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=[call]), AIMessage(content="Act 3 is building.")]))
    middleware = SubAgentMiddleware(backend=StateBackend(), subagents=[subagent], on_background_complete=reports.collect)
    return create_agent(model, middleware=[middleware], checkpointer=InMemorySaver())


async def test_parent_turn_ends_before_background_subagent_reports() -> None:
    gate = asyncio.Event()
    seen: list[RunnableConfig] = []

    async def build(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        seen.append(config)
        await gate.wait()
        return {"messages": [*state["messages"], AIMessage(content="Act 3 built.")]}

    reports = _Reports()
    result = await _parent(build, reports).ainvoke({"messages": [HumanMessage(content="go")]}, _THREAD)

    started = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert "in the background" in started.text
    assert result["messages"][-1].text == "Act 3 is building."
    assert reports.tasks == []

    gate.set()
    report = await reports.first()
    assert report["status"] == "success"
    assert report["result"] == "Act 3 built."
    # The subagent ran on the parent's thread, so thread-scoped backends resolve the parent's.
    assert seen[0]["configurable"]["thread_id"] == "thread-1"
    assert report["config"]["configurable"]["thread_id"] == "thread-1"
    assert not any(key.startswith("__pregel_") for key in report["config"]["configurable"])


async def test_background_subagent_failure_is_reported() -> None:
    async def crash(_state: dict[str, Any]) -> dict[str, Any]:
        msg = "render crashed"
        raise RuntimeError(msg)

    reports = _Reports()
    await _parent(crash, reports).ainvoke({"messages": [HumanMessage(content="go")]}, _THREAD)

    report = await reports.first()
    assert report["status"] == "error"
    assert report["result"] == "RuntimeError: render crashed"


def test_sync_invoke_refuses_background_subagents() -> None:
    async def never(_state: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError

    reports = _Reports()
    result = _parent(never, reports).invoke({"messages": [HumanMessage(content="go")]}, _THREAD)

    refused = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert "async entrypoint" in refused.text
    assert reports.tasks == []


def test_run_in_background_is_offered_only_with_a_callback() -> None:
    subagent: CompiledSubAgent = {"name": "builder", "description": "Builds one act.", "runnable": RunnableLambda(lambda state: state)}
    plain = SubAgentMiddleware(backend=StateBackend(), subagents=[subagent])
    background = SubAgentMiddleware(backend=StateBackend(), subagents=[subagent], on_background_complete=_Reports().collect)

    assert "run_in_background" not in plain.tools[0].args
    assert "run_in_background" in background.tools[0].args
