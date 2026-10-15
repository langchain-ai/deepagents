"""Tests for background tools and `AsyncToolMiddleware`."""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel

from deepagents import background_tool, create_deep_agent
from deepagents.middleware.async_tools import AsyncToolMiddleware, BackgroundTool
from tests.unit_tests.chat_model import GenericFakeChatModel


@background_tool
def add(a: int, b: int) -> int:
    """Add two numbers slowly."""
    return a + b


class _FakeServer:
    """In-memory stand-in for an Agent Server that runs background tool graphs."""

    def __init__(self, graphs: dict[str, Pregel], statuses: Iterator[str] | None = None) -> None:
        self.graphs = graphs
        self.statuses = statuses
        self.values: dict[str, Any] = {}
        self.cancelled: list[tuple[str, str]] = []

    def create_thread(self, **_: Any) -> dict[str, str]:
        return {"thread_id": f"thread-{len(self.values)}"}

    def create_run(self, thread_id: str, assistant_id: str, *, input: dict[str, Any], **_: Any) -> dict[str, str]:  # noqa: A002
        self.values[thread_id] = self.graphs[assistant_id].invoke(input)
        return {"run_id": f"run-{thread_id}", "status": "pending"}

    def get_run(self, thread_id: str, run_id: str) -> dict[str, str]:
        status = next(self.statuses) if self.statuses else "success"
        return {"run_id": run_id, "thread_id": thread_id, "status": status}

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        return {"thread_id": thread_id, "values": self.values[thread_id]}

    def cancel(self, thread_id: str, run_id: str) -> None:
        self.cancelled.append((thread_id, run_id))

    def async_client(self) -> MagicMock:
        client = MagicMock()
        client.threads.create = AsyncMock(side_effect=self.create_thread)
        client.threads.get = AsyncMock(side_effect=self.get_thread)
        client.runs.create = AsyncMock(side_effect=self.create_run)
        client.runs.get = AsyncMock(side_effect=self.get_run)
        client.runs.cancel = AsyncMock(side_effect=self.cancel)
        return client


def _model(*messages: AIMessage) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([*messages, AIMessage(content="done")]))


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_background_tool_is_an_invocable_graph() -> None:
    assert isinstance(add, Pregel)
    assert add.invoke({"a": 1, "b": 2}) == 3
    assert add.graph_id == "add"
    copied = add.copy()
    assert isinstance(copied, BackgroundTool)
    assert copied.tool.name == "add"


def test_background_tool_options() -> None:
    @background_tool(name="sum_numbers", graph_id="summer", url="http://server")
    async def total(numbers: list[int]) -> int:
        """Sum numbers."""
        return sum(numbers)

    assert total.tool.name == "sum_numbers"
    assert total.as_spec() == {"name": "sum_numbers", "description": "Sum numbers.", "graph_id": "summer", "url": "http://server"}


async def test_background_tool_async_body() -> None:
    @background_tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    assert await double.ainvoke({"x": 4}) == 8


def test_middleware_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        AsyncToolMiddleware(tools=[add, add])


async def test_deep_agent_launches_and_is_notified_of_background_tool() -> None:
    server = _FakeServer({"add": add})
    model = _model(_call("add", {"a": 2, "b": 3}, "c1"), _call("check_background_task", {"task_id": "thread-0"}, "c2"))
    with patch("deepagents.middleware.async_subagents.get_client", return_value=server.async_client()):
        agent = create_deep_agent(model=model, tools=[add], checkpointer=InMemorySaver())
        result = await agent.ainvoke({"messages": [HumanMessage("add 2 and 3")]}, {"configurable": {"thread_id": "t"}})

    messages = result["messages"]
    launched = next(m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "c1")
    assert "task_id: thread-0" in launched.content
    reminder = next(m for m in messages if isinstance(m, HumanMessage) and "system-reminder" in m.text)
    assert "finished with status: success" in reminder.text
    checked = next(m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "c2")
    assert json.loads(checked.content) == {"task_id": "thread-0", "tool": "add", "status": "success", "result": "5"}
    assert result["background_tasks"]["thread-0"]["status"] == "success"


async def test_check_blocks_until_task_finishes() -> None:
    server = _FakeServer({"add": add}, statuses=iter(["running", "running", "running", "success"]))
    model = _model(_call("add", {"a": 1, "b": 1}, "c1"), _call("check_background_task", {"task_id": "thread-0", "block": True}, "c2"))
    with (
        patch("deepagents.middleware.async_subagents.get_client", return_value=server.async_client()),
        patch("deepagents.middleware.async_tools._POLL_INTERVAL_SECONDS", 0),
    ):
        agent = create_deep_agent(model=model, tools=[add])
        result = await agent.ainvoke({"messages": [HumanMessage("go")]})

    checked = next(m for m in result["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "c2")
    assert json.loads(checked.content)["result"] == "2"


async def test_cancel_background_task() -> None:
    server = _FakeServer({"add": add}, statuses=iter(["running"] * 10))
    model = _model(_call("add", {"a": 1, "b": 1}, "c1"), _call("cancel_background_task", {"task_id": "thread-0"}, "c2"))
    with patch("deepagents.middleware.async_subagents.get_client", return_value=server.async_client()):
        agent = create_deep_agent(model=model, tools=[add])
        result = await agent.ainvoke({"messages": [HumanMessage("go")]})

    assert server.cancelled == [("thread-0", "run-thread-0")]
    assert result["background_tasks"]["thread-0"]["status"] == "cancelled"
    assert not any(isinstance(m, HumanMessage) and "system-reminder" in m.text for m in result["messages"])
