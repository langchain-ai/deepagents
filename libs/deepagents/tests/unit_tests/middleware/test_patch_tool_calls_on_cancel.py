import asyncio
from typing import TYPE_CHECKING

import langgraph.errors
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from deepagents import create_deep_agent
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from tests.unit_tests.test_end_to_end import FixedGenericFakeChatModel

RunCancelled = getattr(langgraph.errors, "RunCancelled", None)

needs_cancel_handler = pytest.mark.skipif(
    not hasattr(StateGraph, "set_cancel_handler"),
    reason="installed langgraph has no `StateGraph.set_cancel_handler`",
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def _ai(*ids: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(id=i, name="slow", args={}) for i in ids], id="a1")


class TestOnCancel:
    def test_appends_error_for_unanswered_calls(self) -> None:
        messages = [HumanMessage(content="h0", id="h0"), _ai("1", "2")]

        update = PatchToolCallsMiddleware().on_cancel({"messages": messages}, None)

        assert update is not None
        assert [(m.tool_call_id, m.status) for m in update["messages"]] == [("1", "error"), ("2", "error")]

    def test_skips_calls_already_answered(self) -> None:
        messages = [_ai("1", "2"), ToolMessage(content="done", tool_call_id="1", id="t1")]

        update = PatchToolCallsMiddleware().on_cancel({"messages": messages}, None)

        assert update is not None
        assert [m.tool_call_id for m in update["messages"]] == ["2"]

    def test_no_update_when_last_ai_message_is_answered(self) -> None:
        messages = [_ai("1"), ToolMessage(content="done", tool_call_id="1", id="t1")]

        assert PatchToolCallsMiddleware().on_cancel({"messages": messages}, None) is None

    def test_leaves_earlier_history_to_before_agent(self) -> None:
        messages = [_ai("1"), HumanMessage(content="h2", id="h2")]

        assert PatchToolCallsMiddleware().on_cancel({"messages": messages}, None) is None


@tool
async def slow() -> str:
    """Wait for a long time."""
    await asyncio.sleep(10)
    return "done"


@needs_cancel_handler
async def test_cancelled_tool_call_is_answered_at_cancel_time() -> None:
    model = FixedGenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "slow", "args": {}, "id": "call_1", "type": "tool_call"}]),
                AIMessage(content="ok"),
            ]
        )
    )
    agent = create_deep_agent(model=model, tools=[slow], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    run = asyncio.create_task(agent.ainvoke({"messages": [HumanMessage(content="h1")]}, config))
    await asyncio.sleep(0.3)
    run.cancel(RunCancelled("user stop") if RunCancelled else None)
    with pytest.raises(asyncio.CancelledError):
        await run

    messages = (await agent.aget_state(config)).values["messages"]
    assert [m.type for m in messages] == ["human", "ai", "tool"]
    assert messages[-1].tool_call_id == "call_1"
    assert messages[-1].status == "error"

    await agent.ainvoke({"messages": [HumanMessage(content="h2")]}, config)

    sent = model.captured_messages[-1]
    assert [m.type for m in sent if m.type != "system"] == ["human", "ai", "tool", "human"]
