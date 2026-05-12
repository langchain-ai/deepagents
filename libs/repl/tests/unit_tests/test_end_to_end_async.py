from __future__ import annotations

import pytest
from deepagents.graph import create_deep_agent
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # tool decorator resolves type hints at import time
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_repl.middleware import ReplMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def mul(left: int, right: int) -> int:
    return left * right


@tool("async_identity")
async def async_identity(value: str) -> str:
    """Return the provided value from an asynchronous LangChain tool."""
    return value


@tool("get_user_id")
def get_user_id(runtime: ToolRuntime) -> str:
    """Return the configured user identifier from ToolRuntime."""
    return str(runtime.config["configurable"]["user_id"])


async def test_deepagent_with_repl_interpreter() -> None:
    """Basic async test with Repl interpreter."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(mul(6, 7))"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The answer is 42."),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[ReplMiddleware(ptc=[mul])],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to calculate 6 * 7")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["42"]
    assert result["messages"][-1].content == "The answer is 42."
    assert len(model.call_history) == 2
    assert (
        model.call_history[0]["messages"][-1].content
        == "Use the repl to calculate 6 * 7"
    )


async def test_deepagent_with_repl_toolruntime_foreign_function_configurable() -> None:
    """Verify async REPL foreign tool calls receive configurable runtime context."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(get_user_id())"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[ReplMiddleware(ptc=[get_user_id])],
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the repl to inspect configurable runtime")
            ]
        },
        config={"configurable": {"user_id": "user-async"}},
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert tool_messages[0].content_blocks == [{"type": "text", "text": "user-async"}]


@pytest.mark.xfail(reason="Async tools not yet supported.")
async def test_deepagent_with_repl_async_langchain_tool() -> None:
    """Verify that we can call async tools"""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": 'print(async_identity("value"))'},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[ReplMiddleware(ptc=[async_identity])],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to call the async tool")]}
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert tool_messages[0].content_blocks == [{"type": "text", "text": "value"}]
