from __future__ import annotations

import asyncio
import threading

from deepagents.graph import create_deep_agent
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # tool decorator resolves type hints at import time
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_quickjs.middleware import QuickJSMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


@tool
def list_user_ids() -> list[str]:
    """Return example user identifiers for QuickJS bridging tests."""
    return ["user_1", "user_2", "user_3"]


@tool("sync_label")
def sync_label_tool(value: str) -> str:
    """Return a labeled value from a synchronous LangChain tool."""
    return f"sync:{value}"


@tool("async_label")
async def async_label_tool(value: str) -> str:
    """Return a labeled value from an asynchronous LangChain tool."""
    await asyncio.sleep(0)
    return f"async:{value}"


@tool("runtime_configurable")
def runtime_configurable(value: str, runtime: ToolRuntime) -> str:
    """Return configurable runtime data for testing ToolRuntime context propagation."""
    return f"{value}:{runtime.config['configurable']['user_id']}"


async def async_uppercase(value: str) -> str:
    """Return an uppercased value after yielding to the event loop."""
    await asyncio.sleep(0)
    return value.upper()


async def capture_thread_name() -> str:
    """Return the name of the thread running the coroutine."""
    await asyncio.sleep(0)
    return threading.current_thread().name


async def test_deepagent_with_quickjs_interpreter() -> None:
    """Basic async test with QuickJS interpreter."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(6 * 7)"},
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
        middleware=[QuickJSMiddleware()],
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


async def test_deepagent_with_quickjs_json_stringify_foreign_function() -> None:
    """Verify async repl calls bridge Python list returns into JS arrays."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {
                                "code": (
                                    "const ids = list_user_ids();\n"
                                    "print(JSON.stringify(ids));"
                                )
                            },
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
        middleware=[QuickJSMiddleware(ptc=[list_user_ids])],
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the repl to print the available user ids")
            ]
        }
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content_blocks == [
        {"type": "text", "text": '["user_1","user_2","user_3"]'}
    ]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]


async def test_deepagent_with_quickjs_async_foreign_function() -> None:
    """Verify the repl can call sync and async Python helpers in one evaluation."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {
                                "code": (
                                    "print(sync_label('hello'));\n"
                                    "print(async_uppercase('world'));"
                                )
                            },
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
        middleware=[QuickJSMiddleware(ptc=[sync_label_tool, async_uppercase])],
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the repl to call the sync and async helpers")
            ]
        }
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content_blocks == [
        {"type": "text", "text": "sync:hello\nWORLD"}
    ]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]


async def test_quickjs_async_langchain_tool() -> None:
    """Verify the repl supports async LangChain tools alongside sync ones."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {
                                "code": (
                                    "print(sync_label('left'));\n"
                                    "print(async_label('right'));"
                                )
                            },
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
        middleware=[QuickJSMiddleware(ptc=[sync_label_tool, async_label_tool])],
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the repl to call the sync and async tools")
            ]
        }
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content_blocks == [
        {"type": "text", "text": "sync:left\nasync:right"}
    ]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]


async def test_quickjs_async_timeout_error() -> None:
    """Verify the async repl path surfaces QuickJS eval timeouts."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "while (true) {}"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="timeout hit"),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[QuickJSMiddleware(timeout=1)],
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the repl and keep looping until it times out")
            ]
        }
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == (
        "InternalError: interrupted\n    at <eval> (<input>)\n"
    )
    assert result["messages"][-1].content == "timeout hit"


async def test_quickjs_async_toolruntime_configurable_foreign_function() -> None:
    """Verify async QuickJS foreign tool calls see configurable runtime data."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(runtime_configurable('value'))"},
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
        middleware=[QuickJSMiddleware(ptc=[runtime_configurable])],
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the repl to inspect configurable runtime")
            ]
        },
        config={"configurable": {"user_id": "user-123"}},
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert tool_messages[0].content_blocks == [
        {"type": "text", "text": "value:user-123"}
    ]


async def test_quickjs_async_foreign_function_runs_on_daemon_loop_thread() -> None:
    """Verify async foreign functions execute on the background event-loop thread."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(capture_thread_name())"},
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
        middleware=[QuickJSMiddleware(ptc=[capture_thread_name])],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to inspect the async thread")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) == 1
    thread_name = tool_messages[0].content
    assert thread_name != threading.current_thread().name
    assert thread_name.startswith("Thread-")
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]
