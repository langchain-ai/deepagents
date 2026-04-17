from __future__ import annotations

from deepagents.graph import create_deep_agent
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # tool decorator resolves type hints at import time
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_monty.middleware import REPL_SYSTEM_PROMPT, MontyMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def test_deepagent_with_monty_interpreter() -> None:
    """Basic test with monty interpreter."""
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
        middleware=[MontyMiddleware()],
    )

    result = agent.invoke(
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


def test_deepagent_with_monty_math_library() -> None:
    """Verify the repl has access to the Monty math library."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "import math; print(math.ceil(1.8))"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The answer is 2."),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware()],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to calculate 6 + 7")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["2"]
    assert result["messages"][-1].content == "The answer is 2."


def foo(value: str) -> str:
    return f"foo returned {value}!"


def test_deepagent_with_monty_foreign_function() -> None:
    """Verify the repl can call a registered foreign function."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(foo('bar'))"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="foo returned bar!"),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(ptc=[foo])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to call foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["foo returned bar!"]
    assert result["messages"][-1].content == "foo returned bar!"


@tool("foo")
def foo_tool(value: str) -> str:
    """Return a formatted value for testing Monty tool interop."""
    return f"foo returned {value}!"


@tool("get_user_location")
def get_user_location_tool(user_id: int) -> int:
    """Return a deterministic location id for a known user id in tests."""
    return user_id


@tool("get_city_for_location")
def get_city_for_location_tool(location_id: int) -> str:
    """Return a deterministic city for a known location id in tests."""
    return "New York" if location_id == 1 else f"City {location_id}"


@tool("format_user_location")
def format_user_location_tool(user_id: int, location_id: int) -> str:
    """Return a combined user/location string for multi-argument mapping tests."""
    return f"user={user_id}, location={location_id}"


def test_deepagent_with_monty_langchain_tool_foreign_function() -> None:
    """Verify the repl can call a registered LangChain tool."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(foo('bar'))"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="foo returned bar!"),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(ptc=[foo_tool])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to call foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["foo returned bar!"]
    assert result["messages"][-1].content == "foo returned bar!"


def test_deepagent_with_monty_langchain_tools_chain_positional_arguments() -> None:
    """Verify positional Monty arguments map to tool inputs by field name."""
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
                                    "location_id = get_user_location(1)\n"
                                    "city = get_city_for_location(location_id)\n"
                                    "print(city)"
                                )
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The city is New York."),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[
            MontyMiddleware(ptc=[get_user_location_tool, get_city_for_location_tool])
        ],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use the repl to get the city for user 1 via location lookup"
                    )
                )
            ]
        }
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["New York"]
    assert result["messages"][-1].content == "The city is New York."


def test_sync_system_prompt_keeps_sync_tools_non_awaitable() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(ptc=[foo_tool], add_ptc_docs=True)],
    )

    agent.invoke({"messages": [HumanMessage(content="hi")]})

    system_message = model.call_history[0]["messages"][0].content
    rendered_system_message = "".join(block["text"] for block in system_message)
    assert (
        REPL_SYSTEM_PROMPT.split("{external_functions_section}", maxsplit=1)[0]
        in rendered_system_message
    )
    assert "def foo(value: str) -> str:" in rendered_system_message
    assert "async def foo(value: str) -> str:" not in rendered_system_message


def test_deepagent_with_monty_langchain_tool_multiple_positional_arguments() -> None:
    """Verify multiple positional Monty arguments map to tool field names in order."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(format_user_location(7, 3))"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="user=7, location=3"),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(ptc=[format_user_location_tool])],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the repl to format user and location information"
                )
            ]
        }
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["user=7, location=3"]
    assert result["messages"][-1].content == "user=7, location=3"


@tool
def runtime_marker(value: str, runtime: ToolRuntime) -> str:
    """Return runtime metadata for testing ToolRuntime injection."""
    return (
        f"{value}:{runtime.tool_call_id}:{runtime.config['metadata']['langgraph_node']}"
    )


@tool("runtime_configurable")
def runtime_configurable(value: str, runtime: ToolRuntime) -> str:
    """Return configurable runtime data for testing ToolRuntime context propagation."""
    return f"{value}:{runtime.config['configurable']['user_id']}"


def test_monty_toolruntime_foreign_function() -> None:
    """Verify Monty foreign tool calls inherit the enclosing repl ToolRuntime."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": 'print(runtime_marker("value"))'},
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
        middleware=[MontyMiddleware(ptc=[runtime_marker])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to inspect the runtime")]}
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert tool_messages[0].content_blocks == [
        {"type": "text", "text": "value:call_1:tools"}
    ]


def test_monty_toolruntime_foreign_function_configurable() -> None:
    """Verify Monty foreign tool calls receive configurable runtime context."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": 'print(runtime_configurable("value"))'},
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
        middleware=[MontyMiddleware(ptc=[runtime_configurable])],
    )

    result = agent.invoke(
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
