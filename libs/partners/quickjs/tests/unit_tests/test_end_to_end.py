from __future__ import annotations

from deepagents.graph import create_deep_agent
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # tool decorator resolves type hints at import time
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_quickjs.middleware import QuickJSMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def test_deepagent_with_quickjs_interpreter() -> None:
    """Basic test with QuickJS interpreter."""
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


@tool("foo")
def foo_tool(value: str) -> str:
    """Return a formatted value for testing QuickJS tool interop."""
    return f"foo returned {value}!"


def test_quickjs_tool_single_arg_foreign_function() -> None:
    """Verify the repl maps a single positional arg to a single-field tool payload."""
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
        middleware=[QuickJSMiddleware(ptc=[foo_tool])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to call foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["foo returned bar!"]
    assert result["messages"][-1].content == "foo returned bar!"


@tool("join_values")
def join_tool(left: str, right: str) -> str:
    """Join two values for testing positional argument payload mapping."""
    return f"{left}:{right}"


@tool("numbers")
def list_numbers(limit: int) -> list[int]:
    """Return a list of integers from zero up to the provided limit."""
    return list(range(limit))


@tool
def list_user_ids() -> list[str]:
    """Return example user identifiers for testing JSON output from foreign tools."""
    return ["user_1", "user_2", "user_3"]


@tool
def get_user_profile() -> dict[str, str | int]:
    """Return example user profile data for testing object bridging."""
    return {"id": "user_1", "name": "Ada", "age": 37}


@tool("runtime_marker")
def runtime_marker(value: str, runtime: ToolRuntime) -> str:
    """Return runtime metadata for testing ToolRuntime injection."""
    return (
        f"{value}:{runtime.tool_call_id}:{runtime.config['metadata']['langgraph_node']}"
    )


@tool("runtime_configurable")
def runtime_configurable(value: str, runtime: ToolRuntime) -> str:
    """Return configurable runtime data for testing ToolRuntime context propagation."""
    return f"{value}:{runtime.config['configurable']['user_id']}"


def test_quickjs_tool_multi_arg_foreign_function() -> None:
    """Verify the repl maps multiple positional args onto matching tool fields."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(join_values('left', 'right'))"},
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
        middleware=[QuickJSMiddleware(ptc=[join_tool])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to join left and right")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0].content_blocks == [{"type": "text", "text": "left:right"}]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]


def test_quickjs_tool_list_of_ints_foreign_function() -> None:
    """Verify the repl can print array output from a foreign tool returning ints."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(JSON.stringify(numbers(4)))"},
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
        middleware=[QuickJSMiddleware(ptc=[list_numbers])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to print numbers up to 4")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0].content_blocks == [{"type": "text", "text": "[0,1,2,3]"}]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]


def test_quickjs_tool_json_stringify_foreign_function() -> None:
    """Verify the repl transparently bridges Python list returns into JS arrays."""
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

    result = agent.invoke(
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


def test_quickjs_toolruntime_foreign_function() -> None:
    """Verify QuickJS foreign tool calls inherit the enclosing repl ToolRuntime."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(runtime_marker('value'))"},
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
        middleware=[QuickJSMiddleware(ptc=[runtime_marker])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to inspect the runtime")]}
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert tool_messages[0].content_blocks == [
        {"type": "text", "text": "value:call_1:tools"}
    ]


def test_quickjs_memory_limit_error() -> None:
    """Verify the repl surfaces QuickJS memory limit failures."""
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
                                    "const values = [];\n"
                                    "while (true) {\n"
                                    "  values.push('x'.repeat(1024));\n"
                                    "}"
                                )
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="memory limit hit"),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[QuickJSMiddleware(memory_limit=1_000_000)],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the repl and keep allocating memory until it fails"
                )
            ]
        }
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == (
        "InternalError: out of memory\n    at <eval> (<input>:3)\n"
    )
    assert result["messages"][-1].content == "memory limit hit"


def test_quickjs_timeout_error() -> None:
    """Verify the repl surfaces QuickJS eval timeouts."""
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

    result = agent.invoke(
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


def test_quickjs_toolruntime_configurable_foreign_function() -> None:
    """Verify QuickJS foreign tool calls see configurable runtime data."""
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


def test_quickjs_tool_dict_foreign_function() -> None:
    """Verify the repl transparently bridges Python dict returns into JS objects."""
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
                                    "const profile = get_user_profile();\n"
                                    "print(profile.name + ':' + profile.age);"
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
        middleware=[QuickJSMiddleware(ptc=[get_user_profile])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to inspect the user profile")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0].content_blocks == [{"type": "text", "text": "Ada:37"}]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]
