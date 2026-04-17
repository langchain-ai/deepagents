from __future__ import annotations

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


def test_deepagent_with_repl_interpreter() -> None:
    """Basic test with Repl interpreter."""
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
    """Return a formatted value for testing Repl tool interop."""
    return f"foo returned {value}!"


def test_deepagent_with_repl_langchain_tool_single_arg_foreign_function() -> None:
    """Verify the repl maps a single positional arg to a single-field tool payload."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": 'print(foo("bar"))'},
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
        middleware=[ReplMiddleware(ptc=[foo_tool])],
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


@tool("favorite_item_ids")
def favorite_item_ids(user_id: int) -> list[int]:
    """Return example favorite item identifiers for a user."""
    return [101, 202, 303] if user_id == 43 else []


@tool("item_name")
def item_name(item_id: int) -> str:
    """Return the display name for an item."""
    return f"item-{item_id}"


@tool("item_score")
def item_score(item_id: int) -> int:
    """Return the score for an item."""
    return item_id + 1


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


def test_deepagent_with_repl_langchain_tool_multi_arg_foreign_function() -> None:
    """Verify the repl maps multiple positional args onto matching tool fields."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": 'print(join_values("left", "right"))'},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
                AIMessage(content="done"),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        middleware=[ReplMiddleware(ptc=[join_tool])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to join left and right")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0].content_blocks == [{"type": "text", "text": "left:right"}]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]


def test_deepagent_with_repl_langchain_tool_list_of_ints_foreign_function() -> None:
    """Verify the repl can print array output from a foreign tool returning ints."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(numbers(4))"},
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
        middleware=[ReplMiddleware(ptc=[list_numbers])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to print numbers up to 4")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0].content_blocks == [{"type": "text", "text": "[0, 1, 2, 3]"}]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]


def test_deepagent_with_repl_langchain_tool_json_stringify_foreign_function() -> None:
    """Verify the repl can stringify foreign tool output before printing it."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(list_user_ids())"},
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
        middleware=[ReplMiddleware(ptc=[list_user_ids])],
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
        {"type": "text", "text": "['user_1', 'user_2', 'user_3']"}
    ]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]


def test_repl_toolruntime_foreign_function() -> None:
    """Verify REPL foreign tool calls inherit the enclosing repl ToolRuntime."""
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
        middleware=[ReplMiddleware(ptc=[runtime_marker])],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to inspect the runtime")]}
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert tool_messages[0].content_blocks == [
        {"type": "text", "text": "value:call_1:tools"}
    ]


def test_repl_toolruntime_foreign_function_configurable() -> None:
    """Verify REPL foreign tool calls receive configurable runtime context."""
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
        middleware=[ReplMiddleware(ptc=[runtime_configurable])],
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


def test_deepagent_with_repl_parallel_following_get_chain() -> None:
    """Verify chained `get` values can feed parallel foreign function calls."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {
                                "code": """
user_id = 43
item_ids = favorite_item_ids(user_id)
first_item_id = item_ids[0]
print(parallel([task(item_name(first_item_id)), task(item_score(first_item_id))]))
""",
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
        middleware=[ReplMiddleware(ptc=[favorite_item_ids, item_name, item_score])],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use the repl to inspect the first favorite item for user 43"
                    )
                )
            ]
        }
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0].content_blocks == [
        {"type": "text", "text": "['item-101', 102]"}
    ]
    assert result["messages"][-1].content_blocks == [{"type": "text", "text": "done"}]
