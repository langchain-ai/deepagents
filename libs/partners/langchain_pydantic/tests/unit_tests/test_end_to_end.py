from __future__ import annotations

from deepagents.backends.state import StateBackend
from deepagents.graph import create_deep_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore

from langchain_pydantic.middleware import MontyMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _make_runtime() -> ToolRuntime:
    return ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id="tc",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


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

    runtime = _make_runtime()
    backend = StateBackend(runtime)

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(backend=backend)],
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

    runtime = _make_runtime()
    backend = StateBackend(runtime)

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(backend=backend)],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to calculate 6 + 7")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["2"]
    assert result["messages"][-1].content == "The answer is 2."


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

    runtime = _make_runtime()
    backend = StateBackend(runtime)

    agent = create_deep_agent(
        model=model,
        middleware=[
            MontyMiddleware(
                backend=backend,
                external_functions=["foo"],
                external_function_implementations={
                    "foo": lambda value: f"foo returned {value}!"
                },
            )
        ],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to call foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["foo returned bar!"]
    assert result["messages"][-1].content == "foo returned bar!"


@tool
def foo_tool(value: str) -> str:
    """Return a formatted value for testing Monty tool interop."""
    return f"foo returned {value}!"


@tool
def get_user_location_tool(user_id: int) -> int:
    """Return a deterministic location id for a known user id in tests."""
    return user_id


@tool
def get_city_for_location_tool(location_id: int) -> str:
    """Return a deterministic city for a known location id in tests."""
    return "New York" if location_id == 1 else f"City {location_id}"


@tool
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

    runtime = _make_runtime()
    backend = StateBackend(runtime)

    agent = create_deep_agent(
        model=model,
        middleware=[
            MontyMiddleware(
                backend=backend,
                external_functions=["foo"],
                external_function_implementations={"foo": foo_tool},
            )
        ],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the repl to call foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["foo returned bar!"]
    assert result["messages"][-1].content == "foo returned bar!"


def test_deepagent_with_monty_langchain_tools_chain_positional_arguments() -> None:
    """Verify positional Monty arguments are mapped to structured tool inputs by field name."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {
                                "code": "location_id = get_user_location(1)\ncity = get_city_for_location(location_id)\nprint(city)"
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

    runtime = _make_runtime()
    backend = StateBackend(runtime)

    agent = create_deep_agent(
        model=model,
        middleware=[
            MontyMiddleware(
                backend=backend,
                external_functions=["get_user_location", "get_city_for_location"],
                external_function_implementations={
                    "get_user_location": get_user_location_tool,
                    "get_city_for_location": get_city_for_location_tool,
                },
            )
        ],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the repl to get the city for user 1 via location lookup"
                )
            ]
        }
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["New York"]
    assert result["messages"][-1].content == "The city is New York."



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
                            "args": {
                                "code": "print(format_user_location(7, 3))"
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="user=7, location=3"),
            ]
        )
    )

    runtime = _make_runtime()
    backend = StateBackend(runtime)

    agent = create_deep_agent(
        model=model,
        middleware=[
            MontyMiddleware(
                backend=backend,
                external_functions=["format_user_location"],
                external_function_implementations={
                    "format_user_location": format_user_location_tool,
                },
            )
        ],
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
