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


async def test_deepagent_with_monty_interpreter() -> None:
    """Basic async test with monty interpreter."""
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


async def test_deepagent_with_monty_math_library() -> None:
    """Verify the async repl has access to the Monty math library."""
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

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to calculate 6 + 7")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["2"]
    assert result["messages"][-1].content == "The answer is 2."


async def test_deepagent_with_monty_foreign_function() -> None:
    """Verify the async repl can call a registered foreign function."""
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

    result = await agent.ainvoke(
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


async def test_deepagent_with_monty_langchain_tool_foreign_function() -> None:
    """Verify the async repl can call a registered LangChain tool."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(await foo('bar'))"},
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

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to call foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["foo returned bar!"]
    assert result["messages"][-1].content == "foo returned bar!"


@tool
async def async_foo_tool(value: str) -> str:
    """Return a formatted value from an async LangChain tool."""
    return f"async foo returned {value}!"


async def test_deepagent_with_monty_async_langchain_tool_foreign_function() -> None:
    """Verify the async repl can call a registered async LangChain tool."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(await foo('bar'))"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="async foo returned bar!"),
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
                external_function_implementations={"foo": async_foo_tool},
            )
        ],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to call async foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["async foo returned bar!"]
    assert result["messages"][-1].content == "async foo returned bar!"



async def test_deepagent_with_monty_async_langchain_tool_multiple_positional_arguments() -> None:
    """Verify awaited async tools receive positional arguments mapped by field name."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {"code": "print(await async_add(2, 5))"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="7"),
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
                external_functions=["async_add"],
                external_function_implementations={"async_add": async_add_tool},
            )
        ],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to add two numbers")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["7"]
    assert result["messages"][-1].content == "7"
