from __future__ import annotations

from deepagents.graph import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_monty.middleware import REPL_SYSTEM_PROMPT, MontyMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


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

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware()],
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

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware()],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to calculate 6 + 7")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["2"]
    assert result["messages"][-1].content == "The answer is 2."


def foo(value: str) -> str:
    return f"foo returned {value}!"


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

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(ptc=[foo])],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to call foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["foo returned bar!"]
    assert result["messages"][-1].content == "foo returned bar!"


@tool("foo")
async def foo_tool(value: str) -> str:
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

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(ptc=[foo_tool])],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to call foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["foo returned bar!"]
    assert result["messages"][-1].content == "foo returned bar!"


@tool("foo")
async def async_foo_tool(value: str) -> str:
    """Return a formatted value from an async LangChain tool."""
    return f"async foo returned {value}!"


@tool("async_add")
async def async_add_tool(a: int, b: int) -> int:
    """Add two integers asynchronously for positional argument mapping tests."""
    return a + b


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

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(ptc=[async_foo_tool])],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to call async foo on bar")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["async foo returned bar!"]
    assert result["messages"][-1].content == "async foo returned bar!"


async def test_async_system_prompt_marks_sync_tools_as_awaitable() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(ptc=[foo_tool], add_ptc_docs=True)],
    )

    await agent.ainvoke({"messages": [HumanMessage(content="hi")]})

    system_message = model.call_history[0]["messages"][0].content
    rendered_system_message = "".join(block["text"] for block in system_message)
    assert (
        REPL_SYSTEM_PROMPT.split("{external_functions_section}", maxsplit=1)[0]
        in rendered_system_message
    )
    assert "async def foo(value: str) -> str:" in rendered_system_message


async def test_deepagent_with_monty_async_tool_mulitple_args() -> None:
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

    agent = create_deep_agent(
        model=model,
        middleware=[MontyMiddleware(ptc=[async_add_tool])],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use the repl to add two numbers")]}
    )

    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["7"]
