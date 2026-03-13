from __future__ import annotations

from deepagents.graph import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_quickjs.middleware import QuickJSMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


@tool
async def list_user_ids() -> list[str]:
    """Return example user identifiers for async QuickJS bridging tests."""
    return ["user_1", "user_2", "user_3"]


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
