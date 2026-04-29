"""Shared helpers for QuickJS CodSpeed benchmark suites."""

from __future__ import annotations

from collections.abc import (
    Iterator,  # noqa: TC003  # pydantic resolves this annotation at runtime
    Sequence,  # noqa: TC003  # used in runtime-accessed annotations
)
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

if TYPE_CHECKING:
    from langchain_quickjs import REPLMiddleware

CONSOLE_LOG_CODE = "for (let i = 0; i < 200; i += 1) {  console.log(`line-${i}`);}'ok';"
PTC_ONLY_CODE = (
    "const values = [];"
    "for (let i = 0; i < 100; i += 1) {"
    "  values.push(await tools.echoPayload({value: `value-${i}`}));"
    "}"
    "values.length;"
)
PTC_AND_CONSOLE_CODE = (
    "await (async () => {"
    "  const values = [];"
    "  for (let i = 0; i < 20; i += 1) {"
    "    const value = await tools.echoPayload({value: `value-${i}`});"
    "    values.push(value);"
    "    console.log(value);"
    "  }"
    "  return values.length;"
    "})();"
)
THROUGHPUT_ITERATIONS = 200


class FakeChatModel(GenericFakeChatModel):
    """Generic fake chat model whose ``bind_tools`` keeps scripted messages."""

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> FakeChatModel:
        del tools
        return self


@tool("echo_payload")
def echo_payload(value: str) -> str:
    """Echo an input payload for PTC benchmark calls."""
    return value


def tool_call_message(code: str, *, call_id: str = "call_1") -> AIMessage:
    """Build the model message that calls the REPL eval tool."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "eval",
                "args": {"code": code},
                "id": call_id,
                "type": "tool_call",
            },
        ],
    )


def finite_script(code: str, *, repeats: int) -> Iterator[AIMessage]:
    """Build a finite tool-call script for the fake model."""
    messages: list[AIMessage] = []
    for index in range(repeats):
        messages.append(tool_call_message(code, call_id=f"call_{index}"))
        messages.append(AIMessage(content="done"))
    return iter(messages)


def make_agent(
    *,
    code: str,
    middleware: REPLMiddleware,
    repeats: int,
) -> Any:
    """Create a deep agent with a scripted fake chat model."""
    messages = finite_script(code, repeats=repeats)
    return create_deep_agent(
        model=FakeChatModel(messages=messages),
        middleware=[middleware],
    )


def invoke_payload() -> dict[str, list[HumanMessage]]:
    """Return the human message payload used by benchmark runs."""
    return {"messages": [HumanMessage(content="run benchmark workload")]}


def eval_tool_message(result: dict[str, Any]) -> ToolMessage:
    """Return the last eval ToolMessage from an agent result payload."""
    messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name == "eval"
    ]
    assert messages, "expected at least one eval ToolMessage"
    return messages[-1]


def assert_eval_succeeded(result: dict[str, Any]) -> None:
    """Assert the REPL eval did not produce a tool error envelope."""
    tool_message = eval_tool_message(result)
    assert "<error" not in tool_message.content, tool_message.content
