"""Shared helpers for QuickJS CodSpeed benchmark suites."""

from __future__ import annotations

from collections.abc import (
    Iterator,  # noqa: TC003  # pydantic resolves this annotation at runtime
)
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel

if TYPE_CHECKING:
    from langchain_quickjs.middleware import REPLState

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
COUNTER_INIT_CODE = "let counter = 0; String(counter);"
COUNTER_NEXT_CODE = 'typeof counter === "number" ? String(counter += 1) : "missing";'
THROUGHPUT_ITERATIONS = 200


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
    middleware: CodeInterpreterMiddleware,
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


def run_counter_turns(
    *,
    turn_count: int,
    snapshot_between_turns: bool,
) -> list[str]:
    """Run `turn_count` REPL turns and return counter values per turn."""
    middleware = CodeInterpreterMiddleware(
        snapshot_between_turns=snapshot_between_turns
    )
    state: REPLState = {}
    runtime: Any = None  # hooks ignore runtime; `None` keeps this path lightweight
    values: list[str] = []
    try:
        for turn_index in range(turn_count):
            before = middleware.before_agent(state=state, runtime=runtime)
            if before is not None:
                state.update(before)

            repl = middleware._registry.get(middleware._fallback_thread_id)
            code = COUNTER_INIT_CODE if turn_index == 0 else COUNTER_NEXT_CODE
            outcome = repl.eval_sync(code)
            assert outcome.error_type is None, outcome.error_message
            assert outcome.result is not None
            values.append(outcome.result)

            after = middleware.after_agent(state=state, runtime=runtime)
            if after is not None:
                state.update(after)
    finally:
        middleware._registry.close()
    return values


def assert_counter_turn_values(
    *,
    values: list[str],
    snapshot_between_turns: bool,
) -> None:
    """Assert expected counter values with snapshots enabled/disabled."""
    expected = (
        [str(turn_index) for turn_index in range(len(values))]
        if snapshot_between_turns
        else ["0", *(["missing"] * max(0, len(values) - 1))]
    )
    assert values == expected, f"expected {expected}, got {values}"
