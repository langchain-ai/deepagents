"""Unit tests for cross-turn REPL snapshot persistence."""

from __future__ import annotations

from typing import Any, Literal

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel

InvokeMode = Literal["invoke", "ainvoke"]


def _script_two_turns() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "eval",
                    "args": {"code": "const counter = 10"},
                    "id": "call_1",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="turn 1 done"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "eval",
                    "args": {"code": "counter + 1"},
                    "id": "call_2",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="turn 2 done"),
    ]


def _script_two_turns_without_snapshots() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "eval",
                    "args": {"code": "const counter = 10"},
                    "id": "call_1",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="turn 1 done"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "eval",
                    "args": {"code": "typeof counter"},
                    "id": "call_2",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="turn 2 done"),
    ]


async def _invoke_agent(
    agent: Any,
    payload: dict[str, Any],
    config: dict[str, Any],
    invoke_mode: InvokeMode,
) -> dict[str, Any]:
    if invoke_mode == "ainvoke":
        return await agent.ainvoke(payload, config=config)
    return agent.invoke(payload, config=config)


def _eval_tool_message(result: dict[str, Any]) -> ToolMessage:
    messages = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"
    ]
    assert messages, "expected at least one eval ToolMessage"
    return messages[-1]


@pytest.mark.parametrize(
    "invoke_mode",
    ["invoke", "ainvoke"],
    ids=["sync_invoke", "async_ainvoke"],
)
async def test_repl_snapshot_persists_state_between_turns(
    invoke_mode: InvokeMode,
) -> None:
    """REPL state survives across turns on the same thread_id."""
    agent = create_deep_agent(
        model=FakeChatModel(messages=iter(_script_two_turns())),
        middleware=[CodeInterpreterMiddleware()],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "quickjs-snapshot-thread"}}

    first = await _invoke_agent(
        agent,
        {"messages": [HumanMessage(content="set counter to 10 with eval")]},
        config,
        invoke_mode,
    )
    first_eval = _eval_tool_message(first)
    assert "<error" not in first_eval.content, first_eval.content

    second = await _invoke_agent(
        agent,
        {"messages": [HumanMessage(content="read counter and add one")]},
        config,
        invoke_mode,
    )
    second_eval = _eval_tool_message(second)
    assert "<error" not in second_eval.content, second_eval.content
    assert "<result>11</result>" in second_eval.content, second_eval.content


@pytest.mark.parametrize(
    "invoke_mode",
    ["invoke", "ainvoke"],
    ids=["sync_invoke", "async_ainvoke"],
)
async def test_repl_mode_turn_resets_state_between_turns(
    invoke_mode: InvokeMode,
) -> None:
    """In turn mode, turn-2 eval starts with a fresh context."""
    agent = create_deep_agent(
        model=FakeChatModel(messages=iter(_script_two_turns_without_snapshots())),
        middleware=[CodeInterpreterMiddleware(mode="turn")],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "quickjs-no-snapshot-thread"}}

    first = await _invoke_agent(
        agent,
        {"messages": [HumanMessage(content="set counter to 10 with eval")]},
        config,
        invoke_mode,
    )
    first_eval = _eval_tool_message(first)
    assert "<error" not in first_eval.content, first_eval.content

    second = await _invoke_agent(
        agent,
        {"messages": [HumanMessage(content="check whether counter still exists")]},
        config,
        invoke_mode,
    )
    second_eval = _eval_tool_message(second)
    assert "<error" not in second_eval.content, second_eval.content
    assert "<result>undefined</result>" in second_eval.content, second_eval.content


@pytest.mark.parametrize(
    "invoke_mode",
    ["invoke", "ainvoke"],
    ids=["sync_invoke", "async_ainvoke"],
)
async def test_repl_snapshot_persists_top_level_await_binding_between_turns(
    invoke_mode: InvokeMode,
) -> None:
    """Top-level-await bindings persist after cross-turn snapshot restore.

    Historically, `quickjs-rs` dropped lexical bindings created in an eval
    that used top-level `await`. The first turn could read `story`, but
    after `after_agent` snapshot + `before_agent` restore, turn 2 raised
    `ReferenceError: story is not defined`.

    This regression test locks in the fixed behavior: once the first turn
    declares `story` via top-level `await`, the second turn can still read it.
    """
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "eval",
                    "args": {"code": "const story = await Promise.resolve('hi')"},
                    "id": "call_1",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="turn 1 done"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "eval",
                    "args": {"code": "story"},
                    "id": "call_2",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="turn 2 done"),
    ]
    agent = create_deep_agent(
        model=FakeChatModel(messages=iter(script)),
        middleware=[CodeInterpreterMiddleware()],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "quickjs-top-level-await-thread"}}

    first = await _invoke_agent(
        agent,
        {"messages": [HumanMessage(content="define story in eval")]},
        config,
        invoke_mode,
    )
    first_eval = _eval_tool_message(first)
    assert "<error" not in first_eval.content, first_eval.content

    second = await _invoke_agent(
        agent,
        {"messages": [HumanMessage(content="read story from previous turn")]},
        config,
        invoke_mode,
    )
    second_eval = _eval_tool_message(second)
    assert "<error" not in second_eval.content, second_eval.content
    assert "<result>hi</result>" in second_eval.content, second_eval.content


@pytest.mark.parametrize(
    "invoke_mode",
    ["invoke", "ainvoke"],
    ids=["sync_invoke", "async_ainvoke"],
)
async def test_repl_top_level_const_let_persist_across_evals_same_turn(
    invoke_mode: InvokeMode,
) -> None:
    """Top-level ``const``/``let`` bindings persist between evals.

    The runtime is configured with
    ``SourceTransform.TOP_LEVEL_CONST_TO_VAR`` so top-level lexical
    declarations are rewritten to ``var`` and survive on the global object.
    Without it, a second eval on the same live context would raise
    ``ReferenceError`` because lexical bindings are dropped between evals.

    This exercises the in-turn path (two evals, one context, no snapshot
    round-trip in between), which is what the bare-``const`` REPL
    persistence model promises.
    """
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "eval",
                    "args": {"code": "const greeting = 'hello'; let count = 41;"},
                    "id": "call_1",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "eval",
                    "args": {"code": "greeting + (count + 1)"},
                    "id": "call_2",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="done"),
    ]
    agent = create_deep_agent(
        model=FakeChatModel(messages=iter(script)),
        middleware=[CodeInterpreterMiddleware()],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "quickjs-const-let-thread"}}

    result = await _invoke_agent(
        agent,
        {"messages": [HumanMessage(content="declare then read const/let")]},
        config,
        invoke_mode,
    )
    eval_messages = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"
    ]
    assert len(eval_messages) == 2, eval_messages
    for msg in eval_messages:
        assert "<error" not in msg.content, msg.content
    assert "<result>hello42</result>" in eval_messages[-1].content, eval_messages[
        -1
    ].content
