"""Unit tests for cross-turn REPL snapshot persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import pytest
from deepagents import create_deep_agent
from deepagents.backends.state import StateBackend
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_quickjs import REPLMiddleware
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


def _script_two_turns_with_skill() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "eval",
                    "args": {
                        "code": ('const m = await import("@/skills/inc");\nm.inc(10);')
                    },
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
                    "args": {
                        "code": ('const m = await import("@/skills/inc");\nm.inc(41);')
                    },
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
        middleware=[REPLMiddleware()],
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
async def test_repl_without_snapshots_resets_state_between_turns(
    invoke_mode: InvokeMode,
) -> None:
    """When snapshots are disabled, turn-2 eval starts with a fresh context."""
    agent = create_deep_agent(
        model=FakeChatModel(messages=iter(_script_two_turns_without_snapshots())),
        middleware=[REPLMiddleware(snapshot_between_turns=False)],
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
async def test_repl_snapshot_persists_skill_usage_between_turns(
    invoke_mode: InvokeMode,
) -> None:
    """A skill imported before snapshot restore still works on the next turn."""
    backend = StateBackend()
    now = datetime.now(UTC).isoformat()
    skill_files = {
        "/skills/inc/SKILL.md": {
            "content": (
                "---\n"
                "name: inc\n"
                "description: Increment numbers\n"
                "module: index.js\n"
                "---\n"
            ),
            "encoding": "utf-8",
            "created_at": now,
            "modified_at": now,
        },
        "/skills/inc/index.js": {
            "content": "export const inc = (n) => n + 1;\n",
            "encoding": "utf-8",
            "created_at": now,
            "modified_at": now,
        },
    }

    agent = create_deep_agent(
        model=FakeChatModel(messages=iter(_script_two_turns_with_skill())),
        backend=backend,
        skills=["/skills"],
        middleware=[REPLMiddleware(skills_backend=backend)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "quickjs-snapshot-skill-thread"}}

    first = await _invoke_agent(
        agent,
        {
            "messages": [HumanMessage(content="use skill inc")],
            "files": skill_files,
        },
        config,
        invoke_mode,
    )
    first_eval = _eval_tool_message(first)
    assert "<error" not in first_eval.content, first_eval.content
    assert "<result>11</result>" in first_eval.content, first_eval.content

    second = await _invoke_agent(
        agent,
        {"messages": [HumanMessage(content="use skill inc again next turn")]},
        config,
        invoke_mode,
    )
    second_eval = _eval_tool_message(second)
    assert "<error" not in second_eval.content, second_eval.content
    assert "<result>42</result>" in second_eval.content, second_eval.content


@pytest.mark.parametrize(
    "invoke_mode",
    ["invoke", "ainvoke"],
    ids=["sync_invoke", "async_ainvoke"],
)
async def test_repl_snapshot_persists_top_level_await_binding_between_turns(
    invoke_mode: InvokeMode,
) -> None:
    """Top-level-await bindings persist after cross-turn snapshot restore.

    Historically, ``quickjs-rs`` dropped lexical bindings created in an eval
    that used top-level ``await``. The first turn could read ``story``, but
    after ``after_agent`` snapshot + ``before_agent`` restore, turn 2 raised
    ``ReferenceError: story is not defined``.

    This regression test locks in the fixed behavior: once the first turn
    declares ``story`` via top-level ``await``, the second turn can still read it.
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
        middleware=[REPLMiddleware()],
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
