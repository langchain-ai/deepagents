"""Unit tests for cross-turn REPL snapshot persistence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import pytest
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_quickjs import CodeInterpreterMiddleware, _repl
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


def _call(code: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "eval", "args": {"code": code}, "id": call_id, "type": "tool_call"}
        ],
    )


async def _invoke(agent: Any, mode: InvokeMode, config: dict[str, Any]) -> Any:
    payload = {"messages": [HumanMessage(content="Continue.")]}
    if mode == "ainvoke":
        return await agent.ainvoke(payload, config=config)
    return agent.invoke(payload, config=config)


def _results(result: dict[str, Any]) -> list[str]:
    return [
        str(message.content)
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name == "eval"
    ]


def _forbid_allocation(*args: object, **kwargs: object) -> None:
    pytest.fail("A model-only turn must not allocate a QuickJS runtime or worker")


@tool
def lookup() -> str:
    """Return the currently available value."""
    return "available"


@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
@pytest.mark.parametrize("mode", ["thread", "turn", "call"])
@pytest.mark.parametrize("ptc", [None, [], [lookup]])
async def test_model_only_turn_never_allocates_runtime(
    monkeypatch, invoke_mode, mode, ptc
) -> None:
    monkeypatch.setattr(_repl, "Runtime", _forbid_allocation)
    monkeypatch.setattr(_repl, "ThreadWorker", _forbid_allocation)
    middleware = CodeInterpreterMiddleware(mode=mode, ptc=ptc)
    agent = create_agent(
        model=FakeChatModel(messages=iter([AIMessage(content="No eval needed.")])),
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "model-only"}}
    result = await _invoke(agent, invoke_mode, config)
    assert result["messages"][-1].content == "No eval needed."
    assert not agent.get_state(config).values.get("_quickjs_snapshot_payload")


@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
@pytest.mark.parametrize("mode", ["thread", "turn", "call"])
async def test_first_eval_initializes_runtime_and_current_tools(
    invoke_mode, mode
) -> None:
    middleware = CodeInterpreterMiddleware(mode=mode, ptc=[lookup])
    agent = create_agent(
        model=FakeChatModel(
            messages=iter(
                [_call("await tools.lookup({})", "first"), AIMessage(content="Done.")]
            )
        ),
        middleware=[middleware],
    )
    result = await _invoke(agent, invoke_mode, {})
    assert _results(result) == ["<result>available</result>"]
    assert not middleware._registry._slots


@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
@pytest.mark.parametrize("ptc", [None, [], [lookup]])
async def test_signed_snapshot_survives_model_only_turns_and_rebuilt_agent(
    monkeypatch, invoke_mode, ptc
) -> None:
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "signed-lazy-resume"}}

    def build(messages):
        return create_agent(
            model=FakeChatModel(messages=iter(messages)),
            middleware=[
                CodeInterpreterMiddleware(
                    snapshot_signing_key=b"deterministic-test-key", ptc=ptc
                )
            ],
            checkpointer=saver,
        )

    def checkpoint(agent):
        state = agent.get_state(config).values
        return tuple(
            state[key]
            for key in (
                "_quickjs_slot_id",
                "_quickjs_snapshot_payload",
                "_quickjs_snapshot_hmac",
            )
        )

    agent = build(
        [
            _call("const counter = 41", "define"),
            AIMessage(content="Saved."),
            AIMessage(content="No eval."),
        ]
    )
    await _invoke(agent, invoke_mode, config)
    persisted = checkpoint(agent)
    assert all(persisted)
    with monkeypatch.context() as forbidden:
        forbidden.setattr(_repl, "Runtime", _forbid_allocation)
        forbidden.setattr(_repl, "ThreadWorker", _forbid_allocation)
        await _invoke(agent, invoke_mode, config)
        assert checkpoint(agent) == persisted
        rebuilt = build(
            [
                AIMessage(content="Still no eval."),
                _call("counter += 1; counter", "resume"),
                _call("counter += 1; counter", "again"),
                AIMessage(content="Resumed."),
            ]
        )
        await _invoke(rebuilt, invoke_mode, config)
        assert checkpoint(rebuilt) == persisted
    result = await _invoke(rebuilt, invoke_mode, config)
    assert _results(result)[-2:] == ["<result>42</result>", "<result>43</result>"]


@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
@pytest.mark.parametrize("mode", ["thread", "turn", "call"])
async def test_live_bindings_change_before_first_and_subsequent_eval(
    invoke_mode, mode
) -> None:
    @tool("lookup")
    def replacement() -> str:
        """Return a replacement result."""
        return "replacement"

    middleware = CodeInterpreterMiddleware(mode=mode, ptc=["lookup"])
    state = middleware.before_agent({}, None)
    runtime = ToolRuntime(
        state=state,
        context={},
        config={},
        stream_writer=lambda _: None,
        tools=middleware.tools,
        tool_call_id="live-bindings",
        store=None,
    )

    def prepare(tools):
        return middleware._prepare_for_call(SimpleNamespace(state=state, tools=tools))

    async def evaluate(code):
        if invoke_mode == "ainvoke":
            return await middleware.tools[0].ainvoke({"code": code, "runtime": runtime})
        return middleware.tools[0].invoke({"code": code, "runtime": runtime})

    try:
        original = prepare([lookup])
        assert original == prepare([lookup])
        prepare([replacement])
        assert middleware._registry.get_if_exists(state["_quickjs_slot_id"]) is None
        result = await evaluate("await tools.lookup({})")
        assert result.content == "<result>replacement</result>"
        prepare([lookup])
        result = await evaluate("await tools.lookup({})")
        assert result.content == "<result>available</result>"
        prepare([])
        result = await evaluate("typeof tools.lookup")
        assert result.content == "<result>undefined</result>"
    finally:
        middleware.after_agent(state, None)
    assert not middleware._registry._slots
