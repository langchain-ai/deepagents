"""Behavioral coverage for allocating QuickJS only when eval is called."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_quickjs import CodeInterpreterMiddleware, _repl
from tests._common import FakeChatModel

InvokeMode = Literal["invoke", "ainvoke"]


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
    from langchain.tools import ToolRuntime

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
            return await middleware.tools[0].coroutine(runtime=runtime, code=code)
        return middleware.tools[0].func(runtime=runtime, code=code)

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
