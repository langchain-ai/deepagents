"""Contexts left behind by runs that never reach `after_agent` get closed.

A run that raises, is cancelled, or has its stream closed skips
`after_agent`, so its turn's Context stays open. The next run on the thread
must not reuse that Context's JS state (it was never checkpointed).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel
from tests.unit_tests._slots import drop_slots, slots_of, warm_slot

InvokeMode = Literal["invoke", "ainvoke"]


def _eval_call(code: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "eval", "args": {"code": code}, "id": call_id, "type": "tool_call"}
        ],
    )


def _tool_call(name: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {}, "id": call_id, "type": "tool_call"}],
    )


@tool
def boom() -> str:
    """Raise, failing the run before `after_agent`."""
    msg = "boom"
    raise RuntimeError(msg)


@tool
async def hang() -> str:
    """Block until cancelled."""
    await asyncio.Event().wait()
    return "unreachable"


@tool
def ask() -> str:
    """Pause the run on an interrupt."""
    return str(interrupt("continue?"))


def _last_eval(result: dict[str, Any]) -> str:
    evals = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"
    ]
    assert evals
    return str(evals[-1].content)


async def _invoke(agent: Any, payload: Any, config: Any, mode: InvokeMode) -> Any:
    if mode == "ainvoke":
        return await agent.ainvoke(payload, config=config)
    return agent.invoke(payload, config=config)


def _agent(
    script: list[AIMessage], tools: list[Any], mw: CodeInterpreterMiddleware
) -> Any:
    return create_agent(
        model=FakeChatModel(messages=iter(script)),
        tools=tools,
        middleware=[mw],
        checkpointer=InMemorySaver(),
    )


@pytest.mark.parametrize("mode", ["invoke", "ainvoke"])
async def test_next_run_restores_snapshot_after_failed_run(mode: InvokeMode) -> None:
    mw = CodeInterpreterMiddleware()
    agent = _agent(
        [
            _eval_call("var counter = 10", "c1"),
            AIMessage(content="turn 1 done"),
            _eval_call("counter = 99", "c2"),
            _tool_call("boom", "c3"),
            _eval_call("counter", "c4"),
            AIMessage(content="turn 3 done"),
        ],
        [boom],
        mw,
    )
    config = {"configurable": {"thread_id": "failed-run"}}

    await _invoke(agent, {"messages": [HumanMessage("set")]}, config, mode)
    with pytest.raises(RuntimeError, match="boom"):
        await _invoke(agent, {"messages": [HumanMessage("bump")]}, config, mode)
    # The failed run skipped `after_agent`, so its Context (counter = 99) leaked.
    slot = next(iter(slots_of(mw)))
    leaked = slot._context
    assert leaked is not None

    result = await _invoke(agent, {"messages": [HumanMessage("read")]}, config, mode)

    # The new run closed the leaked Context and restored the last snapshot on
    # the same warm slot.
    assert "<result>10</result>" in _last_eval(result), _last_eval(result)
    assert leaked._closed
    assert list(slots_of(mw)) == [slot]
    assert slot._context is not leaked


async def test_next_run_restores_snapshot_after_cancelled_run() -> None:
    mw = CodeInterpreterMiddleware()
    agent = _agent(
        [
            _eval_call("var counter = 10", "c1"),
            AIMessage(content="turn 1 done"),
            _eval_call("counter = 99", "c2"),
            _tool_call("hang", "c3"),
            _eval_call("counter", "c4"),
            AIMessage(content="turn 3 done"),
        ],
        [hang],
        mw,
    )
    config = {"configurable": {"thread_id": "cancelled-run"}}

    await agent.ainvoke({"messages": [HumanMessage("set")]}, config=config)
    task = asyncio.create_task(
        agent.ainvoke({"messages": [HumanMessage("bump")]}, config=config)
    )
    slot = next(iter(slots_of(mw)))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if slot._context is not None:
            break
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slot._context is not None

    result = await agent.ainvoke({"messages": [HumanMessage("read")]}, config=config)

    assert "<result>10</result>" in _last_eval(result), _last_eval(result)


@pytest.mark.parametrize("mode", ["invoke", "ainvoke"])
async def test_resume_after_interrupt_keeps_live_slot(mode: InvokeMode) -> None:
    mw = CodeInterpreterMiddleware()
    agent = _agent(
        [
            _eval_call("var counter = 10", "c1"),
            AIMessage(content="turn 1 done"),
            _eval_call("counter = 42", "c2"),
            _tool_call("ask", "c3"),
            _eval_call("counter", "c4"),
            AIMessage(content="turn 2 done"),
        ],
        [ask],
        mw,
    )
    config = {"configurable": {"thread_id": "interrupted-run"}}

    await _invoke(agent, {"messages": [HumanMessage("set")]}, config, mode)
    paused = await _invoke(agent, {"messages": [HumanMessage("bump")]}, config, mode)
    assert paused["__interrupt__"]
    slot = next(iter(slots_of(mw)))
    live = slot._context
    assert live is not None

    result = await _invoke(agent, Command(resume="yes"), config, mode)

    # Resume doesn't re-run `before_agent`: the live Context (counter = 42) is
    # kept rather than rolled back to the snapshot (counter = 10).
    assert "<result>42</result>" in _last_eval(result), _last_eval(result)
    assert slot._context is live  # snapshotted and kept warm by `after_agent`


@pytest.mark.parametrize("mode", ["invoke", "ainvoke"])
async def test_next_turn_reuses_warm_context(mode: InvokeMode) -> None:
    mw = CodeInterpreterMiddleware()
    agent = _agent(
        [
            _eval_call("var counter = 10", "c1"),
            AIMessage(content="turn 1 done"),
            _eval_call("counter", "c2"),
            AIMessage(content="turn 2 done"),
        ],
        [],
        mw,
    )
    config = {"configurable": {"thread_id": "warm-context"}}

    await _invoke(agent, {"messages": [HumanMessage("set")]}, config, mode)
    slot = next(iter(slots_of(mw)))
    warm = slot._context
    assert warm is not None

    result = await _invoke(agent, {"messages": [HumanMessage("read")]}, config, mode)

    assert "<result>10</result>" in _last_eval(result), _last_eval(result)
    assert slot._context is warm


def test_before_agent_closes_stale_context_and_its_caches() -> None:
    mw = CodeInterpreterMiddleware()
    try:
        state: dict[str, Any] = {"messages": [], "_quickjs_slot_id": "stale"}
        slot = warm_slot(mw, "stale")
        slot.eval_sync("var leaked = 1")
        runtime = slot._runtime
        mw._ptc_tools_by_slot["stale"] = ()

        mw.before_agent(state, MagicMock())  # type: ignore[arg-type]

        assert slot._context is None
        assert slot._runtime is runtime
        assert "stale" not in mw._ptc_tools_by_slot
    finally:
        drop_slots(mw)


async def test_abefore_agent_closes_stale_context() -> None:
    mw = CodeInterpreterMiddleware()
    try:
        state: dict[str, Any] = {"messages": [], "_quickjs_slot_id": "stale"}
        slot = warm_slot(mw, "stale")
        await slot.eval_async("var leaked = 1")
        await mw.abefore_agent(state, MagicMock())  # type: ignore[arg-type]
        assert slot._context is None
    finally:
        drop_slots(mw)


def test_before_agent_keeps_context_only_if_it_matches_the_snapshot() -> None:
    mw = CodeInterpreterMiddleware()
    try:
        state: dict[str, Any] = {"messages": [], "_quickjs_slot_id": "warm"}
        slot = warm_slot(mw, "warm")
        slot.eval_sync("var x = 1")
        first = slot.create_snapshot()
        slot.eval_sync("x = 2")
        second = slot.create_snapshot()
        warm = slot._context

        mw.before_agent({**state, "_quickjs_snapshot_payload": second}, MagicMock())  # type: ignore[arg-type]
        assert slot._context is warm

        # A fork or time travel back to an older snapshot gets a fresh Context.
        mw.before_agent({**state, "_quickjs_snapshot_payload": first}, MagicMock())  # type: ignore[arg-type]
        assert slot._context is None
    finally:
        drop_slots(mw)
