"""Slots left behind by runs that never reach `after_agent` get closed.

A run that raises, is cancelled, or has its stream closed skips
`after_agent`, so its REPL slot stays registered. The next run on the thread
must not reuse that slot's JS state (it was never checkpointed), and idle
slots must not accumulate forever.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs._repl import _Registry
from tests._common import FakeChatModel

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
    # The failed run skipped `after_agent`, so its slot (counter = 99) leaked.
    assert len(mw._registry._slots) == 1
    leaked = next(iter(mw._registry._slots.values()))

    result = await _invoke(agent, {"messages": [HumanMessage("read")]}, config, mode)

    # The new run closed the leaked slot and restored the last snapshot.
    assert "<result>10</result>" in _last_eval(result), _last_eval(result)
    assert leaked.repl._ctx is None
    assert mw._registry._slots == {}


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
    for _ in range(200):
        await asyncio.sleep(0.01)
        if mw._registry._slots:
            break
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(mw._registry._slots) == 1

    result = await agent.ainvoke({"messages": [HumanMessage("read")]}, config=config)

    assert "<result>10</result>" in _last_eval(result), _last_eval(result)
    assert mw._registry._slots == {}


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
    slot = next(iter(mw._registry._slots.values()))

    result = await _invoke(agent, Command(resume="yes"), config, mode)

    # Resume doesn't re-run `before_agent`: the live slot (counter = 42) is
    # kept rather than rolled back to the snapshot (counter = 10).
    assert "<result>42</result>" in _last_eval(result), _last_eval(result)
    assert slot.repl._ctx is None  # closed by `after_agent` at the end
    assert mw._registry._slots == {}


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _registry(clock: _Clock, idle_ttl: float | None = 60.0) -> _Registry:
    return _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
        idle_ttl=idle_ttl,
        clock=clock,
    )


def test_idle_slot_expires_on_next_acquire_for_another_thread() -> None:
    clock = _Clock()
    expired: list[str] = []
    reg = _registry(clock)
    reg.on_expire = expired.append
    try:
        slot = reg.acquire("idle")
        reg.release(slot)

        clock.now += 59
        reg.release(reg.acquire("other"))
        assert "idle" in reg._slots

        clock.now += 2
        with patch.object(reg, "_close_slot", wraps=reg._close_slot) as close_spy:
            reg.release(reg.acquire("other"))
        close_spy.assert_called_once_with(slot)
        assert "idle" not in reg._slots
        assert expired == ["idle"]
    finally:
        reg.close()


async def test_idle_slot_expires_on_aevict() -> None:
    clock = _Clock()
    reg = _registry(clock)
    try:
        await reg.arelease(await reg.aacquire("idle"))
        clock.now += 61
        await reg.aevict("unrelated")
        assert "idle" not in reg._slots
    finally:
        reg.close()


def test_leased_slot_is_never_expired() -> None:
    clock = _Clock()
    reg = _registry(clock)
    try:
        slot = reg.acquire("busy")
        clock.now += 3600
        reg.release(reg.acquire("other"))
        assert reg._slots["busy"] is slot
    finally:
        reg.close()


def test_eval_longer_than_ttl_is_not_closed() -> None:
    clock = _Clock()
    reg = _registry(clock, idle_ttl=1.0)
    try:
        slot = reg.acquire("long")
        slot.repl.eval_sync("var x = 1")
        # 10x the TTL passes mid-eval while other threads hit the registry.
        clock.now += 10
        reg.evict("other")
        reg.release(reg.acquire("other"))
        assert slot.repl.eval_sync("x + 1").result == "2"
        reg.release(slot)  # release refreshes last_used
        reg.release(reg.acquire("other"))
        assert reg._slots["long"] is slot
    finally:
        reg.close()


def test_touch_postpones_expiry() -> None:
    clock = _Clock()
    reg = _registry(clock)
    try:
        reg.release(reg.acquire("t"))
        clock.now += 50
        reg.touch("t")
        clock.now += 50
        reg.release(reg.acquire("other"))
        assert "t" in reg._slots
    finally:
        reg.close()


def test_evicting_a_busy_slot_defers_close_to_release() -> None:
    reg = _registry(_Clock())
    try:
        slot = reg.acquire("busy")
        with patch.object(reg, "_close_slot", wraps=reg._close_slot) as close_spy:
            reg.evict("busy")
            assert "busy" not in reg._slots
            close_spy.assert_not_called()
            assert slot.repl.eval_sync("1 + 1").result == "2"
            reg.release(slot)
        close_spy.assert_called_once_with(slot)
    finally:
        reg.close()


def test_idle_ttl_none_disables_expiry() -> None:
    clock = _Clock()
    reg = _registry(clock, idle_ttl=None)
    try:
        reg.release(reg.acquire("kept"))
        clock.now += 10**9
        reg.release(reg.acquire("other"))
        assert "kept" in reg._slots
    finally:
        reg.close()


def test_expiry_clears_middleware_ptc_caches() -> None:
    clock = _Clock()
    mw = CodeInterpreterMiddleware(idle_ttl=60)
    mw._registry.clock = clock
    try:
        mw._ptc_tools_by_slot["s"] = ()
        mw._ptc_prompt_by_slot["s"] = ((), "")
        mw._registry.release(mw._registry.acquire("s"))
        clock.now += 61
        mw._registry.release(mw._registry.acquire("other"))
        assert "s" not in mw._ptc_tools_by_slot
        assert "s" not in mw._ptc_prompt_by_slot
    finally:
        mw._registry.close()


def test_before_agent_closes_stale_slot_and_its_caches() -> None:
    mw = CodeInterpreterMiddleware()
    try:
        state: dict[str, Any] = {"messages": [], "_quickjs_slot_id": "stale"}
        stale = mw._registry.acquire("stale")
        mw._registry.release(stale)
        mw._ptc_tools_by_slot["stale"] = ()
        mw._ptc_prompt_by_slot["stale"] = ((), "")

        mw.before_agent(state, MagicMock())  # type: ignore[arg-type]

        assert mw._registry.get_if_exists("stale") is None
        assert "stale" not in mw._ptc_tools_by_slot
        assert "stale" not in mw._ptc_prompt_by_slot
    finally:
        mw._registry.close()


async def test_abefore_agent_closes_stale_slot() -> None:
    mw = CodeInterpreterMiddleware()
    try:
        state: dict[str, Any] = {"messages": [], "_quickjs_slot_id": "stale"}
        await mw._registry.arelease(await mw._registry.aacquire("stale"))
        await mw.abefore_agent(state, MagicMock())  # type: ignore[arg-type]
        assert mw._registry.get_if_exists("stale") is None
    finally:
        mw._registry.close()


@pytest.mark.parametrize("bad", [0, -1.0])
def test_idle_ttl_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="idle_ttl"):
        CodeInterpreterMiddleware(idle_ttl=bad)
