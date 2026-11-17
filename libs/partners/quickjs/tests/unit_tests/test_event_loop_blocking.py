"""The caller's event loop keeps running while slots build and close.

Each test runs a 5 ms heartbeat on the caller's loop and makes the slot's
worker do slow, blocking work. A heartbeat gap near that work's duration means
the loop waited on the worker thread.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest
from quickjs_rs import Runtime

from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs._repl import _Registry, _Slot, _ThreadREPL

if TYPE_CHECKING:
    from collections.abc import Awaitable

SLOW = 0.3
# A loop that never waits on the worker stays well under SLOW.
MAX_GAP = 0.15


def _registry() -> _Registry:
    return _Registry(
        memory_limit=64 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )


async def _max_gap(awaitable: Awaitable[Any]) -> tuple[float, Any]:
    """Await `awaitable` under a heartbeat; return the worst gap and result."""
    gaps: list[float] = []

    async def _beat() -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.005)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(_beat())
    await asyncio.sleep(0.02)
    gaps.clear()
    try:
        result = await awaitable
    finally:
        await asyncio.sleep(0.02)
        beat.cancel()
    return max(gaps), result


def _worker_threads(slot_id: str) -> list[threading.Thread]:
    name = f"quickjs-worker-{slot_id[:8]}"
    return [t for t in threading.enumerate() if t.name == name and t.is_alive()]


async def _wait_for_no_worker(slot_id: str) -> None:
    deadline = time.monotonic() + 5.0
    # Threads expose nothing awaitable, so poll.
    while _worker_threads(slot_id) and time.monotonic() < deadline:  # noqa: ASYNC110
        await asyncio.sleep(0.01)
    assert not _worker_threads(slot_id)


@pytest.fixture
def slow_build(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make Runtime creation block its worker thread for SLOW seconds."""
    calls: list[str] = []

    def _create_runtime(self: _Slot) -> Runtime:
        calls.append(threading.current_thread().name)
        time.sleep(SLOW)
        return Runtime(memory_limit=64 * 1024 * 1024)

    monkeypatch.setattr(_Slot, "_create_runtime", _create_runtime)
    return calls


async def test_first_eval_build_does_not_block_loop(slow_build: list[str]) -> None:
    registry = _registry()
    slot = registry.get("build-01")
    gap, outcome = await _max_gap(slot.eval_async("1 + 1"))
    assert gap < MAX_GAP
    assert outcome.result == "2"
    assert slow_build == ["quickjs-worker-build-01"]
    await registry.aevict("build-01")


async def test_concurrent_first_evals_build_one_runtime(slow_build: list[str]) -> None:
    registry = _registry()
    slot = registry.get("shared-1")
    outcomes = await asyncio.gather(
        *(slot.eval_async(f"{i} + 1", fresh=True) for i in range(3))
    )
    assert [o.result for o in outcomes] == ["1", "2", "3"]
    assert slow_build == ["quickjs-worker-shared-1"]
    await registry.aevict("shared-1")


async def test_fresh_context_does_not_block_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    slot = registry.get("fresh-01")
    await slot.eval_async("1")
    original = _ThreadREPL._init

    def _slow_init(self: _ThreadREPL) -> None:
        time.sleep(SLOW)
        original(self)

    monkeypatch.setattr(_ThreadREPL, "_init", _slow_init)
    gap, outcome = await _max_gap(slot.eval_async("1 + 1", fresh=True))
    assert gap < MAX_GAP
    assert outcome.result == "2"
    await registry.aevict("fresh-01")


@pytest.mark.parametrize("mode", ["thread", "turn"])
async def test_after_agent_does_not_block_loop(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mw = CodeInterpreterMiddleware(mode=mode)  # type: ignore[arg-type]
    slot_id = f"after-{mode}"
    slot = mw._registry.get(slot_id)
    await slot.eval_async("globalThis.x = 1")
    original = _ThreadREPL._aclose

    async def _slow_aclose(self: _ThreadREPL) -> None:
        time.sleep(SLOW)  # noqa: ASYNC251  # blocks the worker thread on purpose
        await original(self)

    monkeypatch.setattr(_ThreadREPL, "_aclose", _slow_aclose)
    state: dict[str, Any] = {"_quickjs_slot_id": slot_id}
    gap, _ = await _max_gap(mw.aafter_agent(state, None))  # type: ignore[arg-type]
    assert gap < MAX_GAP
    assert slot.repl is None
    await mw._registry.aevict(slot_id)


async def test_aevict_does_not_block_loop() -> None:
    registry = _registry()
    slot = registry.get("evict-01")
    await slot.eval_async("1")

    async def _linger() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            time.sleep(SLOW)  # noqa: ASYNC251  # blocks the worker thread on purpose
            raise

    lingering = slot.worker.run_async(_linger())
    gap, _ = await _max_gap(registry.aevict("evict-01"))
    assert gap < MAX_GAP
    assert registry.get_if_exists("evict-01") is None
    await _wait_for_no_worker("evict-01")
    assert lingering.cancelled()
