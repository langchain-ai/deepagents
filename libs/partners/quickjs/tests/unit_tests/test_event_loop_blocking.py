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

from langchain_quickjs import CodeInterpreterMiddleware, _repl
from langchain_quickjs._repl import _Registry, _Slot, _SlotConfig
from tests.unit_tests._slots import warm_slot

if TYPE_CHECKING:
    from collections.abc import Awaitable

SLOW = 0.3
# A loop that never waits on the worker stays well under SLOW.
MAX_GAP = 0.15


def _get(registry: _Registry, thread_id: str) -> Any:
    slot = registry.acquire(thread_id, _SlotConfig())
    registry.release(slot)
    return slot


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

    def _slow_runtime(**kwargs: Any) -> Runtime:
        calls.append(threading.current_thread().name)
        time.sleep(SLOW)
        return Runtime(**kwargs)

    monkeypatch.setattr(_repl, "QjsRuntime", _slow_runtime)
    return calls


async def test_first_eval_build_does_not_block_loop(
    registry: _Registry, slow_build: list[str]
) -> None:
    slot = _get(registry, "build-01")
    gap, outcome = await _max_gap(slot.eval_async("1 + 1"))
    assert gap < MAX_GAP
    assert outcome.result == "2"
    assert slow_build == ["quickjs-worker-build-01"]


async def test_concurrent_first_evals_build_one_runtime(
    registry: _Registry, slow_build: list[str]
) -> None:
    slot = _get(registry, "shared-1")
    outcomes = await asyncio.gather(*(slot.eval_async(f"{i} + 1") for i in range(3)))
    assert outcomes[0].result == "1"
    assert all(o.error_type in {None, "ConcurrentEval"} for o in outcomes)
    assert slow_build == ["quickjs-worker-shared-1"]


async def test_fresh_context_does_not_block_loop(
    registry: _Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot = _get(registry, "fresh-01")
    await slot.eval_async("1")
    await slot.aclose_context()
    original = _Slot._install_console

    def _slow_install(self: _Slot) -> None:
        time.sleep(SLOW)
        original(self)

    monkeypatch.setattr(_Slot, "_install_console", _slow_install)
    gap, outcome = await _max_gap(slot.eval_async("1 + 1"))
    assert gap < MAX_GAP
    assert outcome.result == "2"


@pytest.mark.parametrize("mode", ["thread", "turn"])
async def test_after_agent_does_not_block_loop(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mw = CodeInterpreterMiddleware(mode=mode)  # type: ignore[arg-type]
    slot_id = f"after-{mode}"
    slot = warm_slot(mw, slot_id)
    await slot.eval_async("globalThis.x = 1")
    original = _Slot._dispose_context

    def _slow_close(self: _Slot) -> None:
        time.sleep(SLOW)
        original(self)

    monkeypatch.setattr(_Slot, "_dispose_context", _slow_close)
    state: dict[str, Any] = {"_quickjs_slot_id": slot_id}
    gap, _ = await _max_gap(mw.aafter_agent(state, None))  # type: ignore[arg-type]
    assert gap < MAX_GAP
    # `thread` mode keeps the snapshotted Context warm for the next turn.
    assert (slot._context is None) == (mode != "thread")


async def test_closing_an_evicted_slot_does_not_block_loop(
    registry: _Registry,
) -> None:
    registry._max_slots = 1
    slot = _get(registry, "evict-01")
    await slot.eval_async("1")

    async def _linger() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            time.sleep(SLOW)  # noqa: ASYNC251  # blocks the worker thread on purpose
            raise

    lingering = slot._worker.run_async(_linger())
    gap, newer = await _max_gap(registry.aacquire("evict-02", _SlotConfig()))
    registry.release(newer)
    assert gap < MAX_GAP
    assert [thread_id for thread_id, _ in registry._slots] == ["evict-02"]
    await _wait_for_no_worker("evict-01")
    assert lingering.cancelled()
