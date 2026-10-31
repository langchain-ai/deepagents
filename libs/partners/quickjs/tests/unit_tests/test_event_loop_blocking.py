"""The caller's event loop keeps running while slots are built and closed.

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
from langchain_quickjs._repl import _Registry, _ThreadREPL

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
    """Make runtime creation block its worker thread for SLOW seconds."""
    calls: list[str] = []

    async def _acreate_runtime(self: _Registry) -> Runtime:
        calls.append(threading.current_thread().name)
        time.sleep(SLOW)  # noqa: ASYNC251  # blocks the worker thread on purpose
        return Runtime(memory_limit=self.memory_limit)

    monkeypatch.setattr(_Registry, "_acreate_runtime", _acreate_runtime)
    return calls


def _slow_worker_teardown(registry: _Registry, slot_id: str) -> asyncio.Future[None]:
    """Leave a task on the slot's worker that takes SLOW seconds to cancel."""

    async def _linger() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            time.sleep(SLOW)  # noqa: ASYNC251  # blocks the worker thread on purpose
            raise

    return registry._slots[slot_id].worker.run_async(_linger())


async def test_build_does_not_block_loop(slow_build: list[str]) -> None:
    registry = _registry()
    gap, repl = await _max_gap(registry.aget("build-01"))
    assert gap < MAX_GAP
    assert slow_build == ["quickjs-worker-build-01"]
    assert (await repl.eval_async("1 + 1")).result == "2"
    await registry.aevict("build-01")


@pytest.mark.parametrize("mode", ["call", "thread"])
async def test_evict_does_not_block_loop(mode: str) -> None:
    mw = CodeInterpreterMiddleware(mode=mode)  # type: ignore[arg-type]
    slot_id = f"evict-{mode}"
    repl = await mw._registry.aget(slot_id)
    await repl.eval_async("globalThis.x = 1")
    lingering = _slow_worker_teardown(mw._registry, slot_id)
    state: dict[str, Any] = {"_quickjs_slot_id": slot_id}
    gap, _ = await _max_gap(mw.aafter_agent(state, None))  # type: ignore[arg-type]
    assert gap < MAX_GAP
    assert mw._registry.get_if_exists(slot_id) is None
    await _wait_for_no_worker(slot_id)
    assert lingering.cancelled()


async def test_call_mode_reset_does_not_block_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    first = await registry.aget("reset-01")
    original = _ThreadREPL._ainit

    async def _slow_ainit(self: _ThreadREPL) -> None:
        time.sleep(SLOW)  # noqa: ASYNC251  # blocks the worker thread on purpose
        await original(self)

    monkeypatch.setattr(_ThreadREPL, "_ainit", _slow_ainit)
    gap, _ = await _max_gap(registry.areset_repl("reset-01"))
    assert gap < MAX_GAP
    second = registry.get_if_exists("reset-01")
    assert second is not None
    assert second is not first
    await registry.aevict("reset-01")


@pytest.mark.usefixtures("slow_build")
async def test_concurrent_misses_share_one_slot() -> None:
    registry = _registry()
    repls = await asyncio.gather(*(registry.aget("shared-1") for _ in range(3)))
    assert all(r is repls[0] for r in repls)
    assert registry._builds == {}
    # Duplicate builds are closed; only the installed slot's worker remains.
    deadline = time.monotonic() + 5.0
    while len(_worker_threads("shared-1")) > 1 and time.monotonic() < deadline:  # noqa: ASYNC110
        await asyncio.sleep(0.01)
    assert len(_worker_threads("shared-1")) == 1
    await registry.aevict("shared-1")
    await _wait_for_no_worker("shared-1")


@pytest.mark.usefixtures("slow_build")
async def test_cancel_mid_build_leaves_no_worker() -> None:
    registry = _registry()
    task = asyncio.create_task(registry.aget("cancel-1"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _wait_for_no_worker("cancel-1")
    assert registry._slots == {}
    assert registry._builds == {}


@pytest.mark.usefixtures("slow_build")
async def test_evict_mid_build_leaves_registry_empty() -> None:
    registry = _registry()
    task = asyncio.create_task(registry.aget("evicted1"))
    await asyncio.sleep(0.05)
    await registry.aevict("evicted1")
    with pytest.raises(RuntimeError, match="evicted"):
        await task
    await _wait_for_no_worker("evicted1")
    assert registry._slots == {}
    assert registry._builds == {}
