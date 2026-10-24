"""Tests for the in-memory run queue wakeup patch."""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from langgraph_runtime_inmem.database import GLOBAL_STORE, connect
from langgraph_runtime_inmem.ops import Runs

from deepagents_code import _run_queue_wakeup
from deepagents_code._run_queue_wakeup import install_run_queue_wakeup

if TYPE_CHECKING:
    from collections.abc import Iterator

_FAST_PICKUP_SECS = 0.25
"""Upper bound for pickup with the patch; stock waits at least 0.5 s."""


@pytest.fixture
def isolated_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test from stock `Runs` and a fresh wakeup signal.

    Other tests enter the server lifespan, which installs the patch for the
    rest of the session, so strip any wrappers before each test.
    """
    for name in ("next", "put"):
        stock = inspect.unwrap(Runs.__dict__[name].__func__)
        monkeypatch.setattr(Runs, name, staticmethod(stock))
    monkeypatch.setattr(_run_queue_wakeup, "_WAKEUP", _run_queue_wakeup._Wakeup())


@pytest.fixture
def stub_put(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[uuid.UUID]]:
    """Replace the upstream insert with a minimal pending run in the store.

    The real `Runs.put` needs a configured `langgraph_api` server. The wakeup
    only cares that a run is pending once `put` returns, and the real
    `Runs.next` still claims it.

    Yields:
        The thread IDs created, removed from the global store on teardown.
    """
    thread_ids: list[uuid.UUID] = []

    async def put(*_args: object, **_kwargs: object) -> None:  # noqa: RUF029  # upstream `put` is a coroutine function
        thread_id = uuid.uuid4()
        thread_ids.append(thread_id)
        GLOBAL_STORE["threads"].append({"thread_id": thread_id})
        GLOBAL_STORE["runs"].append(
            {
                "run_id": uuid.uuid4(),
                "thread_id": thread_id,
                "status": "pending",
                "created_at": datetime.now(UTC),
            }
        )

    monkeypatch.setattr(Runs, "put", staticmethod(put))
    yield thread_ids
    for key in ("runs", "threads"):
        GLOBAL_STORE[key][:] = [
            r for r in GLOBAL_STORE[key] if r["thread_id"] not in thread_ids
        ]


@pytest.mark.usefixtures("isolated_runs")
def test_installs_on_pinned_runtime() -> None:
    assert install_run_queue_wakeup() is True
    assert install_run_queue_wakeup() is True  # second call is a no-op


@pytest.mark.usefixtures("isolated_runs")
def test_skips_when_upstream_shape_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def next_runs(*, wait: bool) -> None: ...

    original_put = Runs.__dict__["put"]
    monkeypatch.setattr(Runs, "next", staticmethod(next_runs))

    assert install_run_queue_wakeup() is False
    assert Runs.__dict__["put"] is original_put


@pytest.mark.usefixtures("isolated_runs", "stub_put")
async def test_put_wakes_waiting_next() -> None:
    install_run_queue_wakeup()
    # Bind the wakeup to this loop the way the scheduler's first poll does.
    _ = [item async for item in Runs.next(wait=True)]

    async def claim() -> list[object]:
        return [run async for run, _attempt in Runs.next(wait=True)]

    waiter = asyncio.create_task(claim())
    await asyncio.sleep(0.05)  # let the waiter block on the wakeup
    start = time.perf_counter()
    async with connect() as conn:
        await Runs.put(
            conn, uuid.uuid4(), {}, metadata={}, prevent_insert_if_inflight=False
        )
    claimed = await waiter
    elapsed = time.perf_counter() - start

    assert len(claimed) == 1
    assert elapsed < _FAST_PICKUP_SECS
