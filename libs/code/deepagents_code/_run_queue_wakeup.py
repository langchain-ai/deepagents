"""Wake the in-memory run queue when a run is created.

The server runs on `langgraph_runtime_inmem`. Its background scheduler polls
for pending runs: `Runs.next(wait=True)` sleeps 0.5 s, then claims only runs
created before that sleep began. Every user turn and every interrupt resume is
a new run, so each one waits 0.5-1.0 s before a worker starts it.

This patch makes `Runs.put` set an event and makes a waiting `Runs.next` wait
on that event instead of the fixed sleep. The stock `Runs.next` still claims
the runs, so queue order and per-thread locking do not change. The 0.5 s
timeout stays as a fallback, so a missed wakeup costs at most the stock delay.

`Runs.next` and `Runs.put` are private APIs. If their shape is not what this
module expects, the patch logs at debug level and does nothing. Remove this
module when `langgraph_runtime_inmem` wakes the scheduler on put.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.5
"""Fallback wait in seconds; matches the stock `Runs.next` sleep."""

_PATCHED_MARKER = "_dcode_run_queue_wakeup"
"""Attribute set on each wrapper so a second install is a no-op."""


class _Wakeup:
    """Edge-triggered signal from `Runs.put` to a waiting `Runs.next`.

    The event binds to the scheduler loop on the first wait. A put before
    that first wait is not lost: the scheduler then waits the fallback
    interval, as stock does.
    """

    def __init__(self) -> None:
        self._event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def wait(self) -> None:
        """Wait until a run is put, or until the fallback interval passes."""
        if self._event is None:
            self._event = asyncio.Event()
            self._loop = asyncio.get_running_loop()
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(_POLL_INTERVAL):
                await self._event.wait()
        self._event.clear()

    def notify(self) -> None:
        """Wake the scheduler; safe from any thread and after loop close."""
        event, loop = self._event, self._loop
        if event is None or loop is None:
            return
        # The loop can close during shutdown; the wakeup is then moot.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(event.set)


_WAKEUP = _Wakeup()


def _wrap_next(
    original: Callable[[bool, int], AsyncIterator[object]],
) -> Callable[[bool, int], AsyncIterator[object]]:
    """Replace the fixed poll sleep with a wait on `_WAKEUP`.

    Returns:
        A drop-in replacement for the upstream `Runs.next`.
    """

    @functools.wraps(original)
    async def next_runs(wait: bool, limit: int = 1) -> AsyncIterator[object]:
        if wait:
            await _WAKEUP.wait()
        # `wait=False` skips the sleep and claims runs created before now,
        # which includes the run that woke us.
        async for item in original(False, limit):
            yield item

    setattr(next_runs, _PATCHED_MARKER, True)
    return next_runs


def _wrap_put(
    original: Callable[..., Awaitable[object]],
) -> Callable[..., Awaitable[object]]:
    """Notify `_WAKEUP` after each run insert.

    Returns:
        A drop-in replacement for the upstream `Runs.put`.
    """

    @functools.wraps(original)
    async def put(*args: object, **kwargs: object) -> object:
        result = await original(*args, **kwargs)
        _WAKEUP.notify()
        return result

    setattr(put, _PATCHED_MARKER, True)
    return put


def _has_expected_shape(runs: type) -> bool:
    """Check that `Runs.next` and `Runs.put` look like the pinned version.

    Returns:
        `True` when both are static methods with the expected signatures.
    """
    next_attr = runs.__dict__.get("next")
    put_attr = runs.__dict__.get("put")
    if not isinstance(next_attr, staticmethod) or not isinstance(
        put_attr, staticmethod
    ):
        return False
    next_fn, put_fn = next_attr.__func__, put_attr.__func__
    params = list(inspect.signature(next_fn).parameters)
    return (
        inspect.isasyncgenfunction(next_fn)
        and params[:2] == ["wait", "limit"]
        and inspect.iscoroutinefunction(put_fn)
    )


def install_run_queue_wakeup() -> bool:
    """Patch the in-memory run queue to start new runs without the poll delay.

    Returns:
        `True` when the patch is active after the call, else `False`.
    """
    try:
        from langgraph_runtime_inmem.ops import Runs
    except ImportError:
        logger.debug("Run queue wakeup skipped: in-memory runtime not installed")
        return False
    if getattr(Runs.next, _PATCHED_MARKER, False):
        return True
    if not _has_expected_shape(Runs):
        logger.debug("Run queue wakeup skipped: Runs.next/put changed upstream")
        return False
    wrappers = {"next": _wrap_next, "put": _wrap_put}
    for name, wrap in wrappers.items():
        setattr(Runs, name, staticmethod(wrap(getattr(Runs, name))))
    logger.debug("Run queue wakeup installed")
    return True
