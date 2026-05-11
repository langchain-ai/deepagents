"""Server-authoritative round timer.

Replaces the old "operator clicks End when the CLI countdown finishes"
flow. The controller arms the timer in `/api/round/start`; if it expires
without an early end, the timer fires the on-expiry callback (auto-end +
eval) exactly once.

The timer never raises out of the asyncio task — it logs and exits. The
auto-end callback owns its own error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TimerSnapshot:
    """Serializable view of the current timer for `/api/state`."""

    running: bool
    duration_secs: float
    remaining_secs: float
    started_at: float | None


class RoundTimer:
    """One-shot countdown that fires an async callback on expiry."""

    def __init__(self) -> None:
        """Initialize an idle timer (no countdown running)."""
        self._task: asyncio.Task[None] | None = None
        self._started_at: float | None = None
        self._duration_secs: float = 0.0
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Return the asyncio lock, lazy-initialized inside the running loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def snapshot(self) -> TimerSnapshot:
        """Return the current timer state for HTTP polling."""
        if self._task is None or self._task.done() or self._started_at is None:
            return TimerSnapshot(
                running=False,
                duration_secs=self._duration_secs,
                remaining_secs=0.0,
                started_at=None,
            )
        elapsed = time.monotonic() - self._started_at
        remaining = max(0.0, self._duration_secs - elapsed)
        return TimerSnapshot(
            running=True,
            duration_secs=self._duration_secs,
            remaining_secs=remaining,
            started_at=self._started_at,
        )

    async def start(
        self,
        duration_secs: float,
        on_expire: Callable[[], Awaitable[None]],
    ) -> None:
        """Cancel any in-flight countdown and arm a new one.

        Args:
            duration_secs: Round duration in seconds.
            on_expire: Awaitable called exactly once if the timer
                reaches zero without being cancelled.
        """
        async with self._get_lock():
            await self._cancel_locked()
            self._duration_secs = float(duration_secs)
            self._started_at = time.monotonic()
            self._task = asyncio.create_task(self._run(duration_secs, on_expire))

    async def cancel(self) -> None:
        """Cancel the in-flight countdown, if any. Safe to call when idle."""
        async with self._get_lock():
            await self._cancel_locked()

    async def _cancel_locked(self) -> None:
        """Cancel the in-flight task; assumes `self._lock` is held."""
        task = self._task
        self._task = None
        self._started_at = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except RuntimeError:
            logger.debug("Round timer task awaited across loops; ignoring.")
        except Exception:
            logger.exception("Round timer task raised on cancellation")

    async def _run(
        self,
        duration_secs: float,
        on_expire: Callable[[], Awaitable[None]],
    ) -> None:
        """Sleep for `duration_secs`, then call `on_expire`."""
        try:
            await asyncio.sleep(duration_secs)
        except asyncio.CancelledError:
            raise
        try:
            await on_expire()
        except Exception:
            logger.exception("Round timer expiry callback raised")
