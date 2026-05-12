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

TIMER_WARNING_MESSAGES: dict[int, str] = {
    150: "2.5 minutes left",
    60: "1 minute left",
    30: "30 seconds left",
}


@dataclass(frozen=True, slots=True)
class TimerWarning:
    """Warning threshold reached by an active countdown."""

    threshold_secs: int
    message: str


def timer_warning_for_remaining(
    *, duration_secs: float, remaining_secs: float
) -> TimerWarning | None:
    """Return the most recent warning threshold crossed by a countdown.

    Args:
        duration_secs: Total countdown duration.
        remaining_secs: Current countdown time remaining.

    Returns:
        Timer warning metadata when a configured threshold has been reached.
    """
    if remaining_secs <= 0:
        return None
    for threshold_secs in sorted(TIMER_WARNING_MESSAGES):
        if duration_secs >= threshold_secs and remaining_secs <= threshold_secs:
            return TimerWarning(
                threshold_secs=threshold_secs,
                message=TIMER_WARNING_MESSAGES[threshold_secs],
            )
    return None


@dataclass(frozen=True, slots=True)
class TimerSnapshot:
    """Serializable view of the current timer for `/api/state`.

    Construct via `TimerSnapshot.idle(...)` or `TimerSnapshot.active(...)`
    so the `running` / `started_at` invariant is enforced at the call
    site rather than by ad-hoc consumer code.
    """

    running: bool
    duration_secs: float
    remaining_secs: float
    started_at: float | None
    warning: TimerWarning | None = None

    def __post_init__(self) -> None:
        """Reject internally inconsistent snapshots."""
        if self.running and self.started_at is None:
            msg = "running snapshot must have a started_at timestamp"
            raise ValueError(msg)
        if not self.running and self.started_at is not None:
            msg = "idle snapshot must not have a started_at timestamp"
            raise ValueError(msg)
        if not self.running and self.warning is not None:
            msg = "idle snapshot must not have a warning"
            raise ValueError(msg)
        if self.duration_secs < 0 or self.remaining_secs < 0:
            msg = "duration_secs and remaining_secs must be non-negative"
            raise ValueError(msg)

    @classmethod
    def idle(cls, *, duration_secs: float) -> TimerSnapshot:
        """Return a snapshot for an idle (or just-cancelled) timer."""
        return cls(
            running=False,
            duration_secs=duration_secs,
            remaining_secs=0.0,
            started_at=None,
        )

    @classmethod
    def active(
        cls,
        *,
        duration_secs: float,
        remaining_secs: float,
        started_at: float,
    ) -> TimerSnapshot:
        """Return a snapshot for a running countdown."""
        return cls(
            running=True,
            duration_secs=duration_secs,
            remaining_secs=remaining_secs,
            started_at=started_at,
            warning=timer_warning_for_remaining(
                duration_secs=duration_secs,
                remaining_secs=remaining_secs,
            ),
        )


@dataclass(slots=True)
class _ActiveCountdown:
    """Grouped state for an in-flight countdown.

    Held atomically on `RoundTimer._active` so `snapshot()` reads one
    reference rather than three correlated fields (preventing torn
    reads during cancellation).
    """

    task: asyncio.Task[None]
    started_at: float
    duration_secs: float
    start_delay_secs: float


class RoundTimer:
    """One-shot countdown that fires an async callback on expiry."""

    def __init__(self) -> None:
        """Initialize an idle timer (no countdown running)."""
        self._active: _ActiveCountdown | None = None
        self._last_duration_secs: float = 0.0
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Return the asyncio lock, lazy-bound to the running loop.

        Not strictly thread-safe for concurrent first-callers, but FastAPI
        runs on a single event loop so the first `await` happens before
        any second task observes `self._lock is None`.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def snapshot(self) -> TimerSnapshot:
        """Return the current timer state for HTTP polling."""
        active = self._active
        if active is None or active.task.done():
            return TimerSnapshot.idle(duration_secs=self._last_duration_secs)
        elapsed = max(0.0, time.monotonic() - active.started_at)
        remaining = max(0.0, active.duration_secs - elapsed)
        return TimerSnapshot.active(
            duration_secs=active.duration_secs,
            remaining_secs=remaining,
            started_at=active.started_at,
        )

    async def start(
        self,
        duration_secs: float,
        on_expire: Callable[[], Awaitable[None]],
        *,
        start_delay_secs: float = 0.0,
    ) -> None:
        """Cancel any in-flight countdown and arm a new one.

        Args:
            duration_secs: Round duration in seconds. Must be non-negative.
            on_expire: Awaitable called exactly once if the timer
                reaches zero without being cancelled.
            start_delay_secs: Seconds to wait before the visible round
                timer begins. This lets launch countdowns finish before
                player time is consumed.

        Raises:
            ValueError: If `duration_secs` or `start_delay_secs` is negative.
        """
        if duration_secs < 0:
            msg = f"duration_secs must be non-negative, got {duration_secs}"
            raise ValueError(msg)
        if start_delay_secs < 0:
            msg = f"start_delay_secs must be non-negative, got {start_delay_secs}"
            raise ValueError(msg)
        duration = float(duration_secs)
        delay = float(start_delay_secs)
        async with self._get_lock():
            await self._cancel_locked()
            self._last_duration_secs = duration
            task = asyncio.create_task(self._run(duration, on_expire, delay))
            self._active = _ActiveCountdown(
                task=task,
                started_at=time.monotonic() + delay,
                duration_secs=duration,
                start_delay_secs=delay,
            )

    async def cancel(self) -> None:
        """Cancel the in-flight countdown, if any. Safe to call when idle."""
        async with self._get_lock():
            await self._cancel_locked()

    async def _cancel_locked(self) -> None:
        """Cancel the in-flight task; assumes `self._lock` is held."""
        active = self._active
        self._active = None
        if active is None or active.task.done():
            return
        active.task.cancel()
        try:
            await active.task
        except asyncio.CancelledError:
            pass
        except RuntimeError:
            # TestClient teardown swaps event loops between requests; awaiting
            # a task bound to the prior loop raises RuntimeError. Safe to
            # ignore in that narrow case.
            logger.debug("Round timer task awaited across loops; ignoring.")
        except Exception:
            logger.exception("Round timer task raised on cancellation")

    async def _run(
        self,
        duration_secs: float,
        on_expire: Callable[[], Awaitable[None]],
        start_delay_secs: float,
    ) -> None:
        """Sleep for `duration_secs`, then call `on_expire`."""
        try:
            if start_delay_secs:
                await asyncio.sleep(start_delay_secs)
            await asyncio.sleep(duration_secs)
        except asyncio.CancelledError:
            raise
        try:
            await on_expire()
        except Exception:
            logger.exception("Round timer expiry callback raised")
