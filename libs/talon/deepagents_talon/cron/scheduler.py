"""Ticker that runs due cron jobs.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from deepagents_talon.cron.jobs import CronJob, CronJobStore
from deepagents_talon.observability import log_event

logger = logging.getLogger(__name__)

SILENT_SENTINEL = "[SILENT]"
DEFAULT_TICK_SECONDS = 60.0

RunCronJob = Callable[[CronJob], Awaitable[str]]
DeliverCronResult = Callable[[CronJob, str], Awaitable[None]]
NowFactory = Callable[[], datetime]


class PersistentCronScheduler:
    """Persistent minute-granularity cron scheduler.

    Args:
        store: Cron job store.
        run_job: Callback that invokes the agent for a claimed job.
        deliver_result: Callback that delivers non-silent job output.
        tick_seconds: Interval between due-job scans.
        now: Clock override for deterministic tests.
    """

    def __init__(
        self,
        *,
        store: CronJobStore,
        run_job: RunCronJob,
        deliver_result: DeliverCronResult,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        now: NowFactory | None = None,
    ) -> None:
        """Initialize the scheduler without starting the ticker."""
        if tick_seconds <= 0:
            msg = "tick_seconds must be positive"
            raise ValueError(msg)
        self.store = store
        self.run_job = run_job
        self.deliver_result = deliver_result
        self.tick_seconds = tick_seconds
        self.now = now or (lambda: datetime.now(UTC))
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        """Start the scheduler ticker."""
        if self._task is not None and not self._task.done():
            return
        self._stopped.clear()
        self._task = asyncio.create_task(self._ticker(), name="talon:cron")

    async def stop(self) -> None:
        """Stop the scheduler ticker."""
        self._stopped.set()
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def tick_once(self) -> None:
        """Run all jobs due at the current clock value once."""
        current = self.now()
        jobs = self.store.due_jobs(now=current)
        log_event(logger, "cron.tick", due_count=len(jobs), now=current.isoformat())
        for job in jobs:
            await self._run_due_job(job, current)

    async def _ticker(self) -> None:
        while not self._stopped.is_set():
            await self.tick_once()
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.tick_seconds)
            except TimeoutError:
                continue

    async def _run_due_job(self, job: CronJob, now: datetime) -> None:
        claimed = self.store.advance_next_run(job.id, now=now)
        if claimed is None:
            return

        log_event(
            logger,
            "cron.dispatch",
            job_id=claimed.id,
            job_name=claimed.name,
            conversation_id=claimed.origin.conversation_id,
            next_run_at=None if claimed.next_run_at is None else claimed.next_run_at.isoformat(),
        )
        try:
            text = await self.run_job(claimed)
        except Exception as exc:
            logger.exception("Cron job %s failed", claimed.id)
            log_event(
                logger,
                "cron.failure",
                job_id=claimed.id,
                job_name=claimed.name,
                error=str(exc),
            )
            self.store.mark_job_run(
                claimed.id,
                status="error",
                error=str(exc),
                now=self.now(),
            )
            return

        self.store.mark_job_run(claimed.id, status="ok", error=None, now=self.now())
        log_event(
            logger,
            "cron.success",
            job_id=claimed.id,
            job_name=claimed.name,
            silent=_is_silent(text),
            has_delivery=bool(text and not _is_silent(text)),
        )
        if _is_silent(text):
            log_event(
                logger,
                "cron.delivery_suppressed",
                job_id=claimed.id,
                job_name=claimed.name,
            )
            return
        if text:
            try:
                await self.deliver_result(claimed, text)
            except Exception as exc:
                logger.exception("Cron job %s delivery failed", claimed.id)
                log_event(
                    logger,
                    "cron.delivery_failure",
                    job_id=claimed.id,
                    job_name=claimed.name,
                    error=str(exc),
                )
                self.store.mark_job_run(
                    claimed.id,
                    status="error",
                    error=f"delivery failed: {exc}",
                    now=self.now(),
                )
                return
            log_event(
                logger,
                "cron.delivery",
                job_id=claimed.id,
                job_name=claimed.name,
                conversation_id=claimed.origin.conversation_id,
            )


def _is_silent(text: str) -> bool:
    return text.strip().startswith(SILENT_SENTINEL)
