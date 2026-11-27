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
        self._running: dict[str, asyncio.Task[None]] = {}
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
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        runs = list(self._running.values())
        for task in runs:
            task.cancel()
        await asyncio.gather(*runs, return_exceptions=True)
        self._running.clear()

    async def tick_once(self, *, wait: bool = True) -> None:
        """Remove finished jobs, then run all jobs due at the current clock value once.

        Args:
            wait: Wait for runs dispatched by this tick. The ticker keeps scanning
                while they run and coalesces overlapping runs of the same job.
        """
        current = self.now()
        running = frozenset(self._running)
        await self.store.claim_missed_deadlines(now=current, running=running)
        self._discard_finished(current, running)
        for job_id, task in list(self._running.items()):
            if task.done():
                # Cancellation before the coroutine starts skips its finally block.
                self._running.pop(job_id)
                continue
            current_job = self.store.get_job(job_id)
            # A final claimed fire can be disabled with no next run by design.
            if current_job is None or (
                not current_job.enabled and current_job.next_run_at is not None
            ):
                task.cancel()
        jobs = self.store.due_jobs(now=current)
        log_event(logger, "cron.tick", due_count=len(jobs), now=current.isoformat())
        started = []
        notices = await self._pending_notices(jobs)
        for job in [*jobs, *notices]:
            if job.id not in self._running:
                task = asyncio.create_task(self._dispatch(job, current, notice_only=job in notices))
                self._running[job.id] = task
                started.append(task)
        if wait:
            await asyncio.gather(*started)

    async def _pending_question(self, job: CronJob) -> str:
        async with self.store.open_journal() as journal:
            return await journal.pending_question(job)

    async def _pending_notices(self, jobs: list[CronJob]) -> list[CronJob]:
        due = {job.id for job in jobs}
        return [
            job
            for job in self.store.list_jobs()
            if job.notice_state in {"pending", "uncertain"}
            or (
                job.enabled
                and job.mode == "objective"
                and job.id not in due
                and await self._pending_question(job)
            )
        ]

    async def _dispatch(self, job: CronJob, now: datetime, *, notice_only: bool) -> None:
        try:
            if not notice_only:
                await self._run_due_job(job, now)
            elif self.store.get_job(job.id) == job:
                if job.completion_notice:
                    await self._deliver_completion(job)
                elif notice := await self._pending_question(job):
                    await self.deliver_result(job, notice)
        except Exception:
            logger.exception("Failed to dispatch job %s", job.id)
        finally:
            self._running.pop(job.id, None)

    async def _deliver_completion(self, job: CronJob) -> None:
        try:
            await self.deliver_result(job, job.completion_notice)
            if job.notice_state == "pending":
                self.store.transition_notice(job, "delivered")
        except Exception:
            logger.exception("Completion notice delivery interrupted for %s", job.id)

    async def _ticker(self) -> None:
        """Scan for due jobs until stopped, surviving a failed scan.

        A tick reads the store, consults the clock, and dispatches; an
        unexpected raise from any of those would otherwise leave the task
        completed-with-exception, and since nothing awaits it but `stop`, the
        scheduler would go quiet for the life of the process with no more than
        an "exception was never retrieved" warning at collection time. Logging
        and continuing costs one missed scan instead: due jobs stay due, so the
        next tick picks them up.

        `Exception` rather than `BaseException` is deliberate --
        `asyncio.CancelledError` derives from the latter, so cancellation still
        propagates and `stop` keeps working.
        """
        while not self._stopped.is_set():
            try:
                await self.tick_once(wait=False)
            except Exception as exc:
                logger.exception("Cron tick failed")
                # `log_event` JSON-encodes and redacts its fields, so untrusted
                # text off the store cannot forge a log line.
                log_event(logger, "cron.tick_failure", error=str(exc))
            # The wait happens even after a failure, so a persistently broken
            # tick retries on the normal interval instead of spinning.
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.tick_seconds)
            except TimeoutError:
                continue

    def _discard_finished(self, now: datetime, running: frozenset[str]) -> None:
        # One event with the facts rather than a completed/expired split: a job
        # can finish its last run and pass `until` in the same tick, and
        # `last_run_at` already says whether it ever ran.
        for job in self.store.discard_finished(now=now, running=running):
            log_event(
                logger,
                "cron.job_removed",
                job_id=job.id,
                job_name=job.name,
                last_run_at=None if job.last_run_at is None else job.last_run_at.isoformat(),
                until=None if job.until is None else job.until.isoformat(),
            )

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

        finished = self.now()
        if notice := claimed.target_delay(finished):
            text = notice if not text or is_silent(text) else f"{notice}\n{text}"

        self.store.mark_job_run(claimed.id, status="ok", error=None, now=finished)
        log_event(
            logger,
            "cron.success",
            job_id=claimed.id,
            job_name=claimed.name,
            silent=is_silent(text),
            has_delivery=bool(text and not is_silent(text)),
        )
        current = self.store.get_job(claimed.id)
        if current and current.notice_state == "pending":
            await self._deliver_completion(current)
            return
        if is_silent(text):
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


def is_silent(text: str) -> bool:
    """Whether a scheduled result asks to be withheld from the chat.

    Args:
        text: Agent output produced for a scheduled job.

    Returns:
        Whether the text carries the silent sentinel at either end.
    """
    stripped = text.strip()
    return stripped.startswith(SILENT_SENTINEL) or stripped.endswith(SILENT_SENTINEL)
