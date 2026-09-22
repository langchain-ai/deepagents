from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

from deepagents_talon.cron import CronJob, CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.cron.scheduler import PersistentCronScheduler


def _store(tmp_path) -> CronJobStore:
    return CronJobStore(assistant_id="assistant", cron_dir=tmp_path / "cron")


async def test_scheduler_runs_due_job_and_delivers_result(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    job = store.create_job(
        prompt="check status",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )
    delivered: list[tuple[str, str]] = []

    async def run_job(claimed: CronJob) -> str:
        assert claimed.id == job.id
        claimed_job = store.get_job(job.id)
        assert claimed_job is not None
        assert claimed_job.next_run_at is None
        return "done"

    async def deliver_result(claimed: CronJob, text: str) -> None:
        delivered.append((claimed.origin.conversation_id, text))

    scheduler = PersistentCronScheduler(
        store=store,
        run_job=run_job,
        deliver_result=deliver_result,
        now=lambda: now + timedelta(minutes=1),
    )

    await scheduler.tick_once()

    updated = store.get_job(job.id)
    assert updated is not None
    assert updated.last_status == "ok"
    assert updated.last_error is None
    assert delivered == [("chat", "done")]


async def test_scheduler_logs_structured_lifecycle_events(tmp_path, caplog) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    store.create_job(
        prompt="check status",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        name="status",
        now=now,
    )

    scheduler = PersistentCronScheduler(
        store=store,
        run_job=lambda _: _return("done"),
        deliver_result=_deliver_returned_text,
        now=lambda: now + timedelta(minutes=1),
    )

    with caplog.at_level(logging.INFO, logger="deepagents_talon.cron.scheduler"):
        await scheduler.tick_once()

    events = [_event(message)["event"] for message in caplog.messages]
    assert events == ["cron.tick", "cron.dispatch", "cron.success", "cron.delivery"]


async def test_scheduler_suppresses_silent_result(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    store.create_job(
        prompt="quiet heartbeat",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )
    delivered: list[str] = []

    scheduler = PersistentCronScheduler(
        store=store,
        run_job=lambda _: _return("[SILENT] nothing changed"),
        deliver_result=lambda _, text: _append(delivered, text),
        now=lambda: now + timedelta(minutes=1),
    )

    await scheduler.tick_once()

    assert delivered == []
    assert store.list_jobs()[0].last_status == "ok"


async def test_scheduler_suppresses_trailing_silent_result(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    store.create_job(
        prompt="quiet heartbeat",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )
    delivered: list[str] = []

    scheduler = PersistentCronScheduler(
        store=store,
        run_job=lambda _: _return("nothing changed [SILENT]"),
        deliver_result=lambda _, text: _append(delivered, text),
        now=lambda: now + timedelta(minutes=1),
    )

    await scheduler.tick_once()

    assert delivered == []
    assert store.list_jobs()[0].last_status == "ok"


async def test_scheduler_records_error_after_claiming_job(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    job = store.create_job(
        prompt="fail",
        schedule=CronSchedule.parse("every 5m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )

    async def run_job(_: CronJob) -> str:
        msg = "model unavailable"
        raise RuntimeError(msg)

    scheduler = PersistentCronScheduler(
        store=store,
        run_job=run_job,
        deliver_result=_deliver_returned_text,
        now=lambda: now + timedelta(minutes=5),
    )

    await scheduler.tick_once()

    updated = store.get_job(job.id)
    assert updated is not None
    assert updated.last_status == "error"
    assert updated.last_error == "model unavailable"
    assert updated.next_run_at == now + timedelta(minutes=10)


async def test_scheduler_records_delivery_error(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    job = store.create_job(
        prompt="deliver",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )

    async def deliver_result(_: CronJob, __: str) -> None:
        msg = "bridge unavailable"
        raise RuntimeError(msg)

    scheduler = PersistentCronScheduler(
        store=store,
        run_job=lambda _: _return("done"),
        deliver_result=deliver_result,
        now=lambda: now + timedelta(minutes=1),
    )

    await scheduler.tick_once()

    updated = store.get_job(job.id)
    assert updated is not None
    assert updated.last_status == "error"
    assert updated.last_error == "delivery failed: bridge unavailable"


async def _return(value: str) -> str:
    return value


async def _deliver_returned_text(_: CronJob, text: str) -> None:
    await _return(text)


async def _append(values: list[str], value: str) -> None:
    values.append(value)


def _event(message: str) -> dict[str, object]:
    return json.loads(message.removeprefix("talon_event "))


def _is_event(message: str) -> bool:
    return message.startswith("talon_event ")


async def test_ticker_survives_a_failing_tick(tmp_path, caplog) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    job = store.create_job(
        prompt="check status",
        schedule=CronSchedule.parse("every 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )
    ran: list[str] = []
    failures = 2
    ticks = 0
    original_due_jobs = store.due_jobs

    def flaky_due_jobs(*, now: datetime | None = None) -> list[CronJob]:
        nonlocal ticks
        ticks += 1
        if ticks <= failures:
            msg = "store unavailable"
            raise RuntimeError(msg)
        return original_due_jobs(now=now)

    store.due_jobs = flaky_due_jobs  # type: ignore[method-assign]

    scheduler = PersistentCronScheduler(
        store=store,
        run_job=lambda claimed: _append_and_return(ran, claimed.id),
        deliver_result=_deliver_returned_text,
        tick_seconds=0.01,
        now=lambda: now + timedelta(minutes=1),
    )

    with caplog.at_level(logging.INFO, logger="deepagents_talon.cron.scheduler"):
        await scheduler.start()
        for _ in range(200):
            if ran:
                break
            await asyncio.sleep(0.01)
        await scheduler.stop()

    assert ticks > failures, "the ticker must keep scanning after a failed tick"
    assert ran == [job.id]
    # `logger.exception` also lands in caplog as plain text, not a talon_event.
    events = [_event(message)["event"] for message in caplog.messages if _is_event(message)]
    assert events.count("cron.tick_failure") == failures
    assert "cron.dispatch" in events
    assert "Cron tick failed" in caplog.text


async def test_stop_still_cancels_a_guarded_ticker(tmp_path) -> None:
    store = _store(tmp_path)
    scheduler = PersistentCronScheduler(
        store=store,
        run_job=lambda _: _return("done"),
        deliver_result=_deliver_returned_text,
        tick_seconds=0.01,
    )

    await scheduler.start()
    await asyncio.sleep(0.05)
    await scheduler.stop()

    assert scheduler._task is None


async def _append_and_return(values: list[str], value: str) -> str:
    values.append(value)
    return "[SILENT]"


async def test_a_stalled_job_does_not_silence_the_others(tmp_path) -> None:
    """One job that never finishes must not stop the rest of the fleet firing.

    Due jobs run one at a time, so an unbounded run silences every other schedule --
    and because a job's next run time is claimed before it starts, the fires it
    swallows are deleted rather than merely delayed. `run_scheduled_job` bounds the
    run; this pins that the bound is what lets the tick move on.
    """
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    stalled = store.create_job(
        prompt="stall",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )
    healthy = store.create_job(
        prompt="report",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )
    delivered: list[str] = []

    async def run_job(claimed: CronJob) -> str:
        if claimed.id == stalled.id:
            # Stands in for `run_scheduled_job`'s own bound, which the host applies
            # around the agent turn.
            async with asyncio.timeout(0.05):
                await asyncio.Event().wait()
        return "report filed"

    async def deliver_result(_claimed: CronJob, text: str) -> None:
        delivered.append(text)

    scheduler = PersistentCronScheduler(
        store=store,
        run_job=run_job,
        deliver_result=deliver_result,
        now=lambda: now + timedelta(minutes=1),
    )

    await asyncio.wait_for(scheduler.tick_once(), 2)

    assert delivered == ["report filed"]
    stalled_after = store.get_job(stalled.id)
    healthy_after = store.get_job(healthy.id)
    assert stalled_after is not None
    assert healthy_after is not None
    assert stalled_after.last_status == "error"
    assert healthy_after.last_status == "ok"
