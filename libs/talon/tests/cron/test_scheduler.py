from __future__ import annotations

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
