from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deepagents_talon.cron import (
    CronJob,
    CronJobError,
    CronJobStore,
    CronOrigin,
    CronSchedule,
    CronTools,
)
from deepagents_talon.cron.jobs import parse_until
from deepagents_talon.cron.scheduler import PersistentCronScheduler

NOON = datetime(2026, 1, 1, 12, tzinfo=UTC)
ORIGIN = CronOrigin(conversation_id="chat")


def _store(tmp_path) -> CronJobStore:
    return CronJobStore(assistant_id="assistant", cron_dir=tmp_path / "cron")


def _hourly_until(store: CronJobStore, until: datetime) -> CronJob:
    return store.create_job(
        prompt="heartbeat",
        schedule=CronSchedule.parse("every 60m"),
        origin=ORIGIN,
        until=until,
        now=NOON,
    )


class _Runner:
    """Scheduler callbacks that record what ran and what was delivered."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.ran: list[str] = []

    async def run(self, job: CronJob) -> str:
        self.ran.append(job.id)
        if self.fail:
            msg = "boom"
            raise RuntimeError(msg)
        return "done"

    async def deliver(self, _job: CronJob, _text: str) -> None:
        return None


async def _tick(store: CronJobStore, runner: _Runner, at: datetime) -> None:
    scheduler = PersistentCronScheduler(
        store=store, run_job=runner.run, deliver_result=runner.deliver, now=lambda: at
    )
    await scheduler.tick_once()


def test_parse_until_reads_local_wall_clock() -> None:
    # 23:59 EST is 04:59 UTC the next day.
    assert parse_until("2026-12-31 23:59 America/New_York") == datetime(
        2027, 1, 1, 4, 59, tzinfo=UTC
    )


@pytest.mark.parametrize("text", ["2026-12-31", "2026-12-31 23:59", "tomorrow 09:00 UTC"])
def test_parse_until_rejects_incomplete_text(text: str) -> None:
    with pytest.raises(CronJobError):
        parse_until(text)


def test_parse_until_rejects_dates_past_the_supported_range() -> None:
    # 23:59 EST on the last representable day is already year 10000 in UTC.
    with pytest.raises(CronJobError, match="supported date range"):
        parse_until("9999-12-31 23:59 America/New_York")


def test_until_is_rejected_on_one_shot_schedules(tmp_path) -> None:
    with pytest.raises(CronJobError, match="only valid for recurring"):
        _store(tmp_path).create_job(
            prompt="once",
            schedule=CronSchedule.parse("in 30m"),
            origin=ORIGIN,
            until=NOON + timedelta(days=1),
            now=NOON,
        )


def test_until_before_the_first_run_is_rejected(tmp_path) -> None:
    with pytest.raises(CronJobError, match="would never run"):
        _hourly_until(_store(tmp_path), NOON + timedelta(minutes=30))


def test_run_scheduled_exactly_at_until_fires_then_job_finishes(tmp_path) -> None:
    store = _store(tmp_path)
    job = _hourly_until(store, NOON + timedelta(hours=2))

    first = store.advance_next_run(job.id, now=NOON + timedelta(hours=1))
    last = store.advance_next_run(job.id, now=NOON + timedelta(hours=2))

    assert first is not None
    assert first.next_run_at == NOON + timedelta(hours=2)
    assert last is not None
    assert last.enabled is False
    assert last.next_run_at is None


def test_run_due_at_until_survives_tick_latency(tmp_path) -> None:
    store = _store(tmp_path)
    job = _hourly_until(store, NOON + timedelta(hours=1))

    claimed = store.advance_next_run(job.id, now=NOON + timedelta(hours=1, seconds=40))

    assert claimed is not None


def test_run_missed_across_until_is_dropped_not_delivered_late(tmp_path) -> None:
    store = _store(tmp_path)
    job = _hourly_until(store, NOON + timedelta(hours=1))

    # The host was down from before the 13:00 run until 16:00.
    claimed = store.advance_next_run(job.id, now=NOON + timedelta(hours=4))

    assert claimed is None
    stored = store.get_job(job.id)
    assert stored is not None
    assert stored.enabled is False
    assert stored.next_run_at is None


def test_upcoming_stops_at_until(tmp_path) -> None:
    job = _hourly_until(_store(tmp_path), NOON + timedelta(hours=2, minutes=30))

    assert job.upcoming(5) == [NOON + timedelta(hours=1), NOON + timedelta(hours=2)]


def test_until_round_trips_and_older_records_load_without_it(tmp_path) -> None:
    store = _store(tmp_path)
    job = _hourly_until(store, NOON + timedelta(days=1))
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    del payload["jobs"][0]["until"]
    legacy = _store(tmp_path / "legacy")
    legacy.cron_dir.mkdir(mode=0o700, parents=True)
    legacy.path.write_text(json.dumps(payload), encoding="utf-8")

    assert _store(tmp_path).get_job(job.id) == job
    reloaded = legacy.get_job(job.id)
    assert reloaded is not None
    assert reloaded.until is None


def test_edit_sets_clears_and_validates_until(tmp_path) -> None:
    store = _store(tmp_path)
    job = _hourly_until(store, NOON + timedelta(days=1))

    cleared = store.edit_job(job.id, origin=ORIGIN, clear_until=True, now=NOON)
    assert cleared.until is None
    with pytest.raises(CronJobError, match="would never run"):
        store.edit_job(job.id, origin=ORIGIN, until=NOON + timedelta(minutes=5), now=NOON)
    with pytest.raises(CronJobError, match="either until or clear_until"):
        store.edit_job(job.id, origin=ORIGIN, until=NOON, clear_until=True, now=NOON)


def test_tools_accept_until_text_and_report_upcoming(tmp_path) -> None:
    tools = CronTools(store=_store(tmp_path), origin=lambda: ORIGIN)

    created = tools.create_job(
        prompt="lunch",
        schedule="cron 0 12 * * sat,sun America/New_York",
        until="2099-01-01 00:00 America/New_York",
    )
    edited = tools.edit_job(created["id"], until="")

    assert created["until"] is not None
    assert len(created["upcoming"]) == 3
    for run in created["upcoming"]:
        local = datetime.fromisoformat(run).astimezone(ZoneInfo("America/New_York"))
        assert (local.strftime("%a"), local.hour) in {("Sat", 12), ("Sun", 12)}
    assert edited["until"] is None


async def test_job_runs_inside_window_then_is_removed(tmp_path, caplog) -> None:
    store = _store(tmp_path)
    runner = _Runner()
    job = _hourly_until(store, NOON + timedelta(hours=2))

    for hours in (1, 2):
        await _tick(store, runner, NOON + timedelta(hours=hours))
    assert store.get_job(job.id) is not None
    with caplog.at_level(logging.INFO, logger="deepagents_talon.cron.scheduler"):
        await _tick(store, runner, NOON + timedelta(hours=2, minutes=1))

    assert runner.ran == [job.id, job.id]
    assert store.get_job(job.id) is None
    assert any('"cron.job_removed"' in message for message in caplog.messages)


async def test_paused_job_is_removed_once_until_passes_without_running(tmp_path) -> None:
    store = _store(tmp_path)
    runner = _Runner()
    job = _hourly_until(store, NOON + timedelta(hours=1))
    store.edit_job(job.id, origin=ORIGIN, enabled=False, now=NOON)

    await _tick(store, runner, NOON + timedelta(hours=1))
    assert store.get_job(job.id) is not None
    await _tick(store, runner, NOON + timedelta(hours=1, minutes=10))

    assert runner.ran == []
    assert store.get_job(job.id) is None


async def test_downtime_across_until_never_runs_and_is_removed(tmp_path) -> None:
    store = _store(tmp_path)
    runner = _Runner()
    job = _hourly_until(store, NOON + timedelta(hours=1))

    await _tick(store, runner, NOON + timedelta(hours=4))
    await _tick(store, runner, NOON + timedelta(hours=4, minutes=1))

    assert runner.ran == []
    assert store.get_job(job.id) is None


async def test_successful_one_shot_is_removed_on_the_next_tick(tmp_path) -> None:
    store = _store(tmp_path)
    runner = _Runner()
    job = store.create_job(
        prompt="ping", schedule=CronSchedule.parse("in 1m"), origin=ORIGIN, now=NOON
    )

    await _tick(store, runner, NOON + timedelta(minutes=1))
    assert store.get_job(job.id) is not None
    await _tick(store, runner, NOON + timedelta(minutes=2))

    assert store.get_job(job.id) is None


async def test_failed_final_run_is_kept_with_its_error(tmp_path) -> None:
    store = _store(tmp_path)
    runner = _Runner(fail=True)
    job = store.create_job(
        prompt="ping", schedule=CronSchedule.parse("in 1m"), origin=ORIGIN, now=NOON
    )

    await _tick(store, runner, NOON + timedelta(minutes=1))
    await _tick(store, runner, NOON + timedelta(minutes=2))

    kept = store.get_job(job.id)
    assert kept is not None
    assert kept.last_error == "boom"


async def test_finished_job_given_a_new_schedule_is_not_removed(tmp_path) -> None:
    store = _store(tmp_path)
    runner = _Runner()
    job = store.create_job(
        prompt="ping", schedule=CronSchedule.parse("in 1m"), origin=ORIGIN, now=NOON
    )
    await _tick(store, runner, NOON + timedelta(minutes=1))

    store.edit_job(
        job.id,
        origin=ORIGIN,
        schedule=CronSchedule.parse("in 30m"),
        enabled=True,
        now=NOON + timedelta(minutes=1),
    )
    await _tick(store, runner, NOON + timedelta(minutes=2))

    assert store.get_job(job.id) is not None


async def test_until_at_the_largest_date_does_not_stall_the_scheduler(tmp_path) -> None:
    store = _store(tmp_path)
    runner = _Runner()
    far = _hourly_until(store, parse_until("9999-12-31 23:59 UTC"))
    plain = store.create_job(
        prompt="ping", schedule=CronSchedule.parse("in 1h"), origin=ORIGIN, now=NOON
    )

    await _tick(store, runner, NOON + timedelta(hours=1))

    assert sorted(runner.ran) == sorted([far.id, plain.id])


async def test_run_interrupted_after_its_claim_is_kept_until_retention(tmp_path) -> None:
    store = _store(tmp_path)
    job = store.create_job(
        prompt="ping", schedule=CronSchedule.parse("in 1m"), origin=ORIGIN, now=NOON
    )
    # The claim is written, then the process stops before recording an outcome.
    store.advance_next_run(job.id, now=NOON + timedelta(minutes=1))

    await _tick(store, _Runner(), NOON + timedelta(minutes=2))
    kept = store.get_job(job.id)
    assert kept is not None
    assert kept.claimed_at == NOON + timedelta(minutes=1)

    store.prune_completed(retain_for=timedelta(days=1), now=NOON + timedelta(days=2))
    assert store.get_job(job.id) is None


async def test_expired_paused_job_with_an_error_is_pruned_after_retention(tmp_path) -> None:
    store = _store(tmp_path)
    job = _hourly_until(store, NOON + timedelta(hours=2))
    await _tick(store, _Runner(fail=True), NOON + timedelta(hours=1))
    store.edit_job(job.id, origin=ORIGIN, enabled=False, now=NOON + timedelta(hours=1))

    await _tick(store, _Runner(), NOON + timedelta(hours=3))
    kept = store.get_job(job.id)
    assert kept is not None
    assert kept.last_error == "boom"

    store.prune_completed(retain_for=timedelta(days=1), now=NOON + timedelta(days=2))
    assert store.get_job(job.id) is None
