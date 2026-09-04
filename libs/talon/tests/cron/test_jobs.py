from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deepagents_talon.cron import CronJobError, CronJobStore, CronOrigin, CronSchedule, CronTools


def _store(tmp_path, assistant_id: str = "assistant") -> CronJobStore:
    return CronJobStore(assistant_id=assistant_id, cron_dir=tmp_path / "cron")


def test_store_writes_restrictive_permissions(tmp_path) -> None:
    store = _store(tmp_path)

    store.create_job(
        prompt="check status",
        schedule=CronSchedule.parse("in 30m"),
        origin=CronOrigin(conversation_id="chat"),
    )

    assert store.cron_dir.stat().st_mode & 0o777 == 0o700
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_one_shot_job_advances_to_disabled_before_run(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    job = store.create_job(
        prompt="send reminder",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )

    claimed = store.advance_next_run(job.id, now=now + timedelta(minutes=1))

    assert claimed is not None
    assert claimed.enabled is False
    assert claimed.next_run_at is None
    assert store.due_jobs(now=now + timedelta(minutes=1)) == []


def test_recurring_job_advances_before_run_and_honors_repeat_cap(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    job = store.create_job(
        prompt="heartbeat",
        schedule=CronSchedule.parse("every 15m"),
        origin=CronOrigin(conversation_id="chat"),
        repeat_times=2,
        now=now,
    )

    first = store.advance_next_run(job.id, now=now + timedelta(minutes=15))
    second = store.advance_next_run(job.id, now=now + timedelta(minutes=30))

    assert first is not None
    assert first.next_run_at == now + timedelta(minutes=30)
    assert first.repeat.completed == 1
    assert second is not None
    assert second.enabled is False
    assert second.next_run_at is None
    assert second.repeat.completed == 2


def test_store_prunes_only_expired_completed_jobs(tmp_path) -> None:
    now = datetime(2026, 1, 31, 12, tzinfo=UTC)
    store = _store(tmp_path)
    expired = store.create_job(
        prompt="old",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now - timedelta(days=40),
    )
    fresh = store.create_job(
        prompt="fresh",
        schedule=CronSchedule.parse("in 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now - timedelta(days=1),
    )
    active = store.create_job(
        prompt="active",
        schedule=CronSchedule.parse("every 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now - timedelta(days=40),
    )
    store.advance_next_run(expired.id, now=now - timedelta(days=39))
    store.mark_job_run(expired.id, status="ok", now=now - timedelta(days=39))
    store.advance_next_run(fresh.id, now=now)
    store.mark_job_run(fresh.id, status="ok", now=now)

    removed = store.prune_completed(retain_for=timedelta(days=30), now=now)

    assert [job.id for job in removed] == [expired.id]
    assert {job.id for job in store.list_jobs()} == {fresh.id, active.id}


def test_tools_are_scoped_to_current_conversation(tmp_path) -> None:
    store = _store(tmp_path)
    current = CronOrigin(conversation_id="current", channel="whatsapp")
    other = CronOrigin(conversation_id="other", channel="whatsapp")
    tools = CronTools(store=store, origin=lambda: current)
    other_job = store.create_job(
        prompt="other",
        schedule=CronSchedule.parse("every 5m"),
        origin=other,
    )

    created = tools.create_job(prompt="current", schedule="in 5m", name="mine")

    assert [job["id"] for job in tools.list_jobs()] == [created["id"]]
    with pytest.raises(CronJobError):
        tools.edit_job(other_job.id, enabled=False)
    with pytest.raises(CronJobError):
        tools.remove_job(other_job.id)


NEW_YORK = "America/New_York"


def _schedule_round_trip(text: str) -> CronSchedule:
    schedule = CronSchedule.parse(text)
    return CronSchedule.from_dict(schedule.to_dict())


@pytest.mark.parametrize(
    ("text", "kind", "form"),
    [
        ("in 30m", "one_shot", "interval"),
        ("every 15m", "recurring", "interval"),
        ("in 2h", "one_shot", "interval"),
        (f"at 2026-09-04 13:30 {NEW_YORK}", "one_shot", "at"),
        (f"daily at 08:00 {NEW_YORK}", "recurring", "daily"),
        ("daily at 08:00 UTC", "recurring", "daily"),
    ],
)
def test_parse_accepts_supported_forms_and_round_trips(text: str, kind: str, form: str) -> None:
    schedule = CronSchedule.parse(text)

    assert schedule.kind == kind
    assert schedule.form == form
    assert _schedule_round_trip(text) == schedule


def test_parse_preserves_timezone_case_and_canonicalizes_display() -> None:
    schedule = CronSchedule.parse(f"DAILY At 8:00 {NEW_YORK}")

    assert schedule.timezone == NEW_YORK
    assert schedule.local_time == "08:00"
    assert schedule.display == f"daily at 08:00 {NEW_YORK}"


@pytest.mark.parametrize(
    "text",
    [
        "daily at 08:00",
        "at 2026-09-04 13:30",
        "daily at 08:00 America/NewYork",
        "daily at 08:00 EST5EDT",
        "daily at 08:00 +02:00",
        "daily at 08:00 /etc/localtime",
        "daily at 08:00 ../../etc/passwd",
        "daily at 25:00 UTC",
        "daily at 8am UTC",
        "at 2026-02-30 10:00 UTC",
        "at 2026-9-4 10:00 UTC",
        "weekly at 08:00 UTC",
        "0 9 * * *",
    ],
)
def test_parse_rejects_unsupported_schedules(text: str) -> None:
    with pytest.raises(CronJobError):
        CronSchedule.parse(text)


def test_parse_rejects_oversized_schedule_text() -> None:
    with pytest.raises(CronJobError):
        CronSchedule.parse("daily at 08:00 " + "A" * 300)


def test_daily_schedule_holds_local_time_across_spring_forward() -> None:
    schedule = CronSchedule.parse(f"daily at 08:00 {NEW_YORK}")

    before = schedule.next_after(datetime(2026, 3, 7, 13, 30, tzinfo=UTC))
    after = schedule.next_after(before, previous=before)

    assert before == datetime(2026, 3, 8, 12, tzinfo=UTC)
    assert after == datetime(2026, 3, 9, 12, tzinfo=UTC)
    assert before.astimezone(ZoneInfo(NEW_YORK)).hour == 8
    assert after.astimezone(ZoneInfo(NEW_YORK)).hour == 8


def test_daily_schedule_holds_local_time_across_fall_back() -> None:
    schedule = CronSchedule.parse(f"daily at 08:00 {NEW_YORK}")

    instant = schedule.next_after(datetime(2026, 10, 31, 12, 30, tzinfo=UTC))

    assert instant == datetime(2026, 11, 1, 13, tzinfo=UTC)
    assert instant.astimezone(ZoneInfo(NEW_YORK)).hour == 8


def test_daily_schedule_snaps_nonexistent_local_time_forward() -> None:
    schedule = CronSchedule.parse(f"daily at 02:30 {NEW_YORK}")

    instant = schedule.next_after(datetime(2026, 3, 8, 5, tzinfo=UTC))

    assert instant == datetime(2026, 3, 8, 7, tzinfo=UTC)
    assert instant.astimezone(ZoneInfo(NEW_YORK)).strftime("%H:%M") == "03:00"


def test_daily_schedule_fires_ambiguous_local_time_once() -> None:
    schedule = CronSchedule.parse(f"daily at 01:30 {NEW_YORK}")

    first = schedule.next_after(datetime(2026, 11, 1, 4, tzinfo=UTC))
    second = schedule.next_after(first, previous=first)

    assert first == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    assert second == datetime(2026, 11, 2, 6, 30, tzinfo=UTC)


def test_daily_schedule_snaps_sub_hour_gap() -> None:
    schedule = CronSchedule.parse("daily at 02:15 Australia/Lord_Howe")

    instant = schedule.next_after(datetime(2026, 10, 3, 14, tzinfo=UTC))

    assert instant.astimezone(ZoneInfo("Australia/Lord_Howe")).strftime("%H:%M") == "02:30"


def test_daily_schedule_advance_never_returns_current_instant(tmp_path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 9, 3, 18, tzinfo=UTC)
    job = store.create_job(
        prompt="briefing",
        schedule=CronSchedule.parse(f"daily at 08:00 {NEW_YORK}"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )
    due = datetime(2026, 9, 4, 12, tzinfo=UTC)

    claimed = store.advance_next_run(job.id, now=due)

    assert job.next_run_at == due
    assert claimed is not None
    assert claimed.next_run_at == datetime(2026, 9, 5, 12, tzinfo=UTC)
    assert store.due_jobs(now=due) == []


def test_one_shot_wall_clock_job_runs_once(tmp_path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 9, 3, 18, tzinfo=UTC)
    job = store.create_job(
        prompt="call the dentist",
        schedule=CronSchedule.parse(f"at 2026-09-04 13:00 {NEW_YORK}"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )

    claimed = store.advance_next_run(job.id, now=datetime(2026, 9, 4, 17, tzinfo=UTC))

    assert job.next_run_at == datetime(2026, 9, 4, 17, tzinfo=UTC)
    assert claimed is not None
    assert claimed.enabled is False
    assert claimed.next_run_at is None


def test_one_shot_wall_clock_job_in_the_past_is_rejected(tmp_path) -> None:
    store = _store(tmp_path)

    with pytest.raises(CronJobError, match="in the past"):
        store.create_job(
            prompt="too late",
            schedule=CronSchedule.parse(f"at 2026-09-02 13:00 {NEW_YORK}"),
            origin=CronOrigin(conversation_id="chat"),
            now=datetime(2026, 9, 3, 18, tzinfo=UTC),
        )


def test_interval_job_keeps_phase_when_a_tick_runs_late(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    job = store.create_job(
        prompt="heartbeat",
        schedule=CronSchedule.parse("every 15m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )

    claimed = store.advance_next_run(job.id, now=now + timedelta(minutes=15, seconds=40))

    assert claimed is not None
    assert claimed.next_run_at == now + timedelta(minutes=30)


def test_interval_job_catches_up_after_long_downtime(tmp_path) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path)
    job = store.create_job(
        prompt="heartbeat",
        schedule=CronSchedule.parse("every 1m"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )
    back_online = now + timedelta(days=30)

    claimed = store.advance_next_run(job.id, now=back_online)

    assert claimed is not None
    assert claimed.next_run_at == back_online + timedelta(minutes=1)
