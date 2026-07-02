from __future__ import annotations

from datetime import UTC, datetime, timedelta

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


def test_wall_clock_job_persists_timezone_and_local_output(tmp_path) -> None:
    now = datetime(2026, 7, 2, 1, tzinfo=UTC)
    store = _store(tmp_path)

    job = store.create_job(
        prompt="send reminder",
        schedule=CronSchedule.parse("at 8:00pm", timezone="America/Los_Angeles"),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )

    assert job.next_run_at == datetime(2026, 7, 2, 3, tzinfo=UTC)
    assert job.schedule.to_dict()["timezone"] == "America/Los_Angeles"
    assert job.schedule.to_dict()["wall_time"] == "20:00"
    assert job.scheduler_host_clock is not None
    assert CronTools(store=store, origin=lambda: CronOrigin(conversation_id="chat")).list_jobs()[0][
        "next_run_local"
    ] == "2026-07-01 20:00 America/Los_Angeles"


def test_active_window_moves_initial_run_to_next_allowed_time(tmp_path) -> None:
    now = datetime(2026, 7, 2, 0, 20, tzinfo=UTC)
    store = _store(tmp_path)

    job = store.create_job(
        prompt="heartbeat",
        schedule=CronSchedule.parse(
            "every 15m",
            timezone="America/Los_Angeles",
            active_start="08:30",
            active_end="17:27",
        ),
        origin=CronOrigin(conversation_id="chat"),
        now=now,
    )

    assert job.next_run_at == datetime(2026, 7, 2, 15, 30, tzinfo=UTC)


def test_active_window_skip_does_not_consume_repeat_cap(tmp_path) -> None:
    now = datetime(2026, 7, 2, 23, 44, tzinfo=UTC)
    store = _store(tmp_path)
    job = store.create_job(
        prompt="heartbeat",
        schedule=CronSchedule.parse(
            "every 15m",
            timezone="America/Los_Angeles",
            active_start="08:00",
            active_end="17:00",
        ),
        origin=CronOrigin(conversation_id="chat"),
        repeat_times=2,
        now=now,
    )

    claimed = store.advance_next_run(job.id, now=datetime(2026, 7, 3, 0, 1, tzinfo=UTC))

    updated = store.get_job(job.id)
    assert claimed is None
    assert updated is not None
    assert updated.repeat.completed == 0
    assert updated.next_run_at == datetime(2026, 7, 3, 15, tzinfo=UTC)


def test_tools_edit_active_hours_without_replacing_schedule(tmp_path) -> None:
    store = _store(tmp_path)
    current = CronOrigin(conversation_id="chat")
    tools = CronTools(store=store, origin=lambda: current)
    job = store.create_job(
        prompt="heartbeat",
        schedule=CronSchedule.parse("every 15m"),
        origin=current,
        now=datetime(2026, 7, 2, 15, tzinfo=UTC),
    )

    updated = tools.edit_job(
        job.id,
        timezone="America/Los_Angeles",
        active_start="08:00",
        active_end="17:00",
    )
    cleared = tools.edit_job(job.id, active_start="", active_end="")

    assert updated["schedule"]["active_window"] == {
        "timezone": "America/Los_Angeles",
        "start": "08:00",
        "end": "17:00",
    }
    assert updated["schedule"]["display"] == "every 15m"
    assert cleared["schedule"]["active_window"] is None


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
