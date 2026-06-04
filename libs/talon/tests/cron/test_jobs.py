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
