"""Tests for cron.jobs."""

from __future__ import annotations

import pytest

from cron.jobs import parse_duration


class TestParseDuration:
    def test_minutes_short(self) -> None:
        assert parse_duration("30m") == 30

    def test_minutes_long(self) -> None:
        assert parse_duration("45 minutes") == 45

    def test_hours(self) -> None:
        assert parse_duration("2h") == 120

    def test_days(self) -> None:
        assert parse_duration("1d") == 1440

    def test_case_insensitive(self) -> None:
        assert parse_duration("2H") == 120

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("0m")

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("foo")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("")

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("-5m")


from datetime import datetime, timedelta

from cron.jobs import parse_schedule


class TestParseSchedule:
    def test_one_shot_minutes(self) -> None:
        result = parse_schedule("30m")
        assert result["kind"] == "once"
        assert "run_at" in result
        # run_at should be ~30 minutes from now, tz-aware
        run_at = datetime.fromisoformat(result["run_at"])
        assert run_at.tzinfo is not None
        delta = run_at - datetime.now().astimezone()
        assert timedelta(minutes=29) <= delta <= timedelta(minutes=31)
        assert result["display"] == "once in 30m"

    def test_interval(self) -> None:
        result = parse_schedule("every 2h")
        assert result == {"kind": "interval", "minutes": 120, "display": "every 120m"}

    def test_interval_case_insensitive(self) -> None:
        result = parse_schedule("EVERY 15m")
        assert result["kind"] == "interval"
        assert result["minutes"] == 15

    def test_rejects_cron_expression(self) -> None:
        with pytest.raises(ValueError):
            parse_schedule("0 9 * * *")

    def test_rejects_timestamp(self) -> None:
        with pytest.raises(ValueError):
            parse_schedule("2026-04-20T14:00")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            parse_schedule("")

    def test_rejects_every_without_duration(self) -> None:
        with pytest.raises(ValueError):
            parse_schedule("every")


from cron.jobs import compute_next_run


class TestComputeNextRun:
    def test_one_shot_first_run(self) -> None:
        schedule = {"kind": "once", "run_at": "2026-04-20T09:00:00+00:00"}
        assert compute_next_run(schedule) == "2026-04-20T09:00:00+00:00"

    def test_one_shot_already_ran(self) -> None:
        schedule = {"kind": "once", "run_at": "2026-04-20T09:00:00+00:00"}
        assert compute_next_run(schedule, last_run_at="2026-04-20T09:00:01+00:00") is None

    def test_interval_first_run(self) -> None:
        # First run = now + interval
        schedule = {"kind": "interval", "minutes": 60}
        result = compute_next_run(schedule)
        assert result is not None
        next_run = datetime.fromisoformat(result)
        delta = next_run - datetime.now().astimezone()
        assert timedelta(minutes=59) <= delta <= timedelta(minutes=61)

    def test_interval_subsequent_run(self) -> None:
        schedule = {"kind": "interval", "minutes": 30}
        last = "2026-04-20T10:00:00+00:00"
        result = compute_next_run(schedule, last_run_at=last)
        assert result == "2026-04-20T10:30:00+00:00"


from cron.jobs import load_jobs, save_jobs


class TestStorage:
    def test_load_missing_file_returns_empty(self, jobs_path) -> None:
        assert load_jobs(jobs_path) == []

    def test_save_then_load_round_trip(self, jobs_path) -> None:
        job = {"id": "abc", "name": "hello", "prompt": "hi"}
        save_jobs(jobs_path, [job])
        assert load_jobs(jobs_path) == [job]

    def test_save_creates_parent_dir(self, jobs_path) -> None:
        assert not jobs_path.parent.exists()
        save_jobs(jobs_path, [])
        assert jobs_path.parent.exists()

    def test_save_is_atomic_no_tmp_left(self, jobs_path) -> None:
        save_jobs(jobs_path, [{"id": "abc"}])
        leftover = list(jobs_path.parent.glob(".jobs_*.tmp"))
        assert leftover == []

    def test_save_writes_top_level_shape(self, jobs_path) -> None:
        import json

        save_jobs(jobs_path, [{"id": "abc"}])
        data = json.loads(jobs_path.read_text())
        assert set(data.keys()) == {"jobs", "updated_at"}
        assert data["jobs"] == [{"id": "abc"}]

    def test_load_corrupt_file_raises(self, jobs_path) -> None:
        jobs_path.parent.mkdir(parents=True)
        jobs_path.write_text("{not valid json at all")
        with pytest.raises(RuntimeError):
            load_jobs(jobs_path)


from cron.jobs import create_job


class TestCreateJob:
    def test_creates_one_shot_with_repeat_1(self, jobs_path) -> None:
        job = create_job(
            jobs_path,
            prompt="summarize news",
            schedule="30m",
            origin={"chat_id": "123@c.us", "message_id": "msg1"},
        )
        assert job["schedule"]["kind"] == "once"
        assert job["repeat"] == {"times": 1, "completed": 0}
        assert job["enabled"] is True
        assert job["origin"] == {"chat_id": "123@c.us", "message_id": "msg1"}
        assert len(job["id"]) == 12
        assert job["name"] == "summarize news"
        assert job["next_run_at"] == job["schedule"]["run_at"]
        assert job["last_run_at"] is None
        assert job["last_status"] is None

    def test_creates_interval_with_repeat_forever(self, jobs_path) -> None:
        job = create_job(
            jobs_path,
            prompt="daily news",
            schedule="every 2h",
            origin={"chat_id": "123@c.us", "message_id": None},
        )
        assert job["schedule"]["kind"] == "interval"
        assert job["repeat"] == {"times": None, "completed": 0}
        assert job["next_run_at"] is not None

    def test_explicit_name(self, jobs_path) -> None:
        job = create_job(
            jobs_path,
            prompt="long prompt that exceeds the default-label length so we "
                   "must truncate it otherwise names get ugly",
            schedule="30m",
            origin={"chat_id": "123", "message_id": None},
            name="tidy",
        )
        assert job["name"] == "tidy"

    def test_default_name_truncates(self, jobs_path) -> None:
        job = create_job(
            jobs_path,
            prompt="x" * 100,
            schedule="30m",
            origin={"chat_id": "123", "message_id": None},
        )
        assert len(job["name"]) == 50

    def test_explicit_repeat_3(self, jobs_path) -> None:
        job = create_job(
            jobs_path,
            prompt="p",
            schedule="every 10m",
            origin={"chat_id": "123", "message_id": None},
            repeat=3,
        )
        assert job["repeat"] == {"times": 3, "completed": 0}

    def test_persists_to_storage(self, jobs_path) -> None:
        job = create_job(
            jobs_path,
            prompt="p",
            schedule="30m",
            origin={"chat_id": "123", "message_id": None},
        )
        assert load_jobs(jobs_path) == [job]

    def test_rejects_bad_schedule(self, jobs_path) -> None:
        with pytest.raises(ValueError):
            create_job(
                jobs_path,
                prompt="p",
                schedule="0 9 * * *",
                origin={"chat_id": "123", "message_id": None},
            )


from cron.jobs import get_job, list_jobs_for_chat, remove_job, update_job


class TestListJobsForChat:
    def test_empty(self, jobs_path) -> None:
        assert list_jobs_for_chat(jobs_path, "123") == []

    def test_filters_by_chat(self, jobs_path) -> None:
        create_job(jobs_path, prompt="a", schedule="30m",
                   origin={"chat_id": "111", "message_id": None})
        create_job(jobs_path, prompt="b", schedule="30m",
                   origin={"chat_id": "222", "message_id": None})
        rows_111 = list_jobs_for_chat(jobs_path, "111")
        assert len(rows_111) == 1
        assert rows_111[0]["prompt"] == "a"


class TestGetJob:
    def test_hit(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="30m",
                       origin={"chat_id": "123", "message_id": None})
        assert get_job(jobs_path, j["id"])["id"] == j["id"]

    def test_miss(self, jobs_path) -> None:
        assert get_job(jobs_path, "does-not-exist") is None


class TestRemoveJob:
    def test_removes_when_chat_matches(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="30m",
                       origin={"chat_id": "111", "message_id": None})
        assert remove_job(jobs_path, j["id"], chat_id="111") is True
        assert load_jobs(jobs_path) == []

    def test_refuses_when_chat_mismatches(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="30m",
                       origin={"chat_id": "111", "message_id": None})
        assert remove_job(jobs_path, j["id"], chat_id="222") is False
        # Job still present
        assert len(load_jobs(jobs_path)) == 1

    def test_missing_id_returns_false(self, jobs_path) -> None:
        assert remove_job(jobs_path, "nope", chat_id="111") is False


class TestUpdateJob:
    def test_partial_update(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="every 10m",
                       origin={"chat_id": "111", "message_id": None})
        updated = update_job(
            jobs_path, j["id"], chat_id="111",
            name="new name", enabled=False,
        )
        assert updated is not None
        assert updated["name"] == "new name"
        assert updated["enabled"] is False
        assert updated["prompt"] == "p"  # untouched

    def test_reschedule_resets_next_run(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="every 30m",
                       origin={"chat_id": "111", "message_id": None})
        updated = update_job(
            jobs_path, j["id"], chat_id="111", schedule="every 5m",
        )
        assert updated["schedule"]["minutes"] == 5
        assert updated["next_run_at"] != j["next_run_at"]

    def test_repeat_zero_clears_cap(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="every 30m", repeat=3,
                       origin={"chat_id": "111", "message_id": None})
        updated = update_job(jobs_path, j["id"], chat_id="111", repeat=0)
        assert updated["repeat"]["times"] is None

    def test_empty_name_raises(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="30m",
                       origin={"chat_id": "111", "message_id": None})
        with pytest.raises(ValueError):
            update_job(jobs_path, j["id"], chat_id="111", name="   ")

    def test_invalid_schedule_raises(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="30m",
                       origin={"chat_id": "111", "message_id": None})
        with pytest.raises(ValueError):
            update_job(jobs_path, j["id"], chat_id="111", schedule="garbage")

    def test_cross_chat_returns_none(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="30m",
                       origin={"chat_id": "111", "message_id": None})
        assert update_job(
            jobs_path, j["id"], chat_id="222", enabled=False,
        ) is None
        # Unchanged on disk
        assert load_jobs(jobs_path)[0]["enabled"] is True

    def test_missing_id_returns_none(self, jobs_path) -> None:
        assert update_job(
            jobs_path, "nope", chat_id="111", enabled=False,
        ) is None


from cron.jobs import advance_next_run, get_due_jobs, mark_job_run


class TestMarkJobRun:
    def test_one_shot_disables_after_success(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="30m",
                       origin={"chat_id": "1", "message_id": None})
        mark_job_run(jobs_path, j["id"], success=True)
        updated = get_job(jobs_path, j["id"])
        assert updated["last_status"] == "ok"
        assert updated["last_error"] is None
        assert updated["enabled"] is False
        assert updated["repeat"]["completed"] == 1
        assert updated["next_run_at"] is None

    def test_error_recorded(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="30m",
                       origin={"chat_id": "1", "message_id": None})
        mark_job_run(jobs_path, j["id"], success=False, error="boom")
        updated = get_job(jobs_path, j["id"])
        assert updated["last_status"] == "error"
        assert updated["last_error"] == "boom"

    def test_interval_removes_after_repeat_limit(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="every 10m",
                       origin={"chat_id": "1", "message_id": None}, repeat=2)
        mark_job_run(jobs_path, j["id"], success=True)
        assert get_job(jobs_path, j["id"])["repeat"]["completed"] == 1
        mark_job_run(jobs_path, j["id"], success=True)
        assert get_job(jobs_path, j["id"]) is None  # removed

    def test_interval_forever_keeps_going(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="every 10m",
                       origin={"chat_id": "1", "message_id": None})  # repeat=None
        mark_job_run(jobs_path, j["id"], success=True)
        updated = get_job(jobs_path, j["id"])
        assert updated is not None
        assert updated["enabled"] is True
        assert updated["next_run_at"] is not None

    def test_missing_id_is_a_noop(self, jobs_path) -> None:
        # Should not raise
        mark_job_run(jobs_path, "nope", success=True)


class TestAdvanceNextRun:
    def test_interval_advances(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="every 10m",
                       origin={"chat_id": "1", "message_id": None})
        first_next = j["next_run_at"]
        changed = advance_next_run(jobs_path, j["id"])
        assert changed is True
        second_next = get_job(jobs_path, j["id"])["next_run_at"]
        assert second_next != first_next

    def test_one_shot_is_left_alone(self, jobs_path) -> None:
        j = create_job(jobs_path, prompt="p", schedule="30m",
                       origin={"chat_id": "1", "message_id": None})
        first_next = j["next_run_at"]
        changed = advance_next_run(jobs_path, j["id"])
        assert changed is False
        assert get_job(jobs_path, j["id"])["next_run_at"] == first_next


class TestGetDueJobs:
    def test_returns_only_enabled_and_past_due(self, jobs_path) -> None:
        from datetime import datetime, timedelta
        past = (_now_helper() - timedelta(minutes=5)).isoformat()
        future = (_now_helper() + timedelta(minutes=5)).isoformat()

        jobs = [
            {"id": "a", "enabled": True, "next_run_at": past,
             "origin": {"chat_id": "1"}, "schedule": {"kind": "interval", "minutes": 10},
             "repeat": {"times": None, "completed": 0}},
            {"id": "b", "enabled": True, "next_run_at": future,
             "origin": {"chat_id": "1"}, "schedule": {"kind": "interval", "minutes": 10},
             "repeat": {"times": None, "completed": 0}},
            {"id": "c", "enabled": False, "next_run_at": past,
             "origin": {"chat_id": "1"}, "schedule": {"kind": "interval", "minutes": 10},
             "repeat": {"times": None, "completed": 0}},
            {"id": "d", "enabled": True, "next_run_at": None,
             "origin": {"chat_id": "1"}, "schedule": {"kind": "once", "run_at": past},
             "repeat": {"times": 1, "completed": 0}},
        ]
        save_jobs(jobs_path, jobs)
        due = get_due_jobs(jobs_path)
        ids = [j["id"] for j in due]
        assert ids == ["a"]  # b future, c disabled, d no next_run_at

    def test_sorted_by_next_run_at(self, jobs_path) -> None:
        from datetime import timedelta
        t1 = (_now_helper() - timedelta(minutes=5)).isoformat()
        t2 = (_now_helper() - timedelta(minutes=1)).isoformat()
        jobs = [
            {"id": "later", "enabled": True, "next_run_at": t2,
             "origin": {"chat_id": "1"}, "schedule": {"kind": "interval", "minutes": 10},
             "repeat": {"times": None, "completed": 0}},
            {"id": "earlier", "enabled": True, "next_run_at": t1,
             "origin": {"chat_id": "1"}, "schedule": {"kind": "interval", "minutes": 10},
             "repeat": {"times": None, "completed": 0}},
        ]
        save_jobs(jobs_path, jobs)
        assert [j["id"] for j in get_due_jobs(jobs_path)] == ["earlier", "later"]


def _now_helper():
    """Local helper mirroring cron.jobs._now_aware for test timestamps."""
    from datetime import datetime
    return datetime.now().astimezone()
