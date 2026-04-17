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
