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
