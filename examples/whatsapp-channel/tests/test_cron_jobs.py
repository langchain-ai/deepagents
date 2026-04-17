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
