from __future__ import annotations

from datetime import UTC, datetime

import pytest

from deepagents_talon.cron.time import (
    ActiveWindow,
    CronTimeError,
    LocalTimeOfDay,
    active_window_contains,
    next_wall_clock_run,
    parse_local_time,
)


def test_parse_local_time_supported_grammar() -> None:
    assert parse_local_time("20:00") == LocalTimeOfDay(hour=20, minute=0)
    assert parse_local_time("8:00pm") == LocalTimeOfDay(hour=20, minute=0)
    assert parse_local_time("8pm") == LocalTimeOfDay(hour=20, minute=0)
    assert parse_local_time("08:30") == LocalTimeOfDay(hour=8, minute=30)


def test_next_wall_clock_run_uses_next_local_occurrence() -> None:
    now = datetime(2026, 7, 2, 1, tzinfo=UTC)

    result = next_wall_clock_run(
        now=now,
        timezone="America/Los_Angeles",
        time=LocalTimeOfDay(hour=20, minute=0),
    )

    assert result == datetime(2026, 7, 2, 3, tzinfo=UTC)


def test_next_wall_clock_run_uses_first_ambiguous_dst_occurrence() -> None:
    now = datetime(2026, 11, 1, 0, tzinfo=UTC)

    result = next_wall_clock_run(
        now=now,
        timezone="America/Los_Angeles",
        time=LocalTimeOfDay(hour=1, minute=30),
        date=datetime(2026, 11, 1, tzinfo=UTC).date(),
    )

    assert result == datetime(2026, 11, 1, 8, 30, tzinfo=UTC)


def test_next_wall_clock_run_rejects_nonexistent_dst_time() -> None:
    now = datetime(2026, 3, 8, 0, tzinfo=UTC)

    with pytest.raises(CronTimeError, match="does not exist"):
        next_wall_clock_run(
            now=now,
            timezone="America/Los_Angeles",
            time=LocalTimeOfDay(hour=2, minute=30),
            date=datetime(2026, 3, 8, tzinfo=UTC).date(),
        )


def test_active_window_supports_cross_midnight() -> None:
    window = ActiveWindow(
        timezone="America/Los_Angeles",
        start=LocalTimeOfDay(hour=22, minute=0),
        end=LocalTimeOfDay(hour=6, minute=0),
    )

    assert active_window_contains(datetime(2026, 7, 2, 6, tzinfo=UTC), window)
    assert active_window_contains(datetime(2026, 7, 2, 12, 30, tzinfo=UTC), window)
    assert not active_window_contains(datetime(2026, 7, 2, 19, tzinfo=UTC), window)
