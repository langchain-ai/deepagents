from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deepagents_talon.cron.time import (
    ActiveWindow,
    CronTimeError,
    DaysOfWeek,
    LocalTimeOfDay,
    Weekday,
    next_interval_run,
    next_wall_clock_run,
    parse_days_of_week,
    parse_local_time,
    parse_weekday,
)

LA = "America/Los_Angeles"
LA_ZONE = ZoneInfo(LA)


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# --- days of week ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("m", Weekday.MONDAY),
        ("Mon", Weekday.MONDAY),
        ("mondays", Weekday.MONDAY),
        ("tues", Weekday.TUESDAY),
        ("weds", Weekday.WEDNESDAY),
        ("thurs", Weekday.THURSDAY),
        ("fri", Weekday.FRIDAY),
        ("sa", Weekday.SATURDAY),
        ("sun.", Weekday.SUNDAY),
    ],
)
def test_parse_weekday_accepts_common_names(text: str, expected: Weekday) -> None:
    assert parse_weekday(text) is expected


@pytest.mark.parametrize("text", ["", "  ", "s", "tueth", "funday"])
def test_parse_weekday_rejects_unknown_or_ambiguous_names(text: str) -> None:
    with pytest.raises(CronTimeError):
        parse_weekday(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("daily", tuple(Weekday)),
        ("every day", tuple(Weekday)),
        (
            "weekdays",
            (
                Weekday.MONDAY,
                Weekday.TUESDAY,
                Weekday.WEDNESDAY,
                Weekday.THURSDAY,
                Weekday.FRIDAY,
            ),
        ),
        ("weekends", (Weekday.SATURDAY, Weekday.SUNDAY)),
        ("mon wed fri", (Weekday.MONDAY, Weekday.WEDNESDAY, Weekday.FRIDAY)),
        ("monday, wednesday, and friday", (Weekday.MONDAY, Weekday.WEDNESDAY, Weekday.FRIDAY)),
        ("sat/sun", (Weekday.SATURDAY, Weekday.SUNDAY)),
    ],
)
def test_parse_days_of_week_accepts_timer_forms(text: str, expected: tuple[Weekday, ...]) -> None:
    assert parse_days_of_week(text).days == expected


def test_days_of_week_deduplicates_and_sorts() -> None:
    days = DaysOfWeek((Weekday.FRIDAY, Weekday.MONDAY, Weekday.FRIDAY))

    assert days.days == (Weekday.MONDAY, Weekday.FRIDAY)
    assert str(days) == "monday, friday"


def test_days_of_week_contains_dates_and_datetimes() -> None:
    days = DaysOfWeek.weekdays()

    assert days.contains_date(date(2026, 7, 3))  # Friday
    assert days.contains_datetime(_utc(2026, 7, 3, 12))
    assert not days.contains_date(date(2026, 7, 4))  # Saturday


def test_days_of_week_next_date_on_or_after() -> None:
    days = DaysOfWeek((Weekday.MONDAY, Weekday.WEDNESDAY))

    assert days.next_date_on_or_after(date(2026, 7, 6)) == date(2026, 7, 6)
    assert days.next_date_on_or_after(date(2026, 7, 7)) == date(2026, 7, 8)
    assert days.next_date_on_or_after(date(2026, 7, 9)) == date(2026, 7, 13)


@pytest.mark.parametrize("value", ["", "holiday"])
def test_parse_days_of_week_rejects_invalid_values(value: str) -> None:
    with pytest.raises(CronTimeError):
        parse_days_of_week(value)


# --- parse_local_time --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("20:00", (20, 0)),
        ("8:00pm", (20, 0)),
        ("8pm", (20, 0)),
        ("08:30", (8, 30)),
        ("17:27", (17, 27)),
        ("12am", (0, 0)),
        ("12pm", (12, 0)),
        ("11:59pm", (23, 59)),
        ("00:00", (0, 0)),
    ],
)
def test_parse_local_time_accepts_grammar_forms(text: str, expected: tuple[int, int]) -> None:
    parsed = parse_local_time(text)
    assert (parsed.hour, parsed.minute) == expected


@pytest.mark.parametrize(
    "text",
    ["", "  ", "25:00", "8:60pm", "abc", "13pm", "-1:00", "8:00 xm", "0am", "0pm"],
)
def test_parse_local_time_rejects_malformed(text: str) -> None:
    with pytest.raises(CronTimeError):
        parse_local_time(text)


def test_local_time_of_day_str_and_validation() -> None:
    assert str(LocalTimeOfDay(8, 30)) == "08:30"
    assert LocalTimeOfDay(12, 0).total_minutes == 720
    with pytest.raises(CronTimeError):
        LocalTimeOfDay(24, 0)
    with pytest.raises(CronTimeError):
        LocalTimeOfDay(8, 60)


# --- next_wall_clock_run ------------------------------------------------------


def test_next_wall_clock_run_today() -> None:
    now = _utc(2026, 7, 2, 15, 0)  # 8am LA
    target = LocalTimeOfDay(12, 0)  # noon LA = 19:00 UTC
    result = next_wall_clock_run(now=now, timezone=LA, time=target)
    assert result == _utc(2026, 7, 2, 19, 0)


def test_next_wall_clock_run_tomorrow_when_already_passed() -> None:
    now = _utc(2026, 7, 2, 20, 30)  # 1:30pm LA — noon already passed
    target = LocalTimeOfDay(12, 0)
    result = next_wall_clock_run(now=now, timezone=LA, time=target)
    assert result == _utc(2026, 7, 3, 19, 0)


def test_next_wall_clock_run_explicit_date() -> None:
    now = _utc(2026, 7, 2, 15, 0)
    target = LocalTimeOfDay(20, 0)
    result = next_wall_clock_run(
        now=now,
        timezone=LA,
        time=target,
        date=date(2026, 7, 5),
    )
    # July 5 20:00 LA = July 6 03:00 UTC
    assert result == _utc(2026, 7, 6, 3, 0)


def test_next_wall_clock_run_unknown_timezone() -> None:
    with pytest.raises(CronTimeError):
        next_wall_clock_run(
            now=_utc(2026, 7, 2, 15, 0),
            timezone="Mars/Olympus",
            time=LocalTimeOfDay(8, 0),
        )


def test_next_wall_clock_run_rejects_naive_now() -> None:
    with pytest.raises(CronTimeError):
        next_wall_clock_run(
            now=datetime(2026, 7, 2, 15, 0),  # noqa: DTZ001  # naive by design
            timezone=LA,
            time=LocalTimeOfDay(12, 0),
        )


def test_next_interval_run_rejects_naive_previous() -> None:
    with pytest.raises(CronTimeError):
        next_interval_run(
            previous=datetime(2026, 7, 2, 12, 0),  # noqa: DTZ001  # naive by design
            now=_utc(2026, 7, 2, 12, 5),
            minutes=15,
            active_window=None,
        )


# --- DST handling ------------------------------------------------------------


def test_dst_fall_back_ambiguous_uses_first_occurrence() -> None:
    # America/Los_Angeles falls back on 2026-11-01 at 02:00 local.
    # 01:30 is ambiguous (occurs twice). fold=0 selects the PDT instance.
    result = next_wall_clock_run(
        now=_utc(2026, 11, 1, 7, 0),  # just after midnight UTC = 2026-10-31 PT
        timezone=LA,
        time=LocalTimeOfDay(1, 30),
        date=date(2026, 11, 1),
    )
    # PDT offset is UTC-7; 01:30 PDT = 08:30 UTC.
    assert result == _utc(2026, 11, 1, 8, 30)
    assert result.utcoffset() == timedelta(0)


def test_dst_spring_forward_nonexistent_raises() -> None:
    # America/Los_Angeles springs forward on 2026-03-08 at 02:00 local.
    # 02:30 local does not exist on that date.
    with pytest.raises(CronTimeError, match="spring-forward"):
        next_wall_clock_run(
            now=_utc(2026, 3, 8, 9, 0),
            timezone=LA,
            time=LocalTimeOfDay(2, 30),
            date=date(2026, 3, 8),
        )


# --- next_interval_run -------------------------------------------------------


def test_next_interval_run_without_window_advances_by_minutes() -> None:
    previous = _utc(2026, 7, 2, 12, 0)
    now = _utc(2026, 7, 2, 12, 5)
    result = next_interval_run(previous=previous, now=now, minutes=15, active_window=None)
    assert result == previous + timedelta(minutes=15)


def test_next_interval_run_without_window_jumps_after_lag() -> None:
    previous = _utc(2026, 7, 2, 12, 0)
    now = _utc(2026, 7, 2, 12, 30)  # more than 15m after previous
    result = next_interval_run(previous=previous, now=now, minutes=15, active_window=None)
    assert result == now + timedelta(minutes=15)


def test_next_interval_run_rejects_non_positive_minutes() -> None:
    with pytest.raises(CronTimeError):
        next_interval_run(
            previous=_utc(2026, 7, 2, 12, 0),
            now=_utc(2026, 7, 2, 12, 0),
            minutes=0,
            active_window=None,
        )


def _window(start: tuple[int, int], end: tuple[int, int], tz: str = LA) -> ActiveWindow:
    return ActiveWindow(
        timezone=tz,
        start=LocalTimeOfDay(*start),
        end=LocalTimeOfDay(*end),
    )


def test_next_interval_run_inside_window_returns_first_step() -> None:
    window = _window((8, 0), (17, 0))
    previous = _utc(2026, 7, 2, 16, 0)  # 9am LA — inside
    now = _utc(2026, 7, 2, 15, 55)
    result = next_interval_run(previous=previous, now=now, minutes=15, active_window=window)
    assert result == previous + timedelta(minutes=15)


def test_next_interval_run_outside_window_advances_inside() -> None:
    window = _window((8, 0), (17, 0))
    previous = _utc(2026, 7, 2, 16, 50)  # 9:50am LA — inside, last step lands outside
    now = _utc(2026, 7, 2, 16, 45)
    result = next_interval_run(previous=previous, now=now, minutes=15, active_window=window)
    local = result.astimezone(LA_ZONE)
    assert window.contains(LocalTimeOfDay(local.hour, local.minute))


def test_next_interval_run_skips_overnight_inactive_span() -> None:
    # Daytime-only window 08:00-17:00 in LA. Previous at 4:45pm LA (last in-window slot).
    # Next 15m step lands at 17:00 LA — exclusive end — so it falls outside; the
    # run should fast-forward to next day's 08:00 LA.
    window = _window((8, 0), (17, 0))
    previous = _utc(2026, 7, 2, 23, 45)  # 16:45 LA — inside
    now = _utc(2026, 7, 3, 1, 0)
    result = next_interval_run(previous=previous, now=now, minutes=15, active_window=window)
    local = result.astimezone(LA_ZONE)
    # Fast-forward should land at next day's 08:00 LA.
    assert (local.hour, local.minute) == (8, 0)
    assert local.date() == date(2026, 7, 3)
    assert window.contains(LocalTimeOfDay(local.hour, local.minute))


def test_next_interval_run_midnight_crossing_window_includes_late_runs() -> None:
    # 22:00-06:00 crosses midnight; a run at 23:00 LA should be allowed.
    window = _window((22, 0), (6, 0))
    previous = _utc(2026, 7, 2, 5, 30)  # 22:30 previous day LA
    now = _utc(2026, 7, 3, 2, 0)
    result = next_interval_run(previous=previous, now=now, minutes=60, active_window=window)
    local = result.astimezone(LA_ZONE)
    assert window.contains(LocalTimeOfDay(local.hour, local.minute))


def test_active_window_rejects_zero_length() -> None:
    with pytest.raises(CronTimeError):
        ActiveWindow(
            timezone=LA,
            start=LocalTimeOfDay(8, 0),
            end=LocalTimeOfDay(8, 0),
        )


def test_active_window_unknown_timezone_resolves_lazily() -> None:
    window = ActiveWindow(
        timezone="Mars/Olympus",
        start=LocalTimeOfDay(8, 0),
        end=LocalTimeOfDay(17, 0),
    )
    with pytest.raises(CronTimeError):
        window.zone()


def test_active_window_contains_helpers() -> None:
    daytime = _window((8, 0), (17, 0))
    assert daytime.contains(LocalTimeOfDay(12, 0))
    assert not daytime.contains(LocalTimeOfDay(22, 0))
    assert not daytime.crosses_midnight

    overnight = _window((22, 0), (6, 0))
    assert overnight.contains(LocalTimeOfDay(23, 30))
    assert overnight.contains(LocalTimeOfDay(2, 0))
    assert not overnight.contains(LocalTimeOfDay(12, 0))
    assert overnight.crosses_midnight
