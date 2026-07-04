from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from deepagents_talon.cron.days_of_week import (
    DaysOfWeek,
    Weekday,
    parse_days_of_week,
    parse_weekday,
)
from deepagents_talon.cron.time import CronTimeError


def _utc(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


# --- parse_weekday -----------------------------------------------------------


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


@pytest.mark.parametrize("text", ["", "  ", "s", "t", "tueth", "funday"])
def test_parse_weekday_rejects_unknown_or_ambiguous_names(text: str) -> None:
    with pytest.raises(CronTimeError):
        parse_weekday(text)


# --- parse_days_of_week ------------------------------------------------------


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


def test_days_of_week_rejects_empty() -> None:
    with pytest.raises(CronTimeError):
        DaysOfWeek(())


def test_weekday_from_date_and_datetime_agree_with_python_weekday() -> None:
    # 2026-07-03 is a Friday (weekday() == 4)
    assert Weekday.from_date(date(2026, 7, 3)) is Weekday.FRIDAY
    assert Weekday.from_datetime(_utc(2026, 7, 3, 12)) is Weekday.FRIDAY
    assert int(Weekday.FRIDAY) == 4


def test_weekday_label_and_abbreviation() -> None:
    assert Weekday.MONDAY.label == "monday"
    assert Weekday.WEDNESDAY.abbreviation == "wed"
    assert str(Weekday.SUNDAY) == "sunday"
