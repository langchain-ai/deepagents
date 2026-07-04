"""Pure-function day-of-week helpers for cron scheduling.

Provides weekday parsing, an immutable weekday set, and helpers for
determining which weekdays a timer may run. No I/O, no side effects, fully
deterministic and unit-testable without sleeping.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import IntEnum
from typing import TYPE_CHECKING

from deepagents_talon.cron.time import CronTimeError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "DaysOfWeek",
    "Weekday",
    "parse_days_of_week",
    "parse_weekday",
]

_DAYS_IN_WEEK = 7


class Weekday(IntEnum):
    """Day of week using Python's `datetime.weekday()` numbering.

    Args:
        value: Integer day index where Monday is `0` and Sunday is `6`.
    """

    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

    @property
    def label(self) -> str:
        """Return the lowercase full day name."""
        return _WEEKDAY_LABELS[self]

    @property
    def abbreviation(self) -> str:
        """Return the lowercase three-letter day abbreviation."""
        return self.label[:3]

    @classmethod
    def from_date(cls, value: date) -> Weekday:
        """Return the weekday for a calendar date.

        Args:
            value: Calendar date to inspect.

        Returns:
            Matching weekday.
        """
        return cls(value.weekday())

    @classmethod
    def from_datetime(cls, value: datetime) -> Weekday:
        """Return the weekday for a datetime's local date.

        Args:
            value: Datetime to inspect.

        Returns:
            Matching weekday.
        """
        return cls(value.weekday())

    def __str__(self) -> str:
        """Return the lowercase full day name."""
        return self.label


_WEEKDAY_LABELS: dict[Weekday, str] = {
    Weekday.MONDAY: "monday",
    Weekday.TUESDAY: "tuesday",
    Weekday.WEDNESDAY: "wednesday",
    Weekday.THURSDAY: "thursday",
    Weekday.FRIDAY: "friday",
    Weekday.SATURDAY: "saturday",
    Weekday.SUNDAY: "sunday",
}
_WEEKDAY_ALIASES: dict[str, Weekday] = {
    "m": Weekday.MONDAY,
    "mon": Weekday.MONDAY,
    "monday": Weekday.MONDAY,
    "tue": Weekday.TUESDAY,
    "tues": Weekday.TUESDAY,
    "tuesday": Weekday.TUESDAY,
    "w": Weekday.WEDNESDAY,
    "wed": Weekday.WEDNESDAY,
    "weds": Weekday.WEDNESDAY,
    "wednesday": Weekday.WEDNESDAY,
    "th": Weekday.THURSDAY,
    "thu": Weekday.THURSDAY,
    "thur": Weekday.THURSDAY,
    "thurs": Weekday.THURSDAY,
    "thursday": Weekday.THURSDAY,
    "f": Weekday.FRIDAY,
    "fri": Weekday.FRIDAY,
    "friday": Weekday.FRIDAY,
    "sa": Weekday.SATURDAY,
    "sat": Weekday.SATURDAY,
    "saturday": Weekday.SATURDAY,
    "su": Weekday.SUNDAY,
    "sun": Weekday.SUNDAY,
    "sunday": Weekday.SUNDAY,
}


@dataclass(frozen=True, slots=True)
class DaysOfWeek:
    """Immutable set of active weekdays for a timer.

    Args:
        days: Weekdays when a timer may run. Duplicates are removed and the
            stored tuple is sorted Monday through Sunday.
    """

    days: tuple[Weekday, ...]

    def __post_init__(self) -> None:
        """Normalize and validate the weekday collection."""
        try:
            normalized = tuple(sorted({Weekday(day) for day in self.days}, key=int))
        except ValueError as exc:
            msg = "days of week contains an invalid weekday"
            raise CronTimeError(msg) from exc
        if not normalized:
            msg = "days of week must include at least one day"
            raise CronTimeError(msg)
        object.__setattr__(self, "days", normalized)

    @classmethod
    def all(cls) -> DaysOfWeek:
        """Return all seven days.

        Returns:
            Days of week containing Monday through Sunday.
        """
        return cls(tuple(Weekday))

    @classmethod
    def weekdays(cls) -> DaysOfWeek:
        """Return Monday through Friday.

        Returns:
            Business weekdays.
        """
        return cls(
            (
                Weekday.MONDAY,
                Weekday.TUESDAY,
                Weekday.WEDNESDAY,
                Weekday.THURSDAY,
                Weekday.FRIDAY,
            ),
        )

    @classmethod
    def weekends(cls) -> DaysOfWeek:
        """Return Saturday and Sunday.

        Returns:
            Weekend days.
        """
        return cls((Weekday.SATURDAY, Weekday.SUNDAY))

    def contains_date(self, value: date) -> bool:
        """Return whether `value` falls on an active weekday.

        Args:
            value: Calendar date to inspect.

        Returns:
            `True` when the date's weekday is active.
        """
        return Weekday.from_date(value) in self.days

    def contains_datetime(self, value: datetime) -> bool:
        """Return whether `value` falls on an active weekday.

        Args:
            value: Datetime to inspect using its local date.

        Returns:
            `True` when the datetime's weekday is active.
        """
        return Weekday.from_datetime(value) in self.days

    def next_date_on_or_after(self, value: date) -> date:
        """Return the first active date on or after `value`.

        Args:
            value: Starting calendar date.

        Returns:
            `value` when active, otherwise the next active date within one week.
        """
        for offset in range(_DAYS_IN_WEEK):
            candidate = value + timedelta(days=offset)
            if self.contains_date(candidate):
                return candidate
        msg = "days of week must include at least one reachable day"
        raise CronTimeError(msg)

    def __contains__(self, day: object) -> bool:
        """Return whether `day` is active."""
        return day in self.days

    def __iter__(self) -> Iterator[Weekday]:
        """Iterate over active weekdays in Monday-through-Sunday order."""
        return iter(self.days)

    def __str__(self) -> str:
        """Return comma-separated weekday labels."""
        return ", ".join(day.label for day in self.days)


# --- Parsing -----------------------------------------------------------------

_DAY_SEPARATORS = re.compile(r"[\s,;/+&]+")
_DAY_JOINERS = {"and", "or", "on"}
_ALL_DAYS_TEXT = {"all", "all days", "any day", "anyday", "daily", "every day", "everyday"}
_WEEKDAYS_TEXT = {"weekday", "weekdays"}
_WEEKENDS_TEXT = {"weekend", "weekends"}


def parse_weekday(value: str) -> Weekday:
    """Parse a weekday name or abbreviation.

    Supports full day names and common abbreviations such as `mon`, `monday`,
    `thu`, `thurs`, `sat`, and `sunday`.

    Args:
        value: Weekday text.

    Returns:
        Parsed weekday.

    Raises:
        CronTimeError: If the input is empty, ambiguous, or unsupported.
    """
    text = value.strip().lower().rstrip(".")
    if not text:
        msg = "weekday must not be empty"
        raise CronTimeError(msg)
    if text in _WEEKDAY_ALIASES:
        return _WEEKDAY_ALIASES[text]
    if text.endswith("s") and text[:-1] in _WEEKDAY_ALIASES:
        return _WEEKDAY_ALIASES[text[:-1]]
    msg = f"unrecognized weekday: {value!r}"
    raise CronTimeError(msg)


def parse_days_of_week(value: str) -> DaysOfWeek:
    """Parse timer day-of-week text.

    Supports `daily`, `every day`, `weekdays`, `weekends`, and separated day
    names such as `mon wed fri`, `monday, wednesday, friday`, or `sat/sun`.

    Args:
        value: Day-of-week text.

    Returns:
        Parsed immutable day set.

    Raises:
        CronTimeError: If the input is empty or contains an unsupported day.
    """
    text = " ".join(value.strip().lower().split())
    if not text:
        msg = "days of week must not be empty"
        raise CronTimeError(msg)
    if text in _ALL_DAYS_TEXT:
        return DaysOfWeek.all()
    if text in _WEEKDAYS_TEXT:
        return DaysOfWeek.weekdays()
    if text in _WEEKENDS_TEXT:
        return DaysOfWeek.weekends()

    tokens = [token for token in _DAY_SEPARATORS.split(text) if token and token not in _DAY_JOINERS]
    return DaysOfWeek(tuple(parse_weekday(token) for token in tokens))
