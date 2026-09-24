"""Five-field cron expressions with the common `L`, `W`, and `#` extensions.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import calendar
import functools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import MAXYEAR, date, datetime, time
from typing import NamedTuple

from deepagents_talon.cron.errors import CronJobError

_FIELD_COUNT = 5
_EXPRESSION_CACHE_SIZE = 128
"""Bound on distinct cached expressions; the text originates from agent input."""

_SEARCH_HORIZON_MONTHS = 50 * 12
"""Months searched for the next match before giving up.

The rarest satisfiable patterns pin a weekday to February 29 -- for example
`0 0 * 2 5#5`, the fifth Friday of February -- which recurs only when a leap
year starts that month on the right weekday: every 28 years, or 40 across a
skipped century leap year such as 2100. Fifty years covers both, and the walk
visits only the days of months the expression allows.
"""

_SATURDAY = 5
_SUNDAY = 6
_DECEMBER = 12
_MAX_OCCURRENCE = 5

DayTest = Callable[[date, int], bool]
"""Predicate over a calendar day and the last day number of its month."""


class _FieldSpec(NamedTuple):
    name: str
    low: int
    high: int
    aliases: Mapping[str, int]


_MONTH_ALIASES = {name.lower(): number for number, name in enumerate(calendar.month_abbr) if name}
_DAY_ALIASES = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}

_MINUTE = _FieldSpec("minute", 0, 59, {})
_HOUR = _FieldSpec("hour", 0, 23, {})
_DAY_OF_MONTH = _FieldSpec("day-of-month", 1, 31, {})
_MONTH = _FieldSpec("month", 1, 12, _MONTH_ALIASES)
_DAY_OF_WEEK = _FieldSpec("day-of-week", 0, 7, _DAY_ALIASES)

_MACROS = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}


@dataclass(frozen=True, slots=True)
class CronExpression:
    """Parsed cron expression, evaluated against local wall-clock time.

    Args:
        minutes: Matching minutes, ascending.
        hours: Matching hours, ascending.
        months: Matching months, 1-12.
        day_of_month: Day-of-month predicate.
        day_of_week: Day-of-week predicate.
        either_day: Whether a day matches when either day predicate does,
            rather than both. Vixie cron uses "either" only when neither day
            field starts with `*`.
    """

    minutes: tuple[int, ...]
    hours: tuple[int, ...]
    months: frozenset[int]
    day_of_month: DayTest
    day_of_week: DayTest
    either_day: bool

    def matches_day(self, day: date, last: int) -> bool:
        """Report whether the expression fires on `day`.

        Args:
            day: Calendar day to test. Its month is not checked here.
            last: Last day number of `day`'s month.

        Returns:
            Whether the day fields accept `day`.
        """
        by_month_day = self.day_of_month(day, last)
        by_weekday = self.day_of_week(day, last)
        if self.either_day:
            return by_month_day or by_weekday
        return by_month_day and by_weekday


def iter_local_matches(expression: CronExpression, start: datetime) -> Iterator[datetime]:
    """Yield naive local wall-clock times the expression matches, in order.

    Times before `start`, truncated to the minute, are skipped. The walk stops
    at the search horizon, so an exhausted iterator means no match exists.

    Args:
        expression: Parsed expression.
        start: Naive local time to start from, inclusive.

    Yields:
        Matching naive local times.
    """
    floor = start.replace(second=0, microsecond=0)
    for day in _matching_days(expression, floor.date()):
        for hour in expression.hours:
            for minute in expression.minutes:
                candidate = datetime.combine(day, time(hour, minute))
                if candidate >= floor:
                    yield candidate


@functools.lru_cache(maxsize=_EXPRESSION_CACHE_SIZE)
def parse_expression(text: str) -> CronExpression:
    """Parse a five-field cron expression or an `@` macro.

    Args:
        text: Expression such as `*/15 9-17 * * mon-fri` or `@daily`.

    Returns:
        Parsed expression.

    Raises:
        CronJobError: If the expression is malformed or out of range.
    """
    fields = _expand_macro(text).split()
    if len(fields) != _FIELD_COUNT:
        msg = (
            "cron expression needs 5 fields (minute hour day-of-month month "
            f"day-of-week), got {len(fields)} in {text!r}"
        )
        raise CronJobError(msg)
    minute, hour, month_day, month, weekday = (field.lower() for field in fields)
    return CronExpression(
        minutes=tuple(sorted(_parse_list(minute, _MINUTE))),
        hours=tuple(sorted(_parse_list(hour, _HOUR))),
        months=frozenset(_parse_list(month, _MONTH)),
        day_of_month=_parse_day_of_month(month_day),
        day_of_week=_parse_day_of_week(weekday),
        either_day=not (month_day.startswith("*") or weekday.startswith("*")),
    )


def _expand_macro(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("@"):
        return stripped
    expansion = _MACROS.get(stripped.lower())
    if expansion is None:
        known = ", ".join(_MACROS)
        msg = f"unsupported cron macro {stripped!r}; use one of {known}"
        raise CronJobError(msg)
    return expansion


def _matching_days(expression: CronExpression, start: date) -> Iterator[date]:
    year, month = start.year, start.month
    for _ in range(_SEARCH_HORIZON_MONTHS):
        if month in expression.months:
            first = start.day if (year, month) == (start.year, start.month) else 1
            yield from _matching_days_in_month(expression, year, month, first)
        year, month = (year + 1, 1) if month == _DECEMBER else (year, month + 1)
        if year > MAXYEAR:
            return


def _matching_days_in_month(
    expression: CronExpression, year: int, month: int, first: int
) -> Iterator[date]:
    last = calendar.monthrange(year, month)[1]
    for number in range(first, last + 1):
        day = date(year, month, number)
        if expression.matches_day(day, last):
            yield day


def _parse_list(text: str, spec: _FieldSpec) -> set[int]:
    values: set[int] = set()
    for term in text.split(","):
        values |= _parse_term(term, spec)
    return values


def _parse_term(term: str, spec: _FieldSpec) -> set[int]:
    """Expand one `*`, `N`, `A-B`, or stepped term into its values.

    A step on a single value, `A/S`, runs from `A` to the field maximum, as in
    Vixie cron.

    Args:
        term: Comma-free term text.
        spec: Field being parsed.

    Returns:
        Values the term selects.

    Raises:
        CronJobError: If the term is malformed or out of range.
    """
    base, slash, step_text = term.partition("/")
    step = _parse_number(step_text, f"{spec.name} step") if slash else 1
    if step < 1:
        msg = f"cron {spec.name} step must be at least 1, not {step}"
        raise CronJobError(msg)
    if base == "*":
        low, high = spec.low, spec.high
    elif "-" in base:
        start_text, _, end_text = base.partition("-")
        low, high = _parse_value(start_text, spec), _parse_value(end_text, spec)
    else:
        low = _parse_value(base, spec)
        high = spec.high if slash else low
    if low > high:
        msg = f"cron {spec.name} range {base!r} runs backward"
        raise CronJobError(msg)
    return set(range(low, high + 1, step))


def _parse_value(text: str, spec: _FieldSpec) -> int:
    value = spec.aliases.get(text)
    if value is None:
        value = _parse_number(text, spec.name)
    if not spec.low <= value <= spec.high:
        msg = f"cron {spec.name} {text!r} is outside {spec.low}-{spec.high}"
        raise CronJobError(msg)
    return value


def _parse_number(text: str, label: str) -> int:
    if not (text.isascii() and text.isdecimal()):
        msg = f"cron {label} must be a number, not {text!r}"
        raise CronJobError(msg)
    return int(text)


def _parse_day_of_month(text: str) -> DayTest:
    if text == "l":
        return _is_last_day
    if text == "lw":
        return _is_last_weekday
    if text.endswith("w"):
        return functools.partial(_is_nearest_weekday, _parse_value(text[:-1], _DAY_OF_MONTH))
    return functools.partial(_day_in, frozenset(_parse_list(text, _DAY_OF_MONTH)))


def _parse_day_of_week(text: str) -> DayTest:
    if "#" in text:
        weekday_text, _, occurrence_text = text.partition("#")
        occurrence = _parse_number(occurrence_text, "day-of-week occurrence")
        if not 1 <= occurrence <= _MAX_OCCURRENCE:
            msg = f"cron day-of-week occurrence must be 1-{_MAX_OCCURRENCE}, not {occurrence}"
            raise CronJobError(msg)
        return functools.partial(_is_nth_weekday, _parse_weekday(weekday_text), occurrence)
    if text.endswith("l"):
        return functools.partial(_is_final_weekday, _parse_weekday(text[:-1]))
    weekdays = frozenset(value % 7 for value in _parse_list(text, _DAY_OF_WEEK))
    return functools.partial(_weekday_in, weekdays)


def _parse_weekday(text: str) -> int:
    # 7 is Sunday as well as 0.
    return _parse_value(text, _DAY_OF_WEEK) % 7


def _cron_weekday(day: date) -> int:
    # Python numbers Monday 0; cron numbers Sunday 0.
    return (day.weekday() + 1) % 7


def _day_in(days: frozenset[int], day: date, _last: int) -> bool:
    return day.day in days


def _weekday_in(weekdays: frozenset[int], day: date, _last: int) -> bool:
    return _cron_weekday(day) in weekdays


def _is_last_day(day: date, last: int) -> bool:
    return day.day == last


def _is_last_weekday(day: date, last: int) -> bool:
    last_weekday = (day.weekday() + last - day.day) % 7
    target = last - max(0, last_weekday - _SATURDAY + 1)
    return day.day == target


def _is_nearest_weekday(anchor: int, day: date, last: int) -> bool:
    """Match the Monday-Friday day nearest `anchor`, never leaving the month.

    Args:
        anchor: Requested day of the month.
        day: Day being tested.
        last: Last day number of the month.

    Returns:
        Whether `day` is the weekday nearest `anchor`.
    """
    if anchor > last:
        return False
    anchor_weekday = (day.weekday() + anchor - day.day) % 7
    target = anchor
    if anchor_weekday == _SATURDAY:
        target = anchor - 1 if anchor > 1 else anchor + 2
    elif anchor_weekday == _SUNDAY:
        target = anchor + 1 if anchor < last else anchor - 2
    return day.day == target


def _is_final_weekday(weekday: int, day: date, last: int) -> bool:
    return _cron_weekday(day) == weekday and day.day + 7 > last


def _is_nth_weekday(weekday: int, occurrence: int, day: date, _last: int) -> bool:
    return _cron_weekday(day) == weekday and (day.day - 1) // 7 + 1 == occurrence
