"""Pure-function time helpers for cron scheduling.

Provides IANA time zone conversion, local-time parsing, wall-clock
next-occurrence computation, and DST edge-case handling. No I/O, no side
effects, fully deterministic and unit-testable without sleeping.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import IntEnum
from functools import total_ordering
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "ActiveWindow",
    "CronTimeError",
    "DaysOfWeek",
    "LocalTimeOfDay",
    "Weekday",
    "next_interval_run",
    "next_wall_clock_run",
    "parse_days_of_week",
    "parse_local_time",
    "parse_weekday",
]

_MAX_HOUR = 23
_MAX_MINUTE = 59
_AMPM_HOUR_BOUND = 12  # 12-hour clock upper bound before wrapping to 0
_DAYS_IN_WEEK = 7


class CronTimeError(ValueError):
    """Raised when a cron time request is invalid or impossible."""


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


@total_ordering
@dataclass(frozen=True, slots=True)
class LocalTimeOfDay:
    """Local clock time with minute precision.

    Args:
        hour: Hour of day (0-23).
        minute: Minute of hour (0-59).
    """

    hour: int
    minute: int

    def __post_init__(self) -> None:
        """Validate the clock-time components."""
        if not 0 <= self.hour <= _MAX_HOUR:
            msg = f"hour must be 0-23, got {self.hour}"
            raise CronTimeError(msg)
        if not 0 <= self.minute <= _MAX_MINUTE:
            msg = f"minute must be 0-59, got {self.minute}"
            raise CronTimeError(msg)

    @property
    def total_minutes(self) -> int:
        """Minutes since midnight."""
        return self.hour * 60 + self.minute

    def __lt__(self, other: LocalTimeOfDay) -> bool:
        """Compare ordering by minutes since midnight."""
        return self.total_minutes < other.total_minutes

    def __str__(self) -> str:
        """Return a `HH:MM` string.

        Returns:
            Zero-padded `HH:MM` representation.
        """
        return f"{self.hour:02d}:{self.minute:02d}"


@dataclass(frozen=True, slots=True)
class ActiveWindow:
    """Local-time window during which recurring jobs may run.

    Windows that cross midnight (`end <= start`) are supported and
    interpreted as `start <= local < 24:00` OR `00:00 <= local < end`.

    Args:
        timezone: IANA time zone name used to evaluate the window.
        start: Inclusive local start time.
        end: Exclusive local end time.
    """

    timezone: str
    start: LocalTimeOfDay
    end: LocalTimeOfDay

    def __post_init__(self) -> None:
        """Validate the active window."""
        if self.start == self.end:
            msg = "active window start must differ from end"
            raise CronTimeError(msg)

    @property
    def crosses_midnight(self) -> bool:
        """Whether this window spans midnight."""
        return self.end <= self.start

    def contains(self, local: LocalTimeOfDay) -> bool:
        """Return whether `local` falls inside this window.

        Args:
            local: Local clock time to test.

        Returns:
            `True` when `local` is inside the active window.
        """
        if self.crosses_midnight:
            return local >= self.start or local < self.end
        return self.start <= local < self.end

    def zone(self) -> ZoneInfo:
        """Return the zoneinfo for this window.

        Returns:
            Resolved `ZoneInfo`.

        Raises:
            CronTimeError: If the time zone is unknown.
        """
        return _resolve_zone(self.timezone)


# --- Parsing -----------------------------------------------------------------

_AM_PM = re.compile(
    r"^(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<suffix>am|pm)$",
    re.IGNORECASE,
)
_24H = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})$")
_HOUR_ONLY = re.compile(r"^(?P<hour>\d{1,2})$")
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


def parse_local_time(value: str) -> LocalTimeOfDay:
    """Parse a local clock time string.

    Supports the grammar forms: `20:00`, `8:00pm`, `8pm`, `08:30`, `17:27`.

    Args:
        value: Local time text.

    Returns:
        Parsed local time of day.

    Raises:
        CronTimeError: If the input is malformed or out of range.
    """
    text = value.strip().lower()
    if not text:
        msg = "local time must not be empty"
        raise CronTimeError(msg)

    m = _AM_PM.match(text)
    if m:
        hour = int(m.group("hour"))
        minute = int(m.group("minute") or "0")
        if not 1 <= hour <= _AMPM_HOUR_BOUND:
            msg = f"12-hour clock hour out of range: {value!r}"
            raise CronTimeError(msg)
        if hour == _AMPM_HOUR_BOUND:
            hour = 0
        suffix = m.group("suffix").lower()
        if suffix == "pm":
            hour += _AMPM_HOUR_BOUND
        return LocalTimeOfDay(hour=hour, minute=minute)

    m = _24H.match(text)
    if m:
        return LocalTimeOfDay(hour=int(m.group("hour")), minute=int(m.group("minute")))

    m = _HOUR_ONLY.match(text)
    if m and text.isdigit():
        return LocalTimeOfDay(hour=int(m.group("hour")), minute=0)

    msg = f"unrecognized local time format: {value!r}"
    raise CronTimeError(msg)


# --- Zone resolution ---------------------------------------------------------


def _resolve_zone(timezone: str) -> ZoneInfo:
    """Resolve an IANA zone name, re-raising unknown zones as `CronTimeError`."""
    try:
        return ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        msg = f"unknown time zone: {timezone!r}"
        raise CronTimeError(msg) from exc


def _as_utc(dt: datetime) -> datetime:
    """Return `dt` converted to UTC, rejecting naive datetimes."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        msg = "datetime must be timezone-aware"
        raise CronTimeError(msg)
    return dt.astimezone(UTC)


def _local_time_of(dt: datetime) -> LocalTimeOfDay:
    """Return the local clock-time components of `dt`."""
    return LocalTimeOfDay(hour=dt.hour, minute=dt.minute)


# --- Wall-clock next occurrence ----------------------------------------------


def next_wall_clock_run(
    *,
    now: datetime,
    timezone: str,
    time: LocalTimeOfDay,
    date: date | None = None,
) -> datetime:
    """Return the next UTC occurrence of `time` in `timezone`.

    If no `date` is provided, the next occurrence is today when the local
    clock has not yet reached `time`, otherwise tomorrow. When `date` is
    provided, that local calendar date's occurrence is returned.

    DST fall-back ambiguous times resolve to the first occurrence (`fold=0`).
    DST spring-forward nonexistent local times raise a structured error
    rather than silently shifting.

    Args:
        now: Current instant, timezone-aware.
        timezone: IANA time zone for wall-clock interpretation.
        time: Target local clock time.
        date: Optional explicit local calendar date.

    Returns:
        Next occurrence of `time` in `timezone`, as a UTC datetime.

    Raises:
        CronTimeError: If the timezone is unknown, `now` is naive, or the
            requested local time does not exist due to DST spring-forward.
    """
    zone = _resolve_zone(timezone)
    now_utc = _as_utc(now)

    if date is None:
        now_local = now_utc.astimezone(zone)
        candidate_date = now_local.date()
        if _local_time_of(now_local) >= time:
            candidate_date += timedelta(days=1)
        candidate = _build_local(zone, candidate_date, time, fold=0)
        if candidate <= now_utc:
            candidate = _build_local(
                zone,
                candidate_date + timedelta(days=1),
                time,
                fold=0,
            )
    else:
        candidate = _build_local(zone, date, time, fold=0)

    return _as_utc(candidate)


def _build_local(zone: ZoneInfo, target_date: date, time: LocalTimeOfDay, *, fold: int) -> datetime:
    """Construct a local datetime at `time` on `target_date` in `zone`.

    Detects nonexistent spring-forward local times by verifying the
    round-trip wall clock matches the request.
    """
    naive = datetime(  # noqa: DTZ001  # naive constructed intentionally; tzinfo attached next line
        target_date.year,
        target_date.month,
        target_date.day,
        time.hour,
        time.minute,
        fold=fold,
    )
    aware = naive.replace(tzinfo=zone)
    roundtrip = aware.astimezone(UTC).astimezone(zone)
    if roundtrip.date() != target_date or _local_time_of(roundtrip) != time:
        msg = (
            f"local time {time} does not exist on {target_date.isoformat()} "
            f"in {zone!s} (spring-forward gap)"
        )
        raise CronTimeError(msg)
    return aware


# --- Interval next run with optional active window ----------------------------


def next_interval_run(
    *,
    previous: datetime,
    now: datetime,
    minutes: int,
    active_window: ActiveWindow | None,
) -> datetime:
    """Return the next interval run, gated by an optional active window.

    Without an active window, the next run is `previous + minutes`, advanced
    forward until it is strictly greater than `now` (one step is guaranteed).

    With an active window, the next run is advanced forward by `minutes`
    increments until it lands inside the window. When an interval lands
    outside the window, the next in-window time after it is used, which may
    fast-forward across inactive spans (e.g. overnight).

    Args:
        previous: Previous scheduled run instant, timezone-aware.
        now: Current instant, timezone-aware.
        minutes: Interval in minutes (>= 1).
        active_window: Optional local-time window.

    Returns:
        Next run instant as a UTC datetime.

    Raises:
        CronTimeError: If `minutes` is not positive.
    """
    if minutes < 1:
        msg = "interval minutes must be positive"
        raise CronTimeError(msg)

    step = timedelta(minutes=minutes)
    candidate = _as_utc(previous) + step
    now_utc = _as_utc(now)
    if candidate <= now_utc:
        missed = (now_utc - candidate) // step + 1
        candidate += missed * step

    if active_window is None:
        return candidate

    return _advance_into_window(candidate, active_window)


def _advance_into_window(
    candidate: datetime,
    window: ActiveWindow,
) -> datetime:
    """Fast-forward `candidate` to the next window-start when it is outside `window`."""
    zone = window.zone()
    local = candidate.astimezone(zone)
    local_time = _local_time_of(local)
    if window.contains(local_time):
        return candidate

    start_today_utc = _as_utc(_build_local(zone, local.date(), window.start, fold=0))
    if start_today_utc >= candidate:
        return start_today_utc
    return _as_utc(_build_local(zone, local.date() + timedelta(days=1), window.start, fold=0))
