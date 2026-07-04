"""Pure-function time helpers for cron scheduling.

Provides IANA time zone conversion, local-time parsing, wall-clock
next-occurrence computation, and DST edge-case handling. No I/O, no side
effects, fully deterministic and unit-testable without sleeping.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import IntEnum
from functools import total_ordering
from typing import TYPE_CHECKING, NotRequired, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "ActiveWindow",
    "ActiveWindowDict",
    "CronTimeError",
    "DaysOfWeek",
    "LocalTimeOfDay",
    "SchedulerHostClock",
    "SchedulerHostClockDict",
    "Weekday",
    "active_window_contains",
    "capture_scheduler_host_clock",
    "format_local_run",
    "next_daily_wall_clock_run",
    "next_interval_run",
    "next_wall_clock_run",
    "parse_active_window",
    "parse_days_of_week",
    "parse_local_time",
    "parse_weekday",
    "resolve_schedule_timezone",
]

_MAX_HOUR = 23
_MAX_MINUTE = 59
_AMPM_HOUR_BOUND = 12  # 12-hour clock upper bound before wrapping to 0
_DAYS_IN_WEEK = 7
TIMEZONE_ENV = "DEEPAGENTS_TALON_TIMEZONE"


class ActiveWindowDict(TypedDict):
    """Serialized local active-hour window."""

    timezone: str
    start: str
    end: str


class SchedulerHostClockDict(TypedDict):
    """Serialized scheduler host clock context."""

    timezone: str
    utc_offset: str
    computed_at: str
    local_now: NotRequired[str]


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

    @property
    def minutes_after_midnight(self) -> int:
        """Minutes since midnight."""
        return self.total_minutes

    def to_display(self) -> str:
        """Return a `HH:MM` string."""
        return f"{self.hour:02d}:{self.minute:02d}"

    @classmethod
    def from_display(cls, value: str) -> LocalTimeOfDay:
        """Parse a persisted `HH:MM` time value."""
        return parse_local_time(value)

    def __str__(self) -> str:
        """Return a `HH:MM` string.

        Returns:
            Zero-padded `HH:MM` representation.
        """
        return self.to_display()


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

    def to_dict(self) -> ActiveWindowDict:
        """Serialize this active window for disk storage."""
        return {
            "timezone": self.timezone,
            "start": self.start.to_display(),
            "end": self.end.to_display(),
        }

    @classmethod
    def from_dict(cls, data: ActiveWindowDict) -> ActiveWindow:
        """Deserialize an active window from disk."""
        return cls(
            timezone=data["timezone"],
            start=LocalTimeOfDay.from_display(data["start"]),
            end=LocalTimeOfDay.from_display(data["end"]),
        )

    def zone(self) -> ZoneInfo:
        """Return the zoneinfo for this window.

        Returns:
            Resolved `ZoneInfo`.

        Raises:
            CronTimeError: If the time zone is unknown.
        """
        return _resolve_zone(self.timezone)


@dataclass(frozen=True, slots=True)
class SchedulerHostClock:
    """Scheduler host clock context used to compute a UTC run time.

    Args:
        timezone: Time zone name reported by the host clock.
        utc_offset: UTC offset reported by the host clock.
        computed_at: UTC timestamp used as the computation basis.
        local_now: Host-local timestamp for diagnostics.
    """

    timezone: str
    utc_offset: str
    computed_at: datetime
    local_now: datetime

    def to_dict(self) -> SchedulerHostClockDict:
        """Serialize this host clock context for disk storage."""
        return {
            "timezone": self.timezone,
            "utc_offset": self.utc_offset,
            "computed_at": _format_time(self.computed_at),
            "local_now": self.local_now.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: SchedulerHostClockDict) -> SchedulerHostClock:
        """Deserialize scheduler host clock context."""
        computed_at = _parse_time(data["computed_at"])
        local_now = datetime.fromisoformat(data.get("local_now", data["computed_at"]))
        if local_now.tzinfo is None or local_now.utcoffset() is None:
            local_now = local_now.replace(tzinfo=UTC)
        return cls(
            timezone=data["timezone"],
            utc_offset=data["utc_offset"],
            computed_at=computed_at,
            local_now=local_now,
        )


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


def parse_active_window(
    *,
    timezone: str | None,
    active_start: str | None,
    active_end: str | None,
) -> ActiveWindow | None:
    """Parse optional active-hour inputs."""
    if active_start == "" and active_end == "":
        return None
    if active_start is None and active_end is None:
        return None
    if active_start is None or active_end is None:
        msg = "active hours require both active_start and active_end"
        raise CronTimeError(msg)
    return ActiveWindow(
        timezone=resolve_schedule_timezone(timezone),
        start=parse_local_time(active_start),
        end=parse_local_time(active_end),
    )


def resolve_schedule_timezone(value: str | None) -> str:
    """Resolve the IANA time zone for a wall-clock schedule."""
    candidate = value or os.environ.get(TIMEZONE_ENV)
    if candidate is None or not candidate.strip():
        msg = f"timezone is required for local-time schedules; set {TIMEZONE_ENV}"
        raise CronTimeError(msg)
    zone = _resolve_zone(candidate.strip())
    key = getattr(zone, "key", None)
    return key if isinstance(key, str) and key else candidate.strip()


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
        if candidate <= now_utc:
            msg = "dated wall-clock schedule is in the past"
            raise CronTimeError(msg)

    return _as_utc(candidate)


def next_daily_wall_clock_run(
    *,
    previous: datetime | None,
    now: datetime,
    timezone: str,
    time: LocalTimeOfDay,
    active_window: ActiveWindow | None = None,
) -> datetime:
    """Return the next UTC run for a recurring daily wall-clock schedule."""
    now_utc = _as_utc(now)
    zone = _resolve_zone(timezone)
    if previous is None:
        candidate = next_wall_clock_run(now=now_utc, timezone=timezone, time=time)
    else:
        previous_local = _as_utc(previous).astimezone(zone)
        candidate = _as_utc(
            _build_local(zone, previous_local.date() + timedelta(days=1), time, fold=0)
        )
        while candidate <= now_utc:
            candidate_local = candidate.astimezone(zone)
            candidate = _as_utc(
                _build_local(zone, candidate_local.date() + timedelta(days=1), time, fold=0)
            )
    if active_window is not None and not active_window_contains(candidate, active_window):
        msg = "daily wall-clock schedule time falls outside active hours"
        raise CronTimeError(msg)
    return candidate


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


def active_window_contains(value: datetime, active_window: ActiveWindow) -> bool:
    """Return whether a UTC timestamp falls inside an active window."""
    local = _as_utc(value).astimezone(active_window.zone())
    return active_window.contains(LocalTimeOfDay(local.hour, local.minute))


def capture_scheduler_host_clock(now: datetime | None = None) -> SchedulerHostClock:
    """Capture the scheduler host clock context."""
    computed_at = datetime.now(UTC) if now is None else _as_utc(now)
    local_now = computed_at.astimezone()
    return SchedulerHostClock(
        timezone=_timezone_name(local_now),
        utc_offset=_format_offset(local_now),
        computed_at=computed_at,
        local_now=local_now,
    )


def format_local_run(value: datetime, timezone: str) -> str:
    """Format a UTC run timestamp in a schedule time zone."""
    local = _as_utc(value).astimezone(_resolve_zone(timezone))
    return f"{local:%Y-%m-%d %H:%M} {timezone}"


def _timezone_name(value: datetime) -> str:
    tzinfo = value.tzinfo
    key = getattr(tzinfo, "key", None)
    if isinstance(key, str) and key:
        return key
    name = value.tzname()
    return name or "local"


def _format_offset(value: datetime) -> str:
    offset = value.utcoffset()
    if offset is None:
        return "+00:00"
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def _format_time(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _parse_time(value: str) -> datetime:
    return _as_utc(datetime.fromisoformat(value))
