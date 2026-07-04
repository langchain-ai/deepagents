"""Deterministic local-time helpers for Talon cron schedules.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import NotRequired, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIMEZONE_ENV = "DEEPAGENTS_TALON_TIMEZONE"

_LOCAL_TIME_RE = re.compile(
    r"^\s*(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<period>am|pm)?\s*$",
)
_MAX_HOUR = 23
_MAX_MINUTE = 59
_HOURS_PER_HALF_DAY = 12


class CronTimeError(ValueError):
    """Raised when local-time schedule conversion fails."""


class LocalTimeOfDayDict(TypedDict):
    """Serialized local time-of-day value."""

    hour: int
    minute: int


class ActiveWindowDict(TypedDict):
    """Serialized local active-hour window."""

    timezone: str
    start: str
    end: str


class SchedulerHostClockDict(TypedDict):
    """Serialized scheduler host clock context."""

    scheduler_timezone: str
    scheduler_utc_offset: str
    computed_at: str
    local_now: NotRequired[str]
    timezone: NotRequired[str]
    utc_offset: NotRequired[str]


@dataclass(frozen=True, slots=True)
class LocalTimeOfDay:
    """Minute-precision local wall-clock time.

    Args:
        hour: Hour in 24-hour local time.
        minute: Minute in local time.
    """

    hour: int
    minute: int

    def __post_init__(self) -> None:
        """Validate the local time components."""
        if not 0 <= self.hour <= _MAX_HOUR:
            msg = "local time hour must be between 0 and 23"
            raise CronTimeError(msg)
        if not 0 <= self.minute <= _MAX_MINUTE:
            msg = "local time minute must be between 0 and 59"
            raise CronTimeError(msg)

    @property
    def minutes_after_midnight(self) -> int:
        """Return this time as minutes after local midnight."""
        return self.hour * 60 + self.minute

    def to_display(self) -> str:
        """Return this time in `HH:MM` form.

        Returns:
            Zero-padded 24-hour display value.
        """
        return f"{self.hour:02d}:{self.minute:02d}"

    @classmethod
    def from_display(cls, value: str) -> LocalTimeOfDay:
        """Parse a persisted `HH:MM` time value.

        Args:
            value: Stored time value.

        Returns:
            Parsed local time.
        """
        return parse_local_time(value)


@dataclass(frozen=True, slots=True)
class ActiveWindow:
    """Local-time window in which a cron job is allowed to run.

    Args:
        timezone: IANA time zone used to evaluate the window.
        start: Inclusive local start time.
        end: Exclusive local end time.
    """

    timezone: str
    start: LocalTimeOfDay
    end: LocalTimeOfDay

    def __post_init__(self) -> None:
        """Validate that the active window is non-empty."""
        resolve_timezone(self.timezone)
        if self.start == self.end:
            msg = "active hours start and end must be different"
            raise CronTimeError(msg)

    def to_dict(self) -> ActiveWindowDict:
        """Serialize this active window for disk storage.

        Returns:
            JSON-compatible active-window dictionary.
        """
        return {
            "timezone": self.timezone,
            "start": self.start.to_display(),
            "end": self.end.to_display(),
        }

    @classmethod
    def from_dict(cls, data: ActiveWindowDict) -> ActiveWindow:
        """Deserialize an active window from disk.

        Args:
            data: JSON active-window dictionary.

        Returns:
            Parsed active window.
        """
        return cls(
            timezone=data["timezone"],
            start=LocalTimeOfDay.from_display(data["start"]),
            end=LocalTimeOfDay.from_display(data["end"]),
        )


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
        """Serialize this host clock context for disk storage.

        Returns:
            JSON-compatible host clock dictionary.
        """
        return {
            "scheduler_timezone": self.timezone,
            "scheduler_utc_offset": self.utc_offset,
            "computed_at": _format_time(self.computed_at),
            "local_now": self.local_now.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: SchedulerHostClockDict) -> SchedulerHostClock:
        """Deserialize scheduler host clock context.

        Args:
            data: JSON host clock dictionary.

        Returns:
            Parsed host clock context.
        """
        computed_at = _parse_time(data["computed_at"])
        local_now = datetime.fromisoformat(data.get("local_now", data["computed_at"]))
        if local_now.tzinfo is None:
            local_now = local_now.replace(tzinfo=UTC)
        return cls(
            timezone=data.get("scheduler_timezone", data.get("timezone", "local")),
            utc_offset=data.get("scheduler_utc_offset", data.get("utc_offset", "+00:00")),
            computed_at=computed_at,
            local_now=local_now,
        )


def parse_local_time(value: str) -> LocalTimeOfDay:
    """Parse a supported local time value.

    Args:
        value: Time text such as `20:00`, `8:00pm`, `8pm`, or `08:30`.

    Returns:
        Parsed local time.

    Raises:
        CronTimeError: If the time is malformed.
    """
    match = _LOCAL_TIME_RE.match(value.lower())
    if match is None:
        msg = "time must look like '20:00', '8:00pm', or '8pm'"
        raise CronTimeError(msg)

    minute_text = match.group("minute")
    period = match.group("period")
    if minute_text is None and period is None:
        msg = "24-hour times must include minutes, such as '08:30'"
        raise CronTimeError(msg)

    hour = int(match.group("hour"))
    minute = 0 if minute_text is None else int(minute_text)
    if period is None:
        return LocalTimeOfDay(hour=hour, minute=minute)
    if not 1 <= hour <= _HOURS_PER_HALF_DAY:
        msg = "12-hour times must use an hour between 1 and 12"
        raise CronTimeError(msg)
    return LocalTimeOfDay(hour=_convert_12_hour(hour, period), minute=minute)


def parse_active_window(
    *,
    timezone: str | None,
    active_start: str | None,
    active_end: str | None,
) -> ActiveWindow | None:
    """Parse optional active-hour inputs.

    Args:
        timezone: Explicit IANA time zone, or `None` to use the environment default.
        active_start: Inclusive local start time.
        active_end: Exclusive local end time.

    Returns:
        Parsed active window, or `None` when no active hours were supplied.

    Raises:
        CronTimeError: If only one boundary is supplied or the window is invalid.
    """
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
    """Resolve the IANA time zone for a wall-clock schedule.

    Args:
        value: Explicit IANA time zone, or `None` to use `DEEPAGENTS_TALON_TIMEZONE`.

    Returns:
        Validated IANA time zone key.

    Raises:
        CronTimeError: If no IANA time zone is available.
    """
    candidate = value or os.environ.get(TIMEZONE_ENV)
    if candidate is None or not candidate.strip():
        msg = f"timezone is required for local-time schedules; set {TIMEZONE_ENV}"
        raise CronTimeError(msg)
    return resolve_timezone(candidate.strip()).key


def resolve_timezone(value: str) -> ZoneInfo:
    """Load an IANA time zone.

    Args:
        value: IANA time zone key.

    Returns:
        Loaded `ZoneInfo` instance.

    Raises:
        CronTimeError: If the time zone is unknown.
    """
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        msg = f"unknown timezone: {value}"
        raise CronTimeError(msg) from exc


def next_wall_clock_run(
    *,
    now: datetime,
    timezone: str,
    time: LocalTimeOfDay,
    date: date | None = None,
) -> datetime:
    """Return the next UTC run for a local wall-clock schedule.

    Args:
        now: Current timestamp.
        timezone: IANA time zone used for local interpretation.
        time: Local time of day to schedule.
        date: Optional local calendar date.

    Returns:
        UTC timestamp for the next run.

    Raises:
        CronTimeError: If the requested local time is impossible or in the past.
    """
    current = _coerce_utc(now)
    zone = resolve_timezone(timezone)
    local_now = current.astimezone(zone)
    run_date = date or local_now.date()
    target = _localize(run_date, time, zone)
    if date is None and target <= current:
        target = _localize(run_date + timedelta(days=1), time, zone)
    if date is not None and target <= current:
        msg = "dated wall-clock schedule is in the past"
        raise CronTimeError(msg)
    return target


def next_daily_wall_clock_run(
    *,
    previous: datetime | None,
    now: datetime,
    timezone: str,
    time: LocalTimeOfDay,
    active_window: ActiveWindow | None = None,
) -> datetime:
    """Return the next UTC run for a recurring daily wall-clock schedule.

    Args:
        previous: Previously scheduled run, if any.
        now: Current timestamp.
        timezone: IANA time zone used for local interpretation.
        time: Local time of day to schedule.
        active_window: Optional local active-hour gate.

    Returns:
        UTC timestamp for the next daily wall-clock run.
    """
    current = _coerce_utc(now)
    zone = resolve_timezone(timezone)
    if previous is None:
        candidate = next_wall_clock_run(now=current, timezone=timezone, time=time)
    else:
        previous_local = _coerce_utc(previous).astimezone(zone)
        candidate = _localize(previous_local.date() + timedelta(days=1), time, zone)
        while candidate <= current:
            candidate = _localize(candidate.astimezone(zone).date() + timedelta(days=1), time, zone)
    if active_window is not None and not active_window_contains(candidate, active_window):
        msg = "daily wall-clock schedule time falls outside active hours"
        raise CronTimeError(msg)
    return candidate


def next_interval_run(
    *,
    previous: datetime,
    now: datetime,
    minutes: int,
    active_window: ActiveWindow | None,
) -> datetime:
    """Return the next interval run after `now`.

    Args:
        previous: Previous candidate run timestamp.
        now: Current timestamp.
        minutes: Fixed interval in minutes.
        active_window: Optional local active-hour gate.

    Returns:
        UTC timestamp for the next allowed interval run.
    """
    if minutes < 1:
        msg = "interval must be at least 1 minute"
        raise CronTimeError(msg)
    current = _coerce_utc(now)
    candidate = _coerce_utc(previous)
    interval = timedelta(minutes=minutes)
    while candidate <= current:
        candidate += interval
    if active_window is None:
        return candidate
    return next_active_window_time(candidate, active_window)


def active_window_contains(value: datetime, active_window: ActiveWindow) -> bool:
    """Return whether a UTC timestamp falls inside an active window.

    Args:
        value: Timestamp to evaluate.
        active_window: Local active-hour window.

    Returns:
        `True` when the timestamp is inside the local window.
    """
    zone = resolve_timezone(active_window.timezone)
    local = _coerce_utc(value).astimezone(zone)
    minutes = local.hour * 60 + local.minute
    start = active_window.start.minutes_after_midnight
    end = active_window.end.minutes_after_midnight
    if start < end:
        return start <= minutes < end
    return minutes >= start or minutes < end


def next_active_window_time(value: datetime, active_window: ActiveWindow) -> datetime:
    """Move a timestamp forward to the next active window if needed.

    Args:
        value: UTC timestamp to evaluate.
        active_window: Local active-hour window.

    Returns:
        `value` when it is already inside the window, otherwise the next local
        active-window start converted to UTC.
    """
    candidate = _coerce_utc(value)
    if active_window_contains(candidate, active_window):
        return candidate

    zone = resolve_timezone(active_window.timezone)
    local = candidate.astimezone(zone)
    start = active_window.start.minutes_after_midnight
    end = active_window.end.minutes_after_midnight
    current = local.hour * 60 + local.minute
    days = _days_until_window_start(start=start, end=end, current=current)
    run_date = local.date() + timedelta(days=days)
    return _localize(run_date, active_window.start, zone)


def capture_scheduler_host_clock(now: datetime | None = None) -> SchedulerHostClock:
    """Capture the scheduler host clock context.

    Args:
        now: Host-local computation basis, or `None` for the current host time.

    Returns:
        Host clock context for diagnostics.
    """
    local_now = datetime.now().astimezone() if now is None else _coerce_local(now)
    return SchedulerHostClock(
        timezone=_timezone_name(local_now),
        utc_offset=_format_offset(local_now),
        computed_at=local_now.astimezone(UTC),
        local_now=local_now,
    )


def format_local_run(value: datetime, timezone: str) -> str:
    """Format a UTC run timestamp in a schedule time zone.

    Args:
        value: UTC run timestamp.
        timezone: IANA time zone key.

    Returns:
        Local display string for tool output.
    """
    local = _coerce_utc(value).astimezone(resolve_timezone(timezone))
    return f"{local:%Y-%m-%d %H:%M} {timezone}"


def _localize(day: date, value: LocalTimeOfDay, zone: ZoneInfo) -> datetime:
    naive = datetime.combine(day, time(value.hour, value.minute))
    first = naive.replace(tzinfo=zone, fold=0)
    second = naive.replace(tzinfo=zone, fold=1)
    first_valid = _round_trips(first, zone, naive)
    second_valid = _round_trips(second, zone, naive)
    if not first_valid and not second_valid:
        msg = (
            "local time does not exist because of daylight saving transition: "
            f"{day} {value.to_display()}"
        )
        raise CronTimeError(msg)
    if first_valid:
        return first.astimezone(UTC)
    return second.astimezone(UTC)


def _round_trips(value: datetime, zone: ZoneInfo, naive: datetime) -> bool:
    return value.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == naive


def _convert_12_hour(hour: int, period: str) -> int:
    if period == "am":
        return 0 if hour == _HOURS_PER_HALF_DAY else hour
    return _HOURS_PER_HALF_DAY if hour == _HOURS_PER_HALF_DAY else hour + _HOURS_PER_HALF_DAY


def _days_until_window_start(*, start: int, end: int, current: int) -> int:
    if start < end:
        return 0 if current < start else 1
    return 0 if end <= current < start else 1


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
    return _coerce_utc(value).isoformat()


def _parse_time(value: str) -> datetime:
    return _coerce_utc(datetime.fromisoformat(value))


def _coerce_local(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _coerce_utc(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
