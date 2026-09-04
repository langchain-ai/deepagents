"""Persistent cron job records and storage.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, cast

from deepagents_talon.timezones import TimeZoneError, resolve_zone

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

CRON_STORE_VERSION = 1
"""Schema version of `jobs.json`.

A file at any other version is discarded, including the unversioned bare list
that predates this envelope -- treated as v0, since it was never numbered.
"""

MIN_GRANULARITY_MINUTES = 1
MAX_SCHEDULE_TEXT_LENGTH = 200
"""Upper bound on agent-supplied schedule text, which is persisted and echoed back."""

_INTERVAL_TOKENS = 2
_WALL_CLOCK_TOKENS = 4
_TIME_FIELDS = 2
_MAX_HOUR = 23
_MAX_MINUTE = 59
_ISO_DATE_LENGTH = 10
_DAILY_LOOKAHEAD_DAYS = 3
"""Local dates probed when resolving the next daily fire; two always suffice."""

_GAP_PROBE_MINUTES = 1440
"""Bound on the forward probe across a nonexistent local time; real gaps are under 2h."""

_ZONE_CACHE_SIZE = 128
"""Bound on distinct cached timezones; names originate from agent-supplied text."""

_MIN_MONTH = 1
_MAX_MONTH = 12
_MIN_DAY = 1
_MAX_DAY = 31
_MIN_YEAR = 1
_MAX_YEAR = 9999

_SCHEDULE_FORMS_HELP = (
    "schedule must be 'in 30m', 'every 15m', "
    "'at 2026-09-04 13:30 America/New_York', or 'daily at 08:00 America/New_York'"
)

JobStatus = Literal["ok", "error"]
ScheduleKind = Literal["one_shot", "recurring"]
ScheduleForm = Literal["interval", "at", "daily"]

_FileIdentity = tuple[int, int, int]
"""Store file fingerprint: modification time in ns, size, and inode."""

_JOB_STATUSES: tuple[str, ...] = ("ok", "error")
_SCHEDULE_KINDS: tuple[str, ...] = ("one_shot", "recurring")
_SCHEDULE_FORMS: tuple[str, ...] = ("interval", "at", "daily")


class CronJobError(ValueError):
    """Raised when a cron job request is invalid."""


class CronOriginDict(TypedDict):
    """Serialized delivery origin for a scheduled job."""

    conversation_id: str
    channel: str | None
    message_id: str | None


class CronScheduleDict(TypedDict):
    """Serialized schedule definition for a job, tagged by `form`.

    Every field a schedule computes from is stored as an integer so that
    loading a job never reparses text. `display` is the one string, carried
    verbatim and never parsed, so the agent's own phrasing survives a round
    trip: `every 2h` does not come back as `every 120m`.

    Interval schedules carry `minutes`. Wall-clock schedules carry `timezone`,
    `hour`, and `minute`; the one-shot `at` form adds `year`, `month`, `day`.
    """

    form: ScheduleForm
    kind: ScheduleKind
    display: str
    minutes: NotRequired[int]
    timezone: NotRequired[str]
    year: NotRequired[int]
    month: NotRequired[int]
    day: NotRequired[int]
    hour: NotRequired[int]
    minute: NotRequired[int]


class CronScheduleWireDict(TypedDict):
    """Model-facing schedule payload, using readable text for local wall clocks."""

    kind: ScheduleKind
    display: str
    minutes: NotRequired[int]
    form: NotRequired[ScheduleForm]
    timezone: NotRequired[str]
    local_time: NotRequired[str]
    local_date: NotRequired[str]


class CronRepeatDict(TypedDict):
    """Serialized repeat cap for recurring jobs."""

    times: int | None
    completed: int


class CronJobDict(TypedDict):
    """Serialized cron job record.

    Timestamps are whole seconds since the Unix epoch. Cron granularity is one
    minute, so second precision loses nothing real, and `_coerce_utc` truncates
    on the way in to keep disk round trips exact rather than lossy.
    """

    id: str
    assistant_id: str
    name: str
    prompt: str
    schedule: CronScheduleDict
    repeat: CronRepeatDict
    enabled: bool
    created_at: int
    next_run_at: int | None
    last_run_at: int | None
    last_status: JobStatus | None
    last_error: str | None
    origin: CronOriginDict


class CronJobWireDict(TypedDict):
    """Model-facing job payload, using ISO-8601 timestamps."""

    id: str
    assistant_id: str
    name: str
    prompt: str
    schedule: CronScheduleWireDict
    repeat: CronRepeatDict
    enabled: bool
    created_at: str
    next_run_at: str | None
    last_run_at: str | None
    last_status: JobStatus | None
    last_error: str | None
    origin: CronOriginDict


class CronStoreDict(TypedDict):
    """Versioned envelope wrapping the persisted job list."""

    version: int
    jobs: list[CronJobDict]


@dataclass(frozen=True, slots=True)
class CronOrigin:
    """Conversation that receives scheduled job results.

    Args:
        conversation_id: Channel-specific conversation identifier.
        channel: Optional channel provider name used by hosts with multiple channels.
        message_id: Optional source message id that created or edited the job.
    """

    conversation_id: str
    channel: str | None = None
    message_id: str | None = None

    def to_dict(self) -> CronOriginDict:
        """Serialize this origin for disk storage.

        Returns:
            JSON-compatible origin dictionary.
        """
        return {
            "conversation_id": self.conversation_id,
            "channel": self.channel,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> CronOrigin:
        """Deserialize a cron origin from disk.

        Args:
            data: JSON origin dictionary.

        Returns:
            Parsed cron origin.

        Raises:
            CronJobError: If a field is missing or has the wrong type.
        """
        record = _as_record(data)
        return cls(
            conversation_id=_str_field(record, "conversation_id"),
            channel=_optional_str_field(record, "channel"),
            message_id=_optional_str_field(record, "message_id"),
        )


@dataclass(frozen=True, slots=True)
class CronSchedule:
    """Minute-granularity schedule for a cron job.

    !!! warning "Breaking change"
        The wall-clock fields changed shape. `local_time="08:00"` became the
        integer pair `hour=8, minute=0`, and `local_date` went from a
        `YYYY-MM-DD` string to a `date`. Callers that constructed a wall-clock
        schedule directly must be updated; positional callers are affected
        silently, since strings now land where integers are expected. Prefer
        `parse`, which is the supported way to build any form.

    The first three fields keep the argument positions they have always had, so
    `CronSchedule("recurring", 15, "every 15m")` still constructs an interval
    schedule. Wall-clock forms pass `minutes=None`.

    Args:
        kind: Whether the schedule is one-shot or recurring.
        minutes: Delay or interval in minutes, or `None` for wall-clock forms.
        display: Human-readable schedule text. Canonicalized for wall-clock forms.
        form: Which arithmetic computes the next run.
        timezone: IANA timezone name. Wall-clock schedules only.
        hour: Local wall-clock hour, 0-23. Wall-clock schedules only.
        minute: Local wall-clock minute, 0-59. Wall-clock schedules only.
        local_date: Local calendar date. One-shot wall-clock schedules only.
    """

    kind: ScheduleKind
    minutes: int | None
    display: str
    form: ScheduleForm = "interval"
    timezone: str | None = None
    hour: int | None = None
    minute: int | None = None
    local_date: date | None = None

    def __post_init__(self) -> None:
        """Validate the fields required by this schedule form.

        Raises:
            CronJobError: If the field combination is invalid for `form`.
        """
        if len(self.display) > MAX_SCHEDULE_TEXT_LENGTH:
            msg = f"schedule text must be at most {MAX_SCHEDULE_TEXT_LENGTH} characters"
            raise CronJobError(msg)
        if self.form == "interval":
            self._validate_interval()
        else:
            self._validate_wall_clock()

    def _validate_interval(self) -> None:
        if self.minutes is None:
            msg = "interval schedules require a minute count"
            raise CronJobError(msg)
        if self.minutes < MIN_GRANULARITY_MINUTES:
            msg = "cron schedules must be at least 1 minute"
            raise CronJobError(msg)
        if (
            self.timezone is not None
            or self.hour is not None
            or self.minute is not None
            or self.local_date is not None
        ):
            msg = "interval schedules cannot carry wall-clock fields"
            raise CronJobError(msg)

    def _validate_wall_clock(self) -> None:
        if self.minutes is not None:
            msg = "wall-clock schedules cannot carry a minute count"
            raise CronJobError(msg)
        if self.timezone is None or self.hour is None or self.minute is None:
            msg = "wall-clock schedules require a timezone, an hour, and a minute"
            raise CronJobError(msg)
        _resolve_zone(self.timezone)
        if not 0 <= self.hour <= _MAX_HOUR or not 0 <= self.minute <= _MAX_MINUTE:
            msg = f"{self.hour}:{self.minute} is not a valid 24-hour local time"
            raise CronJobError(msg)
        if self.form == "at":
            if self.local_date is None:
                msg = "one-shot wall-clock schedules require a local date"
                raise CronJobError(msg)
        elif self.local_date is not None:
            msg = "daily schedules cannot carry a local date"
            raise CronJobError(msg)

    @classmethod
    def parse(cls, value: str) -> CronSchedule:
        """Parse a supported schedule string.

        Recognized forms are `in 30m`, `every 15m`,
        `at YYYY-MM-DD HH:MM <tz>`, and `daily at HH:MM <tz>`, where `<tz>` is a
        required IANA timezone name such as `America/New_York`.

        Keywords are matched case-insensitively; the timezone token keeps its
        original case because IANA keys are case-sensitive on Linux.

        Args:
            value: Schedule text.

        Returns:
            Parsed schedule.

        Raises:
            CronJobError: If the schedule string is unsupported.
        """
        if len(value) > MAX_SCHEDULE_TEXT_LENGTH:
            msg = f"schedule text must be at most {MAX_SCHEDULE_TEXT_LENGTH} characters"
            raise CronJobError(msg)
        tokens = value.split()
        if len(tokens) == _INTERVAL_TOKENS:
            head = tokens[0].lower()
            if head == "in":
                return cls(
                    kind="one_shot",
                    display=" ".join(tokens),
                    minutes=_parse_duration_minutes(tokens[1]),
                )
            if head == "every":
                return cls(
                    kind="recurring",
                    display=" ".join(tokens),
                    minutes=_parse_duration_minutes(tokens[1]),
                )
        if len(tokens) == _WALL_CLOCK_TOKENS:
            if tokens[0].lower() == "at":
                return cls._at(tokens[1], tokens[2], tokens[3])
            if tokens[0].lower() == "daily" and tokens[1].lower() == "at":
                return cls._daily(tokens[2], tokens[3])
        raise CronJobError(_SCHEDULE_FORMS_HELP)

    @classmethod
    def _at(cls, date_text: str, time_text: str, zone_name: str) -> CronSchedule:
        local_date = _parse_local_date(date_text)
        hour, minute = _parse_local_time(time_text)
        _resolve_zone(zone_name)
        return cls(
            kind="one_shot",
            minutes=None,
            display=f"at {local_date.isoformat()} {_format_local_time(hour, minute)} {zone_name}",
            form="at",
            timezone=zone_name,
            hour=hour,
            minute=minute,
            local_date=local_date,
        )

    @classmethod
    def _daily(cls, time_text: str, zone_name: str) -> CronSchedule:
        hour, minute = _parse_local_time(time_text)
        _resolve_zone(zone_name)
        return cls(
            kind="recurring",
            minutes=None,
            display=f"daily at {_format_local_time(hour, minute)} {zone_name}",
            form="daily",
            timezone=zone_name,
            hour=hour,
            minute=minute,
        )

    def next_after(self, now: datetime, *, previous: datetime | None = None) -> datetime:
        """Return the next scheduled run after `now`.

        The result is always strictly greater than `now` except for a one-shot
        `at` schedule whose instant has already passed, which is reported by
        `CronJobStore` rather than silently rescheduled.

        Args:
            now: Current timestamp.
            previous: Previous `next_run_at`, supplied when advancing a job that
                just fired. Interval schedules stay phase-locked to it instead
                of drifting by the scheduler's tick latency.

        Returns:
            Next run timestamp in UTC.
        """
        if self.form == "interval":
            return self._next_interval(now, previous)
        zone = _resolve_zone(cast("str", self.timezone))
        hour, minute = cast("int", self.hour), cast("int", self.minute)
        if self.form == "at":
            naive = datetime.combine(cast("date", self.local_date), time(hour, minute))
            return _localize(naive, zone).astimezone(UTC)
        return _next_daily_instant(now, zone, hour, minute)

    def _next_interval(self, now: datetime, previous: datetime | None) -> datetime:
        interval = timedelta(minutes=cast("int", self.minutes))
        if previous is None or previous > now:
            return now + interval
        return previous + interval * ((now - previous) // interval + 1)

    def to_dict(self) -> CronScheduleDict:
        """Serialize this schedule for disk storage.

        Only the keys the schedule form uses are emitted, all of them integers
        apart from `display` and the timezone name.

        Returns:
            JSON-compatible schedule dictionary.
        """
        data: CronScheduleDict = {
            "form": self.form,
            "kind": self.kind,
            "display": self.display,
        }
        if self.minutes is not None:
            data["minutes"] = self.minutes
        if self.form != "interval":
            data["timezone"] = cast("str", self.timezone)
            data["hour"] = cast("int", self.hour)
            data["minute"] = cast("int", self.minute)
        if self.local_date is not None:
            data["year"] = self.local_date.year
            data["month"] = self.local_date.month
            data["day"] = self.local_date.day
        return data

    def to_wire(self) -> CronScheduleWireDict:
        """Render this schedule for the model-facing tool payload.

        Local wall clocks become `HH:MM` and `YYYY-MM-DD` text, which reads
        better to a model than bare integers. Only tool calls pay this cost;
        the scheduler's hot path never touches it.

        Returns:
            JSON-compatible schedule dictionary.
        """
        data: CronScheduleWireDict = {"kind": self.kind, "display": self.display}
        if self.minutes is not None:
            data["minutes"] = self.minutes
        if self.form != "interval":
            data["form"] = self.form
            data["timezone"] = cast("str", self.timezone)
            data["local_time"] = _format_local_time(
                cast("int", self.hour), cast("int", self.minute)
            )
        if self.local_date is not None:
            data["local_date"] = self.local_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: object) -> CronSchedule:
        """Deserialize a cron schedule from disk.

        Every field is type-checked before use: the file is a trust boundary,
        and a malformed record must surface as `CronJobError` rather than an
        arbitrary exception from deep inside the scheduler loop.

        Args:
            data: JSON schedule dictionary.

        Returns:
            Parsed cron schedule.

        Raises:
            CronJobError: If a field is missing or has the wrong type.
        """
        record = _as_record(data)
        return cls(
            kind=cast("ScheduleKind", _literal_field(record, "kind", _SCHEDULE_KINDS)),
            minutes=_optional_int_field(record, "minutes"),
            display=_str_field(record, "display"),
            form=cast("ScheduleForm", _literal_field(record, "form", _SCHEDULE_FORMS)),
            timezone=_optional_str_field(record, "timezone"),
            hour=_optional_int_field(record, "hour"),
            minute=_optional_int_field(record, "minute"),
            local_date=_optional_date_fields(record),
        )


@dataclass(frozen=True, slots=True)
class CronRepeat:
    """Optional cap for recurring cron jobs.

    Args:
        times: Maximum scheduled attempts, or `None` for unlimited recurrence.
        completed: Number of intervals already claimed for execution.
    """

    times: int | None = None
    completed: int = 0

    def __post_init__(self) -> None:
        """Validate repeat cap values."""
        if self.times is not None and self.times < 1:
            msg = "repeat cap must be at least 1"
            raise CronJobError(msg)
        if self.completed < 0:
            msg = "repeat completed count cannot be negative"
            raise CronJobError(msg)

    def claim(self) -> CronRepeat:
        """Return repeat state after claiming one scheduled attempt.

        Returns:
            Updated repeat state.
        """
        return replace(self, completed=self.completed + 1)

    @property
    def exhausted(self) -> bool:
        """Whether the repeat cap has been reached."""
        return self.times is not None and self.completed >= self.times

    def to_dict(self) -> CronRepeatDict:
        """Serialize this repeat state for disk storage.

        Returns:
            JSON-compatible repeat dictionary.
        """
        return {"times": self.times, "completed": self.completed}

    @classmethod
    def from_dict(cls, data: object) -> CronRepeat:
        """Deserialize repeat state from disk.

        Args:
            data: JSON repeat dictionary.

        Returns:
            Parsed repeat state.

        Raises:
            CronJobError: If a field has the wrong type.
        """
        record = _as_record(data)
        return cls(
            times=_optional_int_field(record, "times"),
            completed=_optional_int_field(record, "completed") or 0,
        )


@dataclass(frozen=True, slots=True)
class CronJob:
    """Persistent cron job record.

    Args:
        id: Stable job identifier.
        assistant_id: Owning assistant namespace.
        name: Human-readable label.
        prompt: Prompt passed to the agent when the job fires.
        schedule: Job schedule.
        repeat: Optional repeat cap.
        enabled: Whether this job may run.
        created_at: Creation timestamp.
        next_run_at: Next due timestamp, or `None` for completed jobs.
        last_run_at: Last attempted run timestamp.
        last_status: Last run outcome.
        last_error: Last run error text.
        origin: Conversation that receives results.
    """

    id: str
    assistant_id: str
    name: str
    prompt: str
    schedule: CronSchedule
    repeat: CronRepeat
    enabled: bool
    created_at: datetime
    next_run_at: datetime | None
    last_run_at: datetime | None
    last_status: JobStatus | None
    last_error: str | None
    origin: CronOrigin

    def to_dict(self) -> CronJobDict:
        """Serialize this job for disk storage.

        Returns:
            JSON-compatible job dictionary.
        """
        return {
            "id": self.id,
            "assistant_id": self.assistant_id,
            "name": self.name,
            "prompt": self.prompt,
            "schedule": self.schedule.to_dict(),
            "repeat": self.repeat.to_dict(),
            "enabled": self.enabled,
            "created_at": _to_epoch(self.created_at),
            "next_run_at": _to_optional_epoch(self.next_run_at),
            "last_run_at": _to_optional_epoch(self.last_run_at),
            "last_status": self.last_status,
            "last_error": self.last_error,
            "origin": self.origin.to_dict(),
        }

    def to_wire(self) -> CronJobWireDict:
        """Render this job for the model-facing tool payload.

        Identical to `to_dict` except that timestamps are ISO-8601 text and the
        schedule uses `to_wire`, so what the model reads is unchanged by the
        integer disk encoding.

        Returns:
            JSON-compatible job dictionary.
        """
        return {
            "id": self.id,
            "assistant_id": self.assistant_id,
            "name": self.name,
            "prompt": self.prompt,
            "schedule": self.schedule.to_wire(),
            "repeat": self.repeat.to_dict(),
            "enabled": self.enabled,
            "created_at": _format_time(self.created_at),
            "next_run_at": _format_optional_time(self.next_run_at),
            "last_run_at": _format_optional_time(self.last_run_at),
            "last_status": self.last_status,
            "last_error": self.last_error,
            "origin": self.origin.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> CronJob:
        """Deserialize a cron job from disk.

        Args:
            data: JSON job dictionary.

        Returns:
            Parsed cron job.

        Raises:
            CronJobError: If a field is missing or has the wrong type.
        """
        record = _as_record(data)
        return cls(
            id=_str_field(record, "id"),
            assistant_id=_str_field(record, "assistant_id"),
            name=_str_field(record, "name"),
            prompt=_str_field(record, "prompt"),
            schedule=CronSchedule.from_dict(record.get("schedule")),
            repeat=CronRepeat.from_dict(record.get("repeat")),
            enabled=_bool_field(record, "enabled"),
            created_at=_from_epoch(_int_field(record, "created_at")),
            next_run_at=_from_optional_epoch(_optional_int_field(record, "next_run_at")),
            last_run_at=_from_optional_epoch(_optional_int_field(record, "last_run_at")),
            last_status=cast(
                "JobStatus | None",
                _optional_literal_field(record, "last_status", _JOB_STATUSES),
            ),
            last_error=_optional_str_field(record, "last_error"),
            origin=CronOrigin.from_dict(record.get("origin")),
        )


class CronJobStore:
    """JSON-backed store for assistant-scoped cron jobs."""

    def __init__(self, *, assistant_id: str, cron_dir: Path) -> None:
        """Initialize the store.

        Args:
            assistant_id: Owning assistant namespace.
            cron_dir: Directory that contains `jobs.json`.
        """
        self.assistant_id = assistant_id
        self.cron_dir = cron_dir
        self.path = cron_dir / "jobs.json"
        self._cache: list[CronJob] | None = None
        self._cache_identity: _FileIdentity | None = None

    def create_job(  # noqa: PLR0913  # job creation exposes the persisted CRON_JOB fields
        self,
        *,
        prompt: str,
        schedule: CronSchedule,
        origin: CronOrigin,
        name: str = "",
        repeat_times: int | None = None,
        now: datetime | None = None,
    ) -> CronJob:
        """Create and persist a cron job.

        Args:
            prompt: Prompt passed to the agent when the job fires.
            schedule: Job schedule.
            origin: Conversation that receives results.
            name: Human-readable label.
            repeat_times: Optional cap for recurring jobs.
            now: Creation time override for deterministic tests.

        Returns:
            Created job record.
        """
        current = _coerce_utc(now)
        repeat = CronRepeat(times=repeat_times)
        if schedule.kind == "one_shot" and repeat_times is not None:
            msg = "repeat cap is only valid for recurring jobs"
            raise CronJobError(msg)
        job = CronJob(
            id=uuid.uuid4().hex[:12],
            assistant_id=self.assistant_id,
            name=name,
            prompt=prompt,
            schedule=schedule,
            repeat=repeat,
            enabled=True,
            created_at=current,
            next_run_at=_first_run_at(schedule, current),
            last_run_at=None,
            last_status=None,
            last_error=None,
            origin=origin,
        )
        jobs = [*self.list_jobs(), job]
        self._write_jobs(jobs)
        return job

    def list_jobs(self, *, origin: CronOrigin | None = None) -> list[CronJob]:
        """List jobs, optionally scoped to an origin conversation.

        Args:
            origin: Optional origin scope.

        Returns:
            Stored jobs sorted by creation time.
        """
        jobs = self._read_jobs()
        if origin is None:
            return jobs
        return [job for job in jobs if _same_origin_scope(job.origin, origin)]

    def due_jobs(self, *, now: datetime | None = None) -> list[CronJob]:
        """Return enabled jobs due at or before `now`.

        Args:
            now: Current timestamp override for deterministic tests.

        Returns:
            Due jobs sorted by next run time.
        """
        current = _coerce_utc(now)
        return sorted(
            [
                job
                for job in self.list_jobs()
                if job.enabled and job.next_run_at is not None and job.next_run_at <= current
            ],
            key=lambda job: cast("datetime", job.next_run_at),
        )

    def get_job(self, job_id: str, *, origin: CronOrigin | None = None) -> CronJob | None:
        """Return a job by id.

        Args:
            job_id: Job identifier.
            origin: Optional origin scope.

        Returns:
            Matching job, or `None`.
        """
        return next((job for job in self.list_jobs(origin=origin) if job.id == job_id), None)

    def edit_job(  # noqa: PLR0913  # edit mirrors the agent-facing optional fields
        self,
        job_id: str,
        *,
        origin: CronOrigin,
        name: str | None = None,
        prompt: str | None = None,
        schedule: CronSchedule | None = None,
        enabled: bool | None = None,
        repeat_times: int | None = None,
        now: datetime | None = None,
    ) -> CronJob:
        """Edit a job within the current conversation scope.

        Args:
            job_id: Job identifier.
            origin: Required conversation scope.
            name: Optional replacement label.
            prompt: Optional replacement prompt.
            schedule: Optional replacement schedule.
            enabled: Optional enabled flag.
            repeat_times: Optional replacement repeat cap for recurring jobs.
            now: Timestamp used to recalculate `next_run_at` when schedule changes.

        Returns:
            Updated job.

        Raises:
            CronJobError: If no scoped job matches.
        """
        jobs = self.list_jobs()
        updated: CronJob | None = None
        current = _coerce_utc(now)
        result: list[CronJob] = []
        for job in jobs:
            if job.id != job_id or not _same_origin_scope(job.origin, origin):
                result.append(job)
                continue
            next_run_at = (
                _first_run_at(schedule, current) if schedule is not None else job.next_run_at
            )
            new_schedule = schedule or job.schedule
            new_repeat = job.repeat
            if repeat_times is not None:
                if new_schedule.kind != "recurring":
                    msg = "repeat cap is only valid for recurring jobs"
                    raise CronJobError(msg)
                new_repeat = CronRepeat(times=repeat_times)
            updated = replace(
                job,
                name=job.name if name is None else name,
                prompt=job.prompt if prompt is None else prompt,
                schedule=new_schedule,
                repeat=new_repeat,
                enabled=job.enabled if enabled is None else enabled,
                next_run_at=next_run_at,
            )
            result.append(updated)
        if updated is None:
            msg = f"cron job not found in current conversation: {job_id}"
            raise CronJobError(msg)
        self._write_jobs(result)
        return updated

    def remove_job(self, job_id: str, *, origin: CronOrigin) -> CronJob:
        """Remove a job within the current conversation scope.

        Args:
            job_id: Job identifier.
            origin: Required conversation scope.

        Returns:
            Removed job.

        Raises:
            CronJobError: If no scoped job matches.
        """
        jobs = self.list_jobs()
        removed: CronJob | None = None
        result: list[CronJob] = []
        for job in jobs:
            if job.id == job_id and _same_origin_scope(job.origin, origin):
                removed = job
                continue
            result.append(job)
        if removed is None:
            msg = f"cron job not found in current conversation: {job_id}"
            raise CronJobError(msg)
        self._write_jobs(result)
        return removed

    def advance_next_run(self, job_id: str, *, now: datetime | None = None) -> CronJob | None:
        """Claim the next scheduled interval before running a due job.

        Args:
            job_id: Job identifier.
            now: Current timestamp override for deterministic tests.

        Returns:
            Updated claimed job, or `None` if the job is no longer due.
        """
        current = _coerce_utc(now)
        jobs = self.list_jobs()
        claimed: CronJob | None = None
        result: list[CronJob] = []
        for job in jobs:
            if job.id != job_id:
                result.append(job)
                continue
            if not job.enabled or job.next_run_at is None or job.next_run_at > current:
                result.append(job)
                continue
            claimed = _advance_claimed_job(job, current)
            result.append(claimed)
        if claimed is not None:
            self._write_jobs(result)
        return claimed

    def mark_job_run(
        self,
        job_id: str,
        *,
        status: JobStatus,
        error: str | None = None,
        now: datetime | None = None,
    ) -> CronJob | None:
        """Record a job run outcome.

        Args:
            job_id: Job identifier.
            status: Run outcome.
            error: Optional error text.
            now: Timestamp override for deterministic tests.

        Returns:
            Updated job, or `None` if the job no longer exists.
        """
        current = _coerce_utc(now)
        updated: CronJob | None = None
        result: list[CronJob] = []
        for job in self.list_jobs():
            if job.id != job_id:
                result.append(job)
                continue
            updated = replace(
                job,
                last_run_at=current,
                last_status=status,
                last_error=error,
            )
            result.append(updated)
        if updated is not None:
            self._write_jobs(result)
        return updated

    def prune_completed(
        self,
        *,
        retain_for: timedelta,
        now: datetime | None = None,
    ) -> list[CronJob]:
        """Delete completed jobs older than the retention window.

        Args:
            retain_for: Duration to keep disabled jobs after completion.
            now: Current timestamp override for deterministic tests.

        Returns:
            Removed job records.

        Raises:
            CronJobError: If `retain_for` is negative.
        """
        if retain_for < timedelta(0):
            msg = "cron retention window cannot be negative"
            raise CronJobError(msg)

        cutoff = _coerce_utc(now) - retain_for
        kept: list[CronJob] = []
        removed: list[CronJob] = []
        for job in self.list_jobs():
            reference = job.last_run_at or job.created_at
            if not job.enabled and job.next_run_at is None and reference <= cutoff:
                removed.append(job)
            else:
                kept.append(job)
        if removed:
            self._write_jobs(kept)
        return removed

    def _read_jobs(self) -> list[CronJob]:
        """Return the stored jobs, reparsing only when the file changed.

        The parsed list is cached against the identity of the inode it came
        from, so a scheduler tick that finds nothing due costs a `stat` rather
        than reading and deserializing every record. `_ensure_store` still runs
        first: re-tightening permissions on each access is cheap next to a
        parse, and worth keeping on the read path.

        Callers get a shallow copy. `CronJob` is frozen, but the list container
        must not be shared with the cache.

        Returns:
            Stored jobs, or an empty list if the file is absent or unreadable.
        """
        self._ensure_store()
        identity = self._stat_identity()
        if identity is None:
            self._cache = []
            self._cache_identity = None
            return []
        if self._cache is not None and self._cache_identity == identity:
            return list(self._cache)
        jobs, loaded = self._load_jobs()
        self._cache = jobs
        self._cache_identity = loaded
        return list(jobs)

    def _stat_identity(self) -> _FileIdentity | None:
        try:
            info = self.path.stat()
        except OSError:
            return None
        return (info.st_mtime_ns, info.st_size, info.st_ino)

    def _load_jobs(self) -> tuple[list[CronJob], _FileIdentity | None]:
        """Parse the store file, reporting the identity of what was parsed.

        The identity comes from `fstat` on the open handle rather than a second
        `stat` of the path, so a concurrent atomic replace cannot make the cache
        claim newer content than it holds.

        Returns:
            Parsed jobs and the identity of the inode they were read from.
        """
        try:
            with self.path.open("rb") as handle:
                info = os.fstat(handle.fileno())
                raw = handle.read()
        except OSError:
            logger.warning("Could not read cron store %s", self.path, exc_info=True)
            return [], None
        identity = (info.st_mtime_ns, info.st_size, info.st_ino)
        return _decode_store(raw, path=self.path), identity

    def _write_jobs(self, jobs: list[CronJob]) -> None:
        self._ensure_store()
        store: CronStoreDict = {
            "version": CRON_STORE_VERSION,
            "jobs": [job.to_dict() for job in jobs],
        }
        payload = json.dumps(store, indent=2, sort_keys=True)
        fd, name = tempfile.mkstemp(
            dir=self.cron_dir,
            prefix=".jobs.",
            suffix=".tmp",
            text=True,
        )
        tmp_path = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(payload)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            tmp_path.chmod(0o600)
            tmp_path.replace(self.path)
            self.path.chmod(0o600)
            _fsync_dir(self.cron_dir)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        # Seed the cache from what was just written; no reparse on the next read.
        # This trusts that no other process replaced the file between the rename
        # above and this stat. Cron stores are single-writer, and the surrounding
        # read-all/write-all pattern already offered no cross-process guarantee.
        self._cache = list(jobs)
        self._cache_identity = self._stat_identity()

    def _ensure_store(self) -> None:
        self.cron_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.cron_dir.chmod(0o700)
        if self.path.exists():
            self.path.chmod(0o600)


def _advance_claimed_job(job: CronJob, now: datetime) -> CronJob:
    if job.schedule.kind == "one_shot":
        return replace(job, enabled=False, next_run_at=None)

    repeat = job.repeat.claim()
    if repeat.exhausted:
        return replace(job, repeat=repeat, enabled=False, next_run_at=None)

    next_run_at = job.schedule.next_after(now, previous=job.next_run_at)
    return replace(job, repeat=repeat, next_run_at=next_run_at)


def _first_run_at(schedule: CronSchedule, now: datetime) -> datetime:
    """Resolve a schedule's first run, rejecting a one-shot that already passed.

    Args:
        schedule: Schedule being installed on a job.
        now: Current timestamp.

    Returns:
        First run timestamp in UTC.

    Raises:
        CronJobError: If a one-shot wall-clock schedule resolves to the past.
    """
    next_run_at = schedule.next_after(now)
    if schedule.form == "at" and next_run_at <= now:
        msg = (
            f"{schedule.display!r} is in the past (resolves to {next_run_at.isoformat()}; "
            f"now is {now.isoformat()})"
        )
        raise CronJobError(msg)
    return next_run_at


@functools.lru_cache(maxsize=_ZONE_CACHE_SIZE)
def _resolve_zone(name: str) -> ZoneInfo:
    """Look up an IANA timezone, reporting failures as a cron job error.

    Cached because every schedule validates its zone on construction -- which
    includes every deserialization -- and `next_after` resolves it again on
    each fire. The cache is bounded: names arrive in agent-supplied schedule
    text, so an unbounded map would grow with distinct invalid input.

    Args:
        name: Timezone key, such as `America/New_York` or `UTC`.

    Returns:
        Resolved timezone.

    Raises:
        CronJobError: If the name is not a usable IANA region key.
    """
    try:
        return resolve_zone(name)
    except TimeZoneError as exc:
        raise CronJobError(str(exc)) from exc


def _local_time_exists(naive: datetime, zone: ZoneInfo) -> bool:
    """Report whether a naive local time exists in `zone`.

    A round trip through UTC is the only reliable check. Comparing the `fold=0`
    and `fold=1` offsets does not work: they differ both in a spring-forward gap
    and at an ambiguous fall-back time.

    Args:
        naive: Naive local wall-clock time.
        zone: Timezone to interpret it in.

    Returns:
        Whether the wall clock reads `naive` at that instant.
    """
    aware = naive.replace(tzinfo=zone)
    return aware.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == naive


def _localize(naive: datetime, zone: ZoneInfo) -> datetime:
    """Attach `zone` to a naive local wall-clock time.

    A time skipped by a spring-forward transition is snapped forward to the
    first minute that does exist, so `daily at 02:30` fires at 03:00 on the
    transition day rather than being skipped. An ambiguous fall-back time
    resolves to its earlier (`fold=0`) occurrence, so the job fires once.

    Args:
        naive: Naive local wall-clock time.
        zone: Timezone to interpret it in.

    Returns:
        Aware local datetime.

    Raises:
        CronJobError: If no nearby local time exists, which no real zone causes.
    """
    probe = naive
    for _ in range(_GAP_PROBE_MINUTES):
        if _local_time_exists(probe, zone):
            return probe.replace(tzinfo=zone)
        probe += timedelta(minutes=1)
    msg = f"no valid local time near {naive.isoformat()} in {zone.key}"
    raise CronJobError(msg)


def _next_daily_instant(now: datetime, zone: ZoneInfo, hour: int, minute: int) -> datetime:
    """Return the next local `hour:minute` in `zone` strictly after `now`.

    Each candidate is rebuilt from a local date rather than advanced by 24
    hours, which is what keeps the job at the same wall-clock time across
    daylight-saving transitions.

    Args:
        now: Current timestamp.
        zone: Schedule timezone.
        hour: Local hour.
        minute: Local minute.

    Returns:
        Next run timestamp in UTC.

    Raises:
        CronJobError: If no candidate within the lookahead window qualifies.
    """
    local_date = now.astimezone(zone).date()
    for offset in range(_DAILY_LOOKAHEAD_DAYS):
        naive = datetime.combine(local_date + timedelta(days=offset), time(hour, minute))
        instant = _localize(naive, zone).astimezone(UTC)
        if instant > now:
            return instant
    msg = f"could not find a daily run after {now.isoformat()} in {zone.key}"
    raise CronJobError(msg)


def _parse_local_time(value: str) -> tuple[int, int]:
    """Parse a 24-hour local `HH:MM` time.

    Args:
        value: Time text.

    Returns:
        Hour and minute.

    Raises:
        CronJobError: If the text is not a valid 24-hour time.
    """
    parts = value.split(":")
    if len(parts) != _TIME_FIELDS or not all(part.isdecimal() for part in parts):
        msg = f"local time must look like '08:00', not {value!r}"
        raise CronJobError(msg)
    hour, minute = int(parts[0]), int(parts[1])
    if hour > _MAX_HOUR or minute > _MAX_MINUTE:
        msg = f"{value!r} is not a valid 24-hour local time"
        raise CronJobError(msg)
    return hour, minute


def _format_local_time(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def _parse_local_date(value: str) -> date:
    """Parse a local `YYYY-MM-DD` date.

    Args:
        value: Date text.

    Returns:
        Parsed date.

    Raises:
        CronJobError: If the text is not an ISO calendar date.
    """
    if len(value) != _ISO_DATE_LENGTH:
        msg = f"date must look like '2026-09-04', not {value!r}"
        raise CronJobError(msg)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"date must look like '2026-09-04', not {value!r}"
        raise CronJobError(msg) from exc


def _parse_duration_minutes(value: str) -> int:
    text = value.lower()
    if text.endswith("m"):
        return _positive_int(text[:-1])
    if text.endswith("h"):
        return _positive_int(text[:-1]) * 60
    msg = "schedule duration must use 'm' for minutes or 'h' for hours"
    raise CronJobError(msg)


def _positive_int(value: str) -> int:
    if not value.isdecimal():
        msg = "schedule duration must be a positive integer"
        raise CronJobError(msg)
    number = int(value)
    if number < MIN_GRANULARITY_MINUTES:
        msg = "schedule duration must be at least 1 minute"
        raise CronJobError(msg)
    return number


def _same_origin_scope(left: CronOrigin, right: CronOrigin) -> bool:
    return left.conversation_id == right.conversation_id and left.channel == right.channel


def _coerce_utc(value: datetime | None = None) -> datetime:
    """Normalize a timestamp to whole-second UTC.

    Sub-second precision is dropped here rather than at the serialization
    boundary, so a job's in-memory timestamps always match what a disk round
    trip returns. Cron granularity is one minute, so nothing real is lost.

    Args:
        value: Timestamp to normalize. Defaults to the current time.

    Returns:
        Timezone-aware UTC timestamp with `microsecond` zeroed.
    """
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC, microsecond=0)
    return value.astimezone(UTC).replace(microsecond=0)


def _to_epoch(value: datetime) -> int:
    return int(_coerce_utc(value).timestamp())


def _to_optional_epoch(value: datetime | None) -> int | None:
    return None if value is None else _to_epoch(value)


def _from_epoch(value: int) -> datetime:
    try:
        return datetime.fromtimestamp(value, UTC)
    except (OverflowError, OSError, ValueError) as exc:
        msg = f"{value} is not a valid epoch timestamp"
        raise CronJobError(msg) from exc


def _from_optional_epoch(value: int | None) -> datetime | None:
    return None if value is None else _from_epoch(value)


def _format_optional_time(value: datetime | None) -> str | None:
    return None if value is None else _format_time(value)


def _format_time(value: datetime) -> str:
    return _coerce_utc(value).isoformat()


def _decode_store(raw: bytes, *, path: Path) -> list[CronJob]:
    """Decode the store file, treating anything unreadable as an empty store.

    A malformed file must not raise: `_read_jobs` sits under the scheduler's
    tick loop, which has no exception handler, so a throw here would silently
    kill the ticker for the life of the process. Returning empty degrades to
    "no jobs scheduled" and lets the next write heal the file, at the cost of
    dropping whatever could not be read -- hence the loud log.

    Args:
        raw: File contents.
        path: Store path, for diagnostics.

    Returns:
        Parsed jobs, or an empty list if the file cannot be read.
    """
    try:
        data = json.loads(raw)
    except ValueError:
        logger.warning("Discarding cron store %s: not valid JSON", path, exc_info=True)
        return []
    if not isinstance(data, dict):
        logger.warning(
            "Discarding cron store %s: expected a JSON object, found %s",
            path,
            type(data).__name__,
        )
        return []
    version = data.get("version")
    if version != CRON_STORE_VERSION:
        logger.warning(
            "Discarding cron store %s: schema version %r is not %d; scheduled jobs are lost",
            path,
            version,
            CRON_STORE_VERSION,
        )
        return []
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        logger.warning(
            "Discarding cron store %s: 'jobs' must be a list, found %s",
            path,
            type(jobs).__name__,
        )
        return []
    try:
        return [CronJob.from_dict(item) for item in jobs]
    except CronJobError:
        logger.exception("Discarding cron store %s: a job record is malformed", path)
        return []


_MISSING = object()
"""Sentinel distinguishing an absent field from a stored `null`."""

_Record = dict[str, object]


def _as_record(data: object) -> _Record:
    """Confirm a decoded JSON value is an object before reading fields from it.

    Called once per record rather than once per field: re-checking the same
    mapping for every field is what makes a strict loader slower than the text
    parsing it replaced.

    Args:
        data: Value expected to be a JSON object.

    Returns:
        The value as a string-keyed mapping.

    Raises:
        CronJobError: If the value is not a JSON object.
    """
    if not isinstance(data, dict):
        msg = f"cron record must be a JSON object, not {type(data).__name__}"
        raise CronJobError(msg)
    # JSON objects always key by string, so the cast is sound once this holds.
    return cast("_Record", data)


def _str_field(record: _Record, key: str) -> str:
    value = record.get(key, _MISSING)
    if not isinstance(value, str):
        raise _field_error(key, value, "a string")
    return value


def _optional_str_field(record: _Record, key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _field_error(key, value, "a string or null")
    return value


def _int_field(record: _Record, key: str) -> int:
    value = record.get(key, _MISSING)
    # `bool` is a subclass of `int`; a JSON `true` is not an acceptable count.
    if not isinstance(value, int) or isinstance(value, bool):
        raise _field_error(key, value, "an integer")
    return value


def _optional_int_field(record: _Record, key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise _field_error(key, value, "an integer or null")
    return value


def _bool_field(record: _Record, key: str) -> bool:
    value = record.get(key, _MISSING)
    if not isinstance(value, bool):
        raise _field_error(key, value, "a boolean")
    return value


def _literal_field(record: _Record, key: str, allowed: tuple[str, ...]) -> str:
    value = _str_field(record, key)
    if value not in allowed:
        raise _field_error(key, value, f"one of {allowed}")
    return value


def _optional_literal_field(record: _Record, key: str, allowed: tuple[str, ...]) -> str | None:
    value = _optional_str_field(record, key)
    if value is None or value in allowed:
        return value
    raise _field_error(key, value, f"one of {allowed} or null")


def _field_error(key: str, value: object, expected: str) -> CronJobError:
    """Build the error for a field that is absent or of the wrong type.

    Args:
        key: Field name.
        value: Offending value, or `_MISSING` when the field is absent.
        expected: Description of what the field should have held.

    Returns:
        Error to raise at the call site, so the traceback points there.
    """
    if value is _MISSING:
        return CronJobError(f"cron record is missing required field {key!r}")
    return CronJobError(f"cron field {key!r} must be {expected}, not {type(value).__name__}")


def _optional_date_fields(record: _Record) -> date | None:
    """Rebuild a local date from its integer components.

    Args:
        record: Schedule dictionary.

    Returns:
        Parsed date, or `None` when the schedule carries no date.

    Raises:
        CronJobError: If the components are partial, out of range, or not a
            real calendar date.
    """
    year = _optional_int_field(record, "year")
    month = _optional_int_field(record, "month")
    day = _optional_int_field(record, "day")
    if year is None and month is None and day is None:
        return None
    if year is None or month is None or day is None:
        msg = "a schedule date requires all of 'year', 'month', and 'day'"
        raise CronJobError(msg)
    if (
        not _MIN_YEAR <= year <= _MAX_YEAR
        or not _MIN_MONTH <= month <= _MAX_MONTH
        or not _MIN_DAY <= day <= _MAX_DAY
    ):
        msg = f"{year}-{month}-{day} is not a valid calendar date"
        raise CronJobError(msg)
    try:
        return date(year, month, day)
    except ValueError as exc:
        msg = f"{year}-{month}-{day} is not a valid calendar date"
        raise CronJobError(msg) from exc


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
