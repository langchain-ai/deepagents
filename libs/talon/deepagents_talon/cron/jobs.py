"""Persistent cron job records and storage.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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

_SCHEDULE_FORMS_HELP = (
    "schedule must be 'in 30m', 'every 15m', "
    "'at 2026-09-04 13:30 America/New_York', or 'daily at 08:00 America/New_York'"
)

JobStatus = Literal["ok", "error"]
ScheduleKind = Literal["one_shot", "recurring"]
ScheduleForm = Literal["interval", "at", "daily"]


class CronJobError(ValueError):
    """Raised when a cron job request is invalid."""


class CronOriginDict(TypedDict):
    """Serialized delivery origin for a scheduled job."""

    conversation_id: str
    channel: str | None
    message_id: str | None


class CronScheduleDict(TypedDict):
    """Serialized schedule definition for a job.

    Only `kind` and `display` are always present. Interval schedules carry
    `minutes`; wall-clock schedules carry `form`, `timezone`, `local_time`, and
    (for one-shots) `local_date`.
    """

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
    """Serialized cron job record."""

    id: str
    assistant_id: str
    name: str
    prompt: str
    schedule: CronScheduleDict
    repeat: CronRepeatDict
    enabled: bool
    created_at: str
    next_run_at: str | None
    last_run_at: str | None
    last_status: JobStatus | None
    last_error: str | None
    origin: CronOriginDict


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
    def from_dict(cls, data: CronOriginDict) -> CronOrigin:
        """Deserialize a cron origin from disk.

        Args:
            data: JSON origin dictionary.

        Returns:
            Parsed cron origin.
        """
        return cls(
            conversation_id=data["conversation_id"],
            channel=data.get("channel"),
            message_id=data.get("message_id"),
        )


@dataclass(frozen=True, slots=True)
class CronSchedule:
    """Minute-granularity schedule for a cron job.

    The first three fields keep the argument positions this class has always
    had, so `CronSchedule("recurring", 15, "every 15m")` still constructs an
    interval schedule. Wall-clock forms pass `minutes=None`.

    Args:
        kind: Whether the schedule is one-shot or recurring.
        minutes: Delay or interval in minutes, or `None` for wall-clock forms.
        display: Human-readable schedule text. Canonicalized for wall-clock forms.
        form: Which arithmetic computes the next run.
        timezone: IANA timezone name. Wall-clock schedules only.
        local_time: Local `HH:MM` wall-clock time. Wall-clock schedules only.
        local_date: Local `YYYY-MM-DD` date. One-shot wall-clock schedules only.
    """

    kind: ScheduleKind
    minutes: int | None
    display: str
    form: ScheduleForm = "interval"
    timezone: str | None = None
    local_time: str | None = None
    local_date: str | None = None

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
        if self.timezone is not None or self.local_time is not None or self.local_date is not None:
            msg = "interval schedules cannot carry wall-clock fields"
            raise CronJobError(msg)

    def _validate_wall_clock(self) -> None:
        if self.minutes is not None:
            msg = "wall-clock schedules cannot carry a minute count"
            raise CronJobError(msg)
        if self.timezone is None or self.local_time is None:
            msg = "wall-clock schedules require a timezone and a local time"
            raise CronJobError(msg)
        _resolve_zone(self.timezone)
        _parse_local_time(self.local_time)
        if self.form == "at":
            if self.local_date is None:
                msg = "one-shot wall-clock schedules require a local date"
                raise CronJobError(msg)
            _parse_local_date(self.local_date)
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
        local_time = _format_local_time(_parse_local_time(time_text))
        _resolve_zone(zone_name)
        return cls(
            kind="one_shot",
            minutes=None,
            display=f"at {local_date.isoformat()} {local_time} {zone_name}",
            form="at",
            timezone=zone_name,
            local_time=local_time,
            local_date=local_date.isoformat(),
        )

    @classmethod
    def _daily(cls, time_text: str, zone_name: str) -> CronSchedule:
        local_time = _format_local_time(_parse_local_time(time_text))
        _resolve_zone(zone_name)
        return cls(
            kind="recurring",
            minutes=None,
            display=f"daily at {local_time} {zone_name}",
            form="daily",
            timezone=zone_name,
            local_time=local_time,
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
        hour, minute = _parse_local_time(cast("str", self.local_time))
        if self.form == "at":
            local_date = _parse_local_date(cast("str", self.local_date))
            naive = datetime.combine(local_date, time(hour, minute))
            return _localize(naive, zone).astimezone(UTC)
        return _next_daily_instant(now, zone, hour, minute)

    def _next_interval(self, now: datetime, previous: datetime | None) -> datetime:
        interval = timedelta(minutes=cast("int", self.minutes))
        if previous is None or previous > now:
            return now + interval
        return previous + interval * ((now - previous) // interval + 1)

    def to_dict(self) -> CronScheduleDict:
        """Serialize this schedule for disk storage.

        Only the keys the schedule form uses are emitted.

        Returns:
            JSON-compatible schedule dictionary.
        """
        data: CronScheduleDict = {"kind": self.kind, "display": self.display}
        if self.minutes is not None:
            data["minutes"] = self.minutes
        if self.form != "interval":
            data["form"] = self.form
            data["timezone"] = cast("str", self.timezone)
            data["local_time"] = cast("str", self.local_time)
        if self.local_date is not None:
            data["local_date"] = self.local_date
        return data

    @classmethod
    def from_dict(cls, data: CronScheduleDict) -> CronSchedule:
        """Deserialize a cron schedule from disk.

        Args:
            data: JSON schedule dictionary.

        Returns:
            Parsed cron schedule.
        """
        return cls(
            kind=data["kind"],
            minutes=data.get("minutes"),
            display=data["display"],
            form=data.get("form", "interval"),
            timezone=data.get("timezone"),
            local_time=data.get("local_time"),
            local_date=data.get("local_date"),
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
    def from_dict(cls, data: CronRepeatDict) -> CronRepeat:
        """Deserialize repeat state from disk.

        Args:
            data: JSON repeat dictionary.

        Returns:
            Parsed repeat state.
        """
        return cls(times=data.get("times"), completed=data.get("completed", 0))


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
            "created_at": _format_time(self.created_at),
            "next_run_at": _format_optional_time(self.next_run_at),
            "last_run_at": _format_optional_time(self.last_run_at),
            "last_status": self.last_status,
            "last_error": self.last_error,
            "origin": self.origin.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: CronJobDict) -> CronJob:
        """Deserialize a cron job from disk.

        Args:
            data: JSON job dictionary.

        Returns:
            Parsed cron job.
        """
        return cls(
            id=data["id"],
            assistant_id=data["assistant_id"],
            name=data["name"],
            prompt=data["prompt"],
            schedule=CronSchedule.from_dict(data["schedule"]),
            repeat=CronRepeat.from_dict(data["repeat"]),
            enabled=data["enabled"],
            created_at=_parse_time(data["created_at"]),
            next_run_at=_parse_optional_time(data["next_run_at"]),
            last_run_at=_parse_optional_time(data["last_run_at"]),
            last_status=data["last_status"],
            last_error=data["last_error"],
            origin=CronOrigin.from_dict(data["origin"]),
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
        self._ensure_store()
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            msg = "cron jobs file must contain a JSON list"
            raise CronJobError(msg)
        return [CronJob.from_dict(cast("CronJobDict", item)) for item in data]

    def _write_jobs(self, jobs: list[CronJob]) -> None:
        self._ensure_store()
        payload = json.dumps(
            [job.to_dict() for job in jobs],
            indent=2,
            sort_keys=True,
        )
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


def _resolve_zone(name: str) -> ZoneInfo:
    """Look up an IANA timezone by name.

    Legacy POSIX aliases such as `EST5EDT` and bare UTC offsets are rejected:
    they cannot express a region's future daylight-saving rules, so a recurring
    schedule under one would silently drift.

    Args:
        name: Timezone key, such as `America/New_York` or `UTC`.

    Returns:
        Resolved timezone.

    Raises:
        CronJobError: If the name is not a usable IANA region key.
    """
    if name != "UTC" and "/" not in name:
        msg = (
            f"timezone must be an IANA region name such as 'America/New_York', "
            f"or 'UTC', not {name!r}"
        )
        raise CronJobError(msg)
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        msg = f"unknown timezone {name!r}; use an IANA name such as 'America/New_York'"
        raise CronJobError(msg) from exc


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


def _format_local_time(parts: tuple[int, int]) -> str:
    return f"{parts[0]:02d}:{parts[1]:02d}"


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
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_optional_time(value: datetime | None) -> str | None:
    return None if value is None else _format_time(value)


def _format_time(value: datetime) -> str:
    return _coerce_utc(value).isoformat()


def _parse_optional_time(value: str | None) -> datetime | None:
    return None if value is None else _parse_time(value)


def _parse_time(value: str) -> datetime:
    return _coerce_utc(datetime.fromisoformat(value))


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
