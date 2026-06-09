"""Persistent cron job records and storage.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, TypedDict, cast

MIN_GRANULARITY_MINUTES = 1

JobStatus = Literal["ok", "error"]
ScheduleKind = Literal["one_shot", "recurring"]


class CronJobError(ValueError):
    """Raised when a cron job request is invalid."""


class CronOriginDict(TypedDict):
    """Serialized delivery origin for a scheduled job."""

    conversation_id: str
    channel: str | None
    message_id: str | None


class CronScheduleDict(TypedDict):
    """Serialized schedule definition for a job."""

    kind: ScheduleKind
    minutes: int
    display: str


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

    Args:
        kind: Whether the schedule is one-shot or recurring.
        minutes: Delay or interval in minutes.
        display: Human-readable schedule text supplied by the agent.
    """

    kind: ScheduleKind
    minutes: int
    display: str

    def __post_init__(self) -> None:
        """Validate schedule granularity."""
        if self.minutes < MIN_GRANULARITY_MINUTES:
            msg = "cron schedules must be at least 1 minute"
            raise CronJobError(msg)

    @classmethod
    def parse(cls, value: str) -> CronSchedule:
        """Parse a supported schedule string.

        Args:
            value: Schedule text such as `in 30m` or `every 15m`.

        Returns:
            Parsed schedule.

        Raises:
            CronJobError: If the schedule string is unsupported.
        """
        text = " ".join(value.strip().lower().split())
        if text.startswith("in "):
            return cls(kind="one_shot", minutes=_parse_duration_minutes(text[3:]), display=value)
        if text.startswith("every "):
            return cls(kind="recurring", minutes=_parse_duration_minutes(text[6:]), display=value)
        msg = "schedule must look like 'in 30m' or 'every 15m'"
        raise CronJobError(msg)

    def next_after(self, now: datetime) -> datetime:
        """Return the next scheduled run after `now`.

        Args:
            now: Current timestamp.

        Returns:
            Next run timestamp.
        """
        return now + timedelta(minutes=self.minutes)

    def to_dict(self) -> CronScheduleDict:
        """Serialize this schedule for disk storage.

        Returns:
            JSON-compatible schedule dictionary.
        """
        return {"kind": self.kind, "minutes": self.minutes, "display": self.display}

    @classmethod
    def from_dict(cls, data: CronScheduleDict) -> CronSchedule:
        """Deserialize a cron schedule from disk.

        Args:
            data: JSON schedule dictionary.

        Returns:
            Parsed cron schedule.
        """
        return cls(kind=data["kind"], minutes=data["minutes"], display=data["display"])


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
            next_run_at=schedule.next_after(current),
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
            next_run_at = schedule.next_after(current) if schedule is not None else job.next_run_at
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

    next_run_at = cast("datetime", job.next_run_at)
    interval = timedelta(minutes=job.schedule.minutes)
    while next_run_at <= now:
        next_run_at += interval
    return replace(job, repeat=repeat, next_run_at=next_run_at)


def _parse_duration_minutes(value: str) -> int:
    parts = value.split()
    if len(parts) != 1:
        msg = "schedule duration must be a single value such as '30m'"
        raise CronJobError(msg)
    text = parts[0]
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
