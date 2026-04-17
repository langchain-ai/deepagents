"""Cron job storage, CRUD, and schedule parsing for the WhatsApp channel example.

This module has no dependencies outside the stdlib so it remains importable
in tests without pulling in the agent graph or adapter.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

_DURATION_RE = re.compile(
    r"^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$",
    re.IGNORECASE,
)
_UNIT_TO_MINUTES = {"m": 1, "h": 60, "d": 1440}


def parse_duration(s: str) -> int:
    """Parse a duration string like ``"30m"`` / ``"2h"`` / ``"1d"`` into minutes.

    Raises ``ValueError`` on bad input, empty string, or zero/negative values.
    """
    if not isinstance(s, str):
        raise ValueError(f"Duration must be a string, got {type(s).__name__}")
    match = _DURATION_RE.match(s.strip())
    if not match:
        raise ValueError(
            f"Invalid duration: '{s}'. Use formats like '30m', '2h', '1d'."
        )
    value = int(match.group(1))
    if value <= 0:
        raise ValueError(f"Duration must be positive, got {value}")
    unit = match.group(2)[0].lower()
    return value * _UNIT_TO_MINUTES[unit]


from datetime import datetime, timedelta
from typing import Any


def _now_aware() -> datetime:
    """Return current time as a timezone-aware datetime in the local tz."""
    return datetime.now().astimezone()


def parse_schedule(s: str) -> dict[str, Any]:
    """Parse a schedule string.

    Supported forms:
      - ``"30m"`` / ``"2h"`` / ``"1d"``  — one-shot, run once at *now + duration*
      - ``"every 15m"`` / ``"every 2h"`` — recurring interval

    Returns a dict:
      - one-shot: ``{"kind": "once", "run_at": <ISO tz-aware>, "display": "once in 30m"}``
      - interval: ``{"kind": "interval", "minutes": N, "display": "every Nm"}``

    Raises ``ValueError`` on invalid input.
    """
    if not isinstance(s, str):
        raise ValueError(f"Schedule must be a string, got {type(s).__name__}")
    original = s.strip()
    if not original:
        raise ValueError(
            "Schedule is empty. Use '30m', '2h', '1d', or 'every 30m' / 'every 2h'."
        )
    lower = original.lower()
    if lower.startswith("every "):
        duration = original[len("every "):].strip()
        if not duration:
            raise ValueError(
                "'every' requires a duration, e.g. 'every 30m' or 'every 2h'."
            )
        minutes = parse_duration(duration)
        return {"kind": "interval", "minutes": minutes, "display": f"every {minutes}m"}
    try:
        minutes = parse_duration(original)
    except ValueError:
        raise ValueError(
            f"Invalid schedule '{original}'. Use:\n"
            f"  - duration: '30m', '2h', '1d' (one-shot)\n"
            f"  - interval: 'every 30m', 'every 2h' (recurring)"
        )
    run_at = _now_aware() + timedelta(minutes=minutes)
    return {
        "kind": "once",
        "run_at": run_at.isoformat(),
        "display": f"once in {original}",
    }


def compute_next_run(
    schedule: dict[str, Any],
    last_run_at: str | None = None,
) -> str | None:
    """Compute the next run time as an ISO string, or ``None`` if no more runs.

    For ``"once"``: returns ``schedule["run_at"]`` on first call (``last_run_at`` is
    ``None``), ``None`` afterwards.

    For ``"interval"``: first run is ``now + minutes``; subsequent runs are
    ``last_run_at + minutes``.
    """
    kind = schedule["kind"]
    if kind == "once":
        if last_run_at is None:
            return schedule["run_at"]
        return None
    if kind == "interval":
        minutes = int(schedule["minutes"])
        if last_run_at is None:
            base = _now_aware()
        else:
            base = datetime.fromisoformat(last_run_at)
            if base.tzinfo is None:
                base = base.astimezone()
        return (base + timedelta(minutes=minutes)).isoformat()
    raise ValueError(f"Unknown schedule kind: {kind!r}")


logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    """Create *path* (including parents) with 0700 perms. Best-effort on Windows."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass


def load_jobs(jobs_path: Path) -> list[dict[str, Any]]:
    """Load jobs from *jobs_path*. Returns ``[]`` if the file doesn't exist.

    On a first-pass ``json.JSONDecodeError``, retries once with ``strict=False``
    to tolerate bare control characters from legacy writes. If that still
    fails, raises ``RuntimeError``.
    """
    jobs_path = Path(jobs_path)
    if not jobs_path.exists():
        return []
    raw = jobs_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(raw, strict=False)
            logger.warning("cron: recovered jobs.json via strict=False")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"cron: jobs.json is corrupt and unrepairable: {e}") from e
    return list(data.get("jobs", []))


def save_jobs(jobs_path: Path, jobs: list[dict[str, Any]]) -> None:
    """Write *jobs* to *jobs_path* atomically (tempfile + fsync + rename)."""
    jobs_path = Path(jobs_path)
    _ensure_dir(jobs_path.parent)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(jobs_path.parent), prefix=".jobs_", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {"jobs": jobs, "updated_at": _now_aware().isoformat()},
                f,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, jobs_path)
        try:
            os.chmod(jobs_path, 0o600)
        except (OSError, NotImplementedError):
            pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def create_job(
    jobs_path: Path,
    *,
    prompt: str,
    schedule: str,
    origin: dict[str, Any],
    name: str | None = None,
    repeat: int | None = None,
) -> dict[str, Any]:
    """Create, persist, and return a new cron job.

    ``origin`` must contain at least ``chat_id``; ``message_id`` may be ``None``.
    For one-shot schedules, ``repeat`` defaults to 1 (run once). For interval
    schedules, ``repeat=None`` means run forever.
    """
    parsed = parse_schedule(schedule)
    if repeat is not None and repeat <= 0:
        repeat = None
    if parsed["kind"] == "once" and repeat is None:
        repeat = 1

    label = (prompt or "cron job").strip()
    job_name = name.strip() if name else label[:50].strip()
    now = _now_aware().isoformat()

    job = {
        "id": uuid.uuid4().hex[:12],
        "name": job_name,
        "prompt": prompt,
        "schedule": parsed,
        "repeat": {"times": repeat, "completed": 0},
        "enabled": True,
        "created_at": now,
        "next_run_at": compute_next_run(parsed),
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "origin": {
            "chat_id": origin["chat_id"],
            "message_id": origin.get("message_id"),
        },
    }

    jobs = load_jobs(jobs_path)
    jobs.append(job)
    save_jobs(jobs_path, jobs)
    return job


def list_jobs_for_chat(jobs_path: Path, chat_id: str) -> list[dict[str, Any]]:
    """Return all jobs whose ``origin.chat_id`` matches *chat_id*."""
    return [
        j for j in load_jobs(jobs_path)
        if j.get("origin", {}).get("chat_id") == chat_id
    ]


def get_job(jobs_path: Path, job_id: str) -> dict[str, Any] | None:
    """Return the job with id *job_id*, or ``None``."""
    for j in load_jobs(jobs_path):
        if j.get("id") == job_id:
            return j
    return None


def remove_job(jobs_path: Path, job_id: str, *, chat_id: str) -> bool:
    """Remove the job with id *job_id* if it belongs to *chat_id*.

    Returns ``True`` if removed, ``False`` if not found or owned by a
    different chat. The caller cannot distinguish "not found" from
    "belongs to another chat" — this is intentional so users in one chat
    cannot probe for jobs in others.
    """
    jobs = load_jobs(jobs_path)
    for i, j in enumerate(jobs):
        if j.get("id") == job_id and j.get("origin", {}).get("chat_id") == chat_id:
            jobs.pop(i)
            save_jobs(jobs_path, jobs)
            return True
    return False


def mark_job_run(
    jobs_path: Path,
    job_id: str,
    *,
    success: bool,
    error: str | None = None,
) -> None:
    """Record the outcome of a run and compute the next ``next_run_at``.

    For finite-repeat interval jobs, increments ``completed`` and removes the
    job when ``completed >= times``. For one-shot jobs, sets ``enabled=False``
    once ``next_run_at`` is ``None``. A job id that is not found is a no-op.
    """
    jobs = load_jobs(jobs_path)
    for i, j in enumerate(jobs):
        if j.get("id") != job_id:
            continue
        now = _now_aware().isoformat()
        j["last_run_at"] = now
        j["last_status"] = "ok" if success else "error"
        j["last_error"] = None if success else error

        repeat = j.setdefault("repeat", {"times": None, "completed": 0})
        repeat["completed"] = repeat.get("completed", 0) + 1
        times = repeat.get("times")
        kind = j.get("schedule", {}).get("kind")
        # Interval jobs with a finite repeat limit are removed when exhausted.
        # One-shot jobs are kept but disabled so the record is preserved.
        if kind == "interval" and times is not None and times > 0 and repeat["completed"] >= times:
            jobs.pop(i)
            save_jobs(jobs_path, jobs)
            return

        j["next_run_at"] = compute_next_run(j["schedule"], now)
        if j["next_run_at"] is None:
            j["enabled"] = False

        save_jobs(jobs_path, jobs)
        return
    logger.warning("cron: mark_job_run: job %s not found", job_id)


def advance_next_run(jobs_path: Path, job_id: str) -> bool:
    """Advance ``next_run_at`` for an *interval* job to the next future tick.

    Used before a run to give at-most-once semantics for recurring jobs — a
    crash mid-run won't cause a re-fire. One-shot jobs are left unchanged so
    they can retry on restart. Returns ``True`` if ``next_run_at`` was updated.
    """
    jobs = load_jobs(jobs_path)
    for j in jobs:
        if j.get("id") != job_id:
            continue
        kind = j.get("schedule", {}).get("kind")
        if kind != "interval":
            return False
        new_next = compute_next_run(j["schedule"], _now_aware().isoformat())
        if new_next and new_next != j.get("next_run_at"):
            j["next_run_at"] = new_next
            save_jobs(jobs_path, jobs)
            return True
        return False
    return False


def get_due_jobs(jobs_path: Path) -> list[dict[str, Any]]:
    """Return enabled jobs whose ``next_run_at`` is at or before now, FIFO."""
    now = _now_aware()
    due: list[dict[str, Any]] = []
    for j in load_jobs(jobs_path):
        if not j.get("enabled", True):
            continue
        next_run = j.get("next_run_at")
        if not next_run:
            continue
        dt = datetime.fromisoformat(next_run)
        if dt.tzinfo is None:
            dt = dt.astimezone()
        if dt <= now:
            due.append(j)
    due.sort(key=lambda j: j["next_run_at"])
    return due
