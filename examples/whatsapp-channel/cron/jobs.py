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
