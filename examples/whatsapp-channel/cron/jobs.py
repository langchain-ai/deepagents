"""Cron job storage, CRUD, and schedule parsing for the WhatsApp channel example.

This module has no dependencies outside the stdlib so it remains importable
in tests without pulling in the agent graph or adapter.
"""

from __future__ import annotations

import re

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
