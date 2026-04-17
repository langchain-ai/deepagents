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
