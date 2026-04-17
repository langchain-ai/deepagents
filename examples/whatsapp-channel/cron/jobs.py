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
