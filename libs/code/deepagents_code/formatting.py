"""Lightweight text-formatting helpers.

Keep this module free of heavy dependencies so it can be imported anywhere
in the app without pulling in large frameworks.
"""

from __future__ import annotations

import locale
import logging
import subprocess  # noqa: S404
import sys
from datetime import UTC, datetime
from functools import cache

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like `"5s"`, `"2.3s"`, `"5m 12s"`, or `"1h 23m 4s"`.
    """
    rounded = round(seconds, 1)
    if rounded < 60:  # noqa: PLR2004
        if rounded % 1 == 0:
            return f"{int(rounded)}s"
        return f"{rounded:.1f}s"
    minutes, secs = divmod(int(rounded), 60)
    if minutes < 60:  # noqa: PLR2004
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def macos_force_24_hour_time() -> bool | None:
    """Read macOS's "24-Hour Time" preference (`AppleICUForce24HourTime`).

    macOS exposes the clock style as a global preference that does not surface
    through the POSIX `LC_TIME` locale (libc `%X` is 24-hour for every macOS
    locale), so it must be read separately.

    Returns:
        `True`/`False` when the preference is explicitly set, or `None` when it
        is unset (the user has never toggled it) or cannot be read.
    """
    try:
        result = subprocess.run(
            ["/usr/bin/defaults", "read", "-g", "AppleICUForce24HourTime"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Could not read AppleICUForce24HourTime: %s", exc)
        return None
    if result.returncode != 0:
        # Preference unset (key absent) — the user has never toggled it.
        return None
    return result.stdout.strip() == "1"


@cache
def uses_24_hour_clock() -> bool:
    """Whether the system is configured for a 24-hour clock.

    On macOS the clock style lives in the `AppleICUForce24HourTime` system
    preference rather than the POSIX locale, so that is consulted first (see
    `macos_force_24_hour_time`). When that preference is unset we intentionally
    default to a 24-hour clock: macOS resolves the region's 12-/24-hour default
    via ICU/CFLocale, which POSIX cannot read (libc `%X` is 24-hour for every
    macOS locale), so the locale probe below would be meaningless there.

    On every other platform the active `LC_TIME` locale's time representation
    is probed instead.

    The result is cached because resolving it mutates process-global locale
    state via `locale.setlocale(locale.LC_TIME, "")` (scoped to the time
    category, and only ever performed once).

    Returns:
        `True` for a 24-hour clock (also the fallback when the macOS preference
        is unset or no source can be resolved, matching the C locale's 24-hour
        default), `False` for 12-hour.
    """
    if sys.platform == "darwin":
        forced = macos_force_24_hour_time()
        if forced is not None:
            return forced
        # Preference unset: the region default lives in ICU/CFLocale, which the
        # POSIX `%X` probe cannot see. Default to 24-hour rather than misread it.
        return True
    try:
        locale.setlocale(locale.LC_TIME, "")
        # 13:00 renders as "13:..." only where the locale's %X is 24-hour; a
        # 12-hour locale renders "01:..." (often with an AM/PM marker).
        probe = datetime(2000, 1, 1, 13, 0, 0).strftime("%X")  # noqa: DTZ001  # naive probe; tz irrelevant to clock style
    except (locale.Error, ValueError) as exc:
        logger.debug(
            "Could not resolve LC_TIME locale; defaulting to 24-hour clock: %s",
            exc,
        )
        return True
    return "13" in probe


def format_message_timestamp(timestamp: float) -> str | None:
    """Format a message timestamp for display.

    Shows only the time of day for messages from the current local date and
    prefixes the date otherwise. The 12- versus 24-hour clock follows the
    system configuration (see `uses_24_hour_clock`).

    Args:
        timestamp: Unix epoch timestamp.

    Returns:
        A formatted timestamp, or `None` when invalid.
    """
    try:
        dt = datetime.fromtimestamp(timestamp, tz=UTC).astimezone()
    except (ValueError, OSError, OverflowError, TypeError):
        return None
    if uses_24_hour_clock():
        time_str = f"{dt:%H:%M:%S}"
    else:
        time_str = f"{dt.hour % 12 or 12}:{dt:%M:%S} {dt:%p}"
    if dt.date() == datetime.now(tz=dt.tzinfo).date():
        return time_str
    return f"{dt:%b} {dt.day}, {time_str}"
