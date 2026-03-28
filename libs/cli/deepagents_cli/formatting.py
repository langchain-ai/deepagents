"""Lightweight text-formatting helpers.

Keep this module free of heavy dependencies so it can be imported anywhere
in the CLI without pulling in large frameworks.
"""

from __future__ import annotations


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
