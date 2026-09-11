"""Agent-facing clock tool.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from deepagents_talon.timezones import resolve_zone, system_timezone_name

_UNKNOWN_ZONE_NOTE = (
    "The host timezone name could not be determined. The local time and UTC "
    "offset below are correct, but ask the user which IANA timezone to use "
    "before creating a wall-clock schedule."
)


def current_time_snapshot(timezone: str | None = None) -> dict[str, Any]:
    """Describe the current instant in a timezone.

    Args:
        timezone: IANA timezone name. Defaults to the host's local timezone.

    Returns:
        Current date, time, and timezone details.

    Raises:
        TimeZoneError: If `timezone` is not a usable IANA region key.
    """
    now_utc = datetime.now(UTC)
    if timezone is not None:
        return _snapshot(now_utc.astimezone(resolve_zone(timezone)), timezone)
    name = system_timezone_name()
    if name is None:
        # No IANA name available, but the OS still knows the correct offset.
        return _snapshot(now_utc.astimezone(), None) | {"note": _UNKNOWN_ZONE_NOTE}
    return _snapshot(now_utc.astimezone(resolve_zone(name)), name)


def _snapshot(local: datetime, name: str | None) -> dict[str, Any]:
    offset = local.strftime("%z")
    return {
        "utc": local.astimezone(UTC).isoformat(),
        "local": local.isoformat(),
        "date": local.strftime("%Y-%m-%d"),
        "time": local.strftime("%H:%M:%S"),
        "day_of_week": local.strftime("%A"),
        "timezone": name,
        "utc_offset": f"{offset[:3]}:{offset[3:]}" if offset else "+00:00",
        "abbreviation": local.tzname(),
    }


@tool
def current_time(timezone: str | None = None) -> dict[str, Any]:
    """Get the current date, time, and timezone.

    Call this before scheduling anything, or whenever the user says something
    relative like "tomorrow" or "in the morning" -- there is no other way to
    know today's date or which timezone the user is in.

    Args:
        timezone: Optional IANA timezone name, such as `Europe/Berlin`. Defaults
            to the host's local timezone.

    Returns:
        The current `date`, `time`, `day_of_week`, `local` and `utc` timestamps,
        `utc_offset`, `abbreviation`, and IANA `timezone` name. The `timezone`
        value can be passed straight into a `create_job` wall-clock schedule,
        as in `daily at 08:00 <timezone>`. It is `null`, alongside a `note`,
        when the host timezone name could not be determined. Returns an error
        dictionary for an unusable timezone.
    """
    try:
        return current_time_snapshot(timezone)
    except Exception as exc:  # noqa: BLE001  # tool errors are returned to the agent
        return {"error": str(exc)}
