"""Timezone resolution shared by cron schedules and the clock tool.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

if TYPE_CHECKING:
    from collections.abc import Mapping

LOCALTIME_PATH = Path("/etc/localtime")
"""Symlink into the tz database on most Unix hosts."""

TIMEZONE_PATH = Path("/etc/timezone")
"""File naming the host zone on Debian-derived hosts."""

_ZONEINFO_MARKER = "zoneinfo/"


class TimeZoneError(ValueError):
    """Raised when a timezone name is not a usable IANA key."""


def resolve_zone(name: str) -> ZoneInfo:
    """Look up an IANA timezone by name.

    Legacy POSIX aliases such as `EST5EDT` and bare UTC offsets are rejected:
    they cannot express a region's future daylight-saving rules, so a recurring
    schedule under one would silently drift.

    Args:
        name: Timezone key, such as `America/New_York` or `UTC`.

    Returns:
        Resolved timezone.

    Raises:
        TimeZoneError: If the name is not a usable IANA region key.
    """
    if name != "UTC" and "/" not in name:
        msg = (
            f"timezone must be an IANA region name such as 'America/New_York', "
            f"or 'UTC', not {name!r}"
        )
        raise TimeZoneError(msg)
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        msg = f"unknown timezone {name!r}; use an IANA name such as 'America/New_York'"
        raise TimeZoneError(msg) from exc


def system_timezone_name(
    *,
    env: Mapping[str, str] | None = None,
    localtime_path: Path = LOCALTIME_PATH,
    timezone_path: Path = TIMEZONE_PATH,
) -> str | None:
    """Best-effort IANA name for the host's local timezone.

    The standard library exposes no API for this. `datetime.astimezone()` yields
    only an abbreviation such as `EDT`, which carries no future daylight-saving
    rules, so the name is recovered from the host's own configuration instead:
    the `TZ` environment variable, then the `/etc/localtime` symlink target,
    then `/etc/timezone`. Each candidate is validated, so a host configured with
    a name Talon cannot schedule against falls through to the next source.

    Args:
        env: Environment mapping to read `TZ` from. Defaults to `os.environ`.
        localtime_path: Symlink into the tz database.
        timezone_path: File naming the host zone on Debian-derived hosts.

    Returns:
        Validated IANA timezone name, or `None` if no source resolved.
    """
    values = os.environ if env is None else env
    candidates = (
        _candidate_from_env(values),
        _candidate_from_localtime(localtime_path),
        _candidate_from_timezone_file(timezone_path),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolve_zone(candidate)
        except TimeZoneError:
            continue
        return candidate
    return None


def _candidate_from_env(env: Mapping[str, str]) -> str | None:
    # POSIX allows a leading colon, as in `TZ=:America/New_York`.
    return env.get("TZ", "").strip().lstrip(":").strip() or None


def _candidate_from_localtime(path: Path) -> str | None:
    try:
        if not path.is_symlink():
            return None
        target = str(path.readlink())
    except OSError:
        return None
    return _key_from_localtime_target(target)


def _candidate_from_timezone_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _key_from_localtime_target(target: str) -> str | None:
    """Extract the IANA key from a `/etc/localtime` symlink target.

    The tz database root varies by platform -- `/usr/share/zoneinfo` on Debian,
    `/var/db/timezone/zoneinfo` on macOS -- and the link is sometimes relative,
    so the key is whatever follows the last `zoneinfo/` component.

    Args:
        target: Symlink target path.

    Returns:
        Timezone key, or `None` if the target is not a tz database path.
    """
    index = target.rfind(_ZONEINFO_MARKER)
    if index == -1:
        return None
    return target[index + len(_ZONEINFO_MARKER) :].strip("/") or None
