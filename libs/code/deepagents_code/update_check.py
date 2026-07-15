"""Update lifecycle for `deepagents-code`.

Handles version checking against PyPI (with caching), install-method detection,
auto-upgrade execution, config-driven opt-in/out, notification throttling, and
"what's new" tracking.

Most public entry points absorb errors and return sentinel values.
`set_auto_update` raises on write failures so callers can surface
actionable feedback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import operator
import os
import re
import shlex
import shutil
import signal
import sys
import tempfile
import time
import tomllib
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, TextIO

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from deepagents_code._version import PYPI_URL, SDK_PYPI_URL, USER_AGENT, __version__
from deepagents_code.model_config import DEFAULT_CONFIG_PATH, DEFAULT_STATE_DIR

logger = logging.getLogger(__name__)

CACHE_FILE: Path = DEFAULT_STATE_DIR / "latest_version.json"
"""On-disk cache of the latest published dcode/SDK versions and SDK release times.

Populated by `get_latest_version`; reads short-circuit on the cached payload
when it is younger than `CACHE_TTL`. SDK upload timestamps are stored under
`_SDK_RELEASE_TIMES_KEY`.
"""

UPDATE_STATE_FILE: Path = DEFAULT_STATE_DIR / "update_state.json"
"""Persistent flags for the update-notification UX.

Tracks which version the user has been notified about (`notified_version`,
`notified_at`) and the most recent version they've seen the splash for
(`seen_version`, `seen_at`). Read by `should_notify_update` and friends
to suppress repeat notifications across invocations. Auto-update opt-outs
live in `config.toml`, not here.
"""

CACHE_TTL = 86_400  # 24 hours
"""Maximum age in seconds before `CACHE_FILE` entries are considered stale.

A cached `latest_version.json` younger than this is reused without an HTTP
call to PyPI; older payloads trigger a fresh fetch. Set conservatively at
24h since release cadence is on the order of days, not minutes.
"""

INSTALLED_AGE_NOTICE_DAYS = 7
"""Minimum installed-version age before update notices call it out explicitly."""

INSTALLED_STALE_NOTICE_DAYS = 14
"""Minimum installed-version age (days) before the stale-install banner shows."""

_SDK_RELEASE_TIMES_KEY = "sdk_release_times"
"""`CACHE_FILE` key for cached SDK upload timestamps, keyed by version string."""

_RELEASE_PRERELEASE_DEPS_KEY = "release_requires_prereleases"
"""`CACHE_FILE` key for release versions that require pre-release dependencies."""

InstallMethod = Literal["uv", "brew", "other", "unknown"]

FALLBACK_UPGRADE_COMMAND = "uv tool install -U deepagents-code"
"""Generic upgrade hint used when install-method detection fails.

Callers that surface an upgrade command in user-facing text should prefer
`upgrade_command()`; this constant exists so those callers have something
to render when detection raises unexpectedly. The documented install path
is `uv tool install` (see `scripts/install.sh`), so the uv command is the
right display fallback. Uses `uv tool install -U` rather than `uv tool
upgrade` for the same receipt-pin reason documented on `_UPGRADE_COMMANDS`:
showing a user the `upgrade` form would hand them a command that silently
stays on the old version for a pinned install. Execution paths still refuse
unrecognized installs instead of updating a separate environment.
"""

_UPGRADE_COMMANDS: dict[InstallMethod, str] = {
    # Use `uv tool install -U` instead of `uv tool upgrade`: the latter
    # *respects* the requirement string baked into the uv tool receipt by the
    # original install (or by any prior `dependency_refresh_command` that
    # wrote `deepagents-code==<old_version>` into the receipt). When that
    # requirement is pinned, `uv tool upgrade` "succeeds" but re-installs the
    # same pinned version, silently leaving the user behind latest. A bare
    # `uv tool install -U deepagents-code` rewrites the receipt's requirement
    # to an unpinned `deepagents-code` and re-resolves to the latest stable
    # release, which is what users running `/update` actually want.
    # `dependency_refresh_command` builds the inverse command for the
    # explicit "stay on this version, refresh deps" flow.
    "uv": FALLBACK_UPGRADE_COMMAND,
    "brew": "brew upgrade deepagents-code",
}
"""Upgrade commands keyed by install method.

`perform_upgrade` runs only the command matching the detected install method;
no fallback chain. Unknown non-editable installs are refused rather than
upgraded with a different package manager, because that can update a separate
environment from the one currently providing `dcode`.
"""

_UV_PRERELEASE_UPGRADE_COMMAND = f"{FALLBACK_UPGRADE_COMMAND} --prerelease allow"
"""uv upgrade command that opts into alpha/beta/rc release resolution.

Uses `uv tool install -U` (not `uv tool upgrade`) for the same receipt-pin
reason documented on `_UPGRADE_COMMANDS`.
"""

_PRERELEASE_UNSUPPORTED_MESSAGE = (
    "Pre-release updates aren't supported for this install. Reinstall with "
    "pre-releases enabled:\n"
    '  curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_PRERELEASE="allow" bash'
)
"""User-facing reason a pre-release upgrade is refused on non-uv installs.

Points at the install script (uv under the hood) rather than raw uv commands,
since that one-liner is the path we promote.
"""

_UPGRADE_TIMEOUT = 120  # seconds
_BROWSER_RUNTIME_INSTALL_TIMEOUT = 600  # seconds
"""Wall-clock cap for `perform_upgrade` and `perform_install_extra`."""

_TERMINATE_WAIT_TIMEOUT = 5  # seconds
"""Cap on reaping a killed install subprocess so cleanup cannot hang launch."""

INSTALL_SCRIPT_COMMAND = "curl -LsSf https://langch.in/dcode | bash"
"""Promoted public install command for Deep Agents Code."""

UPDATE_LOG_DIR: Path = DEFAULT_STATE_DIR / "update_logs"
"""Directory for persisted update command logs."""

UPDATE_LOG_RETENTION_DAYS = 14
"""Delete update logs older than this many days."""

UPDATE_LOG_MAX_FILES = 10
"""Keep at most this many newest update logs."""

STARTUP_AUTO_UPDATE_FAILURE_COOLDOWN = CACHE_TTL
"""Seconds to suppress same-version startup auto-update retries after failure."""

RESUME_AUTO_UPDATE_GRACE_PERIOD = 7 * 24 * 60 * 60
"""Seconds resumed sessions may bypass the startup auto-update path."""

_STARTUP_AUTO_UPDATE_FAILED_VERSION_KEY = "startup_auto_update_failed_version"
_STARTUP_AUTO_UPDATE_FAILED_AT_KEY = "startup_auto_update_failed_at"
_STARTUP_AUTO_UPDATE_FAILURE_KEYS: tuple[str, ...] = (
    _STARTUP_AUTO_UPDATE_FAILED_VERSION_KEY,
    _STARTUP_AUTO_UPDATE_FAILED_AT_KEY,
)
_RESUME_AUTO_UPDATE_DEFERRED_AT_KEY = "resume_auto_update_deferred_at"

UpgradeProgressCallback = Callable[[str], Awaitable[None] | None]


def _parse_version(v: str) -> Version:
    """Parse a PEP 440 version string into a comparable `Version` object.

    Supports stable (`1.2.3`) and pre-release (`1.2.3a1`, `1.2.3rc2`) versions.

    Args:
        v: Version string like `'1.2.3'` or `'1.2.3a1'`.

    Returns:
        A `packaging.version.Version` instance.
    """
    return Version(v.strip())  # raises InvalidVersion for non-PEP 440 strings


def is_installed_version_at_least(version: str) -> bool:
    """Return whether installed package metadata is at least `version`."""
    try:
        from importlib.metadata import PackageNotFoundError, version as pkg_version

        installed = _parse_version(pkg_version("deepagents-code"))
        target = _parse_version(version)
    except (InvalidVersion, PackageNotFoundError):
        return False
    return installed >= target


def _latest_from_releases(
    releases: Mapping[str, Sequence[object]],
    *,
    include_prereleases: bool,
) -> str | None:
    """Pick the newest version from a PyPI `releases` mapping.

    Skips versions with no uploaded files (empty entries) and, when
    *include_prereleases* is `False`, skips pre-release versions.

    Args:
        releases: The `releases` dict from the PyPI JSON API.
        include_prereleases: Whether to consider pre-release versions.

    Returns:
        The highest matching version string, or `None` if none qualify.
    """
    best: Version | None = None
    best_str: str | None = None
    for ver_str, files in releases.items():
        if not files:
            continue
        try:
            ver = Version(ver_str)
        except InvalidVersion:
            logger.debug("Skipping unparseable release key: %s", ver_str)
            continue
        if not include_prereleases and ver.is_prerelease:
            continue
        if best is None or ver > best:
            best = ver
            best_str = ver_str
    return best_str


def get_cached_update_available() -> tuple[bool, str | None]:
    """Check for updates using only a fresh local cache entry.

    This is the startup fast path: it never contacts PyPI. Stale, missing,
    corrupt, or unparsable cache data is treated as "no cached update answer" so
    callers can launch immediately and let a background update check refresh the
    cache later.

    Returns:
        A `(available, latest)` tuple. `latest` is `None` when the cache cannot
            provide a fresh answer.
    """
    try:
        installed = _parse_version(__version__)
    except InvalidVersion:
        logger.warning(
            "Installed version %r is not PEP 440 compliant; "
            "cache-only update checks disabled for this install",
            __version__,
        )
        return False, None

    cache_key = "version_prerelease" if installed.is_prerelease else "version"
    try:
        if not CACHE_FILE.exists():
            return False, None
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False, None
        checked_at = data.get("checked_at")
        checked_at = _coerce_checked_at(checked_at)
        if checked_at is None:
            return False, None
        if time.time() - checked_at >= CACHE_TTL:
            return False, None
        value = data.get(cache_key)
        if not isinstance(value, str):
            return False, None
        return _parse_version(value) > installed, value
    except (OSError, json.JSONDecodeError, TypeError, InvalidVersion):
        logger.debug("Failed to read cache-only update answer", exc_info=True)
        return False, None


def _coerce_checked_at(value: object) -> float | None:
    """Return a valid epoch timestamp from cached state, or `None`."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    checked_at = float(value)
    if not math.isfinite(checked_at):
        return None
    try:
        datetime.fromtimestamp(checked_at, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    return checked_at


def get_last_update_check_time() -> float | None:
    """Return the epoch time of the last PyPI update check, or `None`.

    Reads the `checked_at` stamp recorded in `CACHE_FILE` when the update cache
    is written (primarily by `get_latest_version`; also seeded by
    `_write_release_requires_prereleases`). Missing, corrupt, or non-numeric
    data fail-soft to `None` so callers can render an "unknown" state without
    contacting the network.
    """
    try:
        if not CACHE_FILE.exists():
            return None
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        checked_at = data.get("checked_at")
    except (OSError, json.JSONDecodeError, TypeError):
        logger.debug("Failed to read last update check time", exc_info=True)
        return None
    return _coerce_checked_at(checked_at)


def _requires_prerelease_dependency(requirements: Sequence[object] | None) -> bool:
    """Return whether any requirement specifier names a pre-release version.

    Accepts the raw `Requires-Dist` list from PyPI metadata, which may contain
    non-string or non-PEP-508 junk; such entries are skipped rather than raising
    so one malformed line cannot poison the whole check.

    The check is intentionally operator- and marker-agnostic: it returns `True`
    if *any* specifier across *any* requirement pins a pre-release version,
    regardless of the operator (`==`, `>=`, even `!=`) or environment markers
    (extras, `python_version`). This errs toward `True`, which is the safe
    direction — opting `uv` into `--prerelease allow` still resolves stable
    releases correctly, so a false positive only widens the candidate set and
    never strands a user. Do not "tighten" this to the dangerous direction
    without revisiting the fallback asymmetry in `release_requires_prereleases`.
    """
    if not requirements:
        return False
    for raw in requirements:
        if not isinstance(raw, str):
            continue
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            logger.debug("Skipping unparseable Requires-Dist entry: %r", raw)
            continue
        for specifier in requirement.specifier:
            try:
                version = Version(specifier.version)
            except InvalidVersion:
                logger.debug(
                    "Skipping unparseable requirement version: %r",
                    specifier.version,
                )
                continue
            if version.is_prerelease:
                return True
    return False


def _atomic_write_cache(data: dict[str, Any]) -> None:
    """Write `data` to `CACHE_FILE` as JSON atomically.

    A plain `write_text` truncates the file before writing, so a crash or a
    concurrent reader/writer can observe a half-written cache. Serializing to a
    sibling temp file and `os.replace`-ing it into place makes the swap atomic,
    so readers always see either the old or new contents — never a partial one.

    Raises:
        OSError: If the cache directory or temp file cannot be written, or the
            atomic replace fails. Callers handle reporting.
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=CACHE_FILE.parent, prefix=".latest_version-", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        tmp_path.replace(CACHE_FILE)
    except OSError:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def _write_release_requires_prereleases(version: str, requires: bool) -> None:
    """Cache whether a release needs uv's pre-release resolver opt-in."""
    try:
        data: dict[str, Any]
        if CACHE_FILE.exists():
            loaded = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {}
        else:
            data = {}
        values = data.get(_RELEASE_PRERELEASE_DEPS_KEY)
        if not isinstance(values, dict):
            values = {}
        values[version] = requires
        data[_RELEASE_PRERELEASE_DEPS_KEY] = values
        data.setdefault("checked_at", time.time())
        _atomic_write_cache(data)
    except (OSError, json.JSONDecodeError, TypeError):
        logger.debug(
            "Failed to write release pre-release dependency cache",
            exc_info=True,
        )


def get_latest_version(
    *,
    bypass_cache: bool = False,
    include_prereleases: bool = False,
) -> str | None:
    """Fetch the latest deepagents-code version from PyPI, with caching.

    Results are cached to `CACHE_FILE` to avoid repeated network calls.
    The cache stores both the latest stable and pre-release versions so a
    single PyPI request serves both code paths.

    Args:
        bypass_cache: Skip the cache and always hit PyPI.
        include_prereleases: When `True`, consider pre-release versions
            (alpha, beta, rc). Stable users should leave this `False`.

    Returns:
        The latest version string, or `None` on any failure.
    """
    cache_key = "version_prerelease" if include_prereleases else "version"
    cached_version: str | None = None

    try:
        if not bypass_cache and CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            fresh = time.time() - data.get("checked_at", 0) < CACHE_TTL
            if fresh and cache_key in data:
                value = data[cache_key]
                cached_version = value if isinstance(value, str) else None
            release_times = data.get("release_times")
            has_installed_release_time = (
                isinstance(release_times, dict) and __version__ in release_times
            )
            if fresh and cache_key in data and has_installed_release_time:
                return cached_version
    except (OSError, json.JSONDecodeError, TypeError):
        logger.debug("Failed to read update-check cache", exc_info=True)

    try:
        import requests
    except ImportError:
        logger.warning(
            "requests package not installed — update checks disabled. "
            "Install with: uv tool install --reinstall -U deepagents-code "
            "--with requests"
        )
        return cached_version

    try:
        resp = requests.get(
            PYPI_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=3,
        )
        resp.raise_for_status()
        payload = resp.json()
        info = payload.get("info")
        if not isinstance(info, dict):
            logger.debug("PyPI response missing object 'info' key")
            return cached_version
        value = info.get("version")
        if not isinstance(value, str):
            logger.debug("PyPI response missing string 'info.version' key")
            return cached_version
        stable = value
        releases: dict[str, list[object]] = payload.get("releases", {})
        if not releases:
            logger.debug("PyPI response missing or empty 'releases' key")
        prerelease = _latest_from_releases(releases, include_prereleases=True)
        stable_requires_prereleases = _requires_prerelease_dependency(
            info.get("requires_dist")
        )
    except (requests.RequestException, OSError, KeyError, json.JSONDecodeError):
        logger.debug("Failed to fetch latest version from PyPI", exc_info=True)
        return cached_version

    release_times = _extract_release_times(
        payload, stable=stable, prerelease=prerelease, installed=__version__
    )

    # Preserve per-version pre-release-dependency entries written by
    # `_write_release_requires_prereleases` for *other* versions; this refresh
    # only knows the answer for `stable`, so merge rather than overwrite the map
    # (otherwise a routine check would evict a cached answer and force a re-fetch
    # that, on a PyPI hiccup, falls back to the unsafe stable-only default).
    prerelease_deps: dict[str, Any] = {}
    try:
        if CACHE_FILE.exists():
            existing = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                cached_deps = existing.get(_RELEASE_PRERELEASE_DEPS_KEY)
                if isinstance(cached_deps, dict):
                    prerelease_deps = cached_deps
    except (OSError, json.JSONDecodeError, TypeError):
        logger.debug("Failed to read cached pre-release deps before refresh")
    prerelease_deps[stable] = stable_requires_prereleases

    try:
        _atomic_write_cache(
            {
                "version": stable,
                "version_prerelease": prerelease,
                "release_times": release_times,
                _RELEASE_PRERELEASE_DEPS_KEY: prerelease_deps,
                "checked_at": time.time(),
            }
        )
    except OSError:
        logger.debug("Failed to write update-check cache", exc_info=True)

    return prerelease if include_prereleases else stable


def release_requires_prereleases(
    version: str | None,
    *,
    bypass_cache: bool = False,
) -> bool:
    """Return whether installing `version` needs uv pre-release resolution.

    Args:
        version: `deepagents-code` version to inspect.
        bypass_cache: Skip cached release metadata and fetch PyPI directly.

    Returns:
        `True` when the release metadata pins or bounds a pre-release dependency.

    Note:
        On any lookup failure (no `requests`, network/parse error) this returns
        `False` — i.e. "stable-only resolution". That is deliberately
        conservative rather than fail-safe: the truly safe default would be to
        allow pre-releases, but a spurious `True` would make
        `prerelease_upgrade_supported` *refuse* the upgrade outright on non-uv
        installs (Homebrew/other), regressing the common case. `False` keeps
        those installs upgradable; the cost is that, during a PyPI outage, a
        stable release that genuinely pins a pre-release dependency may be
        installed stable-only. Failures are logged at `warning` so the blind
        decision is at least visible.
    """
    if not version:
        return False
    try:
        if not bypass_cache and CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                values = data.get(_RELEASE_PRERELEASE_DEPS_KEY)
                if isinstance(values, dict) and isinstance(values.get(version), bool):
                    return values[version]
    except (OSError, json.JSONDecodeError, TypeError):
        logger.debug(
            "Failed to read release pre-release dependency cache",
            exc_info=True,
        )

    try:
        import requests
    except ImportError:
        logger.warning(
            "requests package not installed — cannot check whether v%s pins a "
            "pre-release dependency; assuming stable-only resolution",
            version,
        )
        return False

    try:
        url = f"{PYPI_URL.removesuffix('/json')}/{version}/json"
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=3,
        )
        resp.raise_for_status()
        payload = resp.json()
        info = payload.get("info")
        requires = info.get("requires_dist") if isinstance(info, dict) else None
        result = _requires_prerelease_dependency(requires)
    except (requests.RequestException, OSError, KeyError, json.JSONDecodeError):
        logger.warning(
            "Failed to fetch dependency metadata for v%s from PyPI; assuming "
            "stable-only resolution",
            version,
            exc_info=True,
        )
        return False

    _write_release_requires_prereleases(version, result)
    return result


def _extract_release_times(
    payload: dict[str, Any],
    *,
    stable: str,
    prerelease: str | None,
    installed: str | None = None,
) -> dict[str, str]:
    """Pull `upload_time_iso_8601` for the given versions out of a PyPI payload.

    PyPI lists per-file uploads; the first file's timestamp is used as a
    stand-in for the release's publish time (files typically land within
    seconds of each other). Looks up both versions under `releases[ver]`
    rather than `payload["urls"]`, which reflects the project's
    `info.version` and may not match `stable` when the latest on PyPI is
    a pre-release.

    Args:
        payload: Parsed PyPI JSON response.
        stable: Latest stable version string.
        prerelease: Latest pre-release version string, if any.
        installed: Currently installed version string, if it should be cached.

    Returns:
        Mapping of version string to ISO-8601 upload time. Silently drops
        versions whose timestamp is missing or malformed.
    """
    times: dict[str, str] = {}
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        return times
    for ver in (stable, prerelease, installed):
        if not ver:
            continue
        files = releases.get(ver)
        if not isinstance(files, list) or not files:
            continue
        ts = _upload_time(files[0])
        if ts:
            times[ver] = ts
    return times


def _upload_time(file_entry: object) -> str | None:
    """Return `upload_time_iso_8601` from a PyPI file entry, or `None`."""
    if not isinstance(file_entry, dict):
        return None
    # `isinstance(..., dict)` narrows to `dict[Unknown, Unknown]`, so `.get()`
    # overload resolution is ambiguous. PyPI payloads are str-keyed in practice
    # and the `isinstance(value, str)` check below validates the result anyway.
    value = file_entry.get("upload_time_iso_8601")
    return value if isinstance(value, str) else None


def get_release_time(version: str | None) -> str | None:
    """Return the cached ISO-8601 upload time for `version`, or `None`.

    Only versions captured during a prior `get_latest_version` call are
    available; unknown versions, or a `None` input, return `None`.
    """
    if not version:
        return None
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                times = data.get("release_times")
                if isinstance(times, dict):
                    value = times.get(version)
                    if isinstance(value, str):
                        return value
    except (OSError, json.JSONDecodeError):
        logger.debug("Failed to read release_times from cache", exc_info=True)
    return None


def _format_age_from_iso(iso: str | None) -> str:
    """Return `'released Nd ago'` for an ISO-8601 timestamp, or `""` on failure."""
    if not iso:
        return ""
    from deepagents_code.sessions import format_relative_timestamp

    age = format_relative_timestamp(iso)
    return f"released {age}" if age else ""


def format_release_age(version: str | None) -> str:
    """Return a human-readable age for `version` (e.g., `'released 3d ago'`).

    Returns an empty string when the upload time is unknown (cache entry
    lacks `release_times` for this version, or a `None` version) so callers
    can concatenate unconditionally.
    """
    return _format_age_from_iso(get_release_time(version))


def format_age_suffix(version: str | None) -> str:
    """Return `", released Nd ago"` for `version`, or `""` when unknown.

    The `", "` separator is included so callers can splice the age into a
    parenthetical unconditionally — if the age is unknown, the empty
    string collapses cleanly into the surrounding text.
    """
    age = format_release_age(version)
    return f", {age}" if age else ""


def format_release_age_parenthetical(version: str | None) -> str:
    """Return `" (released Nd ago)"` for `version`, or `""` when unknown."""
    age = format_release_age(version)
    return f" ({age})" if age else ""


def _days_old_from_iso(iso: str | None) -> int | None:
    """Return whole elapsed days for an ISO-8601 timestamp, or `None` on failure."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso).astimezone()
    except (ValueError, TypeError):
        logger.debug(
            "Failed to parse release timestamp %r for installed age",
            iso,
            exc_info=True,
        )
        return None

    days = (datetime.now(tz=dt.tzinfo) - dt).days
    return max(days, 0)


def format_installed_age_suffix(version: str | None) -> str:
    """Return `" (N days old)"` for installed versions at least a week old."""
    days = _days_old_from_iso(get_release_time(version))
    if days is None or days < INSTALLED_AGE_NOTICE_DAYS:
        return ""
    unit = "day" if days == 1 else "days"
    return f" ({days} {unit} old)"


def installed_days_old() -> int | None:
    """Return whole days since the installed version's release, or `None`.

    Cache-only (`get_release_time`), so it never blocks on the network.
    Returns `None` when the release time is unknown (cold cache or an
    unparseable timestamp). Best-effort: any unexpected error degrades to
    `None` rather than propagating, since this runs on the startup path.
    """
    try:
        return _days_old_from_iso(get_release_time(__version__))
    except Exception:
        logger.debug("Failed to compute installed version age", exc_info=True)
        return None


def is_installation_stale() -> bool:
    """Return whether an older installed version should warn about updating.

    `True` only when a fresh cache says an update is available and the installed
    version's release is at least `INSTALLED_STALE_NOTICE_DAYS` old. Returns
    `False` for editable/dev installs (release age is meaningless there), when
    update checks are disabled (the opt-out silences this too), and when the age
    or update answer is unknown.

    Best-effort: any unexpected error degrades to `False` rather than
    propagating, so this cosmetic check can never abort TUI startup.
    """
    from deepagents_code.config import _is_editable_install

    try:
        if _is_editable_install() or not is_update_check_enabled():
            return False
        available, _ = get_cached_update_available()
        if not available:
            return False
        days = installed_days_old()
    except Exception:
        logger.debug("Failed to determine installation staleness", exc_info=True)
        return False
    return days is not None and days >= INSTALLED_STALE_NOTICE_DAYS


def get_sdk_release_time(
    version: str | None, *, bypass_cache: bool = False
) -> str | None:
    """Return the ISO-8601 upload time for `deepagents` SDK `version`.

    Reads from `CACHE_FILE` under `sdk_release_times`, falling back to a
    single PyPI fetch on cache miss and writing the result back so
    subsequent calls stay local.

    Args:
        version: Installed SDK version string.
        bypass_cache: Skip the cache read and always hit PyPI.

            The result is still written back to the cache.

    Returns:
        The ISO-8601 upload timestamp, or `None` on any failure (missing
            version, unresolvable on PyPI, `requests` unavailable, or
            network error).
    """
    if not version:
        return None

    try:
        if not bypass_cache and CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                times = data.get(_SDK_RELEASE_TIMES_KEY)
                if isinstance(times, dict):
                    cached = times.get(version)
                    if isinstance(cached, str):
                        return cached
    except (OSError, json.JSONDecodeError):
        logger.debug("Failed to read sdk release_times from cache", exc_info=True)

    try:
        import requests
    except ImportError:
        logger.debug("requests unavailable — SDK release time lookup disabled")
        return None

    try:
        resp = requests.get(
            SDK_PYPI_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=3,
        )
        resp.raise_for_status()
        payload = resp.json()
        releases = payload.get("releases")
        if not isinstance(releases, dict):
            return None
        files = releases.get(version)
        if not isinstance(files, list) or not files:
            return None
        iso = _upload_time(files[0])
    except (requests.RequestException, OSError, json.JSONDecodeError):
        logger.debug("Failed to fetch SDK release time from PyPI", exc_info=True)
        return None

    if iso:
        _write_sdk_release_time(version, iso)
    return iso


def _write_sdk_release_time(version: str, iso: str) -> None:
    """Merge a single SDK release timestamp into `CACHE_FILE`.

    A corrupt existing cache is overwritten rather than propagating the
    decode error — otherwise every caller would keep paying the PyPI
    round-trip because the write never succeeds.
    """
    data: dict[str, object] = {}
    if CACHE_FILE.exists():
        try:
            raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(
                "SDK release-time cache is corrupt; overwriting", exc_info=True
            )
        except OSError:
            logger.debug("Failed to read SDK release-time cache", exc_info=True)
            return
        else:
            if isinstance(raw, dict):
                data = raw

    times: dict[str, str] = {}
    existing = data.get(_SDK_RELEASE_TIMES_KEY)
    if isinstance(existing, dict):
        times.update(
            {
                k: v
                for k, v in existing.items()
                if isinstance(k, str) and isinstance(v, str)
            }
        )
    times[version] = iso
    data[_SDK_RELEASE_TIMES_KEY] = times
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        logger.debug("Failed to write SDK release time to cache", exc_info=True)


def format_sdk_release_age(version: str | None) -> str:
    """Return a human-readable age for SDK `version` (e.g., `'released 3d ago'`).

    May trigger a single PyPI fetch on cache miss (3s timeout). Returns an
    empty string on any failure so callers can concatenate unconditionally.
    """
    return _format_age_from_iso(get_sdk_release_time(version))


def format_sdk_age_suffix(version: str | None) -> str:
    """Return `", released Nd ago"` for SDK `version`, or `""` when unknown.

    The `", "` separator is included so callers can splice the age into a
    line unconditionally — if the age is unknown, the empty string
    collapses cleanly into the surrounding text. May trigger a single
    PyPI fetch on cache miss.
    """
    age = format_sdk_release_age(version)
    return f", {age}" if age else ""


def _read_update_state() -> dict[str, object]:
    """Read the shared update state file.

    Returns:
        Parsed dict, or empty dict on missing/corrupt file.
    """
    try:
        if UPDATE_STATE_FILE.exists():
            raw = json.loads(UPDATE_STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
    except (OSError, json.JSONDecodeError):
        logger.debug("Failed to read update state file", exc_info=True)
    return {}


def _write_update_state(
    patch: dict[str, object], *, remove_keys: tuple[str, ...] = ()
) -> bool:
    """Merge *patch* into the shared update state file and drop *remove_keys*.

    Args:
        patch: Keys to merge into the existing state.
        remove_keys: Keys to drop from the existing state before writing.

    Returns:
        `True` if the state was persisted, `False` if the write failed (the
            error is logged, not raised, so callers stay fail-soft but can surface
            the miss when a stale state has user-visible consequences).
    """
    data = _read_update_state()
    for key in remove_keys:
        data.pop(key, None)
    data.update(patch)
    try:
        UPDATE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        UPDATE_STATE_FILE.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        logger.warning(
            "Failed to write update state to %s",
            UPDATE_STATE_FILE,
            exc_info=True,
        )
        return False
    return True


def should_defer_startup_auto_update_for_resume() -> bool:
    """Return whether a resumed session is still in its update grace period.

    The first resumed launch starts a fixed grace period. Later resumes do not
    extend it, so a resume-only workflow eventually follows the normal startup
    auto-update path. If the marker cannot be persisted, fail closed and run
    the update path rather than allowing an unbounded bypass.
    """
    data = _read_update_state()
    now = time.time()
    deferred_at = _coerce_checked_at(data.get(_RESUME_AUTO_UPDATE_DEFERRED_AT_KEY))
    if deferred_at is None or deferred_at > now:
        return _write_update_state({_RESUME_AUTO_UPDATE_DEFERRED_AT_KEY: now})
    return now - deferred_at < RESUME_AUTO_UPDATE_GRACE_PERIOD


def clear_resume_auto_update_deferral() -> None:
    """Reset the resume grace period after a normal interactive launch."""
    data = _read_update_state()
    if _RESUME_AUTO_UPDATE_DEFERRED_AT_KEY not in data:
        return
    _write_update_state({}, remove_keys=(_RESUME_AUTO_UPDATE_DEFERRED_AT_KEY,))


def should_skip_startup_auto_update_after_failure(version: str) -> bool:
    """Return whether startup auto-update should skip a recently failed version."""
    data = _read_update_state()
    failed_version = data.get(_STARTUP_AUTO_UPDATE_FAILED_VERSION_KEY)
    failed_at = data.get(_STARTUP_AUTO_UPDATE_FAILED_AT_KEY)
    return bool(
        failed_version == version
        and isinstance(failed_at, (int, float))
        and time.time() - failed_at < STARTUP_AUTO_UPDATE_FAILURE_COOLDOWN
    )


def mark_startup_auto_update_failed(version: str) -> bool:
    """Persist a same-version startup auto-update retry cooldown marker.

    Returns:
        `True` if the marker was written, `False` otherwise.
    """
    return _write_update_state(
        {
            _STARTUP_AUTO_UPDATE_FAILED_VERSION_KEY: version,
            _STARTUP_AUTO_UPDATE_FAILED_AT_KEY: time.time(),
        }
    )


def clear_startup_auto_update_failure(version: str) -> None:
    """Clear a startup auto-update failure marker for `version` if present."""
    data = _read_update_state()
    if data.get(_STARTUP_AUTO_UPDATE_FAILED_VERSION_KEY) != version:
        return
    _write_update_state({}, remove_keys=_STARTUP_AUTO_UPDATE_FAILURE_KEYS)


def should_notify_update(latest: str) -> bool:
    """Return whether the user should be notified about version *latest*.

    Throttles notifications to at most once per `CACHE_TTL` period for a
    given version, preventing repeated banners every session.

    Args:
        latest: The version string to check against.

    Returns:
        `True` if the user should see the update banner, `False` if the
            notification was already shown within the `CACHE_TTL` window.
    """
    data = _read_update_state()
    notified_at = data.get("notified_at", 0)
    notified_version = data.get("notified_version")
    return not (
        isinstance(notified_at, (int, float))
        and notified_version == latest
        and time.time() - notified_at < CACHE_TTL
    )


def mark_update_notified(latest: str) -> None:
    """Record that the user was notified about version *latest*.

    Writes into the shared update state file so a subsequent
    `should_notify_update` call can suppress duplicate banners.

    Args:
        latest: The version string that was shown.
    """
    _write_update_state({"notified_at": time.time(), "notified_version": latest})


def clear_update_notified() -> None:
    """Clear the "already notified" marker so the update modal re-opens next launch.

    Removes both `notified_at` and `notified_version` from the shared
    update state file.
    """
    _write_update_state({}, remove_keys=("notified_at", "notified_version"))


def is_update_available(
    *,
    bypass_cache: bool = False,
    include_prereleases: bool | None = None,
) -> tuple[bool, str | None]:
    """Check whether a newer version of deepagents-code is available.

    When the installed version is a pre-release (e.g. `0.0.35a1`),
    pre-release versions on PyPI are included in the comparison so alpha
    testers are notified of newer alphas and the eventual stable release.
    Stable installs only compare against stable PyPI releases unless
    `include_prereleases` is explicitly set.

    Args:
        bypass_cache: Skip the cache and always hit PyPI.
        include_prereleases: Override whether alpha/beta/rc releases are
            considered. When `None`, this follows the installed version.

    Returns:
        A `(available, latest)` tuple.

            `latest` is the PyPI version string when it was fetched and parsed
            successfully, or `None` when the PyPI check itself fails (network
            error, unparseable response, or non-PEP 440 installed version).
            `available` is `True` only when `latest` is strictly newer than
            the installed version. Callers can therefore distinguish "already
            up to date" (`(False, "1.2.3")`) from "could not reach PyPI"
            (`(False, None)`).
    """
    try:
        installed = _parse_version(__version__)
    except InvalidVersion:
        logger.warning(
            "Installed version %r is not PEP 440 compliant; "
            "update checks disabled for this install",
            __version__,
        )
        return False, None

    include_prereleases = _resolve_include_prereleases(
        include_prereleases,
        installed=installed,
    )
    latest = get_latest_version(
        bypass_cache=bypass_cache,
        include_prereleases=include_prereleases,
    )
    if latest is None:
        return False, None

    try:
        return _parse_version(latest) > installed, latest
    except InvalidVersion:
        logger.debug("Failed to compare versions", exc_info=True)
        return False, None


# ---------------------------------------------------------------------------
# Install method detection
# ---------------------------------------------------------------------------


def _resolve_include_prereleases(
    include_prereleases: bool | None,
    *,
    installed: Version | None = None,
) -> bool:
    """Resolve update channel preference from the requested or installed channel.

    Args:
        include_prereleases: Explicit channel preference, or `None` to infer
            from the installed version.
        installed: Parsed installed version to reuse when the caller already
            has one.

    Returns:
        `True` when pre-release versions should be considered.
    """
    if include_prereleases is not None:
        return include_prereleases
    if installed is None:
        try:
            installed = _parse_version(__version__)
        except InvalidVersion:
            logger.warning(
                "Installed version %r is not PEP 440 compliant; "
                "defaulting to stable-only upgrades",
                __version__,
            )
            return False
    return installed.is_prerelease


def detect_install_method() -> InstallMethod:
    """Detect how `deepagents-code` was installed.

    Checks `sys.prefix` against known paths for uv and Homebrew.

    Returns:
        The detected install method: `'uv'`, `'brew'`, `'other'`, or `'unknown'`
            (editable/dev installs).
    """
    from deepagents_code.config import _is_editable_install

    prefix = sys.prefix
    # uv tool installs live under ~/.local/share/uv/tools/
    if "/uv/tools/" in prefix or "\\uv\\tools\\" in prefix:
        return "uv"
    # Homebrew prefixes
    if any(
        prefix.startswith(p)
        for p in ("/opt/homebrew", "/usr/local/Cellar", "/home/linuxbrew")
    ):
        return "brew"
    # Editable / dev installs — don't auto-upgrade
    if _is_editable_install():
        return "unknown"
    return "other"


def upgrade_command(
    method: InstallMethod | None = None,
    *,
    include_prereleases: bool | None = None,
    version: str | None = None,
) -> str:
    """Return the shell command to upgrade `deepagents-code`.

    Falls back to the documented uv command for display-only guidance.

    Args:
        method: Install method override.

            Auto-detected if `None`.
        include_prereleases: Whether to include alpha/beta/rc releases. When
            `None`, follows the installed version's channel. When `True`,
            returns the uv pre-release command regardless of `method`, since
            only uv can be steered onto the pre-release channel.
        version: Optional exact `deepagents-code` version pin for uv guidance.
    """
    include_prereleases = _resolve_include_prereleases(include_prereleases)
    if version is not None:
        requirement = _dcode_extras_requirement((), version=version)
        cmd = f"uv tool install -U {requirement}"
        if include_prereleases:
            cmd += " --prerelease allow"
        return cmd
    if include_prereleases:
        return _UV_PRERELEASE_UPGRADE_COMMAND
    if method is None:
        method = detect_install_method()
    return _UPGRADE_COMMANDS.get(method, FALLBACK_UPGRADE_COMMAND)


def prerelease_upgrade_supported(
    method: InstallMethod | None = None,
) -> tuple[bool, str | None]:
    """Return whether pre-release upgrades are supported for the install method.

    Pre-release channel switching is only safe for `uv tool` installs, where
    `uv tool upgrade --prerelease allow` re-resolves against the pre-release
    feed. Other package managers can't be steered onto that channel, so callers
    should refuse before promising an upgrade.

    Args:
        method: Install method override.

            Auto-detected if `None`.

    Returns:
        A `(supported, reason)` tuple. `reason` is `None` when supported, else a
        user-facing explanation of why the pre-release upgrade is refused.
    """
    if method is None:
        method = detect_install_method()
    if method != "uv":
        return False, _PRERELEASE_UNSUPPORTED_MESSAGE
    return True, None


_DEPENDENCY_REFRESH_UNSUPPORTED: dict[InstallMethod, str] = {
    "unknown": "Editable install detected — skipping dependency refresh.",
    "brew": (
        "Homebrew install detected — dependency-only refresh is not "
        "supported without upgrading deepagents-code."
    ),
    "other": (
        "Unsupported install method detected — cannot refresh dependencies "
        "without knowing which environment provides `dcode`."
    ),
}
"""Why each non-uv install method can't do a dependency-only refresh."""


def dependency_refresh_supported(
    method: InstallMethod | None = None,
) -> tuple[bool, str | None]:
    """Return whether a dependency-only refresh is possible for the install.

    A dependency refresh reinstalls the *current* `deepagents-code` version with
    upgraded dependency resolution (`uv tool install -U deepagents-code==<v>`).
    Only uv-managed installs can express that without crossing to a newer app
    version, so callers should refuse before prompting or shelling out. This is
    the single source of truth for both the gate in the TUI and the refusal in
    `perform_dependency_refresh`.

    Args:
        method: Install method override. Auto-detected if `None`.

    Returns:
        A `(supported, reason)` tuple. `reason` is `None` when supported, else a
            user-facing explanation of why the refresh is refused.
    """
    if method is None:
        method = detect_install_method()
    if method == "uv":
        return True, None
    return False, _DEPENDENCY_REFRESH_UNSUPPORTED[method]


@dataclass(frozen=True)
class ShadowedDcode:
    """A different dcode entry point is winning on PATH than the one we upgraded.

    Returned by `detect_shadowed_dcode` after a successful upgrade so the TUI can
    warn the user that re-launching will pick up the wrong binary. The most
    common cause is a pre-uv install (e.g. a leftover from a previous
    `pipx`/`pip`-based install) earlier on `PATH` than the uv tool shims.

    A frozen dataclass rather than a `NamedTuple` (unlike the sibling
    `DependencyChange`) so `__post_init__` can enforce the conflict invariant
    the type's name promises: an instance only exists when there genuinely is
    a shadow. The producer already guarantees this, so the check is defensive
    against future direct construction, not a runtime gate on the hot path.
    """

    shadowing_bin: Path
    """Absolute path to the `dcode` (or `deepagents-code`) binary the user's
    `PATH` currently resolves first — the file their next `dcode` will run.

    Reported as the un-followed `shutil.which` result rather than its symlink
    target, since that's the file the user needs to either delete or demote
    on `PATH`.
    """

    upgraded_bin_dir: Path
    """Absolute path to the bin directory uv installed the upgraded shim into.

    Resolved via uv's documented executable-directory precedence (see
    `_uv_tool_bin_dir`).
    """

    def __post_init__(self) -> None:
        """Reject a non-conflict instance — the type's namesake invariant.

        If the shadowing binary already lives in the upgraded bin dir there is
        no shadow, and a warning built from it would tell the user a binary
        shadows itself. `detect_shadowed_dcode` returns `None` in that case, so
        this only fires on a misconstructed instance.

        Raises:
            ValueError: If `shadowing_bin` already resides in `upgraded_bin_dir`.
        """
        if self.shadowing_bin.parent == self.upgraded_bin_dir:
            msg = (
                f"ShadowedDcode requires a real shadow, but {self.shadowing_bin} "
                f"already resides in the upgraded bin dir {self.upgraded_bin_dir}"
            )
            raise ValueError(msg)

    @property
    def upgraded_bin(self) -> Path:
        """Absolute path to the upgraded `dcode` shim uv installed.

        Keeps the `dcode` entry-point name owned by the type rather than
        re-derived at each call site (mirrors `DependencyChange.kind`).
        """
        return self.upgraded_bin_dir / "dcode"


def _uv_tool_bin_dir() -> Path | None:
    """Return the bin directory uv installed the running `dcode` shim into.

    Mirrors uv's documented executable-directory precedence so a custom
    layout (e.g. `XDG_BIN_HOME` set on Linux) is compared against the same
    directory uv would install into. Following uv's reference at
    https://docs.astral.sh/uv/reference/storage/#executable-directory:

    The precedence (single, unbranched code path): `UV_TOOL_BIN_DIR` →
    `XDG_BIN_HOME` → `$XDG_DATA_HOME/../bin` → the final `.local/bin` under
    the home directory. The last candidate is `Path.home() / ".local" / "bin"`,
    which `pathlib` resolves per-platform — `$HOME/.local/bin` on Unix and
    `%USERPROFILE%/.local/bin` on Windows — so one expression satisfies uv's
    documented fallback on both without an `os.name` branch here.

    The first candidate that exists as a directory wins; an existing but
    unusable candidate (read failures, race) is skipped so a transient
    glitch doesn't downgrade the answer to a less-preferred path.

    Returns:
        The absolute, resolved bin directory, or `None` when no candidate
            exists (e.g. a CI install that never created `~/.local/bin`).
    """

    def _from_env(name: str) -> Path | None:
        raw = os.environ.get(name)
        return Path(raw).expanduser() if raw else None

    home = Path.home()
    xdg_data_home_str = os.environ.get("XDG_DATA_HOME")
    xdg_data_home = (
        Path(xdg_data_home_str).expanduser()
        if xdg_data_home_str
        else home / ".local" / "share"
    )

    # Build candidates in uv's documented precedence order. None entries
    # (env var unset) are filtered out before iteration so each remaining
    # candidate is a real path.
    candidates: list[Path | None] = [
        _from_env("UV_TOOL_BIN_DIR"),
        _from_env("XDG_BIN_HOME"),
        # `$XDG_DATA_HOME/../bin` — uv's documented intermediate fallback.
        # Falls back to the spec default (`~/.local/share`) when the env
        # var is unset; the resulting `~/.local/share/../bin` = `~/.local/bin`
        # matches the final fallback below, which is fine — only the first
        # match wins.
        xdg_data_home.parent / "bin",
        home / ".local" / "bin",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            logger.debug(
                "Could not resolve uv tool bin dir candidate %s",
                candidate,
                exc_info=True,
            )
            continue
        if resolved.is_dir():
            return resolved
    return None


def detect_shadowed_dcode() -> ShadowedDcode | None:
    """Return the shadowing dcode entry point on the user's PATH, if any.

    After a successful `uv tool upgrade`, the upgraded binary only takes effect
    on the next launch if the user's `PATH` resolves to uv's tool bin dir for
    `dcode` (and `deepagents-code`). A pre-uv install earlier on `PATH` will
    silently win and report the old version, which looks like "the upgrade
    didn't work" to the user.

    This compares each supported console script against uv's tool bin dir. A
    mismatch means a different binary will run next launch for that entry point.

    Caveat: a `dcode` symlink that lives in some unrelated bin dir but
    points *into* the upgraded tool venv (e.g. a manually-created
    convenience symlink) is reported as shadowing even though the next
    launch would actually run the upgraded entry point. Comparing
    directories rather than resolved targets is intentional — see the
    inline note below for why — and this edge is rare enough that we
    accept a benign false positive over a class of false negatives.

    Returns:
        A `ShadowedDcode` describing the conflict, or `None` when there is no
            shadowing binary (the common case) or when detection is not
            applicable (non-uv install, uv bin dir unknown, no supported entry
            point on `PATH` at all).
    """
    if detect_install_method() != "uv":
        return None
    upgraded_bin_dir = _uv_tool_bin_dir()
    if upgraded_bin_dir is None:
        return None
    # Check every supported entry point. One healthy command name does not
    # prove another command name cannot still be shadowed earlier on PATH.
    for name in ("dcode", "deepagents-code"):
        resolved = shutil.which(name)
        if resolved is None:
            continue
        # Compare the *PATH-entry directory* against uv's bin dir, NOT the
        # symlink target. uv exposes its tool entry points as symlinks under
        # the user's bin dir (e.g. `~/.local/bin/dcode` -> the tool venv at
        # `~/.local/share/uv/tools/deepagents-code/bin/dcode`). Following the
        # link would make every healthy uv install look shadowed, because the
        # resolved parent is the tool venv's bin dir rather than the
        # PATH-visible one. Take the parent of the un-followed `which`
        # result so we answer the question we actually care about: "is uv's
        # bin dir what PATH resolves to?" `Path(...).parent` does not follow
        # the file's symlink. Only the directory is canonicalized, so
        # benign filesystem aliases (case folding, /private/var vs /var on
        # macOS, mount-point synonyms) still compare equal.
        path_dir = Path(resolved).parent
        try:
            canonical_path_dir = path_dir.resolve()
        except OSError:
            # Couldn't canonicalize the PATH-entry directory (e.g. a stale
            # symlink, a vanished mount). Returning `None` here would
            # silently hide a real shadow, so continue to the next candidate
            # name if any; if this was the last (`deepagents-code`), the loop
            # falls through to `None` — an indeterminate result we'd rather
            # surface to a developer than mask, hence `warning`, not `debug`.
            logger.warning(
                "Could not resolve PATH directory for %s at %s",
                name,
                path_dir,
                exc_info=True,
            )
            continue
        if canonical_path_dir == upgraded_bin_dir:
            # This entry point resolves to the directory uv just wrote into.
            # Keep checking the other supported entry point before declaring
            # there is no shadow.
            continue
        return ShadowedDcode(
            shadowing_bin=Path(resolved),
            upgraded_bin_dir=upgraded_bin_dir,
        )
    return None


def detect_shadowed_dcode_safe() -> ShadowedDcode | None:
    """Best-effort `detect_shadowed_dcode` that never raises.

    The shadow check only ever runs to decorate an already-successful upgrade,
    so a defect in detection — or an unexpected error escaping the narrow
    `OSError` guards inside `detect_shadowed_dcode` — must not turn a working
    upgrade into a user-facing failure. Any unexpected exception is logged and
    treated as "no shadow detected", matching the fail-open bias the detector
    already applies internally.

    Returns:
        Whatever `detect_shadowed_dcode` returns, or `None` if it raised.
    """
    try:
        return detect_shadowed_dcode()
    except Exception:
        logger.warning("Shadow detection failed after upgrade", exc_info=True)
        return None


def format_shadowed_dcode_warning(shadow: ShadowedDcode) -> str:
    """Render a user-facing warning for a shadowed-dcode situation.

    Shared by the `/update` slash command, the update-notification "Install
    now" action, and the pre-launch auto-update path so the wording stays
    consistent.

    Args:
        shadow: The shadowing-binary description returned by
            `detect_shadowed_dcode`.

    Returns:
        A plain-text, multi-line warning suitable for either the TUI message
            stream or a Rich `console.print`.
    """
    fix_command = format_shadowed_dcode_fix_command(shadow)
    indented_command = fix_command.replace("\n", "\n  ")
    return (
        "Update installed, but another `dcode` is earlier on your PATH and "
        "will keep running the old version on relaunch:\n"
        f"  Shadowing binary: {shadow.shadowing_bin}\n"
        f"  Upgraded shim:    {shadow.upgraded_bin}\n"
        "After closing dcode, run this to make the upgraded shim win in this "
        "terminal:\n"
        f"  {indented_command}\n"
        "Then relaunch dcode. To make the fix permanent, add the PATH change "
        "to your shell profile, or uninstall the older dcode if you no longer "
        "need it."
    )


def format_shadowed_dcode_fix_command(shadow: ShadowedDcode) -> str:
    """Return a session-scoped shell command to prefer the upgraded shim.

    The command targets the shell that matches the current platform: PowerShell
    on Windows (where `_uv_tool_bin_dir` can resolve `%USERPROFILE%/.local/bin`
    and `export`/`hash` are not valid), and POSIX `sh`/`bash`/`zsh` elsewhere.

    Args:
        shadow: The shadowing-binary description returned by
            `detect_shadowed_dcode`.

    Returns:
        A copy-pasteable shell command that updates only the current terminal
            session and, on POSIX, clears the shell's command-path cache.
    """
    bin_dir = str(shadow.upgraded_bin_dir)
    if os.name == "nt":
        # PowerShell, the default Windows shell. Single-quote the literal path so
        # `$`, `$()`, and backticks in directory names are not expanded, then
        # concatenate the live session PATH outside the literal. No cache flush is
        # needed — PowerShell resolves executables per invocation rather than
        # caching like POSIX shells' `hash`.
        quoted = bin_dir.replace("'", "''")
        return f"$env:PATH = '{quoted};' + $env:PATH"
    quoted = shlex.quote(bin_dir)
    return f"export PATH={quoted}:$PATH\nhash -r 2>/dev/null || true"


def cleanup_update_logs(
    *,
    retention_days: int = UPDATE_LOG_RETENTION_DAYS,
    max_files: int = UPDATE_LOG_MAX_FILES,
) -> None:
    """Remove old update logs while preserving the newest recent logs.

    Args:
        retention_days: Maximum age in days to keep.
        max_files: Maximum number of newest log files to keep.
    """
    try:
        if not UPDATE_LOG_DIR.exists():
            return
        logs = sorted(
            (
                (p, p.stat().st_mtime)
                for p in UPDATE_LOG_DIR.glob("*.log")
                if p.is_file()
            ),
            key=operator.itemgetter(1),
            reverse=True,
        )
        cutoff = time.time() - (retention_days * 86_400)
        for idx, (path, mtime) in enumerate(logs):
            if idx >= max_files or mtime < cutoff:
                path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Failed to clean up update logs", exc_info=True)


def create_update_log_path() -> Path:
    """Return a new timestamped update log path and clean stale logs."""
    cleanup_update_logs()
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return UPDATE_LOG_DIR / f"{stamp}-update.log"


async def _emit_progress(callback: UpgradeProgressCallback | None, line: str) -> None:
    """Send a progress line to *callback*, supporting sync or async callbacks."""
    if callback is None:
        return
    result = callback(line)
    if isinstance(result, Awaitable):
        await result


async def _read_stream(
    stream: asyncio.StreamReader,
    *,
    lines: list[str],
    log_file: TextIO | None,
    progress: UpgradeProgressCallback | None,
) -> None:
    """Read subprocess output, append it to the log file, and emit progress."""
    while True:
        raw = await stream.readline()
        if not raw:
            return
        line = raw.decode(errors="replace").rstrip("\n")
        lines.append(line)
        if log_file is not None:
            with suppress(OSError):
                log_file.write(f"{line}\n")
                log_file.flush()
        await _emit_progress(progress, line)


async def _terminate_install_process(proc: asyncio.subprocess.Process) -> None:
    """Best-effort kill of an install subprocess and its descendants.

    On POSIX the sole caller starts the child in its own session
    (`start_new_session=True`), so `proc.pid` doubles as the process-group id
    and `killpg` reaps descendants too; do not call this on a process not
    started that way or it would signal the caller's own group.

    This is teardown that runs *under* a timeout or cancellation, so it must
    never raise — a stray error here would mask the failure it is cleaning up
    after. Every step therefore swallows benign races (a process that already
    exited, a group we cannot signal) and the final reap is time-bounded so an
    unreapable child cannot hang startup before the TUI.
    """
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            # e.g. EPERM if the group contains a privileged descendant; fall
            # back to killing the direct child so at least it is reaped.
            with suppress(OSError):
                proc.kill()
    elif proc.returncode is None:
        with suppress(OSError):
            proc.kill()
    with suppress(OSError, TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_WAIT_TIMEOUT)
    # Reaping the process is not enough: when this cleanup runs after we stopped
    # draining stdout/stderr (timeout/cancellation), those pipes never reach EOF,
    # so the pipe transports — and the parent subprocess transport — stay open
    # until garbage collection. If the event loop has closed by then,
    # `BaseSubprocessTransport.__del__` retries the close on a dead loop and
    # raises "Event loop is closed" (surfaced as PytestUnraisableExceptionWarning
    # under pytest). Close it eagerly while the loop is still alive. `_transport`
    # is the only handle asyncio exposes, and close() must not break teardown.
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        with suppress(Exception):
            transport.close()


async def _run_install_subprocess(
    cmd: str,
    *,
    progress: UpgradeProgressCallback | None,
    log_path: Path | None,
) -> tuple[bool, str]:
    """Run a shell command, streaming stdout/stderr to *progress* and a log file.

    Shared subprocess plumbing for `perform_upgrade` and
    `perform_install_extra`. Returns `(success, combined_output)` where
    *combined_output* is the concatenated stdout+stderr, stripped.

    On timeout or `OSError`, the process is killed and a synthetic error
    line is emitted both to the log and via *progress*. The wall-clock cap
    is `_UPGRADE_TIMEOUT`.

    Args:
        cmd: Shell command to execute.
        progress: Optional callback invoked for each output line.
        log_path: Optional path to persist command output. Falls back to a
            fresh `create_update_log_path()` when `None`.

    Returns:
        `(success, output)` — *success* is `True` iff the subprocess exited 0.

    Raises:
        asyncio.CancelledError: If the calling task is cancelled.
    """
    timeout = _UPGRADE_TIMEOUT
    if log_path is None:
        log_path = create_update_log_path()

    output_lines: list[str] = []
    proc: asyncio.subprocess.Process | None = None
    log_file: TextIO | None = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(f"$ {cmd}\n")
        log_file.flush()
    except OSError:
        logger.warning(
            "Could not create install log at %s; subprocess output will not be "
            "persisted to disk",
            log_path,
            exc_info=True,
        )
        log_file = None

    try:
        if os.name == "posix":
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
        await asyncio.wait_for(
            asyncio.gather(
                _read_stream(
                    proc.stdout,  # ty: ignore[invalid-argument-type]
                    lines=output_lines,
                    log_file=log_file,
                    progress=progress,
                ),
                _read_stream(
                    proc.stderr,  # ty: ignore[invalid-argument-type]
                    lines=output_lines,
                    log_file=log_file,
                    progress=progress,
                ),
                proc.wait(),
            ),
            timeout=timeout,
        )
    except TimeoutError:
        if proc is not None:
            await _terminate_install_process(proc)
        msg = f"Command timed out after {timeout}s: {cmd}"
        if log_file is not None:
            with suppress(OSError):
                log_file.write(f"{msg}\n")
                log_file.close()
        await _emit_progress(progress, msg)
        logger.warning(msg)
        return False, msg
    except asyncio.CancelledError:
        if proc is not None:
            await _terminate_install_process(proc)
        if log_file is not None:
            with suppress(OSError):
                log_file.close()
        raise
    except OSError as exc:
        if log_file is not None:
            with suppress(OSError):
                log_file.close()
        logger.warning("Failed to execute command: %s", cmd, exc_info=True)
        return False, f"Failed to execute: {cmd}\n{type(exc).__name__}: {exc}"

    if log_file is not None:
        with suppress(OSError):
            log_file.close()
    output = "\n".join(output_lines).strip()
    if proc.returncode == 0:
        return True, output
    logger.warning(
        "Command exited with code %d: %s\n%s",
        proc.returncode,
        cmd,
        output,
    )
    return False, output


async def _install_browser_runtime(
    *,
    progress: UpgradeProgressCallback | None,
    log_path: Path | None,
) -> tuple[bool, str]:
    """Provision Playwright Chromium in the current dcode tool environment.

    Uses an argv subprocess rather than a shell because every argument is fixed.
    The browser download is a distinct, bounded phase after the Python extra has
    been installed. Output is appended to the extra-install log.

    Args:
        progress: Optional callback invoked for each output line.
        log_path: Optional path shared with the extra-install phase.

    Returns:
        `(success, output)` where success is true only for exit code zero.
    """
    args = (sys.executable, "-m", "playwright", "install", "chromium")
    display_cmd = shlex.join(args)
    if log_path is None:
        log_path = create_update_log_path()

    output_lines: list[str] = []
    proc: asyncio.subprocess.Process | None = None
    log_file: TextIO | None = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        log_file.write(f"\n$ {display_cmd}\n")
        log_file.flush()
    except OSError:
        logger.warning(
            "Could not append browser provisioning output to %s",
            log_path,
            exc_info=True,
        )
        log_file = None

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(
            asyncio.gather(
                _read_stream(
                    proc.stdout,  # ty: ignore[invalid-argument-type]
                    lines=output_lines,
                    log_file=log_file,
                    progress=progress,
                ),
                _read_stream(
                    proc.stderr,  # ty: ignore[invalid-argument-type]
                    lines=output_lines,
                    log_file=log_file,
                    progress=progress,
                ),
                proc.wait(),
            ),
            timeout=_BROWSER_RUNTIME_INSTALL_TIMEOUT,
        )
    except TimeoutError:
        if proc is not None:
            proc.kill()
            await proc.wait()
        msg = (
            f"Chromium provisioning timed out after {_BROWSER_RUNTIME_INSTALL_TIMEOUT}s"
        )
        if log_file is not None:
            with suppress(OSError):
                log_file.write(f"{msg}\n")
        await _emit_progress(progress, msg)
        logger.warning(msg)
        return False, msg
    except OSError as exc:
        logger.warning("Failed to provision Chromium", exc_info=True)
        return False, f"Chromium provisioning failed: {type(exc).__name__}: {exc}"
    finally:
        if log_file is not None:
            with suppress(OSError):
                log_file.close()

    output = "\n".join(output_lines).strip()
    if proc.returncode == 0:
        return True, output
    logger.warning("Chromium provisioning exited with code %d", proc.returncode)
    return False, output


async def perform_upgrade(
    *,
    progress: UpgradeProgressCallback | None = None,
    log_path: Path | None = None,
    include_prereleases: bool | None = None,
    target_version: str | None = None,
) -> tuple[bool, str]:
    """Attempt to upgrade `deepagents-code` using the detected install method.

    Only tries the detected method — does not fall back to other package
    managers to avoid cross-environment contamination.

    Args:
        progress: Optional callback invoked for each output line.
        log_path: Optional path to persist command output.
        include_prereleases: Whether to include alpha/beta/rc releases. When
            `None`, follows the installed version's channel and the target
            release's dependency metadata. Pre-release upgrades require the uv
            install method; returns failure otherwise.
        target_version: Release version being installed, used to detect stable
            dcode releases that intentionally depend on pre-release packages.

    Returns:
        `(success, output)` — *output* is the combined stdout/stderr.
    """
    method = detect_install_method()
    if method == "unknown":
        return False, "Editable install detected — skipping auto-update."
    if method == "other":
        return False, (
            "Unsupported install method detected — cannot auto-update without "
            "knowing which environment provides `dcode`. Reinstall with "
            "`uv tool install -U deepagents-code` or upgrade with the package "
            "manager originally used for this install."
        )
    resolved_include_prereleases = _resolve_include_prereleases(include_prereleases)
    pin_target_version: str | None = None
    if (
        not resolved_include_prereleases
        and include_prereleases is None
        and release_requires_prereleases(target_version)
    ):
        resolved_include_prereleases = True
        pin_target_version = target_version
    if resolved_include_prereleases:
        supported, reason = prerelease_upgrade_supported(method)
        if not supported:
            return False, reason or _PRERELEASE_UNSUPPORTED_MESSAGE

    fell_back_to_bare_command = False
    if method == "uv":
        # Prefer the receipt-aware `uv tool install -U` builder so installed
        # extras / `--with` packages survive the upgrade and any stale
        # `==<version>` pin in the receipt is cleared. Fall back to the bare
        # display command when extras or receipt introspection fails — the
        # fallback might drop extras, but a successful unpinned upgrade is
        # still strictly better than a pinned "upgrade" that quietly stays
        # on the old version.
        from deepagents_code.extras_info import ExtrasIntrospectionError

        try:
            cmd = upgrade_install_command(
                include_prereleases=resolved_include_prereleases,
                version=pin_target_version,
            )
        except (ExtrasIntrospectionError, ToolRequirementIntrospectionError) as exc:
            logger.warning(
                "Could not build receipt-aware uv upgrade command (%s: %s); "
                "falling back to the bare command. Installed extras may be "
                "dropped.",
                type(exc).__name__,
                exc,
            )
            fell_back_to_bare_command = True
            cmd = upgrade_command(
                method,
                include_prereleases=resolved_include_prereleases,
                version=pin_target_version,
            )
    else:
        cmd = upgrade_command(
            method,
            include_prereleases=resolved_include_prereleases,
        )

    # Skip brew if binary not on PATH
    if method == "brew" and not shutil.which("brew"):
        return False, "brew not found on PATH."

    success, output = await _run_install_subprocess(
        cmd, progress=progress, log_path=log_path
    )
    if success and fell_back_to_bare_command:
        # Surface the dropped-extras caveat only now that the bare upgrade has
        # actually succeeded. Emitting it before `_run_install_subprocess` ran
        # would misfire on a failed upgrade — telling the user to re-add extras
        # for an install that was left untouched. The log line above is
        # invisible in the TUI, so the progress stream is the user's only window
        # into this.
        await _emit_progress(
            progress,
            "Note: couldn't read your full install configuration; "
            "installed extras or extra packages may not carry over. "
            "Re-add them if a feature stops working after relaunch.",
        )
    return success, output


async def perform_dependency_refresh(
    *,
    progress: UpgradeProgressCallback | None = None,
    log_path: Path | None = None,
    include_prereleases: bool | None = None,
) -> tuple[bool, str]:
    """Refresh dependencies while keeping `deepagents-code` on this version.

    Runs `uv tool install -U deepagents-code==<current>` instead of
    `uv tool upgrade deepagents-code`, so compatible dependency releases can be
    picked up without crossing to a newer app version. Only uv-managed installs
    are supported; other install methods cannot safely express this operation.

    Args:
        progress: Optional callback invoked for each output line.
        log_path: Optional path to persist command output.
        include_prereleases: Whether to include alpha/beta/rc releases. When
            `None`, follows the installed version's channel.

    Returns:
        `(success, output)` — *output* is the combined stdout/stderr, or an
            explanatory message when the install method is unsupported, `uv` is
            unavailable, requirement introspection fails, or the subprocess
            fails or times out.
    """
    supported, reason = dependency_refresh_supported()
    if not supported:
        return False, reason or ""
    if not shutil.which("uv"):
        return False, "`uv` not found on PATH."

    from deepagents_code.extras_info import ExtrasIntrospectionError

    try:
        cmd = dependency_refresh_command(
            include_prereleases=include_prereleases,
        )
    except (
        ExtrasIntrospectionError,
        ToolRequirementIntrospectionError,
        ValueError,
    ) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return await _run_install_subprocess(cmd, progress=progress, log_path=log_path)


async def perform_dependency_refresh_dry_run(
    *,
    progress: UpgradeProgressCallback | None = None,
    log_path: Path | None = None,
    include_prereleases: bool | None = None,
) -> tuple[bool, str]:
    """Resolve a dependency refresh plan without mutating the tool environment.

    `uv tool install` has no `--dry-run`, so this targets the running tool
    environment with `uv pip install --dry-run --python <sys.executable>`. It
    uses the same pinned `deepagents-code` requirement, installed extras, and
    preserved `--with` packages as the real refresh command.

    Args:
        progress: Optional callback invoked for each output line.
        log_path: Optional path to persist command output.
        include_prereleases: Whether to include alpha/beta/rc releases. When
            `None`, follows the installed version's channel.

    Returns:
        `(success, output)` — *output* is the combined stdout/stderr from uv, or
            an explanatory message when the plan cannot be computed safely.
    """
    supported, reason = dependency_refresh_supported()
    if not supported:
        return False, reason or ""
    if not shutil.which("uv"):
        return False, "`uv` not found on PATH."

    from deepagents_code.extras_info import ExtrasIntrospectionError

    try:
        cmd = dependency_refresh_dry_run_command(
            include_prereleases=include_prereleases,
        )
    except (
        ExtrasIntrospectionError,
        ToolRequirementIntrospectionError,
        ValueError,
    ) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return await _run_install_subprocess(cmd, progress=progress, log_path=log_path)


class DependencyChange(NamedTuple):
    """A single package version change parsed from uv's environment-diff output.

    Emitted by both `uv tool upgrade` and `uv tool install -U` (the
    dependency-refresh path), so the wording stays command-agnostic. `old` is
    `None` for a newly added package and `new` is `None` for a removed one; both
    are set for an in-place version bump. The `(None, None)` state is invalid —
    see `kind`.
    """

    name: str
    """Package name exactly as uv reported it in the diff line."""

    old: str | None
    """Version before the change; `None` when the package was newly added."""

    new: str | None
    """Version after the change; `None` when the package was removed."""

    @property
    def kind(self) -> Literal["added", "removed", "bumped"]:
        """Classify the change from which version sides are present.

        Reading the case from here keeps consumers (e.g.
        `format_dependency_changes`) from re-deriving it via field truthiness,
        and turns the meaningless `(None, None)` shape into a hard error instead
        of silently rendering as a removal.

        Returns:
            `"added"` when only `new` is set, `"removed"` when only `old` is set,
                and `"bumped"` when both are set.

        Raises:
            ValueError: If neither `old` nor `new` is set.
        """
        if self.old is None and self.new is None:
            msg = (
                f"DependencyChange {self.name!r} records neither an old nor new version"
            )
            raise ValueError(msg)
        if self.old is None:
            return "added"
        if self.new is None:
            return "removed"
        return "bumped"


_DEP_CHANGE_RE = re.compile(
    r"^\s*([+-])\s+([A-Za-z0-9._-]+)==([^\s(]+)(?:\s+\(.*\))?\s*$"
)
"""Matches uv's environment-diff lines, e.g. ` - langchain-openai==1.3.2`.

The optional trailing group tolerates uv's source annotations for non-PyPI
packages, e.g. ` + example==0.1.0 (from file:///path)`.
"""


def parse_dependency_changes(output: str) -> list[DependencyChange]:
    """Parse package version changes from uv's environment-diff output.

    uv reports environment changes as paired ` - pkg==old` / ` + pkg==new`
    lines; this collapses them into one `DependencyChange` per package,
    preserving first-seen order.

    Args:
        output: Combined stdout/stderr from a `uv tool install`/`upgrade`
            subprocess.

    Returns:
        One entry per package whose version was added, removed, or bumped.
    """
    removed: dict[str, str] = {}
    added: dict[str, str] = {}
    order: list[str] = []
    seen: set[str] = set()
    for line in output.splitlines():
        match = _DEP_CHANGE_RE.match(line)
        if match is None:
            continue
        sign, name, version = match.group(1), match.group(2), match.group(3)
        (removed if sign == "-" else added)[name] = version
        if name not in seen:
            seen.add(name)
            order.append(name)
    return [
        DependencyChange(name=name, old=removed.get(name), new=added.get(name))
        for name in order
    ]


def format_dependency_changes(changes: Sequence[DependencyChange]) -> str:
    """Render dependency changes as an aligned, human-readable block.

    Args:
        changes: Parsed changes from `parse_dependency_changes`.

    Returns:
        A newline-joined, column-aligned summary, or `""` when empty.
    """
    if not changes:
        return ""
    width = max(len(change.name) for change in changes)
    lines: list[str] = []
    for change in changes:
        name = change.name.ljust(width)
        if change.kind == "bumped":
            lines.append(f"  {name}  {change.old} -> {change.new}")
        elif change.kind == "added":
            lines.append(f"  {name}  {change.new} (new)")
        else:  # removed
            lines.append(f"  {name}  {change.old} (removed)")
    return "\n".join(lines)


class ToolRequirementIntrospectionError(RuntimeError):
    """Raised when uv tool requested requirements cannot be preserved."""


_EXTRA_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[-_.A-Za-z0-9]*[A-Za-z0-9])?$")
"""Conservative package-extra name pattern used before shell command display."""


_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[-_.A-Za-z0-9]*[A-Za-z0-9])?$")
"""Conservative package name pattern used before shell command display."""


def is_valid_extra_name(extra: str) -> bool:
    """Return whether `extra` is safe to embed in package-extra syntax.

    Args:
        extra: Candidate extra name from CLI or slash-command input.

    Returns:
        `True` when the value is a conservative PEP 508-style extra name.
    """
    return bool(_EXTRA_NAME_RE.fullmatch(extra))


def is_valid_package_name(package: str) -> bool:
    """Return whether `package` is safe to embed in a `--with` install command.

    Args:
        package: Candidate package name from CLI or slash-command input.

    Returns:
        `True` when the value is a conservative PEP 508-style package name.
    """
    return bool(_PACKAGE_NAME_RE.fullmatch(package))


def _uv_tool_receipt_path(tool_root: Path | None = None) -> Path:
    """Return the uv receipt path for the current tool environment.

    Args:
        tool_root: Optional uv tool environment root. Defaults to `sys.prefix`.

    Returns:
        The expected `uv-receipt.toml` path.
    """
    return (tool_root or Path(sys.prefix)) / "uv-receipt.toml"


def _uv_tool_receipt_data(tool_root: Path | None = None) -> dict[str, Any]:
    """Return parsed uv tool receipt data for the current tool environment.

    Args:
        tool_root: Optional uv tool environment root. Defaults to `sys.prefix`.

    Returns:
        Parsed TOML data from `uv-receipt.toml`.

    Raises:
        ToolRequirementIntrospectionError: If the receipt cannot be read or
            parsed.
    """
    receipt_path = _uv_tool_receipt_path(tool_root)
    try:
        return tomllib.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = f"uv tool receipt not found at {receipt_path}"
        raise ToolRequirementIntrospectionError(msg) from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        msg = f"Could not read uv tool receipt at {receipt_path}: {exc}"
        raise ToolRequirementIntrospectionError(msg) from exc


def _uv_tool_python(tool_root: Path | None = None) -> str | None:
    """Return the Python interpreter recorded in the uv tool receipt.

    Args:
        tool_root: Optional uv tool environment root. Defaults to `sys.prefix`.

    Returns:
        The recorded `[tool].python` value, or `None` when the receipt does not
            pin an interpreter.

    Raises:
        ToolRequirementIntrospectionError: If the receipt cannot be read, parsed,
            or safely re-expressed as a `--python` value.
    """
    data = _uv_tool_receipt_data(tool_root)
    tool = data.get("tool")
    if not isinstance(tool, dict):
        msg = "uv tool receipt is missing `[tool]`"
        raise ToolRequirementIntrospectionError(msg)
    python = tool.get("python")
    if python is None:
        return None
    if not isinstance(python, str) or not python:
        msg = "uv tool receipt contains an invalid `[tool].python` value"
        raise ToolRequirementIntrospectionError(msg)
    return python


def _uv_tool_with_packages(
    *,
    distribution_name: str = "deepagents-code",
    tool_root: Path | None = None,
) -> tuple[str, ...]:
    """Return package names recorded as uv tool `--with` requirements.

    uv records the tool's requested requirements in `uv-receipt.toml`. Reading
    that receipt preserves only packages the user asked uv to keep, avoiding the
    over-broad fallback of promoting every installed transitive dependency to a
    top-level `--with` requirement.

    Args:
        distribution_name: Main tool distribution to exclude from `--with`.
        tool_root: Optional uv tool environment root. Defaults to `sys.prefix`.

    Returns:
        A sorted tuple of validated package names to pass as `--with` values.

    Raises:
        ToolRequirementIntrospectionError: If the receipt cannot be read, parsed,
            or safely re-expressed as package-name `--with` requirements.
    """
    data = _uv_tool_receipt_data(tool_root)
    tool = data.get("tool")
    requirements = tool.get("requirements") if isinstance(tool, dict) else None
    if not isinstance(requirements, list):
        msg = "uv tool receipt is missing `[tool].requirements`"
        raise ToolRequirementIntrospectionError(msg)

    main = canonicalize_name(distribution_name)
    packages: set[str] = set()
    for entry in requirements:
        if not isinstance(entry, dict):
            msg = "uv tool receipt contains a non-table requirement entry"
            raise ToolRequirementIntrospectionError(msg)
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            msg = "uv tool receipt contains a requirement without a package name"
            raise ToolRequirementIntrospectionError(msg)
        if canonicalize_name(name) == main:
            continue
        unsupported_keys = sorted(set(entry) - {"name"})
        if unsupported_keys:
            msg = (
                f"uv tool receipt requirement {name!r} cannot be preserved "
                "automatically; reinstall it manually after refreshing "
                "dependencies"
            )
            raise ToolRequirementIntrospectionError(msg)
        if not is_valid_package_name(name):
            msg = (
                f"Invalid uv tool receipt package name {name!r}: must match "
                f"PEP 508 ({_PACKAGE_NAME_RE.pattern})"
            )
            raise ToolRequirementIntrospectionError(msg)
        packages.add(name)
    return tuple(sorted(packages))


def _dcode_extras_requirement(
    extras: Iterable[str],
    *,
    version: str | None = None,
) -> str:
    """Return the validated `deepagents-code[...]` requirement for a uv install.

    Shared by the extra- and package-install commands so already-installed
    extras survive a `uv tool install` reinstall — a bare `deepagents-code`
    request would replace the tool and drop them. Returns plain
    `deepagents-code` when no extras or version are selected; otherwise the
    shell-quoted requirement form, which keeps zsh from globbing brackets.

    Args:
        extras: Extra names to encode. Each is validated against PEP 508
            grammar before interpolation. This is the authoritative gate for
            caller-supplied extras (`install_extras_command`) and a
            redundant re-check for extras read from distribution metadata
            (`install_package_command`).
        version: Optional exact `deepagents-code` version pin.

    Returns:
        Shell-safe requirement token, e.g. `deepagents-code` or
            `'deepagents-code[baseten,nvidia]==1.0.0'`.

    Raises:
        ValueError: If any extra fails PEP 508 validation.
    """
    names = sorted(set(extras))
    for name in names:
        if not is_valid_extra_name(name):
            msg = (
                f"Invalid extra name {name!r}: must match PEP 508 "
                f"({_EXTRA_NAME_RE.pattern})"
            )
            raise ValueError(msg)
    version_suffix = ""
    if version is not None:
        try:
            parsed = Version(version)
        except InvalidVersion as exc:
            msg = f"Invalid deepagents-code version {version!r}"
            raise ValueError(msg) from exc
        version_suffix = f"=={parsed}"
    extras_part = f"[{','.join(names)}]" if names else ""
    requirement = f"deepagents-code{extras_part}{version_suffix}"
    if not names and version is None:
        return requirement
    return shlex.quote(requirement)


def _uv_tool_install_command(
    *,
    version: str | None,
    include_prereleases: bool | None,
    distribution_name: str,
    extras_to_add: Iterable[str] = (),
    with_packages_to_add: Iterable[str] = (),
    reinstall: bool = False,
) -> str:
    """Return the receipt-preserving `uv tool install -U` command.

    Args:
        version: Optional exact `deepagents-code` version pin.
        include_prereleases: Whether to include alpha/beta/rc releases. When
            `None`, follows the installed version's channel.
        distribution_name: Name of the installed distribution to inspect.
        extras_to_add: Extra names to merge with already-installed extras.
        with_packages_to_add: Package names to merge with the receipt's existing
            `--with` packages. Names already present (compared canonically) are
            not duplicated; genuinely new names are appended after the preserved
            ones. Callers must validate these names before passing them — the
            builder only `shlex.quote`-s them.
        reinstall: When `True`, add `--reinstall` so uv rebuilds the tool
            environment from scratch instead of patching it in place. An
            in-place `-U` upgrade can leave stale files behind (e.g. an old
            `tools.py` or its cached bytecode), which has been observed to
            produce a half-updated env that crashes the next server start with
            an `ImportError`; the preserved `--python` interpreter and `--with`
            packages still apply, so the rebuild keeps the existing tool
            context.

    Raises:
        ExtrasIntrospectionError: If a metadata-sourced extra name fails PEP 508
            validation.
        ValueError: If `version` is not PEP 440 compliant.

    Propagates `ToolRequirementIntrospectionError` if the uv tool receipt's
    interpreter or `--with` packages cannot be determined safely from the tool
    receipt.
    """
    from deepagents_code.extras_info import (
        ExtrasIntrospectionError,
        installed_extra_names,
    )

    extras = set(installed_extra_names(distribution_name, strict=True))
    extras.update(extras_to_add)
    try:
        requirement = _dcode_extras_requirement(extras, version=version)
    except ValueError as exc:
        if str(exc).startswith("Invalid deepagents-code version"):
            raise
        msg = f"Distribution metadata yielded an invalid extra name: {exc}"
        raise ExtrasIntrospectionError(msg) from exc
    cmd = "uv tool install --reinstall -U" if reinstall else "uv tool install -U"
    python = _uv_tool_python()
    if python is not None:
        cmd += f" --python {shlex.quote(python)}"
    cmd += f" {requirement}"
    with_packages = list(_uv_tool_with_packages(distribution_name=distribution_name))
    known = {canonicalize_name(package) for package in with_packages}
    for package in with_packages_to_add:
        if canonicalize_name(package) not in known:
            with_packages.append(package)
            known.add(canonicalize_name(package))
    for package in with_packages:
        cmd += f" --with {shlex.quote(package)}"
    if _resolve_include_prereleases(include_prereleases):
        cmd += " --prerelease allow"
    return cmd


def upgrade_install_command(
    *,
    include_prereleases: bool | None = None,
    distribution_name: str = "deepagents-code",
    version: str | None = None,
) -> str:
    """Return the uv command that upgrades dcode while clearing stale pins.

    Built specifically to avoid the `uv tool upgrade` receipt-pin trap: when
    the tool was originally installed via `uv tool install deepagents-code==X.Y.Z`
    — or when a prior `dependency_refresh_command` rewrote the receipt with a
    version-pinned requirement — `uv tool upgrade deepagents-code` will only
    re-resolve *within* that pin and silently keep the user on the same
    version. Re-running `uv tool install -U deepagents-code[<extras>]` (no
    version pin) rewrites the receipt's requirement to unpinned so the next
    upgrade can actually move forward. Callers can still pass `version` when
    the resolver must allow pre-release dependencies for a stable app target;
    that prevents the root `deepagents-code` package from floating to a newer
    app pre-release. Installed extras and `--with` packages are preserved to
    mirror `dependency_refresh_command`.

    Args:
        include_prereleases: Whether to include alpha/beta/rc releases. When
            `None`, follows the installed version's channel.
        distribution_name: Name of the installed distribution to inspect for
            already-installed extras.
        version: Optional exact target version. Use only when pre-release
            dependency resolution must not also select a root app pre-release.

    Returns:
        Shell command string suitable for execution via the shell.

    Propagates `ExtrasIntrospectionError` if installed extras cannot be
    determined safely from distribution metadata, or a metadata-sourced extra name
    fails PEP 508 validation. Also propagates `ToolRequirementIntrospectionError`
    if the uv tool `--with` packages or interpreter cannot be determined safely
    from the tool receipt. Callers choose whether to treat those errors as
    failures or fall back to a simpler unpinned upgrade command with a
    user-facing warning.
    """
    return _uv_tool_install_command(
        version=version,
        include_prereleases=include_prereleases,
        distribution_name=distribution_name,
    )


def dependency_refresh_command(
    *,
    version: str = __version__,
    include_prereleases: bool | None = None,
    distribution_name: str = "deepagents-code",
) -> str:
    """Return the uv command that refreshes deps for the current dcode version.

    Args:
        version: Exact `deepagents-code` version to keep installed.
        include_prereleases: Whether to include alpha/beta/rc releases. When
            `None`, follows the installed version's channel.
        distribution_name: Name of the installed distribution to inspect for
            already-installed extras.

    Returns:
        Shell command string suitable for execution via the shell.

    Propagates `ExtrasIntrospectionError` if installed extras cannot be
    determined safely from distribution metadata, or a metadata-sourced extra name
    fails PEP 508 validation, and `ToolRequirementIntrospectionError` if the uv
    tool `--with` packages or interpreter cannot be determined safely from the
    tool receipt. `perform_dependency_refresh` converts both into a user-facing
    failure.
    """
    return _uv_tool_install_command(
        version=version,
        include_prereleases=include_prereleases,
        distribution_name=distribution_name,
    )


def dependency_refresh_dry_run_command(
    *,
    version: str = __version__,
    include_prereleases: bool | None = None,
    distribution_name: str = "deepagents-code",
    python: str | None = None,
) -> str:
    """Return the uv command that plans a dependency refresh without installing.

    Args:
        version: Exact `deepagents-code` version to keep installed.
        include_prereleases: Whether to include alpha/beta/rc releases. When
            `None`, follows the installed version's channel.
        distribution_name: Name of the installed distribution to inspect for
            already-installed extras and uv receipt requirements.
        python: Python executable for the target environment. Defaults to the
            running interpreter, which is the current dcode tool environment.

    Returns:
        Shell command string suitable for execution via the shell.

    Raises:
        ToolRequirementIntrospectionError: If the target Python or uv tool receipt
            requirements cannot be determined safely.
    """
    from deepagents_code.extras_info import installed_extra_names

    target_python = python or sys.executable
    if not target_python:
        msg = "Could not determine the running Python executable"
        raise ToolRequirementIntrospectionError(msg)
    extras = installed_extra_names(distribution_name, strict=True)
    requirement = _dcode_extras_requirement(extras, version=version)
    cmd = (
        "uv pip install --dry-run --python "
        f"{shlex.quote(target_python)} -U {requirement}"
    )
    with_packages = _uv_tool_with_packages(distribution_name=distribution_name)
    for package in with_packages:
        cmd += f" {shlex.quote(package)}"
    if _resolve_include_prereleases(include_prereleases):
        cmd += " --prerelease allow"
    return cmd


def install_package_command(
    package: str,
    *,
    distribution_name: str = "deepagents-code",
) -> str:
    """Return the shell command that adds a package to the dcode tool env.

    The result is built for *execution* (via `perform_install_package`), not for
    display — surfacing raw `uv tool` invocations to the user is intentionally
    avoided. `package` is validated here against PEP 508 grammar and then
    `shlex.quote`-d by the shared builder: the validation already blocks shell
    metacharacters, so the quoting is defense in depth that keeps the command
    safe even if the pattern is later loosened.

    Delegates to `_uv_tool_install_command` (the same builder the extras path
    uses), passing the new package as a `--with` requirement. That builder folds
    already-installed extras into the pinned `deepagents-code[...]` requirement,
    and preserves the uv-managed Python interpreter and the receipt's existing
    `--with` packages. Without this, reinstalling to add a second package would
    replace the tool with a plain `deepagents-code` (dropping extras the user
    added through `/install <extra>`), rebuild with only the newest `--with`
    package (dropping previously configured custom providers), or silently
    downgrade when the latest stable app depends on prerelease packages.

    Like the extras path (`_install_extra_uv_tool_command`), passes
    `reinstall=True` so the upgrade rebuilds the tool environment cleanly; see
    `_uv_tool_install_command`'s `reinstall` parameter for why an in-place
    upgrade is unsafe.

    Args:
        package: Package name to install into the existing tool environment.
        distribution_name: Name of the installed distribution to inspect for
            already-installed extras and uv receipt requirements.

    Returns:
        Shell command string suitable for execution via the shell.

    Raises:
        ValueError: If `package` fails PEP 508 validation.

    Propagates `ExtrasIntrospectionError` if installed extras cannot be
    determined safely from distribution metadata (or a metadata-sourced extra
    name fails PEP 508 validation), and `ToolRequirementIntrospectionError` if
    the uv tool receipt's interpreter or `--with` packages cannot be determined
    safely.
    """
    if not _PACKAGE_NAME_RE.fullmatch(package):
        msg = (
            f"Invalid package name {package!r}: must match PEP 508 "
            f"({_PACKAGE_NAME_RE.pattern})"
        )
        raise ValueError(msg)
    return _uv_tool_install_command(
        version=__version__,
        include_prereleases=True,
        distribution_name=distribution_name,
        with_packages_to_add=(package,),
        reinstall=True,
    )


def install_extras_command(extras: Iterable[str]) -> str:
    """Return the install-script command that installs dcode extras.

    Args:
        extras: Extra names to include in the tool reinstall. Validated by
            `_dcode_extras_requirement`, which raises `ValueError` on any name
            that fails PEP 508 validation.

    Returns:
        Shell command string suitable for display in error messages.
    """
    names = sorted(extras)
    _dcode_extras_requirement(names)
    if not names:
        return INSTALL_SCRIPT_COMMAND
    extras_env = shlex.quote(",".join(names))
    return (
        f"curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_EXTRAS={extras_env} bash"
    )


def install_extra_command(
    extra: str,
    *,
    distribution_name: str = "deepagents-code",
) -> str:
    """Return the install-script command that adds `extra` to dcode.

    The promoted install path is the install script (see `scripts/install.sh`).
    This helper is display-only and avoids uv receipt introspection so
    unsupported installs can surface method-specific guidance before any uv
    receipt is read. Already-detected extras from distribution metadata are
    included when available, so following the command does not drop them.

    Args:
        extra: The extra name (e.g. `'quickjs'`, `'daytona'`, `'fireworks'`).
            Validated internally against PEP 508 grammar before interpolation
            into the shell command.
        distribution_name: Name of the installed distribution to inspect for
            already-installed extras.

    Returns:
        Shell command string suitable for display in error messages.

    Raises:
        ValueError: If `extra` fails PEP 508 validation.
    """
    if not is_valid_extra_name(extra):
        msg = (
            f"Invalid extra name {extra!r}: must match PEP 508 "
            f"({_EXTRA_NAME_RE.pattern})"
        )
        raise ValueError(msg)
    from deepagents_code.extras_info import installed_extra_names

    extras = installed_extra_names(distribution_name)
    extras.add(extra)
    return install_extras_command(extras)


def install_extra_recovery_command(extra: str) -> str:
    """Return a manual recovery command for the current install method.

    uv-managed installs can preserve the uv receipt's Python interpreter and
    `--with` requirements, so their recovery command uses the same uv path as
    the automatic installer. Unsupported methods keep the install-script command
    and deliberately avoid reading uv receipts.

    Args:
        extra: Extra name to add.

    Returns:
        Shell command string suitable for display in error messages.

    Propagates `ValueError` if `extra` fails PEP 508 validation, and (on the uv
    path) `ExtrasIntrospectionError` if installed extras cannot be determined
    safely or `ToolRequirementIntrospectionError` if the uv receipt's
    interpreter or `--with` packages cannot be preserved safely.
    """
    if detect_install_method() == "uv":
        return _install_extra_uv_tool_command(extra)
    return install_extra_command(extra)


def _install_extra_uv_tool_command(
    extra: str,
    *,
    distribution_name: str = "deepagents-code",
) -> str:
    """Return the receipt-preserving uv command that installs one dcode extra.

    Pins the running `deepagents-code` version and allows prerelease dependency
    resolution so adding an extra cannot make uv backtrack to an older app release.
    Passes `reinstall=True` so the upgrade rebuilds the tool environment from
    scratch rather than patching it in place; see `_uv_tool_install_command`'s
    `reinstall` parameter for why an in-place upgrade is unsafe.

    Args:
        extra: The extra name to add. Validated against PEP 508 grammar before
            interpolation into the shell command.
        distribution_name: Name of the installed distribution to inspect for
            already-installed extras and uv receipt requirements.

    Raises:
        ValueError: If `extra` fails PEP 508 validation.

    Propagates `ExtrasIntrospectionError` if installed extras cannot be
    determined safely from distribution metadata, and
    `ToolRequirementIntrospectionError` if the uv tool receipt's interpreter or
    `--with` packages cannot be preserved safely.
    """
    if not is_valid_extra_name(extra):
        msg = (
            f"Invalid extra name {extra!r}: must match PEP 508 "
            f"({_EXTRA_NAME_RE.pattern})"
        )
        raise ValueError(msg)
    return _uv_tool_install_command(
        version=__version__,
        include_prereleases=True,
        distribution_name=distribution_name,
        extras_to_add=(extra,),
        reinstall=True,
    )


def editable_extra_hint(extra: str) -> str:
    """Return the canonical action hint for editable installs missing an extra.

    Shared by every site that detects an editable install and points the user
    at the correct `uv tool install --editable` invocation, so wording stays
    consistent and the literal `[<extra>]` bracket fragment is centrally
    defined (callers that print through Rich markup must still escape it).
    """
    return (
        "Rerun your `uv tool install --editable` command with "
        f"`--with 'deepagents-code[{extra}]'` added so the extra is "
        "resolved against the editable source."
    )


def editable_package_hint(package: str) -> str:
    """Return the canonical action hint for editable installs needing a package.

    Editable installs can't have packages added automatically, so this points
    the user at adding it to their own development environment. Phrased without
    a raw install command, since surfacing `uv tool` invocations to the user is
    intentionally avoided.
    """
    return (
        f"Add '{package}' to your editable checkout's environment (the one your "
        "editable install of dcode runs from), then relaunch."
    )


async def perform_install_extra(
    extra: str,
    *,
    progress: UpgradeProgressCallback | None = None,
    log_path: Path | None = None,
) -> tuple[bool, str]:
    """Add `extra` to the installed dcode tool environment.

    Runs `uv tool install --reinstall -U 'deepagents-code[<extras>]==<current>'
    --prerelease allow`, preserving any extras that are already installed. For
    the `browser` extra, a successful package install is followed by bounded
    Playwright Chromium provisioning in the same tool environment.
    Editable installs are refused — the caller should rerun their
    `uv tool install --editable` command with `--with 'deepagents-code[<extra>]'`
    added so the extra is resolved against the editable source.

    Args:
        extra: The extra name to install. Must satisfy `is_valid_extra_name`;
            invalid names are rejected without invoking uv (defense in depth
            against shell injection via the `--force`/`--yes` bypass paths).
        progress: Optional callback invoked for each output line.
        log_path: Optional path to persist command output.

    Returns:
        `(success, output)` — *output* is the combined stdout/stderr, or an
            explanatory error message when the install method is unsupported
            or `extra` is malformed.
    """
    if not is_valid_extra_name(extra):
        return False, (
            f"Invalid extra name {extra!r}: must match {_EXTRA_NAME_RE.pattern}"
        )
    method = detect_install_method()
    if method == "unknown":
        return False, (
            "Editable install detected — cannot add extras automatically.\n"
            + editable_extra_hint(extra)
        )
    if method == "brew":
        # Homebrew formula doesn't expose extras; uv tool install is the
        # right escape hatch but would conflict with the brew-managed binary.
        return False, (
            "Homebrew install detected — extras are not supported via brew. "
            f"Reinstall with `{install_extra_command(extra)}` to switch to a "
            "uv-managed tool install with extras."
        )
    if method == "other":
        return False, (
            "Unsupported install method detected — cannot add extras without "
            "knowing which environment provides `dcode`. Reinstall with "
            f"`{install_extra_command(extra)}` to switch to a uv-managed tool "
            "install with extras."
        )

    if not shutil.which("uv"):
        return False, (
            "`uv` not found on PATH. Reinstall dcode following the docs, or "
            "install uv (https://docs.astral.sh/uv/) so extras can be added."
        )

    from deepagents_code.extras_info import ExtrasIntrospectionError

    try:
        cmd = _install_extra_uv_tool_command(extra)
    except (
        ExtrasIntrospectionError,
        ToolRequirementIntrospectionError,
        ValueError,
    ) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    success, output = await _run_install_subprocess(
        cmd, progress=progress, log_path=log_path
    )
    if not success or extra != "browser":
        return success, output

    await _emit_progress(progress, "Provisioning Chromium for browser support...")
    runtime_success, runtime_output = await _install_browser_runtime(
        progress=progress,
        log_path=log_path,
    )
    combined = "\n".join(part for part in (output, runtime_output) if part)
    if not runtime_success:
        detail = runtime_output or "Playwright exited without diagnostic output"
        return False, (
            "The browser extra was installed, but Chromium provisioning failed: "
            f"{detail}"
        )
    return True, combined


async def perform_install_package(
    package: str,
    *,
    progress: UpgradeProgressCallback | None = None,
    log_path: Path | None = None,
) -> tuple[bool, str]:
    """Add an arbitrary `package` to the installed dcode tool environment.

    Runs `uv tool install --reinstall -U 'deepagents-code[<extras>]==<current>'
    --with <package> --prerelease allow`, the escape hatch for a provider whose
    package is not a `deepagents-code` extra (e.g. a custom or in-house
    `class_path` model). Already-installed extras are preserved so the reinstall
    does not drop them.
    Editable installs are refused
    — the caller should rerun their `uv tool install --editable` command with
    `--with <package>` added so it resolves against the editable source.

    Args:
        package: The package name to install. Must satisfy
            `is_valid_package_name`; invalid names are rejected without invoking
            uv (defense in depth against shell injection via the
            `--force`/`--yes` bypass paths).
        progress: Optional callback invoked for each output line.
        log_path: Optional path to persist command output.

    Returns:
        `(success, output)` — on success, *output* is the combined
            stdout/stderr from the install. On failure it is an explanatory
            message: when the install method is unsupported, `package` is
            malformed, `uv` is unavailable, or the install subprocess fails or
            times out.
    """
    if not is_valid_package_name(package):
        return False, (
            f"Invalid package name {package!r}: must match {_PACKAGE_NAME_RE.pattern}"
        )
    method = detect_install_method()
    if method == "unknown":
        return False, (
            "Editable install detected — cannot add packages automatically.\n"
            + editable_package_hint(package)
        )
    if method == "brew":
        return False, (
            "Homebrew install detected — packages can't be added to a brew "
            "install. Reinstall dcode as a uv-managed tool (see the "
            "installation docs) to enable adding packages."
        )
    if method == "other":
        return False, (
            "Unsupported install method detected — cannot add packages without "
            "knowing which environment provides `dcode`. Reinstall dcode "
            "as a uv-managed tool (see the installation docs) to enable "
            "adding packages."
        )

    if not shutil.which("uv"):
        return False, (
            "Package installs require uv, which was not found. Reinstall "
            "dcode following the installation docs so packages can be "
            "added."
        )

    from deepagents_code.extras_info import ExtrasIntrospectionError

    try:
        cmd = install_package_command(package)
    except ValueError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except (ExtrasIntrospectionError, ToolRequirementIntrospectionError) as exc:
        # Distinct from a malformed package name: the running distribution's own
        # metadata, or the uv tool receipt, could not be read or parsed. Leave a
        # breadcrumb so the cause is recoverable from logs, even though the user
        # message is unchanged.
        logger.warning(
            "Could not introspect installed extras or uv receipt for package "
            "install of %r",
            package,
            exc_info=True,
        )
        return False, f"{type(exc).__name__}: {exc}"
    return await _run_install_subprocess(cmd, progress=progress, log_path=log_path)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def is_update_check_enabled() -> bool:
    """Return whether update checks are enabled.

    Checks `DEEPAGENTS_CODE_NO_UPDATE_CHECK` env var and the `[update].check` key
    in `config.toml`.

    Defaults to enabled.
    """
    from deepagents_code._env_vars import NO_UPDATE_CHECK

    if os.environ.get(NO_UPDATE_CHECK):
        return False
    return _read_update_config().get("check", True)


def is_auto_update_enabled() -> bool:
    """Return whether auto-update is enabled.

    Opt-out via `DEEPAGENTS_CODE_AUTO_UPDATE=0` env var or
    `[update].auto_update = false` in `config.toml`.

    Defaults to `True`.

    Unrecognized env values (neither truthy nor falsy) are ignored with a
    warning and fall through to the config read below.

    If `config.toml` exists but cannot be parsed, returns `False` (fail-closed):
    a corrupt file may hold an explicit opt-out, so it is not treated as the
    permissive default. A genuinely absent config falls through to `True`.

    Always disabled for editable installs.
    """
    from deepagents_code._env_vars import AUTO_UPDATE, classify_env_bool
    from deepagents_code.config import _is_editable_install

    if _is_editable_install():
        return False
    if AUTO_UPDATE in os.environ:
        raw = os.environ[AUTO_UPDATE]
        classified = classify_env_bool(raw)
        if classified is not None:
            return classified
        # Unrecognized boolean token: warn and fall through to the config read
        # below (which itself fails closed on a corrupt config), mirroring
        # `config_manifest._coerce_env`. With the opt-out default an absent or
        # default config leaves auto-update on, so an ignored disable attempt
        # (e.g. a typo like `ture`) must be surfaced rather than swallowed.
        logger.warning("Ignoring %s=%r (expected bool)", AUTO_UPDATE, raw)
    try:
        config = _read_update_config_strict()
    except _ConfigReadError:
        # The config exists but cannot be parsed. Fail *closed* here even though
        # the default is opt-out: a corrupt file may hold an explicit
        # `auto_update = false`, and silently re-enabling auto-update (which
        # upgrades and re-execs the process) against an unreadable opt-out is
        # worse than skipping the upgrade. A genuinely absent config still
        # falls through to the opt-out default below.
        logger.warning(
            "Could not read [update] config; disabling auto-update until it is "
            "readable",
            exc_info=True,
        )
        return False
    return config.get("auto_update", True)


def set_auto_update(enabled: bool) -> None:
    """Persist the auto-update preference to `config.toml`.

    Writes `[update].auto_update` so the setting survives across sessions.

    Args:
        enabled: Whether auto-update should be enabled.
    """
    import contextlib
    import tempfile
    from pathlib import Path

    import tomli_w

    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    else:
        data = {}

    if "update" not in data:
        data["update"] = {}
    data["update"]["auto_update"] = enabled

    fd, tmp_path = tempfile.mkstemp(dir=DEFAULT_CONFIG_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            tomli_w.dump(data, f)
        Path(tmp_path).replace(DEFAULT_CONFIG_PATH)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp_path).unlink()
        raise


class _ConfigReadError(Exception):
    """Internal: `config.toml` exists but could not be read or parsed.

    Lets callers that care about the difference (e.g. `is_auto_update_enabled`,
    which fails closed) distinguish a corrupt config from a genuinely absent
    one. A missing file is *not* an error and returns an empty config.
    """


def _read_update_config_strict() -> dict[str, bool]:
    """Read `[update]` section from `config.toml`, surfacing read errors.

    Returns:
        A dict of boolean config values; empty when the file is absent.

    Raises:
        _ConfigReadError: When the file exists but cannot be opened or parsed.
    """
    if not DEFAULT_CONFIG_PATH.exists():
        return {}
    try:
        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise _ConfigReadError from exc
    section = data.get("update", {})
    return {k: v for k, v in section.items() if isinstance(v, bool)}


def _read_update_config() -> dict[str, bool]:
    """Read `[update]` section from `config.toml`.

    Returns:
        A dict of boolean config values, empty on missing/unreadable file.
    """
    try:
        return _read_update_config_strict()
    except _ConfigReadError:
        logger.warning("Could not read [update] config — using defaults", exc_info=True)
        return {}


def is_auto_update_explicitly_set() -> bool:
    """Return whether the user explicitly chose an auto-update preference.

    `True` when `DEEPAGENTS_CODE_AUTO_UPDATE` holds a recognized boolean or
    `[update].auto_update` is present in `config.toml`. Distinguishes a
    deliberate opt-in/out from the implicit opt-out default.
    """
    from deepagents_code._env_vars import AUTO_UPDATE, classify_env_bool

    if (
        AUTO_UPDATE in os.environ
        and classify_env_bool(os.environ[AUTO_UPDATE]) is not None
    ):
        return True
    return "auto_update" in _read_update_config()


def should_announce_auto_update_default() -> bool:
    """Return whether to show the one-time auto-update default migration notice.

    `True` when no explicit env/config preference is set (so auto-update is on
    only *implicitly*, via the opt-out default) and the notice has not been
    acknowledged yet. This does not itself verify that auto-update is enabled;
    callers must gate on `is_auto_update_enabled` first (e.g. an editable
    install has no explicit preference but never auto-updates).
    """
    if is_auto_update_explicitly_set():
        return False
    return not _read_update_state().get("auto_update_default_acknowledged", False)


def mark_auto_update_default_acknowledged() -> bool:
    """Record that the one-time auto-update default migration notice was shown.

    Returns:
        `True` if the acknowledgement was persisted. `False` means the state
            write failed, so the notice will fire again on the next launch;
            callers should surface that rather than letting the repeat
            look like a bug.
    """
    return _write_update_state({"auto_update_default_acknowledged": True})


def _note_install_baseline() -> None:
    """Pre-acknowledge the auto-update default notice for a fresh install.

    The migration notice (`should_announce_auto_update_default`) is intended to
    warn users who ran dcode *before* auto-update became the opt-out default; a
    brand-new install never experienced the old behavior, so the notice is
    meaningless to it. Call this on the first launch ever (see
    `should_show_whats_new`) so the notice never leaks into a new install — the
    notice itself fires pre-TUI in `_run_startup_auto_update`.

    Writes nothing when the user already set an explicit preference.
    """
    if is_auto_update_explicitly_set():
        return
    if not mark_auto_update_default_acknowledged():
        # Fail-soft: the same unwritable state dir also drops the adjacent
        # `seen_version` write, so the install stays "first run ever" and the
        # stamp is retried next launch. Log the operation for context — the
        # generic write warning in `_write_update_state` can't say which write
        # failed when both fire back-to-back.
        logger.debug(
            "Could not stamp install baseline; the auto-update default notice "
            "will be re-evaluated on the next launch",
        )


# ---------------------------------------------------------------------------
# "What's new" tracking
# ---------------------------------------------------------------------------


def get_seen_version() -> str | None:
    """Return the last version the user saw the "what's new" banner for."""
    value = _read_update_state().get("seen_version")
    return value if isinstance(value, str) else None


def mark_version_seen(version: str) -> None:
    """Record that the user has seen the "what's new" banner for *version*."""
    _write_update_state({"seen_version": version, "seen_at": time.time()})


def should_show_whats_new() -> bool:
    """Return `True` if this is the first launch on a newer version."""
    seen = get_seen_version()
    if seen is None:
        # First run ever — mark current as seen, don't show banner. This is the
        # canonical fresh-install signal, so also pre-acknowledge the
        # auto-update default migration notice (which only applies to users who
        # predate the opt-out default) before it can fire on a later launch.
        _note_install_baseline()
        mark_version_seen(__version__)
        return False
    try:
        return _parse_version(__version__) > _parse_version(seen)
    except InvalidVersion:
        logger.debug("Failed to compare versions for what's-new check", exc_info=True)
        return False
