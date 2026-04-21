"""Update lifecycle for `deepagents-cli`.

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
import os
import shutil
import sys
import time
import tomllib
from typing import TYPE_CHECKING, Any, Literal

from packaging.version import InvalidVersion, Version

from deepagents_cli._version import PYPI_URL, SDK_PYPI_URL, USER_AGENT, __version__

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_cli.model_config import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

CACHE_FILE: Path = DEFAULT_CONFIG_DIR / "latest_version.json"
UPDATE_STATE_FILE: Path = DEFAULT_CONFIG_DIR / "update_state.json"
CACHE_TTL = 86_400  # 24 hours

_SDK_RELEASE_TIMES_KEY = "sdk_release_times"
"""`CACHE_FILE` key for cached SDK upload timestamps, keyed by version string."""

InstallMethod = Literal["uv", "pip", "brew", "unknown"]

FALLBACK_UPGRADE_COMMAND = "pip install --upgrade deepagents-cli"
"""Generic upgrade hint used when install-method detection fails.

Callers that surface an upgrade command in user-facing text should prefer
`upgrade_command()`; this constant exists so those callers have something
to render when detection raises unexpectedly.
"""

_UPGRADE_COMMANDS: dict[InstallMethod, str] = {
    "uv": "uv tool upgrade deepagents-cli",
    "brew": "brew upgrade deepagents-cli",
    "pip": FALLBACK_UPGRADE_COMMAND,
}
"""Upgrade commands keyed by install method.

`perform_upgrade` runs only the command matching the detected install method;
no fallback chain.
"""

_UPGRADE_TIMEOUT = 120  # seconds


def _parse_version(v: str) -> Version:
    """Parse a PEP 440 version string into a comparable `Version` object.

    Supports stable (`1.2.3`) and pre-release (`1.2.3a1`, `1.2.3rc2`) versions.

    Args:
        v: Version string like `'1.2.3'` or `'1.2.3a1'`.

    Returns:
        A `packaging.version.Version` instance.
    """
    return Version(v.strip())  # raises InvalidVersion for non-PEP 440 strings


def _latest_from_releases(
    releases: dict[str, list[object]],
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


def get_latest_version(
    *,
    bypass_cache: bool = False,
    include_prereleases: bool = False,
) -> str | None:
    """Fetch the latest deepagents-cli version from PyPI, with caching.

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

    try:
        if not bypass_cache and CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            fresh = time.time() - data.get("checked_at", 0) < CACHE_TTL
            if fresh and cache_key in data:
                return data[cache_key]
    except (OSError, json.JSONDecodeError, TypeError):
        logger.debug("Failed to read update-check cache", exc_info=True)

    try:
        import requests
    except ImportError:
        logger.warning(
            "requests package not installed — update checks disabled. "
            "Install with: pip install requests"
        )
        return None

    try:
        resp = requests.get(
            PYPI_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=3,
        )
        resp.raise_for_status()
        payload = resp.json()
        stable: str = payload["info"]["version"]
        releases: dict[str, list[object]] = payload.get("releases", {})
        if not releases:
            logger.debug("PyPI response missing or empty 'releases' key")
        prerelease = _latest_from_releases(releases, include_prereleases=True)
    except (requests.RequestException, OSError, KeyError, json.JSONDecodeError):
        logger.debug("Failed to fetch latest version from PyPI", exc_info=True)
        return None

    release_times = _extract_release_times(
        payload, stable=stable, prerelease=prerelease
    )

    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps(
                {
                    "version": stable,
                    "version_prerelease": prerelease,
                    "release_times": release_times,
                    "checked_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Failed to write update-check cache", exc_info=True)

    return prerelease if include_prereleases else stable


def _extract_release_times(
    payload: dict[str, Any], *, stable: str, prerelease: str | None
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

    Returns:
        Mapping of version string to ISO-8601 upload time. Silently drops
        versions whose timestamp is missing or malformed.
    """
    times: dict[str, str] = {}
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        return times
    for ver in (stable, prerelease):
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
    value = file_entry.get("upload_time_iso_8601")  # type: ignore[call-overload]
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
    from deepagents_cli.sessions import format_relative_timestamp

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


def _write_update_state(patch: dict[str, object]) -> None:
    """Merge *patch* into the shared update state file (shallow, top-level only).

    Args:
        patch: Keys to merge into the existing state.
    """
    data = _read_update_state()
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


def is_update_available(*, bypass_cache: bool = False) -> tuple[bool, str | None]:
    """Check whether a newer version of deepagents-cli is available.

    When the installed version is a pre-release (e.g. `0.0.35a1`),
    pre-release versions on PyPI are included in the comparison so alpha
    testers are notified of newer alphas and the eventual stable release.
    Stable installs only compare against stable PyPI releases.

    Args:
        bypass_cache: Skip the cache and always hit PyPI.

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

    include_prereleases = installed.is_prerelease
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


def detect_install_method() -> InstallMethod:
    """Detect how `deepagents-cli` was installed.

    Checks `sys.prefix` against known paths for uv and Homebrew.

    Returns:
        The detected install method: `'uv'`, `'brew'`, `'pip'`, or `'unknown'`
            (editable/dev installs).
    """
    from deepagents_cli.config import _is_editable_install

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
    return "pip"


def upgrade_command(method: InstallMethod | None = None) -> str:
    """Return the shell command to upgrade `deepagents-cli`.

    Falls back to the pip command for unrecognized install methods.

    Args:
        method: Install method override.

            Auto-detected if `None`.
    """
    if method is None:
        method = detect_install_method()
    return _UPGRADE_COMMANDS.get(method, _UPGRADE_COMMANDS["pip"])


async def perform_upgrade() -> tuple[bool, str]:
    """Attempt to upgrade `deepagents-cli` using the detected install method.

    Only tries the detected method — does not fall back to other package
    managers to avoid cross-environment contamination.

    Returns:
        `(success, output)` — *output* is the combined stdout/stderr.
    """
    method = detect_install_method()
    if method == "unknown":
        return False, "Editable install detected — skipping auto-update."

    cmd = _UPGRADE_COMMANDS.get(method)
    if cmd is None:
        return False, f"No upgrade command for install method: {method}"

    # Skip brew if binary not on PATH
    if method == "brew" and not shutil.which("brew"):
        return False, "brew not found on PATH."

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_UPGRADE_TIMEOUT
        )
        output = (stdout or b"").decode() + (stderr or b"").decode()
        if proc.returncode == 0:
            return True, output.strip()
        logger.warning(
            "Upgrade via %s exited with code %d: %s",
            method,
            proc.returncode,
            output.strip(),
        )
        return False, output.strip()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        msg = f"Upgrade command timed out after {_UPGRADE_TIMEOUT}s: {cmd}"
        logger.warning(msg)
        return False, msg
    except OSError:
        logger.warning("Failed to execute upgrade command: %s", cmd, exc_info=True)
        return False, f"Failed to execute: {cmd}"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def is_update_check_enabled() -> bool:
    """Return whether update checks are enabled.

    Checks `DEEPAGENTS_CLI_NO_UPDATE_CHECK` env var and the `[update].check` key
    in `config.toml`.

    Defaults to enabled.
    """
    from deepagents_cli._env_vars import NO_UPDATE_CHECK

    if os.environ.get(NO_UPDATE_CHECK):
        return False
    return _read_update_config().get("check", True)


def is_auto_update_enabled() -> bool:
    """Return whether auto-update is enabled.

    Opt-in via `DEEPAGENTS_CLI_AUTO_UPDATE=1` env var or
    `[update].auto_update = true` in `config.toml`.

    Defaults to `False`.

    Always disabled for editable installs.
    """
    from deepagents_cli._env_vars import AUTO_UPDATE
    from deepagents_cli.config import _is_editable_install

    if _is_editable_install():
        return False
    if os.environ.get(AUTO_UPDATE, "").lower() in {"1", "true", "yes"}:
        return True
    return _read_update_config().get("auto_update", False)


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


def _read_update_config() -> dict[str, bool]:
    """Read `[update]` section from `config.toml`.

    Returns:
        A dict of boolean config values, empty on missing/unreadable file.
    """
    try:
        if not DEFAULT_CONFIG_PATH.exists():
            return {}
        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
        section = data.get("update", {})
        return {k: v for k, v in section.items() if isinstance(v, bool)}
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read [update] config — using defaults", exc_info=True)
        return {}


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
        # First run ever — mark current as seen, don't show banner.
        mark_version_seen(__version__)
        return False
    try:
        return _parse_version(__version__) > _parse_version(seen)
    except InvalidVersion:
        logger.debug("Failed to compare versions for what's-new check", exc_info=True)
        return False
