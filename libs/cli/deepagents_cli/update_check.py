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
from typing import TYPE_CHECKING, Literal

from packaging.version import InvalidVersion, Version

from deepagents_cli._version import PYPI_URL, USER_AGENT, __version__

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_cli.model_config import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

CACHE_FILE: Path = DEFAULT_CONFIG_DIR / "latest_version.json"
UPDATE_STATE_FILE: Path = DEFAULT_CONFIG_DIR / "update_state.json"
CACHE_TTL = 86_400  # 24 hours

InstallMethod = Literal["uv", "pip", "brew", "unknown"]

_UPGRADE_COMMANDS: dict[InstallMethod, str] = {
    "uv": "uv tool upgrade deepagents-cli",
    "brew": "brew upgrade deepagents-cli",
    "pip": "pip install --upgrade deepagents-cli",
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

    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps(
                {
                    "version": stable,
                    "version_prerelease": prerelease,
                    "checked_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Failed to write update-check cache", exc_info=True)

    return prerelease if include_prereleases else stable


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

            `available` is `True` when the PyPI version is strictly newer than
            the installed version; `latest` is the version string (or `None`
            when the check fails).
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
        if _parse_version(latest) > installed:
            return True, latest
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
