"""Background update check for deepagents-cli.

Compares the installed version against PyPI and caches the result
(see `CACHE_TTL`). All errors are silently swallowed to avoid disrupting
user experience.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from deepagents_cli._version import __version__

if TYPE_CHECKING:
    from pathlib import Path
from deepagents_cli.model_config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)

PYPI_URL = "https://pypi.org/pypi/deepagents-cli/json"
CACHE_FILE: Path = DEFAULT_CONFIG_DIR / "latest_version.json"
CACHE_TTL = 86_400  # 24 hours
USER_AGENT = f"deepagents-cli/{__version__} update-check"


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable integer tuple.

    Args:
        v: Version string like `'1.2.3'`.

    Returns:
        Tuple of integers, e.g. `(1, 2, 3)`.

    """
    return tuple(int(x) for x in v.strip().split("."))


def get_latest_version() -> str | None:
    """Fetch the latest deepagents-cli version from PyPI, with caching.

    Results are cached to `CACHE_FILE` to avoid repeated network calls.

    Returns:
        The latest version string, or `None` on any failure.
    """
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if time.time() - data.get("checked_at", 0) < CACHE_TTL:
                return data["version"]
    except Exception:
        logger.debug("Failed to read update-check cache", exc_info=True)

    try:
        import requests

        resp = requests.get(
            PYPI_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=3,
        )
        resp.raise_for_status()
        latest: str = resp.json()["info"]["version"]
    except Exception:
        logger.debug("Failed to fetch latest version from PyPI", exc_info=True)
        return None

    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps({"version": latest, "checked_at": time.time()}),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("Failed to write update-check cache", exc_info=True)

    return latest


def is_update_available() -> tuple[bool, str | None]:
    """Check whether a newer version of deepagents-cli is available.

    Returns:
        A `(available, latest)` tuple. `available` is `True` when
        the PyPI version is strictly newer than the installed version;
        `latest` is the version string (or `None` when the check fails).
    """
    latest = get_latest_version()
    if latest is None:
        return False, None

    try:
        if _parse_version(latest) > _parse_version(__version__):
            return True, latest
    except (ValueError, TypeError):
        logger.debug("Failed to compare versions", exc_info=True)

    return False, None
