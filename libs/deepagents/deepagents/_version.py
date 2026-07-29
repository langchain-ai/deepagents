"""Version information for `deepagents` (SDK)."""

from __future__ import annotations

import json
import logging
from importlib.metadata import Distribution, distributions

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

# Do not remove the `x-release-please-version` annotation below — release-please
# uses it to keep `__version__` in sync with `pyproject.toml` on every release
# PR. Without it, `__version__` silently drifts behind the package version. See
# `.github/RELEASING.md` > Version Bumping.
__version__ = "0.7.0"  # x-release-please-version


def _distribution_name(dist: Distribution) -> str:
    """Return a normalized distribution name, or `""` when unavailable."""
    name = getattr(dist, "name", None)
    if not isinstance(name, str):
        return ""
    return name.lower().replace("_", "-")


def _direct_url_marks_editable(dist: Distribution) -> bool:
    """Return whether `dist` has PEP 610 metadata marking it editable."""
    try:
        raw = dist.read_text("direct_url.json")
    except (FileNotFoundError, OSError, UnicodeDecodeError, TypeError):
        return False
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    dir_info = data.get("dir_info")
    if not isinstance(dir_info, dict):
        return False
    return bool(dir_info.get("editable", False))


def _is_editable_install() -> bool:
    """Whether the installed `deepagents` distribution is an editable install.

    Scans every installed distribution named `deepagents` for PEP 610
    `direct_url.json` with `dir_info.editable: true`.

    A bare `importlib.metadata.distribution("deepagents")` lookup is not enough:
    when the process cwd is the source tree, a local `*.egg-info` directory can
    shadow the real editable install in site-packages and omit `direct_url.json`.
    Missing packaging metadata or parse failures are treated as non-editable so
    version reporting stays best-effort and never crashes agent construction.
    """
    try:
        for dist in distributions():
            if _distribution_name(dist) != "deepagents":
                continue
            if _direct_url_marks_editable(dist):
                return True
    except (OSError, TypeError, ValueError):
        logger.debug(
            "Failed to read editable install info from PEP 610 metadata",
            exc_info=True,
        )
    return False


def _with_editable_local_version(value: str) -> str:
    """Add an `editable` local segment to a normalized version string.

    Args:
        value: Base package version.

    Returns:
        The version with an `editable` local segment, or the original value when
        it cannot be parsed.
    """
    try:
        parsed = Version(value)
    except InvalidVersion:
        return value
    local = f"{parsed.local}.editable" if parsed.local else "editable"
    return f"{parsed.public}+{local}"


def _lc_version() -> str:
    """Version string for LangSmith `lc_versions.deepagents` metadata.

    Uses the package release marker from `__version__`. Editable installs append
    a PEP 440 `+editable` local segment so traces can distinguish workspace
    checkouts from published wheels without inventing a different base version.
    """
    if _is_editable_install():
        return _with_editable_local_version(__version__)
    return __version__
