"""Version information for `deepagents` (SDK)."""

from __future__ import annotations

import json
import logging
from functools import cache
from importlib.metadata import Distribution, distributions

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

# Do not remove the `x-release-please-version` annotation below — release-please
# uses it to keep `__version__` in sync with `pyproject.toml` on every release
# PR. Without it, `__version__` silently drifts behind the package version. See
# `.github/RELEASING.md` > Version Bumping.
__version__ = "0.7.0"  # x-release-please-version


def _distribution_name(dist: Distribution) -> str:
    """Return `dist`'s lowercased, hyphenated name, or `""` when unavailable.

    `Distribution.name` is a property that reads and parses `METADATA`, so it can
    raise rather than return `None` — a distribution with non-UTF-8 `METADATA`
    raises `UnicodeDecodeError`. Those failures are contained here so one
    unreadable distribution cannot abort the caller's whole scan.
    """
    try:
        name = dist.name
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        # `KeyError` is future-proofing: on a metadata-less `*.dist-info`,
        # `Distribution.name` returns `None` today but is documented to raise
        # `KeyError` in a later Python (`DeprecationWarning` as of 3.14).
        logger.debug("Could not read distribution name; skipping", exc_info=True)
        return ""
    if not isinstance(name, str):
        # A partial install can leave a `*.dist-info` with no `METADATA`, for
        # which `Distribution.name` is currently `None`.
        return ""
    return name.lower().replace("_", "-")


def _direct_url_marks_editable(dist: Distribution) -> bool:
    """Return whether `dist` has PEP 610 metadata marking it editable.

    Unreadable or unparseable `direct_url.json` reports non-editable rather than
    raising, so one bad distribution cannot abort the caller's scan.
    """
    try:
        raw = dist.read_text("direct_url.json")
    except (OSError, TypeError, ValueError):
        # `read_text` already suppresses the ordinary absent-file cases, so
        # reaching here means the path exists but is unusable (`OSError`) or holds
        # non-UTF-8 bytes (`UnicodeDecodeError`, a `ValueError`). `TypeError` is
        # unreachable through the stdlib's `PathDistribution` and is kept only as
        # insurance against third-party `Distribution` implementations.
        logger.debug("Could not read direct_url.json", exc_info=True)
        return False
    if not raw:
        # Normal and common: non-editable installs omit the file entirely, as
        # does the source-tree `*.egg-info`. Deliberately not logged.
        return False
    try:
        data = json.loads(raw)
    except (RecursionError, ValueError):
        # `ValueError` covers `json.JSONDecodeError`; pathologically nested
        # input raises `RecursionError`, which is not a `ValueError`.
        logger.debug("Malformed direct_url.json", exc_info=True)
        return False
    if not isinstance(data, dict):
        return False
    dir_info = data.get("dir_info")
    if not isinstance(dir_info, dict):
        return False
    return dir_info.get("editable") is True


def _is_editable_install() -> bool:
    """Whether any installed distribution named `deepagents` is editable.

    Scans every installed distribution named `deepagents` for PEP 610
    `direct_url.json` with `dir_info.editable: true`.

    A bare `importlib.metadata.distribution("deepagents")` lookup is not enough.
    Because `setuptools` leaves a gitignored `deepagents.egg-info/` in the source
    tree after any local build, and the cwd is on `sys.path` for `-c`, `-m`, and
    the REPL, that `*.egg-info` is discovered *first* when running from the
    checkout. It carries no `direct_url.json`, so the single-lookup form reports
    "not editable" and never consults the real editable install in
    site-packages. Scanning past it is what makes detection correct.

    Per-distribution metadata failures are contained by the helpers above, so one
    unreadable distribution cannot end the scan early and mask a later editable
    one. Missing or unparseable metadata reports non-editable, keeping version
    reporting best-effort rather than a source of construction failures.
    """
    try:
        for dist in distributions():
            if _distribution_name(dist) != "deepagents":
                continue
            if _direct_url_marks_editable(dist):
                return True
    except (OSError, TypeError, ValueError):
        # Backstop for `distributions()` itself: it yields lazily, so a broken
        # `sys.path` entry surfaces mid-iteration and cannot be contained
        # per-distribution the way the two helpers above are.
        logger.debug(
            "Stopped scanning installed distributions for PEP 610 metadata",
            exc_info=True,
        )
    return False


def _with_editable_local_version(value: str) -> str:
    """Return `value` with an `editable` PEP 440 local segment appended.

    The base version is re-emitted in canonical PEP 440 form, so the result can
    differ from `value` beyond the added segment (`1.0.0-alpha1` becomes
    `1.0.0a1+editable`). An existing local segment is preserved and extended
    rather than clobbered (`1.0+build` becomes `1.0+build.editable`), which makes
    this non-idempotent. Unparseable input is returned unchanged.
    """
    try:
        parsed = Version(value)
    except InvalidVersion:
        return value
    local = f"{parsed.local}.editable" if parsed.local else "editable"
    return f"{parsed.public}+{local}"


@cache
def _lc_version() -> str:
    """Version string for LangSmith `lc_versions.deepagents` metadata.

    Uses the release version from `__version__`. Editable installs get an
    `editable` PEP 440 local segment so traces can distinguish workspace
    checkouts from published wheels without inventing a different base version.

    Cached: install layout is fixed for the life of the process, and this runs on
    every `create_deep_agent()` call, where the underlying metadata scan is a
    measurable share of graph construction.
    """
    if _is_editable_install():
        return _with_editable_local_version(__version__)
    return __version__
