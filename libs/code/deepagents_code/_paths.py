"""Filesystem-path helpers shared across Deep Agents Code."""

from __future__ import annotations

import errno
import logging
import os
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

_MISSING_ERRNOS = {errno.ENOENT, errno.ENOTDIR}


class PathState(StrEnum):
    """Whether a probed path exists, is absent, or could not be read.

    A `StrEnum` so the value serializes directly to JSON without a custom
    encoder.
    """

    EXISTS = "exists"
    """The path is present on disk."""

    MISSING = "missing"
    """The path is absent (and its parents are readable)."""

    UNREADABLE = "unreadable"
    """Existence could not be determined because `Path.stat()` raised.

    Typically EACCES when a parent directory denies traversal. Kept distinct
    from `MISSING` so diagnostics can flag it as a genuine problem rather than
    a not-yet-created path.
    """


def classify_path(path: Path) -> PathState:
    """Classify a path as existing, missing, or unreadable.

    Args:
        path: Filesystem path to probe.

    Returns:
        `PathState.EXISTS` for a present path, `PathState.MISSING` for expected
            absent-path errors, and `PathState.UNREADABLE` when `Path.stat()`
            raises another `OSError` (e.g. a parent directory denies traversal).
            The error is logged at debug level so an unreadable path is never
            silently indistinguishable from a missing one.
    """
    try:
        path.stat()
    except OSError as exc:
        if exc.errno in _MISSING_ERRNOS:
            return PathState.MISSING
        logger.debug("Could not stat %s", path, exc_info=True)
        return PathState.UNREADABLE
    else:
        return PathState.EXISTS


def get_deepagents_home() -> Path:
    """Return the absolute user-level Deep Agents directory."""
    configured = os.environ.get("DEEPAGENTS_HOME")
    path = Path(configured).expanduser() if configured else Path.home() / ".deepagents"
    return path if path.is_absolute() else Path.cwd() / path
