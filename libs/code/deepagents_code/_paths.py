"""Filesystem-path classification shared by the diagnostic CLI commands.

`dcode doctor` and `dcode config path` both probe whether config locations
exist. `Path.exists()` ignores the "absent" error families (ENOENT, ENOTDIR,
ELOOP, EBADF) but propagates others — notably EACCES when a parent directory
denies traversal — so a bare `.exists()` can crash the very command meant to
diagnose a broken install. `classify_path` centralizes the guard and reports
an unreadable path as a distinct state instead of conflating it with "missing".
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


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
    """Existence could not be determined because `Path.exists()` raised.

    Typically EACCES when a parent directory denies traversal. Kept distinct
    from `MISSING` so diagnostics can flag it as a genuine problem rather than
    a not-yet-created path.
    """


def classify_path(path: Path) -> PathState:
    """Classify a path as existing, missing, or unreadable.

    Args:
        path: Filesystem path to probe.

    Returns:
        `PathState.EXISTS` or `PathState.MISSING` for a readable path, and
            `PathState.UNREADABLE` when `Path.exists()` raises (e.g. a parent
            directory denies traversal). The error is logged at debug level
            so an unreadable path is never silently indistinguishable
            from a missing one.
    """
    try:
        return PathState.EXISTS if path.exists() else PathState.MISSING
    except OSError:
        logger.debug("Could not stat %s", path, exc_info=True)
        return PathState.UNREADABLE
