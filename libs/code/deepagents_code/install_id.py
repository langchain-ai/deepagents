"""Stable, anonymous per-install identifier.

A random ID generated once and persisted under the profile state directory. It
carries no user information (no email, account, hostname, or path); it only lets
requests from the same install share a provider-side grouping key, such as an
OpenAI `prompt_cache_key`.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import uuid
from contextlib import suppress
from pathlib import Path

logger = logging.getLogger(__name__)

INSTALL_ID_FILENAME = "install_id"
"""Filename under the profile state directory that holds the install ID."""

_INSTALL_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
"""Shape of a valid install ID (`uuid4().hex`)."""

_cache: dict[Path, str] = {}
_lock = threading.Lock()


def get_install_id(state_dir: Path | None = None) -> str:
    """Return this install's anonymous ID, creating and persisting it if needed.

    The value is read once per process and cached. If the state directory
    cannot be read or written, a random ID is used for the rest of the process
    instead, so callers always get a usable value.

    Performs blocking file I/O on first call; async callers should offload.

    Args:
        state_dir: Optional state directory override for tests.

    Returns:
        A 32-character lowercase hex string.
    """
    if state_dir is None:
        from deepagents_code.model_config import DEFAULT_STATE_DIR

        state_dir = DEFAULT_STATE_DIR
    path = state_dir / INSTALL_ID_FILENAME
    with _lock:
        cached = _cache.get(path)
        if cached is None:
            cached = _read_install_id(path) or _create_install_id(path)
            _cache[path] = cached
        return cached


def _read_install_id(path: Path) -> str | None:
    """Return the persisted install ID, or `None` when missing or malformed."""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("Could not read install ID at %s", path, exc_info=True)
        return None
    if _INSTALL_ID_PATTERN.fullmatch(value):
        return value
    logger.warning("Ignoring malformed install ID at %s; regenerating", path)
    return None


def _create_install_id(path: Path) -> str:
    """Generate a new install ID and persist it (best effort).

    Returns:
        The new ID, whether or not it could be written.
    """
    value = uuid.uuid4().hex
    try:
        _write_install_id(path, value)
    except OSError:
        logger.warning(
            "Could not persist install ID at %s; using a per-process ID",
            path,
            exc_info=True,
        )
    return value


def _write_install_id(path: Path, value: str) -> None:
    """Write `value` to `path` atomically via a sibling temp file.

    Raises:
        OSError: If the directory or temp file cannot be written, or the
            replace fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=".install_id-", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{value}\n")
        tmp_path.replace(path)
    except OSError:
        with suppress(OSError):
            tmp_path.unlink()
        raise
