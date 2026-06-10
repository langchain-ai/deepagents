"""Shared debug-logging configuration for verbose file-based tracing.

When the `DEEPAGENTS_CODE_DEBUG` environment variable is set, modules that handle
streaming or remote communication can enable detailed file-based logging. This
helper centralizes the setup so the env-var name, file path, and format are
defined in one place.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from deepagents_code._env_vars import (
    DEBUG,
    DEBUG_FILE,
    DEFAULT_DEBUG_FILE,
    is_env_truthy,
)

_DEBUG_HANDLER_ATTR = "_deepagents_code_debug_handler"


def configure_debug_logging(target: logging.Logger) -> None:
    """Attach a file handler to *target* when `DEEPAGENTS_CODE_DEBUG` is set.

    Intended to be called once on the `deepagents_code` package logger; child
    module loggers reach the same file via propagation, so individual modules do
    not configure logging themselves.

    The log file defaults to `DEFAULT_DEBUG_FILE` but can be overridden with
    `DEEPAGENTS_CODE_DEBUG_FILE`. The handler appends (`mode='a'`) so logs
    are preserved across separate process runs. Calling this again with the same
    resolved path is a no-op: the existing tagged handler is reused rather than
    stacking duplicates. If the resolved path changes, the stale handler is
    closed and replaced.

    Does nothing when `DEEPAGENTS_CODE_DEBUG` is not truthy (see `is_env_truthy`).

    Args:
        target: Logger to configure.
    """
    if not is_env_truthy(DEBUG):
        return

    debug_path = Path(os.environ.get(DEBUG_FILE, DEFAULT_DEBUG_FILE))
    for existing in list(target.handlers):
        if not (
            isinstance(existing, logging.FileHandler)
            and getattr(existing, _DEBUG_HANDLER_ATTR, False)
        ):
            continue
        if Path(existing.baseFilename) == debug_path:
            # Already configured for this path; reuse rather than duplicate.
            target.setLevel(logging.DEBUG)
            return
        # The debug path changed; drop the stale handler before re-attaching so
        # we don't leak its file descriptor or fan logs out to two files.
        target.removeHandler(existing)
        existing.close()

    try:
        handler = logging.FileHandler(str(debug_path), mode="a")
    except OSError as exc:
        print(  # noqa: T201
            f"Warning: could not open debug log file {debug_path}: {exc}",
            file=sys.stderr,
        )
        return
    setattr(handler, _DEBUG_HANDLER_ATTR, True)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)


def installed_debug_log_path() -> Path | None:
    """Return the path of the active debug log file, or `None` if not logging.

    Reflects the file handler actually attached by `configure_debug_logging`,
    not the current `DEEPAGENTS_CODE_DEBUG` env value. The two diverge when the
    variable is set after import — e.g. via a project/global `.env` loaded during
    settings bootstrap — in which case the variable reads truthy but no handler
    was installed and no log file exists. Callers that surface "full error in
    <path>" hints must use this rather than the env var to avoid pointing users
    at a file that was never created.
    """
    package_logger = logging.getLogger(__package__ or "deepagents_code")
    for handler in package_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, _DEBUG_HANDLER_ATTR, False
        ):
            return Path(handler.baseFilename)
    return None
