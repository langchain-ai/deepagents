"""Shared debug-logging configuration for verbose file-based tracing.

When the `DEEPAGENTS_CLI_DEBUG` environment variable is set, modules that handle
streaming or remote communication can enable detailed file-based logging. This
helper centralizes the setup so the env-var name, file path, and format are
defined in one place.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from deepagents_cli._env_vars import DEBUG, DEBUG_FILE, is_env_truthy


def configure_debug_logging(target: logging.Logger) -> None:
    """Attach a file handler to *target* when `DEEPAGENTS_CLI_DEBUG` is set.

    The log file defaults to `'/tmp/deepagents_debug.log'` but can be overridden
    with `DEEPAGENTS_CLI_DEBUG_FILE`. The handler appends so that multiple
    modules share the same log file across a session.

    Does nothing when `DEEPAGENTS_CLI_DEBUG` is not truthy (see `is_env_truthy`).

    Args:
        target: Logger to configure.
    """
    if not is_env_truthy(DEBUG):
        return

    debug_path = Path(
        os.environ.get(
            DEBUG_FILE,
            "/tmp/deepagents_debug.log",  # noqa: S108
        )
    )
    try:
        handler = logging.FileHandler(str(debug_path), mode="a")
    except OSError as exc:
        import sys

        print(  # noqa: T201
            f"Warning: could not open debug log file {debug_path}: {exc}",
            file=sys.stderr,
        )
        return
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
