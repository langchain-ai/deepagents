"""Stderr marker emission used by the langgraph server graph entry point.

Lives in its own module so unit tests can exercise the marker contract
without triggering `server_graph.make_graph()` at import time.
"""

from __future__ import annotations

import logging
import sys
import traceback

logger = logging.getLogger(__name__)

STARTUP_ERROR_MARKER = "DEEPAGENTS_STARTUP_ERROR:"
"""Stderr marker the parent app scans for in `server._extract_startup_error_marker`
to upgrade an opaque "Server process exited with code N" into a one-line summary.
Format is `{STARTUP_ERROR_MARKER}{single-line message}`."""


def emit_startup_failure(exc: BaseException) -> None:
    """Report a server graph startup failure to the parent app process.

    Emits two stderr outputs: the full traceback for logs/debugging, then a
    single-line `{STARTUP_ERROR_MARKER}{type}: {summary}` line that
    `server._extract_startup_error_marker` parses to upgrade an opaque
    "Server process exited with code N" into an actionable summary.

    Args:
        exc: The exception raised during graph initialization.
    """
    logger.critical("Failed to initialize server graph", exc_info=exc)
    print(  # noqa: T201  # stderr fallback — logger may not reach parent process
        f"Failed to initialize server graph: {exc}\n{traceback.format_exc()}",
        file=sys.stderr,
    )
    # Marker contract is single-line; guard against multi-line/empty `str(exc)`
    # and include the type so e.g. `ValueError` and `RuntimeError` are
    # distinguishable in the parent's truncated summary.
    exc_lines = str(exc).splitlines()
    summary = exc_lines[0] if exc_lines else "<no message>"
    print(  # noqa: T201
        f"{STARTUP_ERROR_MARKER}{type(exc).__name__}: {summary}",
        file=sys.stderr,
    )
