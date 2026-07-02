r"""In-memory ring buffer of recent log records for the in-app Debug Console.

A lightweight `logging.Handler` keeps the most recent formatted log records in a
bounded `deque` so the Debug Console (`Ctrl+\`) can show a live tail without
requiring the opt-in file logging from `_debug.configure_debug_logging`. The
handler is installed once on the `deepagents_code` package logger (see
`__init__.py`); child module loggers reach it via propagation.

The buffer only appends in-memory strings, so it is cheap enough to keep always
on without affecting startup performance.
"""

from __future__ import annotations

import logging
from collections import deque

_BUFFER_HANDLER_ATTR = "_deepagents_code_log_buffer_handler"
"""Marker attribute tagging the handler so install is idempotent."""

DEFAULT_CAPACITY = 1000
"""Maximum number of records retained in the ring buffer."""

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class InMemoryLogBuffer(logging.Handler):
    """Logging handler retaining the most recent formatted records in memory."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        """Create the handler with a bounded backing `deque`.

        Args:
            capacity: Maximum records to retain; oldest are dropped first.
        """
        super().__init__()
        self._records: deque[str] = deque(maxlen=capacity)
        self._total = 0
        self.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        """Append the formatted *record* to the ring buffer."""
        try:
            self._records.append(self.format(record))
            self._total += 1
        except Exception:  # noqa: BLE001  # never let logging crash the app
            self.handleError(record)

    @property
    def total_emitted(self) -> int:
        """Total records ever emitted (monotonic; survives buffer eviction)."""
        return self._total

    def lines_since(self, index: int) -> list[str]:
        """Return retained lines whose absolute emission index is `>= index`.

        Absolute indices are stable even as old records are evicted, so callers
        can poll incrementally without re-reading lines they already consumed.

        Args:
            index: Absolute emission index to start from.

        Returns:
            The formatted lines from *index* onward that are still retained.
        """
        start_abs = self._total - len(self._records)
        offset = index - start_abs
        if offset <= 0:
            return list(self._records)
        if offset >= len(self._records):
            return []
        return list(self._records)[offset:]


_buffer: InMemoryLogBuffer | None = None


def install_log_buffer(
    target: logging.Logger, capacity: int = DEFAULT_CAPACITY
) -> InMemoryLogBuffer:
    """Attach the in-memory buffer handler to *target* (idempotent).

    Lowers *target*'s level to at most `INFO` so the console shows a useful tail
    even when `DEEPAGENTS_CODE_DEBUG` is off; never raises the level, so the
    `DEBUG` level set by `configure_debug_logging` is preserved. Records that
    propagate to the root logger stay invisible because the root's
    last-resort handler only emits `WARNING` and above.

    Args:
        target: Logger to attach the buffer to (the package logger).
        capacity: Maximum records to retain.

    Returns:
        The installed (or already-installed) buffer handler.
    """
    global _buffer  # noqa: PLW0603  # module-level singleton accessor
    for existing in target.handlers:
        if isinstance(existing, InMemoryLogBuffer) and getattr(
            existing, _BUFFER_HANDLER_ATTR, False
        ):
            _buffer = existing
            return existing

    handler = InMemoryLogBuffer(capacity)
    setattr(handler, _BUFFER_HANDLER_ATTR, True)
    handler.setLevel(logging.DEBUG)
    target.addHandler(handler)
    if target.level == logging.NOTSET or target.level > logging.INFO:
        target.setLevel(logging.INFO)
    _buffer = handler
    return handler


def get_log_buffer() -> InMemoryLogBuffer | None:
    """Return the installed buffer handler, or `None` if not yet installed."""
    return _buffer
