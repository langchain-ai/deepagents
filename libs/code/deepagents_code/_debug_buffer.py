r"""In-memory ring buffer of recent log records for the in-app Debug Console.

A lightweight `logging.Handler` keeps the most recent structured log records in
bounded per-level `deque`s so the Debug Console (`Ctrl+\`) can show a live tail
without requiring the opt-in file logging from `_debug.configure_debug_logging`.
Partitioning retention by level means a burst of `DEBUG` output cannot evict the
rarer `INFO`/`WARNING`/`ERROR` records the level filter needs. The handler is
installed once on the `deepagents_code` package logger (see `__init__.py`); child
module loggers reach it via propagation.

Installation is negligible, and each emitted record only appends a structured
record to a bounded `deque`, so the buffer is cheap enough to keep always on.
"""

from __future__ import annotations

import logging
import operator
import os
from collections import deque
from dataclasses import dataclass

from deepagents_code._debug import LOG_LEVELS
from deepagents_code._env_vars import LOG_LEVEL

DEFAULT_CAPACITY = 1000
"""Maximum number of records retained per level in the ring buffer."""

_FALLBACK_LEVEL_BUCKET = "_other"
"""Retention bucket for records whose level name is not in `LOG_LEVELS`.

Non-standard level names are retained here rather than discarded at
classification time, and sharing one bucket keeps the number of per-level deques
bounded no matter how many distinct custom names appear. Like every bucket, this
one is bounded to *capacity* and evicts oldest-first, so — unlike the standard
levels, which each get isolated retention — distinct custom levels compete for a
single budget and can evict one another."""

_DATE_FORMAT = "%H:%M:%S"
_FORMATTER = logging.Formatter(datefmt=_DATE_FORMAT)


def retention_bucket_for_level(level: str) -> str:
    """Return the bounded retention bucket key for a log level name."""
    return level if level in LOG_LEVELS else _FALLBACK_LEVEL_BUCKET


@dataclass(frozen=True, slots=True)
class InMemoryLogRecord:
    """Structured log record retained by the in-memory debug buffer."""

    timestamp: str
    level: str
    levelno: int
    logger: str
    message: str

    @property
    def plain_line(self) -> str:
        """The record in the legacy plain-text debug-console format."""
        return f"{self.timestamp} {self.level} {self.logger} {self.message}"


class InMemoryLogBuffer(logging.Handler):
    """Logging handler retaining the most recent structured records in memory.

    Retention is partitioned by level: each recognized level name (see
    `LOG_LEVELS`) keeps its own bounded `deque` of *capacity* records, while
    unrecognized names share the `_FALLBACK_LEVEL_BUCKET` deque. A burst of
    high-volume records at one recognized level (typically `DEBUG`) therefore
    cannot evict the rarer, higher-severity records at other recognized levels,
    so the Debug Console's level filter can still surface them. Each retained
    record is tagged with its monotonic emission sequence number so snapshots
    stay chronologically ordered across levels.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        """Create the handler with a bounded backing `deque` per level.

        Args:
            capacity: Maximum records to retain per level; oldest are dropped
                first.

        Raises:
            ValueError: If *capacity* is less than 1. A zero-capacity buffer
                would silently discard every record while `total_emitted` kept
                climbing, so it is rejected at the boundary.
        """
        if capacity < 1:
            msg = f"capacity must be >= 1, got {capacity}"
            raise ValueError(msg)
        super().__init__()
        self._capacity = capacity
        self._levels: dict[str, deque[tuple[int, InMemoryLogRecord]]] = {}
        self._total = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Append the structured *record* to its level's ring buffer."""
        self.acquire()
        try:
            structured = self._make_record(record)
            bucket = self._level_bucket(structured.level)
            bucket.append((self._total, structured))
            self._total += 1
        except Exception:  # noqa: BLE001  # never let logging crash the app
            self.handleError(record)
        finally:
            self.release()

    def _level_bucket(self, level: str) -> deque[tuple[int, InMemoryLogRecord]]:
        """Return the retention deque for *level*, creating it on first use."""
        key = retention_bucket_for_level(level)
        bucket = self._levels.get(key)
        if bucket is None:
            bucket = deque(maxlen=self._capacity)
            self._levels[key] = bucket
        return bucket

    @property
    def total_emitted(self) -> int:
        """Total records ever emitted (monotonic; survives buffer eviction)."""
        self.acquire()
        try:
            return self._total
        finally:
            self.release()

    def snapshot_records_since(self, index: int) -> tuple[list[InMemoryLogRecord], int]:
        """Return retained structured records and the next absolute index.

        Absolute indices are stable even as old records are evicted, so callers
        can poll incrementally without re-reading records they already consumed.
        Returning the records and the resume index together under the handler
        lock prevents a concurrent append from being skipped between separate
        reads.

        Args:
            index: Absolute emission index to start from.

        Returns:
            A tuple of the structured records from *index* onward that are still
            retained, and the current total emitted count to resume from.
        """
        self.acquire()
        try:
            return self._records_since_unlocked(index), self._total
        finally:
            self.release()

    def _records_since_unlocked(self, index: int) -> list[InMemoryLogRecord]:
        """Return retained records from *index* while the handler lock is held.

        Merges the per-level deques and restores chronological order using each
        record's emission sequence number, so callers see one ordered tail even
        though retention is partitioned by level.
        """
        tagged = [
            entry
            for bucket in self._levels.values()
            for entry in bucket
            if entry[0] >= index
        ]
        tagged.sort(key=operator.itemgetter(0))
        return [record for _seq, record in tagged]

    @staticmethod
    def _make_record(record: logging.LogRecord) -> InMemoryLogRecord:
        """Convert a standard logging record into a structured debug record.

        Returns:
            Structured record for display and filtering.
        """
        message = record.getMessage()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = _FORMATTER.formatException(record.exc_info)
            if record.exc_text:
                message = f"{message}\n{record.exc_text}"
        if record.stack_info:
            message = f"{message}\n{_FORMATTER.formatStack(record.stack_info)}"
        return InMemoryLogRecord(
            timestamp=_FORMATTER.formatTime(record, _DATE_FORMAT),
            level=record.levelname,
            levelno=record.levelno,
            logger=record.name,
            message=message,
        )


_buffer: InMemoryLogBuffer | None = None


def install_log_buffer(
    target: logging.Logger, capacity: int = DEFAULT_CAPACITY
) -> InMemoryLogBuffer:
    """Attach the in-memory buffer handler to *target* (idempotent).

    Lowers *target*'s level to at most `INFO` so the console shows a useful tail
    even when `DEEPAGENTS_CODE_DEBUG` is off; never raises the level. In
    `__init__.py` this runs *before* `configure_debug_logging`, which then sets
    the final level (honoring `DEEPAGENTS_CODE_DEBUG` and
    `DEEPAGENTS_CODE_LOG_LEVEL`) over this `INFO` floor — so any startup warnings
    `configure_debug_logging` emits are captured by the already-installed buffer.
    On a fresh `NOTSET` logger the `NOTSET` branch forces `INFO` without
    consulting `DEEPAGENTS_CODE_LOG_LEVEL`; the `> INFO and no env` branch only
    matters on reconfiguration (e.g. an `importlib.reload`), where it preserves
    an explicit `DEEPAGENTS_CODE_LOG_LEVEL` rather than clobbering it with `INFO`.

    Lowering the level does not spill log output onto the terminal: because this
    handler is present in the propagation chain, `Logger.callHandlers` finds a
    handler (`found > 0`) and Python's `lastResort` stderr handler is never
    consulted. The exception is an embedding process that attaches its own
    `INFO`-or-lower handler to the root logger, which would then see the
    propagated records.

    Note: this runs as an import-time side effect (see `__init__.py`), so every
    `import deepagents_code` attaches the handler and may lower the package
    logger's level to `INFO` for the lifetime of the process.

    Args:
        target: Logger to attach the buffer to (the package logger).
        capacity: Maximum records to retain.

    Returns:
        The installed (or already-installed) buffer handler.
    """
    global _buffer  # noqa: PLW0603  # module-level singleton accessor
    for existing in target.handlers:
        if isinstance(existing, InMemoryLogBuffer):
            _buffer = existing
            return existing

    handler = InMemoryLogBuffer(capacity)
    handler.setLevel(logging.DEBUG)
    target.addHandler(handler)
    if target.level == logging.NOTSET or (
        target.level > logging.INFO and not os.environ.get(LOG_LEVEL)
    ):
        target.setLevel(logging.INFO)
    _buffer = handler
    return handler


def get_log_buffer() -> InMemoryLogBuffer | None:
    """Return the installed buffer handler.

    Returns:
        The installed buffer handler, or `None` if not yet installed.
    """
    return _buffer
