"""Tests for the in-memory log ring buffer backing the Debug Console."""

from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING

import pytest

import deepagents_code._debug_buffer as debug_buffer
from deepagents_code._debug_buffer import InMemoryLogBuffer

if TYPE_CHECKING:
    from collections.abc import Generator


def _record(name: str, message: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


@pytest.fixture
def _restore_global_buffer() -> Generator[None]:
    """Preserve the module-level singleton across install-mutating tests.

    Yields:
        None; the original singleton is restored on teardown.
    """
    original = debug_buffer._buffer
    try:
        yield
    finally:
        debug_buffer._buffer = original


class TestInMemoryLogBuffer:
    def test_captures_exception_traceback_in_message(self) -> None:
        buffer = InMemoryLogBuffer()
        try:
            msg = "boom-detail"
            raise ValueError(msg)  # noqa: TRY301  # deliberately raised to capture a traceback
        except ValueError:
            record = logging.LogRecord(
                name="deepagents_code.x",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="handler failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        buffer.emit(record)

        records, _total = buffer.snapshot_records_since(0)
        message = records[0].message
        # The exception text is appended to the message as multiple lines.
        assert "handler failed" in message
        assert "Traceback" in message
        assert "boom-detail" in message
        assert "\n" in message

    def test_debug_flood_does_not_evict_other_levels(self) -> None:
        buffer = InMemoryLogBuffer(capacity=3)
        buffer.emit(_record("deepagents_code", "keep-info", level=logging.INFO))
        buffer.emit(_record("deepagents_code", "keep-warning", level=logging.WARNING))
        for i in range(10):  # flood DEBUG well past the per-level capacity
            buffer.emit(_record("deepagents_code", f"debug{i}", level=logging.DEBUG))

        records, total = buffer.snapshot_records_since(0)
        messages = [record.message for record in records]

        # The rarer, higher-severity records survive the DEBUG flood ...
        assert "keep-info" in messages
        assert "keep-warning" in messages
        # ... while DEBUG stays bounded to its own capacity (last 3).
        assert [m for m in messages if m.startswith("debug")] == [
            "debug7",
            "debug8",
            "debug9",
        ]
        # Merged output stays chronological via the emission sequence.
        assert messages == ["keep-info", "keep-warning", "debug7", "debug8", "debug9"]
        assert total == 12

    def test_per_level_bound_is_independent(self) -> None:
        buffer = InMemoryLogBuffer(capacity=2)
        for i in range(5):
            buffer.emit(_record("deepagents_code", f"info{i}", level=logging.INFO))
        for i in range(5):
            buffer.emit(_record("deepagents_code", f"err{i}", level=logging.ERROR))

        records, _total = buffer.snapshot_records_since(0)

        # Each level keeps only its own last `capacity` records.
        assert [record.message for record in records] == [
            "info3",
            "info4",
            "err3",
            "err4",
        ]

    def test_unknown_level_names_share_one_bounded_bucket(self) -> None:
        buffer = InMemoryLogBuffer(capacity=2)
        # Custom numeric levels have level names like "Level 25"; none are in
        # LOG_LEVELS, so they must share the fallback bucket rather than each
        # getting its own unbounded deque.
        for i in range(4):
            buffer.emit(_record("deepagents_code", f"custom{i}", level=25 + i))

        records, _total = buffer.snapshot_records_since(0)

        assert [record.message for record in records] == ["custom2", "custom3"]

    def test_merge_restores_chronological_order_when_levels_overflow(self) -> None:
        """Interleaved levels that both overflow still merge in emission order.

        Unlike `test_per_level_bound_is_independent`, the two levels are emitted
        interleaved, so a merge that concatenated the buckets instead of sorting
        by emission sequence would regroup the tail by level and fail here.
        """
        buffer = InMemoryLogBuffer(capacity=2)
        for i in range(3):  # INFO and ERROR alternate; both overflow capacity=2
            buffer.emit(_record("deepagents_code", f"info{i}", level=logging.INFO))
            buffer.emit(_record("deepagents_code", f"err{i}", level=logging.ERROR))

        records, _total = buffer.snapshot_records_since(0)

        # Each level keeps only its last 2, but the merged tail is chronological
        # (info2 precedes err2 in emission order), not grouped by level.
        assert [record.message for record in records] == [
            "info1",
            "err1",
            "info2",
            "err2",
        ]

    def test_incremental_snapshot_after_eviction_across_levels(self) -> None:
        """Resuming from a prior index skips consumed records but no retained one.

        Reproduces the Debug Console's poll loop: snapshot, then snapshot again
        from the returned resume index after further emits have evicted records
        from one level's bucket. The second snapshot must return only records
        emitted since the resume index, chronologically, with nothing already
        consumed reappearing -- even though the first poll's records are still
        retained in their (un-flooded) buckets.
        """
        buffer = InMemoryLogBuffer(capacity=3)
        buffer.emit(_record("deepagents_code", "info0", level=logging.INFO))
        buffer.emit(_record("deepagents_code", "warn0", level=logging.WARNING))

        first, resume = buffer.snapshot_records_since(0)
        assert [record.message for record in first] == ["info0", "warn0"]
        assert resume == 2

        # Flood DEBUG past its own capacity; the INFO/WARNING buckets are
        # untouched, so info0/warn0 remain retained but already consumed.
        for i in range(5):
            buffer.emit(_record("deepagents_code", f"debug{i}", level=logging.DEBUG))
        buffer.emit(_record("deepagents_code", "info1", level=logging.INFO))

        second, resume2 = buffer.snapshot_records_since(resume)
        messages = [record.message for record in second]

        # Only records emitted since `resume`, in chronological order ...
        assert messages == ["debug2", "debug3", "debug4", "info1"]
        # ... and the still-retained first-poll records are not re-yielded.
        assert "info0" not in messages
        assert "warn0" not in messages
        assert resume2 == 8

    def test_snapshot_is_safe_during_concurrent_emit(self) -> None:
        buffer = InMemoryLogBuffer(capacity=50)
        stop = threading.Event()
        errors: list[BaseException] = []

        def emit_records() -> None:
            i = 0
            while not stop.is_set():
                try:
                    buffer.emit(_record("deepagents_code", f"msg{i}"))
                except BaseException as exc:  # noqa: BLE001  # report thread failures
                    errors.append(exc)
                    stop.set()
                i += 1

        thread = threading.Thread(target=emit_records)
        thread.start()
        try:
            for _ in range(500):
                records, total = buffer.snapshot_records_since(0)
                # No torn read: the resume index is never behind the records.
                assert total >= len(records)
        finally:
            stop.set()
            thread.join(timeout=1)

        assert not errors
