"""Tests for the in-memory log ring buffer backing the Debug Console."""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import deepagents_code._debug_buffer as debug_buffer
from deepagents_code._debug_buffer import (
    InMemoryLogBuffer,
    get_log_buffer,
    install_log_buffer,
)

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


def _lines(buffer: InMemoryLogBuffer, index: int) -> list[str]:
    """Return the retained plain-text lines from *index* onward."""
    records, _total = buffer.snapshot_records_since(index)
    return [record.plain_line for record in records]


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
    def test_captures_and_formats_records(self) -> None:
        buffer = InMemoryLogBuffer()
        buffer.emit(_record("deepagents_code.x", "hello"))
        lines = _lines(buffer, 0)
        assert len(lines) == 1
        assert "hello" in lines[0]
        assert "INFO" in lines[0]
        assert "deepagents_code.x" in lines[0]

    def test_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            InMemoryLogBuffer(capacity=0)
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            InMemoryLogBuffer(capacity=-1)

    def test_records_carry_numeric_level(self) -> None:
        buffer = InMemoryLogBuffer()
        buffer.emit(_record("deepagents_code.x", "hello", level=logging.WARNING))
        records, _total = buffer.snapshot_records_since(0)
        assert records[0].level == "WARNING"
        assert records[0].levelno == logging.WARNING

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

    def test_emit_never_raises_on_malformed_record(self) -> None:
        """A record that can't be formatted is dropped, not propagated.

        `emit` upholds logging's "never crash the caller" contract: a bad
        printf-style record routes to `handleError` instead of raising, and the
        dropped record must not inflate `total_emitted` (the `+= 1` sits after
        the append that raised).
        """
        buffer = InMemoryLogBuffer()
        bad = logging.LogRecord(
            name="deepagents_code.x",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="%d",  # %-format expects an int; the str arg raises in getMessage
            args=("not-an-int",),
            exc_info=None,
        )
        with patch.object(buffer, "handleError") as handle_error:
            buffer.emit(bad)  # must not raise
        handle_error.assert_called_once_with(bad)
        records, total = buffer.snapshot_records_since(0)
        assert records == []
        assert total == 0
        assert buffer.total_emitted == 0

    def test_is_bounded_dropping_oldest(self) -> None:
        buffer = InMemoryLogBuffer(capacity=3)
        for i in range(5):
            buffer.emit(_record("deepagents_code", f"msg{i}"))
        lines = _lines(buffer, 0)
        assert len(lines) == 3
        assert "msg2" in lines[0]
        assert "msg4" in lines[-1]
        assert buffer.total_emitted == 5

    def test_snapshot_uses_absolute_index(self) -> None:
        buffer = InMemoryLogBuffer(capacity=10)
        for i in range(4):
            buffer.emit(_record("deepagents_code", f"msg{i}"))
        tail = _lines(buffer, 2)
        assert len(tail) == 2
        assert "msg2" in tail[0]
        assert _lines(buffer, 4) == []
        assert len(_lines(buffer, -5)) == 4

    def test_snapshot_after_eviction(self) -> None:
        buffer = InMemoryLogBuffer(capacity=2)
        for i in range(5):
            buffer.emit(_record("deepagents_code", f"msg{i}"))
        assert len(_lines(buffer, 0)) == 2
        assert _lines(buffer, 4) == [_lines(buffer, 0)[-1]]

    def test_snapshot_since_returns_records_and_next_index(self) -> None:
        buffer = InMemoryLogBuffer(capacity=10)
        for i in range(3):
            buffer.emit(_record("deepagents_code", f"msg{i}"))

        records, total = buffer.snapshot_records_since(1)

        assert total == 3
        assert len(records) == 2
        assert "msg1" in records[0].message
        assert "msg2" in records[1].message

    def test_snapshot_records_since_returns_structured_records(self) -> None:
        buffer = InMemoryLogBuffer(capacity=10)
        buffer.emit(_record("deepagents_code.x", "hello", level=logging.WARNING))

        records, total = buffer.snapshot_records_since(0)

        assert total == 1
        assert len(records) == 1
        assert records[0].level == "WARNING"
        assert records[0].logger == "deepagents_code.x"
        assert records[0].message == "hello"
        assert records[0].plain_line in _lines(buffer, 0)

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


class TestInstallLogBuffer:
    @pytest.mark.usefixtures("_restore_global_buffer")
    def test_install_is_idempotent(self) -> None:
        logger = logging.getLogger("deepagents_code._test_idempotent")
        logger.handlers = []
        first = install_log_buffer(logger)
        second = install_log_buffer(logger)
        assert first is second
        installed = [h for h in logger.handlers if isinstance(h, InMemoryLogBuffer)]
        assert len(installed) == 1
        assert get_log_buffer() is first

    @pytest.mark.usefixtures("_restore_global_buffer")
    def test_install_lowers_level_to_info_only(self) -> None:
        logger = logging.getLogger("deepagents_code._test_level_notset")
        logger.handlers = []
        logger.setLevel(logging.NOTSET)
        install_log_buffer(logger)
        assert logger.level == logging.INFO

    @pytest.mark.usefixtures("_restore_global_buffer")
    def test_install_preserves_debug_level(self) -> None:
        logger = logging.getLogger("deepagents_code._test_level_debug")
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        install_log_buffer(logger)
        assert logger.level == logging.DEBUG

    @pytest.mark.usefixtures("_restore_global_buffer")
    def test_install_lowers_high_level_without_env(self) -> None:
        # A logger above INFO with no explicit env override drops to INFO so the
        # always-on tail stays useful even when DEEPAGENTS_CODE_DEBUG is off.
        logger = logging.getLogger("deepagents_code._test_level_high_no_env")
        logger.handlers = []
        logger.setLevel(logging.WARNING)
        with patch.dict(os.environ, {}, clear=True):
            install_log_buffer(logger)
        assert logger.level == logging.INFO

    @pytest.mark.usefixtures("_restore_global_buffer")
    def test_install_preserves_explicit_env_log_level(self) -> None:
        logger = logging.getLogger("deepagents_code._test_level_warning")
        logger.handlers = []
        logger.setLevel(logging.WARNING)
        with patch.dict(os.environ, {"DEEPAGENTS_CODE_LOG_LEVEL": "WARNING"}):
            install_log_buffer(logger)
        assert logger.level == logging.WARNING

    @pytest.mark.usefixtures("_restore_global_buffer")
    def test_captures_propagated_records(self) -> None:
        logger = logging.getLogger("deepagents_code._test_capture")
        logger.handlers = []
        logger.setLevel(logging.NOTSET)
        buffer = install_log_buffer(logger)
        before = buffer.total_emitted
        logger.info("captured-line")
        assert buffer.total_emitted == before + 1
        assert any("captured-line" in line for line in _lines(buffer, before))
