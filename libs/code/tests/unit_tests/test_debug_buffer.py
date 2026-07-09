"""Tests for the in-memory log ring buffer backing the Debug Console."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

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
        lines = buffer.lines_since(0)
        assert len(lines) == 1
        assert "hello" in lines[0]
        assert "INFO" in lines[0]
        assert "deepagents_code.x" in lines[0]

    def test_is_bounded_dropping_oldest(self) -> None:
        buffer = InMemoryLogBuffer(capacity=3)
        for i in range(5):
            buffer.emit(_record("deepagents_code", f"msg{i}"))
        lines = buffer.lines_since(0)
        assert len(lines) == 3
        assert "msg2" in lines[0]
        assert "msg4" in lines[-1]
        assert buffer.total_emitted == 5

    def test_lines_since_uses_absolute_index(self) -> None:
        buffer = InMemoryLogBuffer(capacity=10)
        for i in range(4):
            buffer.emit(_record("deepagents_code", f"msg{i}"))
        tail = buffer.lines_since(2)
        assert len(tail) == 2
        assert "msg2" in tail[0]
        assert buffer.lines_since(4) == []
        assert len(buffer.lines_since(-5)) == 4

    def test_lines_since_after_eviction(self) -> None:
        buffer = InMemoryLogBuffer(capacity=2)
        for i in range(5):
            buffer.emit(_record("deepagents_code", f"msg{i}"))
        assert len(buffer.lines_since(0)) == 2
        assert buffer.lines_since(4) == [buffer.lines_since(0)[-1]]

    def test_snapshot_since_returns_lines_and_next_index(self) -> None:
        buffer = InMemoryLogBuffer(capacity=10)
        for i in range(3):
            buffer.emit(_record("deepagents_code", f"msg{i}"))

        lines, total = buffer.snapshot_since(1)

        assert total == 3
        assert len(lines) == 2
        assert "msg1" in lines[0]
        assert "msg2" in lines[1]

    def test_lines_since_is_safe_during_concurrent_emit(self) -> None:
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
                _lines, _total = buffer.snapshot_since(0)
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
    def test_captures_propagated_records(self) -> None:
        logger = logging.getLogger("deepagents_code._test_capture")
        logger.handlers = []
        logger.setLevel(logging.NOTSET)
        buffer = install_log_buffer(logger)
        before = buffer.total_emitted
        logger.info("captured-line")
        assert buffer.total_emitted == before + 1
        assert any("captured-line" in line for line in buffer.lines_since(before))
