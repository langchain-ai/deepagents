"""Tests for the macOS terminal stderr guard."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from deepagents_code._terminal_stderr import (
    TerminalStderrGuard,
    _stderr_targets_stdout,
    stdout_driver_class,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_stderr(message: str) -> None:
    os.write(2, message.encode())


def test_suppresses_stderr_only_while_terminal_is_owned(tmp_path: Path) -> None:
    output_path = tmp_path / "stderr.log"
    output_fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    original_stderr = os.dup(2)
    try:
        os.dup2(output_fd, 2)
        guard = TerminalStderrGuard(enabled=True)
        guard.resume()
        _write_stderr("hidden while active\n")
        guard.pause()
        _write_stderr("visible while paused\n")
        guard.resume()
        _write_stderr("hidden after resume\n")
        guard.close()
        _write_stderr("visible after close\n")
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)
        os.close(output_fd)

    assert output_path.read_text() == "visible while paused\nvisible after close\n"


def test_install_preserves_redirected_stderr(tmp_path: Path) -> None:
    output_path = tmp_path / "stderr.log"
    output_fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    original_stderr = os.dup(2)
    try:
        os.dup2(output_fd, 2)
        with patch("deepagents_code._terminal_stderr.sys.platform", "darwin"):
            guard = TerminalStderrGuard.install()
        _write_stderr("visible while redirected\n")
        guard.close()
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)
        os.close(output_fd)

    assert output_path.read_text() == "visible while redirected\n"


def test_install_is_noop_outside_macos() -> None:
    with (
        patch("deepagents_code._terminal_stderr.sys.platform", "linux"),
        patch("deepagents_code._terminal_stderr.os.isatty") as isatty,
    ):
        guard = TerminalStderrGuard.install()

    assert guard.active is False
    isatty.assert_not_called()


async def test_stdout_driver_renders_to_stdout() -> None:  # noqa: RUF029
    """The subclass must flip the stream Textual's stock driver writes to.

    Must be async: `Driver.__init__` calls `asyncio.get_running_loop()`. The
    drivers are built under a patched `signal.signal` because `LinuxDriver`
    registers process-global SIGTSTP/SIGCONT handlers it never removes.
    """
    from textual.drivers.linux_driver import LinuxDriver

    app = SimpleNamespace()
    driver_class = stdout_driver_class()
    assert driver_class is not None
    with patch("signal.signal"):
        stock = LinuxDriver(app)  # ty: ignore[invalid-argument-type]
        driver = driver_class(app)  # ty: ignore[invalid-argument-type]

    # Pin the upstream contract too: if Textual renames `_file`, the subclass
    # writes a dead attribute and the TUI silently renders to the guarded
    # stderr. That must fail here rather than in a user's terminal.
    assert stock._file is sys.__stderr__
    assert driver._file is sys.__stdout__  # ty: ignore[unresolved-attribute]


def test_stdout_driver_class_declines_when_stdout_is_unusable() -> None:
    """A missing or closed stdout must not yield a driver that renders blind."""
    with patch("deepagents_code._terminal_stderr.sys.__stdout__", None):
        assert stdout_driver_class() is None

    with patch(
        "deepagents_code._terminal_stderr.sys.__stdout__",
        SimpleNamespace(closed=True),
    ):
        assert stdout_driver_class() is None


def test_stdout_driver_class_defers_to_explicit_textual_driver() -> None:
    """An explicit `TEXTUAL_DRIVER` must win over the guard's override."""
    with patch("textual.constants.DRIVER", "my.module:MyDriver"):
        assert stdout_driver_class() is None


def test_stdout_driver_rejects_moved_textual_internals() -> None:
    """Textual moving its output stream must raise, not render to /dev/null."""
    driver_class = stdout_driver_class()
    assert driver_class is not None

    def _init_without_file(*_: object, **__: object) -> None:
        return

    with (
        patch("signal.signal"),
        patch.object(driver_class.__mro__[1], "__init__", _init_without_file),
        pytest.raises(RuntimeError, match="no longer stores its output stream"),
    ):
        driver_class(SimpleNamespace())  # ty: ignore[invalid-argument-type]


def test_same_terminal_requires_matching_device_and_inode() -> None:
    same = SimpleNamespace(st_dev=1, st_ino=2)
    other = SimpleNamespace(st_dev=1, st_ino=3)
    with (
        patch("deepagents_code._terminal_stderr.os.isatty", return_value=True),
        patch("deepagents_code._terminal_stderr.os.fstat", side_effect=[same, other]),
    ):
        assert _stderr_targets_stdout() is False

    with (
        patch("deepagents_code._terminal_stderr.os.isatty", return_value=True),
        patch("deepagents_code._terminal_stderr.os.fstat", side_effect=[same, same]),
    ):
        assert _stderr_targets_stdout() is True
