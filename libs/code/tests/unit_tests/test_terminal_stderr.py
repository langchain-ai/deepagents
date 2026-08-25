"""Tests for the macOS terminal stderr guard."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from deepagents_code._terminal_stderr import TerminalStderrGuard, _stderr_targets_stdout

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
