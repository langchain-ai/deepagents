"""Tests for terminal capability detection."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from deepagents_code import terminal_capabilities

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def _fake_tty() -> Iterator[None]:
    """Convince the probe both streams are attached to a tty."""
    fake_stdin = MagicMock()
    fake_stdin.isatty.return_value = True
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = True
    with (
        patch.object(terminal_capabilities.sys, "stdin", fake_stdin),
        patch.object(terminal_capabilities.sys, "stdout", fake_stdout),
        patch.object(terminal_capabilities.sys, "platform", "linux"),
    ):
        yield


class TestOverrideSupportsKittyKeyboardProtocol:
    """Tests for the explicit environment-variable override."""


class TestTerminalIdentitySupportsKittyKeyboardProtocol:
    """Tests for the conservative terminal-identity heuristic."""


class TestSupportsKittyKeyboardProtocolShortCircuits:
    """Short-circuit branches before env heuristics are consulted."""


class TestSupportsKittyKeyboardProtocolDetection:
    """Tests for override and heuristic behavior on an attached tty."""
