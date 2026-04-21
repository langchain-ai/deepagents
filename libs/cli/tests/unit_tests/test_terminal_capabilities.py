"""Tests for terminal capability detection."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from deepagents_cli import terminal_capabilities
from deepagents_cli._env_vars import KITTY_KEYBOARD
from deepagents_cli.terminal_capabilities import (
    _override_supports_kitty_keyboard_protocol,
    _terminal_identity_supports_kitty_keyboard_protocol,
    supports_kitty_keyboard_protocol,
)

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

    def test_returns_true_for_truthy_value(self) -> None:
        """Truthy override values should force kitty support on."""
        assert (
            _override_supports_kitty_keyboard_protocol({KITTY_KEYBOARD: "true"}) is True
        )

    def test_returns_false_for_falsy_value(self) -> None:
        """Falsy override values should force kitty support off."""
        assert (
            _override_supports_kitty_keyboard_protocol({KITTY_KEYBOARD: "off"}) is False
        )

    def test_returns_none_for_auto(self) -> None:
        """`auto` should defer to heuristic detection."""
        assert (
            _override_supports_kitty_keyboard_protocol({KITTY_KEYBOARD: "auto"}) is None
        )

    def test_returns_none_for_invalid_value(self) -> None:
        """Unknown override values should be ignored."""
        assert (
            _override_supports_kitty_keyboard_protocol({KITTY_KEYBOARD: "maybe"})
            is None
        )


class TestTerminalIdentitySupportsKittyKeyboardProtocol:
    """Tests for the conservative terminal-identity heuristic."""

    def test_detects_kitty_via_window_id(self) -> None:
        """`KITTY_WINDOW_ID` is a strong signal that kitty owns the pty."""
        assert (
            _terminal_identity_supports_kitty_keyboard_protocol(
                {"KITTY_WINDOW_ID": "17", "TERM": "xterm-256color"}
            )
            is True
        )

    def test_detects_kitty_via_term(self) -> None:
        """Kitty exports `TERM=xterm-kitty` by default."""
        assert (
            _terminal_identity_supports_kitty_keyboard_protocol({"TERM": "xterm-kitty"})
            is True
        )

    def test_detects_ghostty_via_term(self) -> None:
        """Ghostty exports `TERM=xterm-ghostty` when its terminfo is available."""
        assert (
            _terminal_identity_supports_kitty_keyboard_protocol(
                {"TERM": "xterm-ghostty"}
            )
            is True
        )

    def test_does_not_auto_detect_wezterm(self) -> None:
        """WezTerm leaves kitty-keyboard support behind a user setting."""
        assert (
            _terminal_identity_supports_kitty_keyboard_protocol(
                {"TERM": "xterm-256color", "TERM_PROGRAM": "WezTerm"}
            )
            is False
        )

    def test_does_not_auto_detect_iterm(self) -> None:
        """iTerm2 can disable app-controlled key reporting, so stay conservative."""
        assert (
            _terminal_identity_supports_kitty_keyboard_protocol(
                {"TERM": "xterm-256color", "TERM_PROGRAM": "iTerm.app"}
            )
            is False
        )


class TestSupportsKittyKeyboardProtocolShortCircuits:
    """Short-circuit branches before env heuristics are consulted."""

    def test_returns_false_when_stdin_not_a_tty(self) -> None:
        """Non-interactive stdin should always disable kitty support."""
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = False
        fake_stdout = MagicMock()
        fake_stdout.isatty.return_value = True
        with (
            patch.object(terminal_capabilities.sys, "stdin", fake_stdin),
            patch.object(terminal_capabilities.sys, "stdout", fake_stdout),
        ):
            assert supports_kitty_keyboard_protocol() is False

    def test_returns_false_when_stdout_not_a_tty(self) -> None:
        """Piped stdout should always disable kitty support."""
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True
        fake_stdout = MagicMock()
        fake_stdout.isatty.return_value = False
        with (
            patch.object(terminal_capabilities.sys, "stdin", fake_stdin),
            patch.object(terminal_capabilities.sys, "stdout", fake_stdout),
        ):
            assert supports_kitty_keyboard_protocol() is False

    def test_returns_false_on_windows(self) -> None:
        """Skip kitty detection on Windows."""
        with patch.object(terminal_capabilities.sys, "platform", "win32"):
            assert supports_kitty_keyboard_protocol() is False


class TestSupportsKittyKeyboardProtocolDetection:
    """Tests for override and heuristic behavior on an attached tty."""

    def test_override_can_force_true(self) -> None:
        """Users can opt into the `Shift+Enter` hint explicitly."""
        with _fake_tty(), patch.dict(os.environ, {KITTY_KEYBOARD: "1"}, clear=True):
            assert supports_kitty_keyboard_protocol() is True

    def test_override_can_force_false(self) -> None:
        """Explicit disable should win over positive terminal identity signals."""
        with (
            _fake_tty(),
            patch.dict(
                os.environ,
                {KITTY_KEYBOARD: "0", "KITTY_WINDOW_ID": "17", "TERM": "xterm-kitty"},
                clear=True,
            ),
        ):
            assert supports_kitty_keyboard_protocol() is False

    def test_invalid_override_falls_back_to_term_heuristic(self) -> None:
        """Invalid overrides should not disable known-safe detection."""
        with (
            _fake_tty(),
            patch.dict(
                os.environ,
                {KITTY_KEYBOARD: "maybe", "TERM": "xterm-kitty"},
                clear=True,
            ),
        ):
            assert supports_kitty_keyboard_protocol() is True

    def test_returns_false_for_unrecognized_terminal(self) -> None:
        """Unknown terminals should keep the legacy newline hint."""
        with (
            _fake_tty(),
            patch.dict(
                os.environ,
                {"TERM": "xterm-256color"},
                clear=True,
            ),
        ):
            assert supports_kitty_keyboard_protocol() is False
