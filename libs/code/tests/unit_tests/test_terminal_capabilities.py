"""Tests for terminal capability detection."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from deepagents_code import terminal_capabilities
from deepagents_code._env_vars import KITTY_KEYBOARD
from deepagents_code.terminal_capabilities import (
    _inside_terminal_multiplexer,
    _override_supports_kitty_keyboard_protocol,
    _terminal_identity_supports_kitty_keyboard_protocol,
    is_iterm2,
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

    def test_returns_none_for_empty_string(self) -> None:
        """Empty string should defer to heuristic detection, not be treated as off."""
        assert _override_supports_kitty_keyboard_protocol({KITTY_KEYBOARD: ""}) is None

    def test_normalizes_whitespace_and_case(self) -> None:
        """Values lifted from `.env` files routinely carry trailing whitespace."""
        assert (
            _override_supports_kitty_keyboard_protocol({KITTY_KEYBOARD: " TRUE "})
            is True
        )
        assert (
            _override_supports_kitty_keyboard_protocol({KITTY_KEYBOARD: "Off\n"})
            is False
        )


class TestInsideTerminalMultiplexer:
    """Tests for multiplexer detection."""

    def test_detects_tmux_via_socket_var(self) -> None:
        """Tmux exports `TMUX` into every pane it owns."""
        assert _inside_terminal_multiplexer({"TMUX": "/tmp/tmux-0/default,1,0"}) is True

    def test_detects_tmux_via_term(self) -> None:
        """A pane's `TERM` is rewritten even when `TMUX` is stripped."""
        assert _inside_terminal_multiplexer({"TERM": "tmux-256color"}) is True

    def test_detects_screen_via_term(self) -> None:
        """GNU Screen (and tmux's compatibility default) use a `screen` TERM."""
        assert _inside_terminal_multiplexer({"TERM": "screen-256color"}) is True

    def test_detects_screen_via_sty_var(self) -> None:
        """GNU Screen exports `STY` even when `TERM` is overridden."""
        assert _inside_terminal_multiplexer({"STY": "12345.pts-0.host"}) is True

    def test_bare_terminal_is_not_a_multiplexer(self) -> None:
        """A terminal owning its own pty should not be flagged."""
        assert _inside_terminal_multiplexer({"TERM": "xterm-kitty"}) is False


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

    def test_ignores_kitty_window_id_inherited_by_a_tmux_pane(self) -> None:
        """Tmux inherits `KITTY_WINDOW_ID` but never forwards the protocol."""
        assert (
            _terminal_identity_supports_kitty_keyboard_protocol(
                {
                    "KITTY_WINDOW_ID": "17",
                    "TERM": "tmux-256color",
                    "TMUX": "/tmp/tmux-0/default,1,0",
                }
            )
            is False
        )

    def test_ignores_kitty_term_under_screen(self) -> None:
        """A `screen` TERM outranks a stale kitty marker in the environment."""
        assert (
            _terminal_identity_supports_kitty_keyboard_protocol(
                {"KITTY_WINDOW_ID": "17", "TERM": "screen-256color"}
            )
            is False
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


class TestIsIterm2:
    """Tests for iTerm2 terminal identification."""

    def test_detects_lc_terminal(self) -> None:
        """`LC_TERMINAL` survives an SSH hop, so it is checked first."""
        with patch.object(os, "isatty", return_value=True):
            assert is_iterm2({"LC_TERMINAL": "iTerm2"}) is True

    def test_detects_term_program(self) -> None:
        """A local iTerm2 window exports `TERM_PROGRAM`."""
        with patch.object(os, "isatty", return_value=True):
            assert is_iterm2({"TERM_PROGRAM": "iTerm.app"}) is True

    def test_rejects_other_terminals(self) -> None:
        """Another terminal must not inherit iTerm2's workarounds."""
        with patch.object(os, "isatty", return_value=True):
            assert is_iterm2({"TERM_PROGRAM": "Ghostty"}) is False

    def test_requires_a_tty(self) -> None:
        """Inherited markers in CI would otherwise look like a live iTerm2."""
        with patch.object(os, "isatty", return_value=False):
            assert is_iterm2({"TERM_PROGRAM": "iTerm.app"}) is False

    def test_stays_true_inside_a_multiplexer(self) -> None:
        """A forwarded escape is interpreted by the outer terminal, not the pane."""
        with patch.object(os, "isatty", return_value=True):
            assert (
                is_iterm2(
                    {
                        "TERM_PROGRAM": "iTerm.app",
                        "TERM": "tmux-256color",
                        "TMUX": "/tmp/tmux-0/default,1,0",
                    }
                )
                is True
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

    def test_override_can_force_true_inside_tmux(self) -> None:
        """Tmux 3.6 users who know their setup works can still opt in."""
        with (
            _fake_tty(),
            patch.dict(
                os.environ,
                {
                    KITTY_KEYBOARD: "1",
                    "TERM": "tmux-256color",
                    "TMUX": "/tmp/tmux-0/default,1,0",
                },
                clear=True,
            ),
        ):
            assert supports_kitty_keyboard_protocol() is True

    def test_tmux_pane_under_kitty_falls_back_to_legacy_hint(self) -> None:
        """Without an override, a kitty-hosted tmux pane must not claim support."""
        with (
            _fake_tty(),
            patch.dict(
                os.environ,
                {
                    "KITTY_WINDOW_ID": "17",
                    "TERM": "tmux-256color",
                    "TMUX": "/tmp/tmux-0/default,1,0",
                },
                clear=True,
            ),
        ):
            assert supports_kitty_keyboard_protocol() is False

    def test_screen_pane_under_kitty_falls_back_to_legacy_hint(self) -> None:
        """A Screen window that overrode `TERM` still must not claim support."""
        with (
            _fake_tty(),
            patch.dict(
                os.environ,
                {
                    "KITTY_WINDOW_ID": "17",
                    "TERM": "xterm-kitty",
                    "STY": "12345.pts-0.host",
                },
                clear=True,
            ),
        ):
            assert supports_kitty_keyboard_protocol() is False

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
