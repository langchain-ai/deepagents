"""Tests for terminal capability detection."""

from __future__ import annotations

from unittest.mock import patch

from deepagents_cli import terminal_capabilities
from deepagents_cli.terminal_capabilities import (
    _parse_kitty_response,
    supports_kitty_keyboard_protocol,
)


class TestParseKittyResponse:
    """Tests for `_parse_kitty_response` pattern matching.

    Keeping this pure-bytes test separate from the full tty probe lets
    us exercise every response shape without needing a real terminal.
    """

    def test_detects_flags_reply_with_digits(self) -> None:
        """`CSI ? <flags> u` is the canonical positive response."""
        assert _parse_kitty_response(b"\x1b[?15u\x1b[?62;1;4c") is True

    def test_detects_flags_reply_with_no_digits(self) -> None:
        """Some terminals report zero flags as `CSI ? u` (no digits)."""
        assert _parse_kitty_response(b"\x1b[?u\x1b[?62c") is True

    def test_ignores_da1_only_response(self) -> None:
        """Terminals without kitty kbd only reply to DA1; no `?...u` segment."""
        assert _parse_kitty_response(b"\x1b[?62;1;4c") is False

    def test_ignores_empty_buffer(self) -> None:
        """No response at all (terminal silent)."""
        assert _parse_kitty_response(b"") is False

    def test_ignores_unrelated_escape_sequences(self) -> None:
        """Incidental escape sequences in the buffer should not false-positive."""
        assert _parse_kitty_response(b"\x1b[2J\x1b[H\x1b[?62c") is False


class TestSupportsKittyKeyboardProtocol:
    """Tests for the public capability probe.

    The function is `functools.cache`-wrapped, so each test clears the
    cache to observe fresh behaviour.
    """

    def setup_method(self) -> None:
        """Clear the module-level cache before each test."""
        supports_kitty_keyboard_protocol.cache_clear()

    def teardown_method(self) -> None:
        """Clear again so leftover mocked results don't leak to other tests."""
        supports_kitty_keyboard_protocol.cache_clear()

    def test_returns_false_when_stdin_not_a_tty(self) -> None:
        """Non-interactive stdin (pipes, redirects, CI) short-circuits to False."""
        with (
            patch.object(terminal_capabilities.sys.stdin, "isatty", return_value=False),
            patch.object(terminal_capabilities.sys.stdout, "isatty", return_value=True),
        ):
            assert supports_kitty_keyboard_protocol() is False

    def test_returns_false_when_stdout_not_a_tty(self) -> None:
        """Piping stdout (e.g. `deepagents ... | tee log`) short-circuits to False."""
        stdin = terminal_capabilities.sys.stdin
        stdout = terminal_capabilities.sys.stdout
        with (
            patch.object(stdin, "isatty", return_value=True),
            patch.object(stdout, "isatty", return_value=False),
        ):
            assert supports_kitty_keyboard_protocol() is False

    def test_returns_false_on_windows(self) -> None:
        """Skip the probe on Windows — ConPTY kitty support is inconsistent."""
        with patch.object(terminal_capabilities.sys, "platform", "win32"):
            assert supports_kitty_keyboard_protocol() is False
