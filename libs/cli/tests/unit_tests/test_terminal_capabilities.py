"""Tests for terminal capability detection."""

from __future__ import annotations

import contextlib
import termios
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from deepagents_cli import terminal_capabilities
from deepagents_cli.terminal_capabilities import (
    _MAX_RESPONSE_BYTES,
    _parse_kitty_response,
    supports_kitty_keyboard_protocol,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


_FAKE_FD = 99
"""Fake fd the mocked syscalls operate on — never hits the kernel."""


@contextlib.contextmanager
def _fake_tty() -> Iterator[None]:
    """Convince the probe both streams are a real tty.

    pytest's stdin capture wrapper raises from `fileno()` and isn't
    patchable via `patch.object`, so we swap `sys.stdin`/`sys.stdout`
    wholesale with `MagicMock`s that return a fake fd. Every syscall
    the probe makes is mocked by the caller, so the fake fd is inert.
    """
    fake_stdin = MagicMock()
    fake_stdin.isatty.return_value = True
    fake_stdin.fileno.return_value = _FAKE_FD
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = True
    fake_stdout.fileno.return_value = _FAKE_FD
    with (
        patch.object(terminal_capabilities.sys, "stdin", fake_stdin),
        patch.object(terminal_capabilities.sys, "stdout", fake_stdout),
        patch.object(terminal_capabilities.sys, "platform", "linux"),
    ):
        yield


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


class TestSupportsKittyKeyboardProtocolShortCircuits:
    """Short-circuit branches that never reach the tty query."""

    def test_returns_false_when_stdin_not_a_tty(self) -> None:
        """Non-interactive stdin (pipes, redirects, CI) short-circuits to False."""
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
        """Piping stdout (e.g. `deepagents ... | tee log`) short-circuits to False."""
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
        """Skip the probe on Windows — ConPTY kitty support is inconsistent."""
        with patch.object(terminal_capabilities.sys, "platform", "win32"):
            assert supports_kitty_keyboard_protocol() is False


class TestTtyProbePath:
    """Tests for the termios / select / os.read portion of the probe.

    These patch the low-level syscalls so we exercise the cleanup
    contract (must restore tty state even on failure) without needing a
    real terminal attached.
    """

    def test_tcgetattr_error_returns_false_and_does_not_write(self) -> None:
        """If `tcgetattr` can't read tty state, bail before writing the query.

        A failed `tcgetattr` means we don't have a snapshot to restore
        to; issuing the query would leave the terminal in an unknown
        state. The probe must short-circuit cleanly.
        """
        with (
            _fake_tty(),
            patch("termios.tcgetattr", side_effect=termios.error("no tty")),
            patch("os.write") as write_mock,
        ):
            assert supports_kitty_keyboard_protocol() is False
            assert write_mock.call_count == 0

    def test_select_timeout_returns_false_and_restores_tty(self) -> None:
        """Terminal silent → return False, still restore original tty state."""
        sentinel_attrs = object()
        with (
            _fake_tty(),
            patch("termios.tcgetattr", return_value=sentinel_attrs),
            patch("tty.setcbreak"),
            patch("os.write"),
            patch("select.select", return_value=([], [], [])),
            patch("termios.tcsetattr") as tcsetattr_mock,
        ):
            assert supports_kitty_keyboard_protocol() is False
            tcsetattr_mock.assert_called_once_with(
                _FAKE_FD, termios.TCSANOW, sentinel_attrs
            )

    def test_os_read_error_restores_tty_state(self) -> None:
        """An `OSError` mid-read must still restore tty state via `finally`."""
        sentinel_attrs = object()
        with (
            _fake_tty(),
            patch("termios.tcgetattr", return_value=sentinel_attrs),
            patch("tty.setcbreak"),
            patch("os.write"),
            patch("select.select", return_value=([_FAKE_FD], [], [])),
            patch("os.read", side_effect=OSError("EIO")),
            patch("termios.tcsetattr") as tcsetattr_mock,
        ):
            assert supports_kitty_keyboard_protocol() is False
            tcsetattr_mock.assert_called_once()

    def test_positive_response_parsed_from_streamed_bytes(self) -> None:
        """A terminal replying with `CSI ? 15 u` must be detected as supporting."""
        replies = [b"\x1b[?15u\x1b[?62;1;4c"]

        def read_stub(_fd: int, _n: int) -> bytes:
            return replies.pop(0) if replies else b""

        with (
            _fake_tty(),
            patch("termios.tcgetattr", return_value=object()),
            patch("tty.setcbreak"),
            patch("os.write"),
            patch("select.select", return_value=([_FAKE_FD], [], [])),
            patch("os.read", side_effect=read_stub),
            patch("termios.tcsetattr"),
        ):
            assert supports_kitty_keyboard_protocol() is True

    def test_response_is_capped_at_max_bytes(self) -> None:
        """A rogue terminal streaming forever must not exceed the byte cap."""
        read_calls: list[int] = []

        def read_stub(_fd: int, n: int) -> bytes:
            read_calls.append(n)
            return b"x" * n

        with (
            _fake_tty(),
            patch("termios.tcgetattr", return_value=object()),
            patch("tty.setcbreak"),
            patch("os.write"),
            patch("select.select", return_value=([_FAKE_FD], [], [])),
            patch("os.read", side_effect=read_stub),
            patch("termios.tcsetattr"),
        ):
            assert supports_kitty_keyboard_protocol() is False
            # Bytes read are capped by the `len(buffer) < _MAX_RESPONSE_BYTES`
            # loop guard — exact total depends on per-read chunk size but
            # must stay within a small multiple of the cap.
            assert sum(read_calls) < _MAX_RESPONSE_BYTES * 2

    def test_tcsetattr_failure_is_swallowed(self) -> None:
        """If the tty restore itself fails, the probe must still return normally."""
        with (
            _fake_tty(),
            patch("termios.tcgetattr", return_value=object()),
            patch("tty.setcbreak"),
            patch("os.write"),
            patch("select.select", return_value=([], [], [])),
            patch("termios.tcsetattr", side_effect=termios.error("restore failed")),
        ):
            # Should not raise; probe returns False since no response arrived.
            assert supports_kitty_keyboard_protocol() is False
