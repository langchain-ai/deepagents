"""Tests for `deepagents_cli.terminal_escape`."""

from __future__ import annotations

import io
import logging
import pathlib

import pytest

from deepagents_cli import terminal_escape
from deepagents_cli.terminal_escape import (
    TerminalProgressState,
    _validate_progress,
    clear_terminal_progress,
    reset_terminal_background,
    set_terminal_background,
    set_terminal_progress,
    write_osc,
    write_terminal_escape,
)


@pytest.fixture(autouse=True)
def _reset_active_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level terminal state sentinels between tests."""
    monkeypatch.setattr(terminal_escape, "_progress_active", False)
    monkeypatch.setattr(terminal_escape, "_terminal_background_active", False)
    monkeypatch.setattr(terminal_escape, "_atexit_registered", False)
    monkeypatch.delenv(terminal_escape.NO_TERMINAL_ESCAPE, raising=False)


class _FakeTTY(io.StringIO):
    """`StringIO` with a context-manager that doesn't truncate on close."""

    def __enter__(self) -> _FakeTTY:  # noqa: PYI034  # _FakeTTY is a test helper
        return self

    def __exit__(self, *exc: object) -> None:
        pass


class TestWriteTerminalEscape:
    """Tests for `write_terminal_escape`."""

    def test_writes_to_tty_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert write_terminal_escape("\x1b[?25l") is True
        assert fake.getvalue() == "\x1b[?25l"

    def test_falls_back_to_stderr_when_tty_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: None)
        monkeypatch.setattr(terminal_escape, "_is_stream_tty", lambda _stream: True)
        buf = io.StringIO()
        monkeypatch.setattr("sys.__stderr__", buf)
        assert write_terminal_escape("\x1b]9;4;0;0\a") is True
        assert buf.getvalue() == "\x1b]9;4;0;0\a"

    def test_no_op_when_no_tty_and_stderr_redirected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: None)
        monkeypatch.setattr(terminal_escape, "_is_stream_tty", lambda _stream: False)
        assert write_terminal_escape("\x1b]9;4;0;0\a") is False

    def test_no_op_when_disabled_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(terminal_escape.NO_TERMINAL_ESCAPE, "1")
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert write_terminal_escape("\x1b]9;4;0;0\a") is False
        assert fake.getvalue() == ""

    def test_no_op_for_empty_sequence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert write_terminal_escape("") is False
        assert fake.getvalue() == ""

    def test_oserror_during_write_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _RaisingTTY(_FakeTTY):
            def write(self, _data: str) -> int:
                msg = "disconnected"
                raise OSError(msg)

        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: _RaisingTTY())
        monkeypatch.setattr(terminal_escape, "_is_stream_tty", lambda _stream: False)
        assert write_terminal_escape("\x1b]9;4;0;0\a") is False

    def test_real_open_tty_oserror_falls_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercise the real `_open_tty` body, not a stub."""

        def _raising_open(*_args: object, **_kwargs: object) -> None:
            msg = "no tty"
            raise OSError(msg)

        monkeypatch.setattr(pathlib.Path, "open", _raising_open)
        monkeypatch.setattr(terminal_escape, "_is_stream_tty", lambda _stream: False)
        assert write_terminal_escape("\x1b]9;4;0;0\a") is False

    def test_stderr_fallback_write_failure_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When `/dev/tty` is missing and stderr write raises, return `False`."""

        class _RaisingStderr:
            def write(self, _data: str) -> int:
                msg = "stderr closed"
                raise OSError(msg)

            def flush(self) -> None: ...

        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: None)
        monkeypatch.setattr(terminal_escape, "_is_stream_tty", lambda _stream: True)
        monkeypatch.setattr("sys.__stderr__", _RaisingStderr())
        assert write_terminal_escape("\x1b]9;4;0;0\a") is False

    def test_unicode_payload_round_trips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSC payloads with non-ASCII bytes are written verbatim."""
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert write_osc("9;4", "テスト") is True
        assert fake.getvalue() == "\x1b]9;4;テスト\a"


class TestWriteOsc:
    """Tests for `write_osc`."""

    def test_default_terminator_is_bel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        write_osc("9;4", "3;0")
        assert fake.getvalue() == "\x1b]9;4;3;0\a"

    def test_st_terminator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        write_osc("9;4", "0;0", st=True)
        assert fake.getvalue() == "\x1b]9;4;0;0\x1b\\"

    def test_empty_payload_omits_separator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        write_osc("0")
        assert fake.getvalue() == "\x1b]0\a"


class TestValidateProgress:
    """Tests for `_validate_progress`."""

    def test_clear_state_normalizes_to_zero(self) -> None:
        assert _validate_progress(50, TerminalProgressState.CLEAR) == 0

    def test_indeterminate_normalizes_to_zero(self) -> None:
        assert _validate_progress(50, TerminalProgressState.INDETERMINATE) == 0

    def test_determinate_passthrough(self) -> None:
        assert _validate_progress(42, TerminalProgressState.NORMAL) == 42

    def test_determinate_clamps_low(self) -> None:
        assert _validate_progress(-10, TerminalProgressState.NORMAL) == 0

    def test_determinate_clamps_high(self) -> None:
        assert _validate_progress(250, TerminalProgressState.ERROR) == 100

    def test_none_progress_for_determinate_becomes_zero(self) -> None:
        assert _validate_progress(None, TerminalProgressState.NORMAL) == 0

    def test_non_numeric_progress_coerces_to_zero(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Bad types are logged + treated as zero, never raised."""
        with caplog.at_level(logging.DEBUG, logger=terminal_escape.__name__):
            assert _validate_progress("nope", TerminalProgressState.NORMAL) == 0  # type: ignore[arg-type]
        assert any("non-numeric" in record.message for record in caplog.records)

    def test_clear_with_nonzero_progress_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Supplying progress to CLEAR/INDETERMINATE is observable misuse."""
        with caplog.at_level(logging.DEBUG, logger=terminal_escape.__name__):
            assert _validate_progress(42, TerminalProgressState.CLEAR) == 0
        assert any("ignoring progress" in record.message for record in caplog.records)


class TestSetTerminalProgress:
    """Tests for `set_terminal_progress` / `clear_terminal_progress`."""

    def test_normal_progress_writes_state_and_percent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert set_terminal_progress(75, state=TerminalProgressState.NORMAL) is True
        assert fake.getvalue() == "\x1b]9;4;1;75\a"

    def test_indeterminate_emits_zero_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        set_terminal_progress(state=TerminalProgressState.INDETERMINATE)
        assert fake.getvalue() == "\x1b]9;4;3;0\a"

    def test_clear_emits_clear_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert clear_terminal_progress() is True
        assert fake.getvalue() == "\x1b]9;4;0;0\a"

    def test_fires_unconditionally_regardless_of_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No terminal allowlist — unsupported terminals ignore the sequence."""
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        monkeypatch.delenv("WT_SESSION", raising=False)
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert set_terminal_progress(state=TerminalProgressState.INDETERMINATE) is True
        assert fake.getvalue() == "\x1b]9;4;3;0\a"

    def test_active_sentinel_set_and_cleared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        registered: list[object] = []
        monkeypatch.setattr("atexit.register", lambda fn: registered.append(fn) or fn)
        set_terminal_progress(state=TerminalProgressState.INDETERMINATE)
        assert terminal_escape._progress_active is True
        assert registered == [terminal_escape._atexit_clear]
        clear_terminal_progress()
        assert terminal_escape._progress_active is False

    def test_atexit_registered_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        registered: list[object] = []
        monkeypatch.setattr("atexit.register", lambda fn: registered.append(fn) or fn)
        set_terminal_progress(state=TerminalProgressState.INDETERMINATE)
        set_terminal_progress(50, state=TerminalProgressState.NORMAL)
        clear_terminal_progress()
        set_terminal_progress(state=TerminalProgressState.INDETERMINATE)
        assert registered == [terminal_escape._atexit_clear]

    def test_failed_write_does_not_register_atexit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: None)
        monkeypatch.setattr(terminal_escape, "_is_stream_tty", lambda _stream: False)
        registered: list[object] = []
        monkeypatch.setattr("atexit.register", lambda fn: registered.append(fn) or fn)
        assert set_terminal_progress(state=TerminalProgressState.INDETERMINATE) is False
        assert registered == []
        assert terminal_escape._progress_active is False


class TestTerminalBackground:
    """Tests for `OSC 11` / `OSC 111` terminal background helpers."""

    def test_set_background_writes_osc_11_with_st(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert set_terminal_background("#11121D") is True
        assert fake.getvalue() == "\x1b]11;#11121D\x1b\\"
        assert terminal_escape._terminal_background_active is True

    def test_reset_background_writes_osc_111_with_st(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        monkeypatch.setattr(terminal_escape, "_terminal_background_active", True)
        assert reset_terminal_background() is True
        assert fake.getvalue() == "\x1b]111\x1b\\"
        assert terminal_escape._terminal_background_active is False

    def test_empty_background_is_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert set_terminal_background("") is False
        assert fake.getvalue() == ""
        assert terminal_escape._terminal_background_active is False

    def test_set_background_registers_atexit_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        registered: list[object] = []
        monkeypatch.setattr("atexit.register", lambda fn: registered.append(fn) or fn)
        set_terminal_background("#11121D")
        set_terminal_background("#F5F5F7")
        assert registered == [terminal_escape._atexit_clear]

    def test_background_respects_terminal_escape_opt_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(terminal_escape.NO_TERMINAL_ESCAPE, "1")
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert set_terminal_background("#11121D") is False
        assert fake.getvalue() == ""
        assert terminal_escape._terminal_background_active is False


class TestAtexitClear:
    """`_atexit_clear` should only emit clears for active terminal state."""

    def test_emits_clear_when_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        monkeypatch.setattr(terminal_escape, "_progress_active", True)
        terminal_escape._atexit_clear()
        assert fake.getvalue() == "\x1b]9;4;0;0\a"

    def test_skips_when_not_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[bool] = []
        monkeypatch.setattr(
            terminal_escape,
            "clear_terminal_progress",
            lambda: called.append(True) or False,
        )
        monkeypatch.setattr(terminal_escape, "_progress_active", False)
        terminal_escape._atexit_clear()
        assert called == []

    def test_emits_background_reset_when_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        monkeypatch.setattr(terminal_escape, "_terminal_background_active", True)
        terminal_escape._atexit_clear()
        assert fake.getvalue() == "\x1b]111\x1b\\"

    def test_emits_progress_and_background_clear_when_both_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        monkeypatch.setattr(terminal_escape, "_progress_active", True)
        monkeypatch.setattr(terminal_escape, "_terminal_background_active", True)
        terminal_escape._atexit_clear()
        assert fake.getvalue() == "\x1b]9;4;0;0\a\x1b]111\x1b\\"

    def test_background_reset_runs_even_when_progress_clear_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[bool] = []

        def _raise() -> bool:
            msg = "progress clear failed"
            raise RuntimeError(msg)

        monkeypatch.setattr(terminal_escape, "clear_terminal_progress", _raise)
        monkeypatch.setattr(
            terminal_escape,
            "reset_terminal_background",
            lambda: called.append(True) or True,
        )
        monkeypatch.setattr(terminal_escape, "_progress_active", True)
        monkeypatch.setattr(terminal_escape, "_terminal_background_active", True)
        terminal_escape._atexit_clear()
        assert called == [True]
