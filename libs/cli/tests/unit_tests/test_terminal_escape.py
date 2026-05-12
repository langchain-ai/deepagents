"""Tests for `deepagents_cli.terminal_escape`."""

from __future__ import annotations

import io

import pytest

from deepagents_cli import terminal_escape
from deepagents_cli.terminal_escape import (
    TerminalProgressState,
    _validate_progress,
    clear_terminal_progress,
    set_terminal_progress,
    write_osc,
    write_terminal_escape,
)


@pytest.fixture(autouse=True)
def _reset_active_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-level progress sentinel between tests."""
    monkeypatch.setattr(terminal_escape, "_progress_active", False)
    monkeypatch.setattr(terminal_escape, "_atexit_registered", False)
    monkeypatch.delenv(terminal_escape.FORCE_TERMINAL_PROGRESS, raising=False)
    monkeypatch.delenv(terminal_escape.NO_TERMINAL_ESCAPE, raising=False)
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)


def _enable_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mark the test environment as OSC 9;4 compatible."""
    monkeypatch.setenv("WT_SESSION", "test-session")


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


class TestSetTerminalProgress:
    """Tests for `set_terminal_progress` / `clear_terminal_progress`."""

    def test_normal_progress_writes_state_and_percent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_progress(monkeypatch)
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert set_terminal_progress(75, state=TerminalProgressState.NORMAL) is True
        assert fake.getvalue() == "\x1b]9;4;1;75\a"

    def test_indeterminate_emits_zero_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_progress(monkeypatch)
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        set_terminal_progress(state=TerminalProgressState.INDETERMINATE)
        assert fake.getvalue() == "\x1b]9;4;3;0\a"

    def test_clear_emits_clear_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_progress(monkeypatch)
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert clear_terminal_progress() is True
        assert fake.getvalue() == "\x1b]9;4;0;0\a"

    def test_iterm_does_not_emit_progress_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert set_terminal_progress(state=TerminalProgressState.INDETERMINATE) is False
        assert fake.getvalue() == ""

    def test_force_progress_env_allows_unrecognized_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(terminal_escape.FORCE_TERMINAL_PROGRESS, "1")
        fake = _FakeTTY()
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
        assert set_terminal_progress(state=TerminalProgressState.INDETERMINATE) is True
        assert fake.getvalue() == "\x1b]9;4;3;0\a"

    def test_active_sentinel_set_and_cleared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_progress(monkeypatch)
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
        _enable_progress(monkeypatch)
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
        _enable_progress(monkeypatch)
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: None)
        monkeypatch.setattr(terminal_escape, "_is_stream_tty", lambda _stream: False)
        registered: list[object] = []
        monkeypatch.setattr("atexit.register", lambda fn: registered.append(fn) or fn)
        assert set_terminal_progress(state=TerminalProgressState.INDETERMINATE) is False
        assert registered == []
        assert terminal_escape._progress_active is False


class TestAtexitClear:
    """`_atexit_clear` should only emit a clear when progress was set."""

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
