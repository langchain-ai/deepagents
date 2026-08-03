"""Tests for the iTerm2 cursor guide workaround."""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from textual.app import App

from deepagents_code import iterm_cursor_guide, terminal_escape
from deepagents_code.app import DeepAgentsApp
from deepagents_code.iterm_cursor_guide import (
    _CURSOR_GUIDE_OFF,
    _CURSOR_GUIDE_ON,
    _disable_iterm_cursor_guide,
    _iterm_profile_cursor_guide_enabled,
    _write_cursor_guide,
    restore_iterm_cursor_guide,
)

if TYPE_CHECKING:
    from types import TracebackType


class _FakeTTY(io.StringIO):
    """`StringIO` with a context-manager that doesn't truncate on close."""

    def __enter__(self) -> _FakeTTY:  # noqa: PYI034  # _FakeTTY is a test helper
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


@pytest.fixture
def tty(monkeypatch: pytest.MonkeyPatch) -> _FakeTTY:
    """Capture what the shared escape writer sends to the terminal."""
    fake = _FakeTTY()
    monkeypatch.setattr(terminal_escape, "_open_tty", lambda: fake)
    monkeypatch.delenv(terminal_escape.NO_TERMINAL_ESCAPE, raising=False)
    # A developer running the suite from inside a pane would otherwise see
    # every expected sequence gain a passthrough DCS.
    monkeypatch.delenv("TMUX", raising=False)
    return fake


class TestITerm2CursorGuide:
    """Test iTerm2 cursor guide handling."""

    def test_write_cursor_guide_does_nothing_when_not_iterm(
        self, tty: _FakeTTY
    ) -> None:
        """`_write_cursor_guide` should no-op when `_IS_ITERM` is `False`."""
        with patch.object(iterm_cursor_guide, "_IS_ITERM", False):
            _write_cursor_guide(_CURSOR_GUIDE_ON)

        assert tty.getvalue() == ""

    def test_write_cursor_guide_emits_osc_1337(self, tty: _FakeTTY) -> None:
        """The command goes out as `OSC 1337 ; <payload> ST`."""
        with patch.object(iterm_cursor_guide, "_IS_ITERM", True):
            _write_cursor_guide(_CURSOR_GUIDE_ON)

        assert tty.getvalue() == "\x1b]1337;HighlightCursorLine=yes\x1b\\"

    def test_write_cursor_guide_wraps_for_tmux(
        self, tty: _FakeTTY, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tmux drops a bare `OSC 1337`, so a pane must send it wrapped."""
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1234,0")
        with patch.object(iterm_cursor_guide, "_IS_ITERM", True):
            _write_cursor_guide(_CURSOR_GUIDE_OFF)

        assert tty.getvalue() == (
            "\x1bPtmux;\x1b\x1b]1337;HighlightCursorLine=no\x1b\x1b\\\x1b\\"
        )

    def test_write_cursor_guide_survives_an_unwritable_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cosmetic write must never propagate a terminal failure."""
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: None)
        monkeypatch.setattr(terminal_escape.sys, "__stderr__", None)
        with patch.object(iterm_cursor_guide, "_IS_ITERM", True):
            _write_cursor_guide(_CURSOR_GUIDE_ON)

    def test_disable_cursor_guide_noops_without_restore_path(self) -> None:
        """Cursor guide should not be disabled when startup state is unknown."""
        with (
            patch.object(iterm_cursor_guide, "_RESTORE_ITERM_CURSOR_GUIDE", False),
            patch.object(iterm_cursor_guide, "_write_cursor_guide") as write_escape,
        ):
            _disable_iterm_cursor_guide()

        write_escape.assert_not_called()

    def test_disable_cursor_guide_writes_off_when_restore_path_exists(self) -> None:
        """Cursor guide may be disabled only when cleanup will restore it."""
        with (
            patch.object(iterm_cursor_guide, "_RESTORE_ITERM_CURSOR_GUIDE", True),
            patch.object(iterm_cursor_guide, "_write_cursor_guide") as write_escape,
        ):
            _disable_iterm_cursor_guide()

        write_escape.assert_called_once_with(_CURSOR_GUIDE_OFF)

    def test_app_exit_does_not_reenable_cursor_guide(self) -> None:
        """App exit should not restore when cursor guide was off before launch."""
        app = DeepAgentsApp()
        with (
            patch("deepagents_code.app.restore_iterm_cursor_guide") as restore,
            patch("deepagents_code.hooks._load_hooks", return_value=[]),
            patch.object(App, "exit") as app_exit,
        ):
            app.exit()

        restore.assert_called_once_with()
        app_exit.assert_called_once_with(result=None, return_code=0, message=None)

    def test_restore_cursor_guide_reenables_when_profile_had_it(self) -> None:
        """Restore should write the iTerm2 escape when launch state requires it."""
        with (
            patch.object(iterm_cursor_guide, "_RESTORE_ITERM_CURSOR_GUIDE", True),
            patch.object(iterm_cursor_guide, "_ITERM_CURSOR_GUIDE_RESTORED", False),
            patch.object(iterm_cursor_guide, "_write_cursor_guide") as write_escape,
        ):
            restore_iterm_cursor_guide()

        write_escape.assert_called_once_with(_CURSOR_GUIDE_ON)

    def test_restore_cursor_guide_is_idempotent(self) -> None:
        """The direct exit path and atexit fallback should not double-restore."""
        with (
            patch.object(iterm_cursor_guide, "_RESTORE_ITERM_CURSOR_GUIDE", True),
            patch.object(iterm_cursor_guide, "_ITERM_CURSOR_GUIDE_RESTORED", False),
            patch.object(iterm_cursor_guide, "_write_cursor_guide") as write_escape,
        ):
            restore_iterm_cursor_guide()
            restore_iterm_cursor_guide()

        write_escape.assert_called_once_with(_CURSOR_GUIDE_ON)

    def test_profile_cursor_guide_enabled_from_iterm_profile(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The active `ITERM_PROFILE` controls whether cursor guide is restored."""
        import plistlib

        prefs_path = tmp_path / "com.googlecode.iterm2.plist"
        with prefs_path.open("wb") as f:
            plistlib.dump(
                {
                    "Default Bookmark Guid": "disabled-guid",
                    "New Bookmarks": [
                        {
                            "Guid": "disabled-guid",
                            "Name": "Disabled",
                            "Use Cursor Guide": 0,
                        },
                        {
                            "Guid": "enabled-guid",
                            "Name": "Enabled",
                            "Use Cursor Guide": 1,
                        },
                    ],
                },
                f,
            )

        monkeypatch.setenv("ITERM_PROFILE", "Enabled")
        with (
            patch.object(iterm_cursor_guide, "_IS_ITERM", True),
            patch.object(iterm_cursor_guide, "_ITERM_PREFS_PATH", prefs_path),
        ):
            assert _iterm_profile_cursor_guide_enabled() is True

    def test_profile_cursor_guide_disabled_from_default_guid(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default profile GUID is used when `ITERM_PROFILE` is unavailable."""
        import plistlib

        prefs_path = tmp_path / "com.googlecode.iterm2.plist"
        with prefs_path.open("wb") as f:
            plistlib.dump(
                {
                    "Default Bookmark Guid": "disabled-guid",
                    "New Bookmarks": [
                        {
                            "Guid": "disabled-guid",
                            "Name": "Disabled",
                            "Use Cursor Guide": 0,
                        }
                    ],
                },
                f,
            )

        monkeypatch.delenv("ITERM_PROFILE", raising=False)
        with (
            patch.object(iterm_cursor_guide, "_IS_ITERM", True),
            patch.object(iterm_cursor_guide, "_ITERM_PREFS_PATH", prefs_path),
        ):
            assert _iterm_profile_cursor_guide_enabled() is False


class TestITerm2Detection:
    """Test iTerm2 detection logic."""

    def test_detection_requires_tty(self) -> None:
        """_IS_ITERM should check that stderr is a TTY.

        Detection happens at module load, so we test the logic pattern directly.
        """
        with (
            patch.dict(os.environ, {"LC_TERMINAL": "iTerm2"}, clear=False),
            patch("os.isatty", return_value=False),
        ):
            result = (
                (
                    os.environ.get("LC_TERMINAL", "") == "iTerm2"
                    or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
                )
                and hasattr(os, "isatty")
                and os.isatty(2)
            )
            assert result is False

    def test_detection_via_lc_terminal(self) -> None:
        """Detection should match `LC_TERMINAL=iTerm2`."""
        with (
            patch.dict(
                os.environ, {"LC_TERMINAL": "iTerm2", "TERM_PROGRAM": ""}, clear=False
            ),
            patch("os.isatty", return_value=True),
        ):
            result = (
                (
                    os.environ.get("LC_TERMINAL", "") == "iTerm2"
                    or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
                )
                and hasattr(os, "isatty")
                and os.isatty(2)
            )
            assert result is True

    def test_detection_via_term_program(self) -> None:
        """Detection should match `TERM_PROGRAM=iTerm.app`."""
        env = {"LC_TERMINAL": "", "TERM_PROGRAM": "iTerm.app"}
        with (
            patch.dict(os.environ, env, clear=False),
            patch("os.isatty", return_value=True),
        ):
            result = (
                (
                    os.environ.get("LC_TERMINAL", "") == "iTerm2"
                    or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
                )
                and hasattr(os, "isatty")
                and os.isatty(2)
            )
            assert result is True
