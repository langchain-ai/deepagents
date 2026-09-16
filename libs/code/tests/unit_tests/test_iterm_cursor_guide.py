"""Tests for the iTerm2 cursor guide workaround."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from deepagents_code import iterm_cursor_guide
from deepagents_code.iterm_cursor_guide import (
    _ITERM_CURSOR_GUIDE_OFF,
    _ITERM_CURSOR_GUIDE_ON,
    _disable_iterm_cursor_guide,
    _write_iterm_escape,
    restore_iterm_cursor_guide,
)


class TestITerm2CursorGuide:
    """Test iTerm2 cursor guide handling."""

    def test_escape_sequences_are_valid(self) -> None:
        """Escape sequences should be properly formatted OSC 1337 commands.

        Format: OSC (ESC ]) + "1337;" + command + ST (ESC backslash)
        """
        assert _ITERM_CURSOR_GUIDE_OFF.startswith("\x1b]1337;")
        assert _ITERM_CURSOR_GUIDE_OFF.endswith("\x1b\\")
        assert "HighlightCursorLine=no" in _ITERM_CURSOR_GUIDE_OFF

        assert _ITERM_CURSOR_GUIDE_ON.startswith("\x1b]1337;")
        assert _ITERM_CURSOR_GUIDE_ON.endswith("\x1b\\")
        assert "HighlightCursorLine=yes" in _ITERM_CURSOR_GUIDE_ON

    def test_write_iterm_escape_does_nothing_when_not_iterm(self) -> None:
        """_write_iterm_escape should no-op when `_IS_ITERM` is `False`."""
        mock_stderr = MagicMock()
        with (
            patch.object(iterm_cursor_guide, "_IS_ITERM", False),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
            mock_stderr.write.assert_not_called()

    def test_write_iterm_escape_writes_sequence_when_iterm(self) -> None:
        """_write_iterm_escape should write sequence when in iTerm2."""
        mock_stderr = io.StringIO()
        with (
            patch.object(iterm_cursor_guide, "_IS_ITERM", True),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
            assert mock_stderr.getvalue() == _ITERM_CURSOR_GUIDE_ON

    def test_write_iterm_escape_handles_oserror_gracefully(self) -> None:
        """_write_iterm_escape should not raise on `OSError`."""
        mock_stderr = MagicMock()
        mock_stderr.write.side_effect = OSError("Broken pipe")
        with (
            patch.object(iterm_cursor_guide, "_IS_ITERM", True),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)

    def test_write_iterm_escape_handles_none_stderr(self) -> None:
        """_write_iterm_escape should handle `None` `__stderr__` gracefully."""
        with (
            patch.object(iterm_cursor_guide, "_IS_ITERM", True),
            patch("sys.__stderr__", None),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)

    def test_disable_cursor_guide_noops_without_restore_path(self) -> None:
        """Cursor guide should not be disabled when startup state is unknown."""
        with (
            patch.object(iterm_cursor_guide, "_RESTORE_ITERM_CURSOR_GUIDE", False),
            patch.object(iterm_cursor_guide, "_write_iterm_escape") as write_escape,
        ):
            _disable_iterm_cursor_guide()

        write_escape.assert_not_called()

    def test_restore_cursor_guide_reenables_when_profile_had_it(self) -> None:
        """Restore should write the iTerm2 escape when launch state requires it."""
        with (
            patch.object(iterm_cursor_guide, "_RESTORE_ITERM_CURSOR_GUIDE", True),
            patch.object(iterm_cursor_guide, "_ITERM_CURSOR_GUIDE_RESTORED", False),
            patch.object(iterm_cursor_guide, "_write_iterm_escape") as write_escape,
        ):
            restore_iterm_cursor_guide()

        write_escape.assert_called_once_with(_ITERM_CURSOR_GUIDE_ON)


class TestITerm2Detection:
    """Test iTerm2 detection logic."""
