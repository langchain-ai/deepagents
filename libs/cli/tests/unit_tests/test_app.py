"""Unit tests for DeepAgentsApp."""

import io
import os
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_cli.app import (
    _ITERM_CURSOR_GUIDE_OFF,
    _ITERM_CURSOR_GUIDE_ON,
    DeepAgentsApp,
    _write_iterm_escape,
)


class TestAppBindings:
    """Test app keybindings."""

    def test_toggle_tool_output_has_ctrl_e_binding(self) -> None:
        """Ctrl+E should be bound to toggle_tool_output with priority."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        ctrl_e = bindings_by_key.get("ctrl+e")

        assert ctrl_e is not None
        assert ctrl_e.action == "toggle_tool_output"
        assert ctrl_e.priority is True

    def test_ctrl_o_not_bound_to_toggle_tool_output(self) -> None:
        """Ctrl+O should not exist (replaced by Ctrl+E)."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        assert "ctrl+o" not in bindings_by_key


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
        """_write_iterm_escape should no-op when _IS_ITERM is False."""
        mock_stderr = MagicMock()
        with (
            patch("deepagents_cli.app._IS_ITERM", False),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
            mock_stderr.write.assert_not_called()

    def test_write_iterm_escape_writes_sequence_when_iterm(self) -> None:
        """_write_iterm_escape should write sequence when in iTerm2."""
        mock_stderr = io.StringIO()
        with (
            patch("deepagents_cli.app._IS_ITERM", True),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
            assert mock_stderr.getvalue() == _ITERM_CURSOR_GUIDE_ON

    def test_write_iterm_escape_handles_oserror_gracefully(self) -> None:
        """_write_iterm_escape should not raise on OSError."""
        mock_stderr = MagicMock()
        mock_stderr.write.side_effect = OSError("Broken pipe")
        with (
            patch("deepagents_cli.app._IS_ITERM", True),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)

    def test_write_iterm_escape_handles_none_stderr(self) -> None:
        """_write_iterm_escape should handle None __stderr__ gracefully."""
        with (
            patch("deepagents_cli.app._IS_ITERM", True),
            patch("sys.__stderr__", None),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)


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
        """Detection should match LC_TERMINAL=iTerm2."""
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
        """Detection should match TERM_PROGRAM=iTerm.app."""
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


class TestModalScreenEscapeDismissal:
    """Test that escape key dismisses modal screens."""

    @staticmethod
    @pytest.mark.asyncio
    async def test_escape_dismisses_modal_screen() -> None:
        """Escape should dismiss any active ModalScreen.

        The app's action_interrupt binding intercepts escape with priority=True.
        When a modal screen is active, it should dismiss the modal rather than
        performing the default interrupt behavior.
        """

        class SimpleModal(ModalScreen[str | None]):
            """A simple test modal."""

            BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

            def compose(self) -> ComposeResult:
                yield Static("Test Modal")

            def action_cancel(self) -> None:
                self.dismiss(None)

        class TestApp(App[None]):
            """Test app with escape -> action_interrupt binding."""

            BINDINGS: ClassVar[list[BindingType]] = [
                Binding("escape", "interrupt", "Interrupt", priority=True)
            ]

            def __init__(self) -> None:
                super().__init__()
                self.modal_dismissed = False
                self.interrupt_called = False

            def compose(self) -> ComposeResult:
                yield Container()

            def action_interrupt(self) -> None:
                if isinstance(self.screen, ModalScreen):
                    self.screen.dismiss(None)
                    return
                self.interrupt_called = True

            def show_modal(self) -> None:
                def on_dismiss(_result: str | None) -> None:
                    self.modal_dismissed = True

                self.push_screen(SimpleModal(), on_dismiss)

        app = TestApp()
        async with app.run_test() as pilot:
            app.show_modal()
            await pilot.pause()

            # Escape should dismiss the modal, not call interrupt
            await pilot.press("escape")
            await pilot.pause()

            assert app.modal_dismissed is True
            assert app.interrupt_called is False
