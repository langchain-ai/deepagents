"""Tests for the cross-agent thread resume prompt."""

from __future__ import annotations

from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets.thread_agent_switch import (
    ThreadAgentSwitchChoice,
    ThreadAgentSwitchPromptScreen,
)


class _ThreadAgentSwitchTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestThreadAgentSwitchPromptScreen:
    """Content and dismissal behavior for the cross-agent prompt."""

    @staticmethod
    def _screen() -> tuple[ThreadAgentSwitchPromptScreen, MagicMock]:
        screen = ThreadAgentSwitchPromptScreen(
            thread_id="thread-123",
            current_agent="coder",
            thread_agent="researcher",
        )
        dismiss = MagicMock()
        screen.dismiss = dismiss  # ty: ignore[invalid-assignment]
        return screen, dismiss

    def test_action_cancel_stays_on_current_thread(self) -> None:
        """Esc resolves to the safe no-op outcome."""
        screen, dismiss = self._screen()

        screen.action_cancel()

        dismiss.assert_called_once_with("cancel")

    async def test_escape_cancels_mounted_modal(self) -> None:
        """The app-level Esc binding resolves to the safe cancel outcome."""
        app = _ThreadAgentSwitchTestApp()
        async with app.run_test() as pilot:
            outcomes: list[ThreadAgentSwitchChoice | None] = []
            app.push_screen(
                ThreadAgentSwitchPromptScreen(
                    thread_id="thread-123",
                    current_agent="coder",
                    thread_agent="researcher",
                ),
                outcomes.append,
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert outcomes == ["cancel"]

    async def test_help_line_is_not_clipped_in_a_narrow_terminal(self) -> None:
        """The `Esc: cancel` half of the help line must survive wrapping.

        The help string is wider than the dialog once the terminal drops below
        roughly 48 columns. A fixed one-row height silently truncated the
        wrapped remainder, hiding how to back out of the switch precisely when
        the dialog was hardest to read.
        """
        app = _ThreadAgentSwitchTestApp()
        async with app.run_test(size=(40, 24)) as pilot:
            app.push_screen(
                ThreadAgentSwitchPromptScreen(
                    thread_id="thread-123",
                    current_agent="coder",
                    thread_agent="researcher",
                )
            )
            await pilot.pause()
            await pilot.pause()

            help_widget = app.screen.query_one(".thread-agent-switch-help")
            viewport = app.screen.size
            needed = help_widget.get_content_height(
                viewport,
                viewport,
                help_widget.size.width,
            )

            assert help_widget.size.width < len(
                f"Enter: switch and resume {get_glyphs().separator} Esc: cancel"
            )
            assert needed > 1, "expected the help line to wrap at this width"
            assert help_widget.size.height >= needed
