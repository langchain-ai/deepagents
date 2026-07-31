"""Tests for the cross-agent thread resume prompt."""

from __future__ import annotations

from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static

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

    def test_body_explains_agents_restart_and_default_scope(self) -> None:
        """The prompt makes every material consequence explicit."""
        screen, _ = self._screen()

        body = screen._body_text()

        assert "thread-123" in body
        assert "researcher" in body
        assert "coder" in body
        assert "server will restart" in body
        assert "saved default agent will not change" in body

    def test_modal_binds_confirmation_cancel_and_quit(self) -> None:
        """Keyboard bindings match the sibling cwd-switch modal."""
        bindings = [
            binding
            for binding in ThreadAgentSwitchPromptScreen.BINDINGS
            if isinstance(binding, Binding)
        ]
        bindings_by_key = {binding.key: binding for binding in bindings}

        assert bindings_by_key["enter"].action == "switch"
        assert bindings_by_key["escape"].action == "cancel"
        assert bindings_by_key["ctrl+c"].action == "quit_or_interrupt"
        assert bindings_by_key["ctrl+d"].action == "quit_app"

    def test_action_switch_confirms(self) -> None:
        """Enter resolves to the explicit switch action."""
        screen, dismiss = self._screen()

        screen.action_switch()

        dismiss.assert_called_once_with("switch")

    def test_action_cancel_stays_on_current_thread(self) -> None:
        """Esc resolves to the safe no-op outcome."""
        screen, dismiss = self._screen()

        screen.action_cancel()

        dismiss.assert_called_once_with("cancel")

    async def test_real_keypresses_resolve_modal(self) -> None:
        """The mounted modal receives the confirmation key."""
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

            await pilot.press("enter")
            await pilot.pause()

            assert outcomes == ["switch"]

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
                "Enter: switch and resume · Esc: cancel"
            )
            assert needed > 1, "expected the help line to wrap at this width"
            assert help_widget.size.height >= needed
