"""Tests for the large-context resume compaction prompt."""

from __future__ import annotations

from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static

from deepagents_code.tui.widgets.resume_compaction import (
    ResumeCompactionChoice,
    ResumeCompactionPromptScreen,
)


class _ResumeCompactionTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestResumeCompactionPromptScreen:
    """Verify the three explicit resume outcomes."""

    @staticmethod
    def _screen() -> tuple[ResumeCompactionPromptScreen, MagicMock]:
        screen = ResumeCompactionPromptScreen(
            context_tokens=350_000,
            threshold=300_000,
        )
        dismiss = MagicMock()
        screen.dismiss = dismiss
        return screen, dismiss

    def test_body_reports_context_and_threshold(self) -> None:
        screen, _ = self._screen()

        assert "350.0K context tokens" in screen._body_text()
        assert "300.0K-token resume threshold" in screen._body_text()
        assert "Continue conversation with compaction" in screen._body_text()
        assert "Continue without compaction" in screen._body_text()
        assert "Esc — Cancel" in screen._body_text()

    def test_bindings_expose_all_choices(self) -> None:
        bindings = [
            binding
            for binding in ResumeCompactionPromptScreen.BINDINGS
            if isinstance(binding, Binding)
        ]
        actions = {binding.action for binding in bindings}

        assert "compact" in actions
        assert "continue_without_compaction" in actions
        assert "cancel_resume" in actions

    def test_compact_action_dismisses_compact(self) -> None:
        screen, dismiss = self._screen()

        screen.action_compact()

        dismiss.assert_called_once_with("compact")

    def test_continue_action_dismisses_continue(self) -> None:
        screen, dismiss = self._screen()

        screen.action_continue_without_compaction()

        dismiss.assert_called_once_with("continue")

    def test_cancel_action_dismisses_cancel(self) -> None:
        screen, dismiss = self._screen()

        screen.action_cancel()

        dismiss.assert_called_once_with("cancel")

    async def test_keyboard_choices_resolve_modal(self) -> None:
        app = _ResumeCompactionTestApp()
        outcomes: list[ResumeCompactionChoice | None] = []

        async with app.run_test() as pilot:
            app.push_screen(
                ResumeCompactionPromptScreen(
                    context_tokens=350_000,
                    threshold=300_000,
                ),
                outcomes.append,
            )
            await pilot.pause()
            await pilot.press("w")
            await pilot.pause()

        assert outcomes == ["continue"]
