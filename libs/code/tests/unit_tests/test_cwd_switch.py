"""Tests for the cwd switch prompt screen."""

from __future__ import annotations

from unittest.mock import MagicMock

from deepagents_code.widgets.cwd_switch import CwdSwitchPromptScreen


class TestCwdSwitchPromptScreen:
    """Dismissal outcomes for the resume cwd-switch prompt."""

    @staticmethod
    def _screen() -> tuple[CwdSwitchPromptScreen, MagicMock]:
        screen = CwdSwitchPromptScreen(
            current_cwd="/a/current",
            thread_cwd="/b/target",
        )
        dismiss = MagicMock()
        screen.dismiss = dismiss  # ty: ignore[invalid-assignment]
        return screen, dismiss

    def test_body_mentions_project_settings_only_when_detected(self) -> None:
        """Project settings copy is conditional on detected changes."""
        unchanged, _ = self._screen()
        changed = CwdSwitchPromptScreen(
            current_cwd="/a/current",
            thread_cwd="/b/target",
            project_settings_change_detected=True,
        )

        assert "Project settings will refresh" not in unchanged._body_text()
        assert "Project settings will refresh" in changed._body_text()

    def test_action_switch_dismisses_switch(self) -> None:
        """Enter / switch resolves the prompt to `switch`."""
        screen, dismiss = self._screen()
        screen.action_switch()
        dismiss.assert_called_once_with("switch")

    def test_action_stay_dismisses_stay(self) -> None:
        """Explicit stay resolves the prompt to `stay`."""
        screen, dismiss = self._screen()
        screen.action_stay()
        dismiss.assert_called_once_with("stay")

    def test_action_cancel_treated_as_stay(self) -> None:
        """Esc / cancel is the safe default and resolves to `stay`.

        The app owns a priority Esc binding, so the screen must define
        `action_cancel` to control the cancel outcome rather than relying on a
        bare `escape` binding.
        """
        screen, dismiss = self._screen()
        screen.action_cancel()
        dismiss.assert_called_once_with("stay")
