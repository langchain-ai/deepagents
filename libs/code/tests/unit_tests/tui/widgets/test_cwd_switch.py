"""Tests for the cwd switch prompt screen."""

from __future__ import annotations

from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static

from deepagents_code.tui.widgets.cwd_switch import (
    CwdSwitchChoice,
    CwdSwitchPromptScreen,
)


class _CwdSwitchTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


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

        assert "project-specific config" not in unchanged._body_text()
        assert "project-specific config" in changed._body_text()

    def test_modal_binds_resume_and_quit_shortcuts(self) -> None:
        """The modal handles resume keys and delegates quit shortcuts."""
        bindings = [b for b in CwdSwitchPromptScreen.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}

        assert bindings_by_key["enter"].action == "switch"
        assert bindings_by_key["escape"].action == "stay"
        assert bindings_by_key["ctrl+c"].action == "quit_or_interrupt"
        assert bindings_by_key["ctrl+d"].action == "quit_app"

    def test_prompt_is_focusable(self) -> None:
        """The modal must own focus so its key bindings work after /threads."""
        screen, _ = self._screen()

        assert screen.can_focus is True
        assert screen.can_focus_children is False

    def test_on_mount_focuses_screen(self) -> None:
        """Mounting the modal claims focus from the dismissed thread selector."""
        screen, _ = self._screen()
        focus = MagicMock()
        screen.focus = focus  # ty: ignore[invalid-assignment]

        screen.on_mount()

        focus.assert_called_once_with()

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


class TestCwdSwitchAbortOption:
    """The launch-time `-r` resume prompt adds a third `abort` option."""

    def test_abort_binding_present(self) -> None:
        """The modal binds `a` to the abort action."""
        bindings = [b for b in CwdSwitchPromptScreen.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}

        assert bindings_by_key["a"].action == "abort"

    def test_help_and_body_mention_abort_only_when_allowed(self) -> None:
        """The abort affordance is described only when `allow_abort` is set."""
        without = CwdSwitchPromptScreen(current_cwd="/a", thread_cwd="/b")
        with_abort = CwdSwitchPromptScreen(
            current_cwd="/a", thread_cwd="/b", allow_abort=True
        )

        assert "new session" not in without._body_text()
        assert "new session" in with_abort._body_text()

    def test_action_abort_dismisses_abort_when_allowed(self) -> None:
        """Abort resolves the prompt to `abort` when offered."""
        screen = CwdSwitchPromptScreen(
            current_cwd="/a", thread_cwd="/b", allow_abort=True
        )
        dismiss = MagicMock()
        screen.dismiss = dismiss  # ty: ignore[invalid-assignment]

        screen.action_abort()

        dismiss.assert_called_once_with("abort")

    def test_action_abort_is_noop_when_not_allowed(self) -> None:
        """Abort does nothing when the prompt was not opened with `allow_abort`."""
        screen = CwdSwitchPromptScreen(current_cwd="/a", thread_cwd="/b")
        dismiss = MagicMock()
        screen.dismiss = dismiss  # ty: ignore[invalid-assignment]

        screen.action_abort()

        dismiss.assert_not_called()

    async def test_pressing_a_aborts_when_allowed(self) -> None:
        """Pressing `a` resolves the prompt to `abort` when offered."""
        app = _CwdSwitchTestApp()
        async with app.run_test() as pilot:
            outcomes: list[CwdSwitchChoice | None] = []
            app.push_screen(
                CwdSwitchPromptScreen(
                    current_cwd="/a", thread_cwd="/b", allow_abort=True
                ),
                outcomes.append,
            )
            await pilot.pause()

            await pilot.press("a")
            await pilot.pause()

            assert outcomes == ["abort"]
