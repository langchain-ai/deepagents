"""Tests for the resume confirmation prompt screen."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.widgets.resume_confirm import ResumeConfirmPromptScreen


class _ResumeConfirmTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestResumeConfirmPromptScreen:
    """Dismissal outcomes for the `-r` resume confirmation prompt."""

    async def test_enter_dismisses_with_resume(self) -> None:
        """Pressing Enter resolves the prompt to `True` (resume)."""
        app = _ResumeConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []
            app.push_screen(
                ResumeConfirmPromptScreen(thread_id="abc12345"),
                outcomes.append,
            )
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert outcomes == [True]

    async def test_escape_dismisses_with_abort(self) -> None:
        """Pressing Esc resolves the prompt to `False` (start new session)."""
        app = _ResumeConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []
            app.push_screen(
                ResumeConfirmPromptScreen(thread_id="abc12345"),
                outcomes.append,
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert outcomes == [False]

    async def test_action_cancel_dismisses_with_abort(self) -> None:
        """`action_cancel` aborts — the path taken by the app's priority Esc."""
        app = _ResumeConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []
            screen = ResumeConfirmPromptScreen(thread_id="abc12345")
            app.push_screen(screen, outcomes.append)
            await pilot.pause()

            screen.action_cancel()
            await pilot.pause()

            assert outcomes == [False]

    async def test_renders_thread_metadata(self) -> None:
        """Thread id, agent, and directory are surfaced when provided."""
        app = _ResumeConfirmTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                ResumeConfirmPromptScreen(
                    thread_id="abc12345",
                    agent_name="coder",
                    thread_cwd="/work/project",
                )
            )
            await pilot.pause()

            body = str(app.screen.query_one(".resume-confirm-body").render())
            assert "abc12345" in body
            assert "coder" in body
            assert "Directory:" in body

    def test_body_omits_optional_metadata_when_absent(self) -> None:
        """Agent and directory rows are conditional on supplied metadata."""
        minimal = ResumeConfirmPromptScreen(thread_id="abc12345")
        body = minimal._body_text()

        assert "abc12345" in body
        assert "Agent:" not in body
        assert "Directory:" not in body
