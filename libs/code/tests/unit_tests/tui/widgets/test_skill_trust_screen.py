"""Tests for the `SkillTrustScreen` out-of-bounds-skill approval modal."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.tui.widgets.skill_trust import SkillTrustScreen


class _SkillTrustTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestSkillTrustScreen:
    """Behavior tests for `SkillTrustScreen`.

    The Enter=allow / Esc=deny mapping is a security default (Esc must never
    grant read access), so each route is pinned explicitly.
    """

    async def test_enter_dismisses_with_true(self) -> None:
        """Pressing Enter approves the out-of-bounds directory."""
        app = _SkillTrustTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []
            app.push_screen(
                SkillTrustScreen("demo", "/shared/skills/demo"), outcomes.append
            )
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert outcomes == [True]

    async def test_escape_dismisses_with_false(self) -> None:
        """Pressing Esc denies (never an implicit approval)."""
        app = _SkillTrustTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []
            app.push_screen(
                SkillTrustScreen("demo", "/shared/skills/demo"), outcomes.append
            )
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert outcomes == [False]

    async def test_action_cancel_dismisses_with_false(self) -> None:
        """`action_cancel` denies — the path the app's priority Esc binding takes.

        The app owns a priority `escape` binding that, for an active
        `ModalScreen`, dispatches to `action_cancel` if present and otherwise
        falls through to `dismiss(None)`. Without an `action_cancel` returning
        `False`, real-app Esc would silently None-dismiss instead of an explicit
        deny.
        """
        app = _SkillTrustTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []
            screen = SkillTrustScreen("demo", "/shared/skills/demo")
            app.push_screen(screen, outcomes.append)
            await pilot.pause()
            screen.action_cancel()
            await pilot.pause()
            assert outcomes == [False]

    async def test_renders_skill_name_and_target(self) -> None:
        """The skill name and resolved target directory are surfaced in the body."""
        app = _SkillTrustTestApp()
        async with app.run_test() as pilot:
            app.push_screen(SkillTrustScreen("demo", "/shared/skills/demo"))
            await pilot.pause()
            bodies = app.screen.query(".skill-trust-body")
            assert len(bodies) == 1
            rendered = str(bodies.first().render())
            assert "demo" in rendered
            assert "/shared/skills/demo" in rendered
