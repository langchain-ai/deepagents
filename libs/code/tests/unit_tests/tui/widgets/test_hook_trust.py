"""Tests for the project-hooks trust modal."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.tui.widgets.hook_trust import HookTrustChoice, HookTrustScreen


class _HookTrustTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestHookTrustScreen:
    """Trust outcomes must remain explicit and fail closed."""

    @staticmethod
    def _screen() -> HookTrustScreen:
        return HookTrustScreen(
            project_root="/workspace/project",
            config_path="/workspace/project/.deepagents/hooks.json",
        )

    async def _press(self, key: str) -> list[HookTrustChoice | None]:
        app = _HookTrustTestApp()
        async with app.run_test() as pilot:
            outcomes: list[HookTrustChoice | None] = []
            app.push_screen(self._screen(), outcomes.append)
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
            return outcomes

    async def test_enter_allows_once(self) -> None:
        assert await self._press("enter") == ["allow_once"]

    async def test_a_always_allows(self) -> None:
        assert await self._press("a") == ["always_allow"]

    async def test_escape_denies(self) -> None:
        assert await self._press("escape") == ["deny"]

    async def test_action_cancel_denies(self) -> None:
        app = _HookTrustTestApp()
        async with app.run_test() as pilot:
            outcomes: list[HookTrustChoice | None] = []
            screen = self._screen()
            app.push_screen(screen, outcomes.append)
            await pilot.pause()
            screen.action_cancel()
            await pilot.pause()
            assert outcomes == ["deny"]

    async def test_renders_workspace_and_config_path(self) -> None:
        app = _HookTrustTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                HookTrustScreen(
                    project_root="/workspace/[project]",
                    config_path="/workspace/[project]/.deepagents/hooks.json",
                )
            )
            await pilot.pause()
            bodies = app.screen.query(".hook-trust-body")
            assert len(bodies) == 1
            rendered = str(bodies.first().render())
            assert "/workspace/[project]" in rendered
            assert ".deepagents/hooks.json" in rendered
