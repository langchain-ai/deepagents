"""Tests for the plugin reload confirmation modal."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.tui.widgets.plugin_reload import PluginReloadPromptScreen


class _PluginReloadTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestPluginReloadPromptScreen:
    async def test_enter_chooses_reload(self) -> None:
        app = _PluginReloadTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []
            app.push_screen(PluginReloadPromptScreen(), outcomes.append)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert outcomes == ["reload"]

    async def test_escape_chooses_later(self) -> None:
        app = _PluginReloadTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []
            app.push_screen(PluginReloadPromptScreen(), outcomes.append)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert outcomes == ["later"]

    async def test_action_cancel_chooses_later(self) -> None:
        app = _PluginReloadTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []
            screen = PluginReloadPromptScreen()
            app.push_screen(screen, outcomes.append)
            await pilot.pause()

            screen.action_cancel()
            await pilot.pause()

            assert outcomes == ["later"]

    async def test_explains_reload_scope(self) -> None:
        app = _PluginReloadTestApp()
        async with app.run_test() as pilot:
            app.push_screen(PluginReloadPromptScreen())
            await pilot.pause()

            body = str(app.screen.query_one(".plugin-reload-body").render())
            assert "plugin skills and MCP tools" in body
