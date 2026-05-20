"""Tests for the MCP reconnect confirmation modal."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.widgets.mcp_reconnect import MCPReconnectPromptScreen


class _ReconnectTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestMCPReconnectPromptScreen:
    """Behavior tests for `MCPReconnectPromptScreen`."""

    async def test_enter_dismisses_with_reconnect(self) -> None:
        """Pressing Enter chooses `reconnect`."""
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            app.push_screen(MCPReconnectPromptScreen("notion"), on_dismiss)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert outcomes == ["reconnect"]

    async def test_escape_dismisses_with_later(self) -> None:
        """Pressing Esc chooses `later` (no implicit reconnect)."""
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            app.push_screen(MCPReconnectPromptScreen("notion"), on_dismiss)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert outcomes == ["later"]

    async def test_renders_server_name(self) -> None:
        """The server name is surfaced in the modal title."""
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            app.push_screen(MCPReconnectPromptScreen("notion"))
            await pilot.pause()

            titles = app.screen.query(".mcp-reconnect-title")
            assert len(titles) == 1
            assert "notion" in str(titles.first().render())
