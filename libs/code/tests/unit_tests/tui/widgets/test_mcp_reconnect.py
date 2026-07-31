"""Tests for the MCP reconnect confirmation modals."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from deepagents_code.tui.widgets.mcp_reconnect import (
    MCPDisableReconnectPromptScreen,
    MCPReconnectForceConfirmScreen,
    MCPReconnectPromptScreen,
)


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

    async def test_action_cancel_dismisses_with_later(self) -> None:
        """`action_cancel` defers — the path taken by the app's Esc handler.

        `DeepAgentsApp.action_interrupt` (a priority `escape` binding)
        fires before the modal's own `escape` binding. When the active
        screen is a `ModalScreen`, it dispatches to `action_cancel` if
        present, else falls through to `dismiss(None)`. Without an
        `action_cancel` that defers, real-app Esc would silently
        None-dismiss instead of choosing `later`, breaking the
        "log into another server first" workflow.
        """
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            screen = MCPReconnectPromptScreen("notion")
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            screen.action_cancel()
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


class TestMCPDisableReconnectPromptScreen:
    """Behavior tests for `MCPDisableReconnectPromptScreen`."""

    async def test_enter_dismisses_with_reconnect(self) -> None:
        """Pressing Enter chooses `reconnect`."""
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            app.push_screen(MCPDisableReconnectPromptScreen(["filesystem"]), on_dismiss)
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

            app.push_screen(MCPDisableReconnectPromptScreen(["filesystem"]), on_dismiss)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert outcomes == ["later"]

    async def test_action_cancel_dismisses_with_later(self) -> None:
        """`action_cancel` defers — the path taken by the app's Esc handler."""
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            screen = MCPDisableReconnectPromptScreen(["filesystem"])
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            screen.action_cancel()
            await pilot.pause()

            assert outcomes == ["later"]

    async def test_renders_server_names(self) -> None:
        """Every pending server name appears in the body."""
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                MCPDisableReconnectPromptScreen(["filesystem", "notion"]),
            )
            await pilot.pause()

            bodies = app.screen.query(".mcp-reconnect-body")
            assert len(bodies) == 1
            rendered = str(bodies.first().render())
            assert "filesystem" in rendered
            assert "notion" in rendered

    async def test_shares_styling_with_login_prompt(self) -> None:
        """Both prompts inherit one style contract from the shared base.

        The two screens used to duplicate a 34-line CSS block, so a tweak
        to one silently diverged the other. They now share `DEFAULT_CSS`
        via `_ReconnectPromptScreen`, which Textual resolves through the
        MRO — this pins that the shared rules actually reach both.
        """
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            for screen in (
                MCPDisableReconnectPromptScreen(["filesystem"]),
                MCPReconnectPromptScreen("notion"),
            ):
                app.push_screen(screen)
                await pilot.pause()

                dialog = app.screen.query_one(Vertical)
                assert dialog.styles.width is not None
                assert dialog.styles.width.value == 64
                help_rows = app.screen.query(".mcp-reconnect-help")
                assert str(help_rows.first().render()) == (
                    "Enter to reconnect, Esc to defer"
                )

                app.pop_screen()
                await pilot.pause()


class TestMCPReconnectForceConfirmScreen:
    """Behavior tests for `MCPReconnectForceConfirmScreen`."""

    async def test_enter_dismisses_with_true(self) -> None:
        """Enter confirms the force-reconnect."""
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []

            def on_dismiss(result: bool | None) -> None:
                outcomes.append(result)

            app.push_screen(MCPReconnectForceConfirmScreen(), on_dismiss)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert outcomes == [True]

    async def test_escape_dismisses_with_false(self) -> None:
        """Esc cancels the force-reconnect."""
        app = _ReconnectTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []

            def on_dismiss(result: bool | None) -> None:
                outcomes.append(result)

            app.push_screen(MCPReconnectForceConfirmScreen(), on_dismiss)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert outcomes == [False]
