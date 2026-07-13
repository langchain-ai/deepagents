"""Tests for the post-install restart confirmation modal."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.tui.widgets.restart_prompt import RestartPromptScreen


class _RestartTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestRestartPromptScreen:
    """Behavior tests for `RestartPromptScreen`."""

    async def test_enter_dismisses_with_restart(self) -> None:
        """Pressing Enter chooses `restart`."""
        app = _RestartTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            app.push_screen(
                RestartPromptScreen("fireworks", verb="Installed"), on_dismiss
            )
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert outcomes == ["restart"]

    async def test_escape_dismisses_with_later(self) -> None:
        """Pressing Esc chooses `later` (no implicit restart)."""
        app = _RestartTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            app.push_screen(
                RestartPromptScreen("fireworks", verb="Installed"), on_dismiss
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert outcomes == ["later"]

    async def test_action_cancel_dismisses_with_later(self) -> None:
        """`action_cancel` defers — the path taken by the app's Esc handler.

        `DeepAgentsApp.action_interrupt` (a priority `escape` binding) fires
        before the modal's own `escape` binding. When the active screen is a
        `ModalScreen`, it dispatches to `action_cancel` if present, else falls
        through to `dismiss(None)`. Without an `action_cancel` that defers,
        real-app Esc would silently None-dismiss instead of choosing `later`,
        which the caller cannot distinguish from a programmatic dismiss.
        """
        app = _RestartTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            screen = RestartPromptScreen("fireworks", verb="Installed")
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            screen.action_cancel()
            await pilot.pause()

            assert outcomes == ["later"]

    async def test_renders_label(self) -> None:
        """The installed extra/package label is surfaced in the modal title."""
        app = _RestartTestApp()
        async with app.run_test() as pilot:
            app.push_screen(RestartPromptScreen("langchain-custom", verb="Installed"))
            await pilot.pause()

            titles = app.screen.query(".restart-prompt-title")
            assert len(titles) == 1
            assert "langchain-custom" in str(titles.first().render())

    async def test_renders_custom_verb(self) -> None:
        """A caller-supplied `verb` replaces the install-flow default in the title."""
        app = _RestartTestApp()
        async with app.run_test() as pilot:
            app.push_screen(RestartPromptScreen("Tavily API key", verb="Saved"))
            await pilot.pause()

            title = str(app.screen.query_one(".restart-prompt-title").render())
            assert "Saved" in title
            assert "Tavily API key" in title
            assert "Installed" not in title

    async def test_renders_default_body(self) -> None:
        """With no `body` override, the generic restart copy is shown."""
        app = _RestartTestApp()
        async with app.run_test() as pilot:
            app.push_screen(RestartPromptScreen("fireworks", verb="Installed"))
            await pilot.pause()

            body = str(app.screen.query_one(".restart-prompt-body").render())
            assert body == RestartPromptScreen._DEFAULT_BODY

    async def test_renders_body_override(self) -> None:
        """A caller-supplied `body` replaces the default explanatory line."""
        app = _RestartTestApp()
        override = "Restart the server to enable web search, or defer with `/restart`."
        async with app.run_test() as pilot:
            app.push_screen(
                RestartPromptScreen("Tavily API key", verb="Saved", body=override)
            )
            await pilot.pause()

            body = str(app.screen.query_one(".restart-prompt-body").render())
            assert body == override
