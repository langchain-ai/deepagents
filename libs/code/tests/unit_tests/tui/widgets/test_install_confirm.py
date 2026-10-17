"""Tests for the `/install --package` confirmation modal."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.content import Content
from textual.style import Style as TStyle
from textual.widgets import Static

from deepagents_code.tui.widgets.install_confirm import (
    InstallPackageConfirmScreen,
    InstallProviderConfirmScreen,
)


class _InstallConfirmTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


def _assert_pypi_link(content: Content, package: str) -> None:
    """Assert that only `package` links to its PyPI project page."""
    links = [
        (content.plain[span.start : span.end], span.style.link)
        for span in content.spans
        if isinstance(span.style, TStyle) and span.style.link
    ]
    assert links == [(package, f"https://pypi.org/project/{package}/")]


class TestInstallPackageConfirmScreen:
    """Behavior tests for `InstallPackageConfirmScreen`."""

    async def test_escape_dismisses_with_false(self) -> None:
        """Pressing Esc cancels (no implicit install)."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []

            def on_dismiss(result: bool | None) -> None:
                outcomes.append(result)

            app.push_screen(InstallPackageConfirmScreen("langchain-custom"), on_dismiss)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert outcomes == [False]

    async def test_action_cancel_dismisses_with_false(self) -> None:
        """`action_cancel` cancels — the path taken by the app's Esc handler.

        `DeepAgentsApp.action_interrupt` (a priority `escape` binding) fires
        before the modal's own `escape` binding. When the active screen is a
        `ModalScreen`, it dispatches to `action_cancel` if present, else falls
        through to `dismiss(None)`. Without an `action_cancel` that returns
        `False`, real-app Esc would silently None-dismiss, which the caller
        cannot distinguish from a programmatic dismiss.
        """
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []

            def on_dismiss(result: bool | None) -> None:
                outcomes.append(result)

            screen = InstallPackageConfirmScreen("langchain-custom")
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            screen.action_cancel()
            await pilot.pause()

            assert outcomes == [False]


class TestInstallProviderConfirmScreen:
    """Behavior tests for `InstallProviderConfirmScreen`."""

    async def test_escape_dismisses_with_false(self) -> None:
        """Pressing Esc cancels (no implicit install)."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []

            app.push_screen(
                InstallProviderConfirmScreen(
                    "baseten", "baseten", "baseten:moonshotai/Kimi-K2.7-Code"
                ),
                outcomes.append,
            )
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            assert outcomes == [False]

    async def test_renders_add_key_body_without_model_spec(self) -> None:
        """The `/auth` path (no model spec) frames the install around a key.

        Omitting `model_spec` switches the body to the "add a key" wording and
        drops the model-centric "To use ..." copy, since the manager installs a
        provider so a credential can be added, not to switch to a model.
        """
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            app.push_screen(InstallProviderConfirmScreen("litellm", "litellm"))
            await pilot.pause()

            bodies = app.screen.query(".install-confirm-body")
            assert len(bodies) == 1
            content = bodies.first().render()
            assert isinstance(content, Content)
            assert "add a key" in content.plain
            assert "langchain-litellm" in content.plain
            assert "To use" not in content.plain
            _assert_pypi_link(content, "langchain-litellm")


class TestLinkHoverAndClick:
    """Link affordance tests shared by both install confirmation screens."""
