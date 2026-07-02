"""Tests for the `/install --package` confirmation modal."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.widgets.install_confirm import (
    InstallPackageConfirmScreen,
    InstallProviderConfirmScreen,
)


class _InstallConfirmTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestInstallPackageConfirmScreen:
    """Behavior tests for `InstallPackageConfirmScreen`."""

    async def test_enter_dismisses_with_true(self) -> None:
        """Pressing Enter confirms the install."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []

            def on_dismiss(result: bool | None) -> None:
                outcomes.append(result)

            app.push_screen(InstallPackageConfirmScreen("langchain-custom"), on_dismiss)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert outcomes == [True]

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

    async def test_renders_package_name(self) -> None:
        """The package name is surfaced in the modal body."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            app.push_screen(InstallPackageConfirmScreen("langchain-custom"))
            await pilot.pause()

            bodies = app.screen.query(".install-confirm-body")
            assert len(bodies) == 1
            assert "langchain-custom" in str(bodies.first().render())


class TestInstallProviderConfirmScreen:
    """Behavior tests for `InstallProviderConfirmScreen`."""

    async def test_enter_dismisses_with_true(self) -> None:
        """Pressing Enter confirms the provider install."""
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
            await pilot.press("enter")
            await pilot.pause()

            assert outcomes == [True]

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

    async def test_renders_model_and_extra(self) -> None:
        """The model spec and extra are surfaced in the modal body."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                InstallProviderConfirmScreen(
                    "baseten", "baseten", "baseten:moonshotai/Kimi-K2.7-Code"
                )
            )
            await pilot.pause()

            bodies = app.screen.query(".install-confirm-body")
            assert len(bodies) == 1
            rendered = str(bodies.first().render())
            assert "baseten:moonshotai/Kimi-K2.7-Code" in rendered
            assert "baseten" in rendered

    async def test_renders_add_key_body_without_model_spec(self) -> None:
        """The `/auth` path (no model spec) frames the install around a key.

        Omitting `model_spec` switches the body to the "add a key" wording and
        drops the model-centric "To use ..." copy, since the manager installs a
        provider so a credential can be added, not to switch to a model.
        """
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            app.push_screen(InstallProviderConfirmScreen("baseten", "baseten"))
            await pilot.pause()

            bodies = app.screen.query(".install-confirm-body")
            assert len(bodies) == 1
            rendered = str(bodies.first().render())
            assert "add a key" in rendered
            assert "baseten" in rendered
            assert "To use" not in rendered
