"""Tests for the `/install --package` confirmation modal."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _link_span_reverses(content: Content) -> list[bool]:
    """Return the `reverse` flag of every link-styled span in `content`."""
    reverses: list[bool] = []
    for span in content.spans:
        style = span.style
        if isinstance(style, TStyle) and style.link:
            reverses.append(bool(style.reverse))
    return reverses


def _link_span_reverses(content: Content) -> list[bool]:
    """Return the `reverse` flag of every link-styled span in `content`."""
    reverses: list[bool] = []
    for span in content.spans:
        style = span.style
        if isinstance(style, TStyle) and style.link:
            reverses.append(bool(style.reverse))
    return reverses


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
            content = bodies.first().render()
            assert isinstance(content, Content)
            assert "langchain-custom" in content.plain
            _assert_pypi_link(content, "langchain-custom")


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
            content = bodies.first().render()
            assert isinstance(content, Content)
            assert "baseten:moonshotai/Kimi-K2.7-Code" in content.plain
            assert "langchain-baseten" in content.plain
            _assert_pypi_link(content, "langchain-baseten")

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

    def _move_event(self, url: str | None) -> SimpleNamespace:
        """Build a minimal mouse-move event over a link (or plain text)."""
        return SimpleNamespace(style=SimpleNamespace(link=url, meta={}))

    def _click_event(self, url: str | None) -> SimpleNamespace:
        """Build a minimal click event over a link (or plain text)."""
        return SimpleNamespace(
            style=SimpleNamespace(link=url),
            app=SimpleNamespace(notify=MagicMock()),
            stop=MagicMock(),
        )

    async def test_hover_shows_pointer_and_highlights_link(self) -> None:
        """Hovering the link sets a pointer cursor and highlights the text."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallPackageConfirmScreen("langchain-custom")
            app.push_screen(screen)
            await pilot.pause()

            screen.on_mouse_move(
                self._move_event("https://pypi.org/project/langchain-custom/")  # ty: ignore
            )
            await pilot.pause()

            assert screen.styles.pointer == "pointer"
            content = screen.query_one(".install-confirm-body", Static).render()
            assert isinstance(content, Content)
            reverses = _link_span_reverses(content)
            assert reverses
            assert all(reverses)

    async def test_leaving_link_clears_pointer_and_highlight(self) -> None:
        """Moving off the link restores the default cursor and plain style."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallPackageConfirmScreen("langchain-custom")
            app.push_screen(screen)
            await pilot.pause()

            screen.on_mouse_move(
                self._move_event("https://pypi.org/project/langchain-custom/")  # ty: ignore
            )
            screen.on_mouse_move(self._move_event(None))  # ty: ignore
            await pilot.pause()

            assert screen.styles.pointer == "default"
            content = screen.query_one(".install-confirm-body", Static).render()
            assert isinstance(content, Content)
            reverses = _link_span_reverses(content)
            assert reverses
            assert not any(reverses)

    async def test_on_leave_clears_hover_state(self) -> None:
        """`on_leave` resets cursor and highlight when the mouse exits."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallPackageConfirmScreen("langchain-custom")
            app.push_screen(screen)
            await pilot.pause()

            screen.on_mouse_move(
                self._move_event("https://pypi.org/project/langchain-custom/")  # ty: ignore
            )
            screen.on_leave()
            await pilot.pause()

            assert screen.styles.pointer == "default"
            content = screen.query_one(".install-confirm-body", Static).render()
            assert isinstance(content, Content)
            reverses = _link_span_reverses(content)
            assert reverses
            assert not any(reverses)

    def test_click_on_link_opens_url(self) -> None:
        """A click on the package link routes through `open_style_link`."""
        screen = InstallPackageConfirmScreen("langchain-custom")
        event = self._click_event("https://pypi.org/project/langchain-custom/")
        with patch(
            "deepagents_code.tui.widgets.install_confirm.open_style_link"
        ) as mock_open:
            screen.on_click(event)  # ty: ignore

        mock_open.assert_called_once_with(event)

    def test_provider_screen_click_on_link_opens_url(self) -> None:
        """The provider screen shares the same click-through behavior."""
        screen = InstallProviderConfirmScreen("baseten", "baseten")
        event = self._click_event("https://pypi.org/project/langchain-baseten/")
        with patch(
            "deepagents_code.tui.widgets.install_confirm.open_style_link"
        ) as mock_open:
            screen.on_click(event)  # ty: ignore

        mock_open.assert_called_once_with(event)
