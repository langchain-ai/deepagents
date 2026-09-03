"""Tests for the `/install --package` confirmation modal."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.content import Content
from textual.style import Style as TStyle
from textual.widgets import Static

from deepagents_code.tui.widgets.install_confirm import (
    InstallPackageConfirmScreen,
    InstallProviderConfirmScreen,
)

if TYPE_CHECKING:
    from textual.screen import Screen


class _InstallConfirmTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


def _body_of(screen: Screen[bool]) -> Static:
    """Return the body widget of an install confirmation screen."""
    return screen.query_one(".install-confirm-body", Static)


def _rendered(widget: Static) -> Content:
    """Return a widget's rendered `Content`."""
    content = widget.render()
    assert isinstance(content, Content)
    return content


def _link_cells(widget: Static) -> list[tuple[int, int]]:
    """Return every widget-relative cell whose rendered style carries a link.

    Driving hover and click from real cell offsets (rather than synthetic
    events carrying a URL of the test's own choosing) means these tests fail
    if the link is dropped, mis-spanned, or never reaches the rendered output.
    """
    return [
        (x, y)
        for y in range(widget.region.height)
        for x in range(widget.region.width)
        if getattr(widget.get_style_at(x, y), "link", None)
    ]


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

    async def test_hover_over_link_shows_pointer_and_highlight(self) -> None:
        """Hovering the link sets a pointer cursor and highlights the text."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallPackageConfirmScreen("langchain-custom")
            app.push_screen(screen)
            await pilot.pause()
            body = _body_of(screen)

            await pilot.hover(body, offset=_link_cells(body)[0])

            assert screen.styles.pointer == "pointer"
            reverses = _link_span_reverses(_rendered(body))
            assert reverses
            assert all(reverses)

    async def test_hover_off_link_clears_pointer_and_highlight(self) -> None:
        """Moving onto plain body text restores the cursor and plain style."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallPackageConfirmScreen("langchain-custom")
            app.push_screen(screen)
            await pilot.pause()
            body = _body_of(screen)
            cells = _link_cells(body)
            linked = set(cells)

            await pilot.hover(body, offset=cells[0])
            plain = next(
                (x, y)
                for y in range(body.region.height)
                for x in range(body.region.width)
                if (x, y) not in linked
            )
            await pilot.hover(body, offset=plain)

            assert screen.styles.pointer == "default"
            reverses = _link_span_reverses(_rendered(body))
            assert reverses
            assert not any(reverses)

    async def test_on_leave_clears_hover_state(self) -> None:
        """`on_leave` resets cursor and highlight when the mouse exits."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallPackageConfirmScreen("langchain-custom")
            app.push_screen(screen)
            await pilot.pause()
            body = _body_of(screen)

            await pilot.hover(body, offset=_link_cells(body)[0])
            screen.on_leave()
            await pilot.pause()

            assert screen.styles.pointer == "default"
            reverses = _link_span_reverses(_rendered(body))
            assert reverses
            assert not any(reverses)

    async def test_link_covers_exactly_the_package_name(self) -> None:
        """The clickable region spans the package name and nothing else."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallPackageConfirmScreen("langchain-custom")
            app.push_screen(screen)
            await pilot.pause()

            assert len(_link_cells(_body_of(screen))) == len("langchain-custom")

    async def test_click_on_link_opens_pypi_url(self) -> None:
        """Clicking the package name opens its PyPI page in a browser."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallPackageConfirmScreen("langchain-custom")
            app.push_screen(screen)
            await pilot.pause()
            body = _body_of(screen)

            with patch(
                "deepagents_code.tui.widgets._links.webbrowser.open",
                return_value=True,
            ) as mock_open:
                await pilot.click(body, offset=_link_cells(body)[0])

            mock_open.assert_called_once_with(
                "https://pypi.org/project/langchain-custom/"
            )

    async def test_provider_click_opens_resolved_package_url(self) -> None:
        """The provider screen links the resolved distribution, not the extra.

        `nvidia`'s extra is `nvidia`, but its distribution is
        `langchain-nvidia-ai-endpoints` -- so this also pins the
        multi-underscore normalization in `provider_package_name`.
        """
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallProviderConfirmScreen("nvidia", "nvidia")
            app.push_screen(screen)
            await pilot.pause()
            body = _body_of(screen)

            with patch(
                "deepagents_code.tui.widgets._links.webbrowser.open",
                return_value=True,
            ) as mock_open:
                await pilot.click(body, offset=_link_cells(body)[0])

            mock_open.assert_called_once_with(
                "https://pypi.org/project/langchain-nvidia-ai-endpoints/"
            )

    async def test_provider_hover_highlights_link_beside_model_spec(self) -> None:
        """With a model spec, hover highlights the link and not the bold model."""
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallProviderConfirmScreen(
                "baseten", "baseten", "baseten:moonshotai/Kimi-K2.7-Code"
            )
            app.push_screen(screen)
            await pilot.pause()
            body = _body_of(screen)

            await pilot.hover(body, offset=_link_cells(body)[0])

            assert screen.styles.pointer == "pointer"
            content = _rendered(body)
            assert all(_link_span_reverses(content))
            model_spans = [
                span
                for span in content.spans
                if content.plain[span.start : span.end]
                == "baseten:moonshotai/Kimi-K2.7-Code"
            ]
            assert model_spans
            assert not any(
                isinstance(span.style, TStyle) and span.style.reverse
                for span in model_spans
            )

    async def test_uncurated_provider_renders_package_unlinked(self) -> None:
        """An unknown provider shows the extra as plain text, not a bad link.

        Extras are not distribution names, so linking one would point at an
        unrelated PyPI project rather than the integration being installed.
        """
        app = _InstallConfirmTestApp()
        async with app.run_test() as pilot:
            screen = InstallProviderConfirmScreen("not-a-provider", "some-extra")
            app.push_screen(screen)
            await pilot.pause()
            body = _body_of(screen)

            content = _rendered(body)
            assert "some-extra" in content.plain
            assert not _link_cells(body)
            assert not [
                span
                for span in content.spans
                if isinstance(span.style, TStyle) and span.style.link
            ]
