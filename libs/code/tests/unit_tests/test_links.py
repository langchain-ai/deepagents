"""Unit tests for style-link click handling."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

from deepagents_code.widgets._links import (
    event_targets_link,
    open_style_link,
    open_url_async,
)

if TYPE_CHECKING:
    from textual.app import App


def _move_event(
    *, link: str | None = None, meta: dict | None = None
) -> SimpleNamespace:
    """Build a minimal mouse-move-like event for hover tests."""
    return SimpleNamespace(style=SimpleNamespace(link=link, meta=meta or {}))


def test_event_targets_link_detects_osc8_link() -> None:
    """A Rich `Style(link=...)` span counts as a link."""
    assert event_targets_link(_move_event(link="https://example.com")) is True  # ty: ignore


def test_event_targets_link_detects_markdown_click_action() -> None:
    """Markdown `@click=link(...)` meta actions count as links."""
    event = _move_event(meta={"@click": "link('https://example.com')"})
    assert event_targets_link(event) is True  # ty: ignore


def test_event_targets_link_ignores_plain_text() -> None:
    """Plain hovered text is not a link."""
    assert event_targets_link(_move_event()) is False  # ty: ignore


def test_event_targets_link_ignores_other_click_actions() -> None:
    """Non-link `@click` actions are not treated as links."""
    assert event_targets_link(_move_event(meta={"@click": "toggle()"})) is False  # ty: ignore


def _event_with_link(url: str) -> SimpleNamespace:
    """Build a minimal click-like event object for tests."""
    return SimpleNamespace(
        style=SimpleNamespace(link=url),
        app=SimpleNamespace(notify=MagicMock()),
        stop=MagicMock(),
    )


def _event_with_meta(meta: dict[str, str]) -> SimpleNamespace:
    """Build a minimal click event whose URL comes from style metadata."""
    return SimpleNamespace(
        style=SimpleNamespace(link=None, meta=meta),
        app=SimpleNamespace(notify=MagicMock()),
        stop=MagicMock(),
    )


def test_open_style_link_opens_browser_and_stops_event() -> None:
    """Safe links should open, toast confirmation, and stop event propagation."""
    event = _event_with_link("https://example.com")

    with patch("deepagents_code.widgets._links.webbrowser.open") as mock_open:
        mock_open.return_value = True
        open_style_link(event)  # ty: ignore

    mock_open.assert_called_once_with("https://example.com")
    event.stop.assert_called_once()
    event.app.notify.assert_called_once()
    args, kwargs = event.app.notify.call_args
    assert args[0] == "Opening URL in default browser: https://example.com"
    assert kwargs["severity"] == "information"
    assert kwargs["markup"] is False


def test_open_style_link_notifies_from_event_widget_app() -> None:
    """Real Textual click events expose the app through `event.widget.app`."""
    notify = MagicMock()
    event = SimpleNamespace(
        style=SimpleNamespace(link="https://example.com", meta={}),
        widget=SimpleNamespace(app=SimpleNamespace(notify=notify)),
        stop=MagicMock(),
    )

    with patch("deepagents_code.widgets._links.webbrowser.open", return_value=True):
        open_style_link(event)  # ty: ignore

    notify.assert_called_once()
    args, kwargs = notify.call_args
    assert args[0] == "Opening URL in default browser: https://example.com"
    assert kwargs["severity"] == "information"
    assert kwargs["markup"] is False
    event.stop.assert_called_once()


async def test_open_url_async_can_toast_on_success() -> None:
    """Async link opening can opt into the same success toast."""
    notify = MagicMock()
    app = cast("App[None]", SimpleNamespace(notify=notify))

    with patch("deepagents_code.widgets._links.webbrowser.open", return_value=True):
        opened = await open_url_async(
            "https://example.com",
            app=app,
            notify_on_success=True,
        )

    assert opened is True
    notify.assert_called_once()
    args, kwargs = notify.call_args
    assert args[0] == "Opening URL in default browser: https://example.com"
    assert kwargs["severity"] == "information"
    assert kwargs["markup"] is False


def test_open_style_link_opens_markdown_link_action() -> None:
    """Markdown `@click=link(...)` metadata should open like Rich link styles."""
    event = _event_with_meta({"@click": "link('https://example.com/docs')"})

    with patch("deepagents_code.widgets._links.webbrowser.open") as mock_open:
        mock_open.return_value = True
        open_style_link(event)  # ty: ignore

    mock_open.assert_called_once_with("https://example.com/docs")
    event.stop.assert_called_once()
    event.app.notify.assert_called_once()


def test_open_style_link_no_toast_when_browser_does_not_open() -> None:
    """When the browser backend declines, no toast is shown and event bubbles."""
    event = _event_with_link("https://example.com")

    with patch("deepagents_code.widgets._links.webbrowser.open") as mock_open:
        mock_open.return_value = False
        open_style_link(event)  # ty: ignore

    mock_open.assert_called_once_with("https://example.com")
    event.stop.assert_not_called()
    event.app.notify.assert_not_called()


def test_open_style_link_ignores_malformed_markdown_link_action() -> None:
    """Malformed Markdown link metadata should not reach the browser opener."""
    event = _event_with_meta({"@click": "link(https://example.com)"})

    with patch("deepagents_code.widgets._links.webbrowser.open") as mock_open:
        open_style_link(event)  # ty: ignore

    mock_open.assert_not_called()
    event.stop.assert_not_called()
    event.app.notify.assert_not_called()


def test_open_style_link_blocks_suspicious_url_with_markup_disabled() -> None:
    """Suspicious links should notify with markup parsing disabled."""
    event = _event_with_link("https://example.com/\u200b[admin]")

    with patch("deepagents_code.widgets._links.webbrowser.open") as mock_open:
        open_style_link(event)  # ty: ignore

    mock_open.assert_not_called()
    event.stop.assert_not_called()
    event.app.notify.assert_called_once()
    _, kwargs = event.app.notify.call_args
    assert kwargs["severity"] == "warning"
    assert kwargs["markup"] is False
