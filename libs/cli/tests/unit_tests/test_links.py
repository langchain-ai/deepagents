"""Unit tests for style-link click handling."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deepagents_cli.widgets._links import open_style_link


def _event_with_link(url: str) -> SimpleNamespace:
    """Build a minimal click-like event object for tests."""
    return SimpleNamespace(
        style=SimpleNamespace(link=url),
        app=SimpleNamespace(notify=MagicMock()),
        stop=MagicMock(),
    )


def test_open_style_link_opens_browser_and_stops_event() -> None:
    """Safe links should open and stop event propagation."""
    event = _event_with_link("https://example.com")

    with patch("deepagents_cli.widgets._links.webbrowser.open") as mock_open:
        open_style_link(event)  # type: ignore[arg-type]

    mock_open.assert_called_once_with("https://example.com")
    event.stop.assert_called_once()
    event.app.notify.assert_not_called()


def test_open_style_link_blocks_suspicious_url_with_markup_disabled() -> None:
    """Suspicious links should notify with markup parsing disabled."""
    event = _event_with_link("https://example.com/\u200b[admin]")

    with patch("deepagents_cli.widgets._links.webbrowser.open") as mock_open:
        open_style_link(event)  # type: ignore[arg-type]

    mock_open.assert_not_called()
    event.stop.assert_not_called()
    event.app.notify.assert_called_once()
    _, kwargs = event.app.notify.call_args
    assert kwargs["severity"] == "warning"
    assert kwargs["markup"] is False
