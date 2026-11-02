"""Unit tests for style-link click handling."""

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code._env_vars import SHOW_URL_OPEN_TOAST
from deepagents_code.tui.widgets._links import (
    event_targets_link,
    open_checked_url_async,
    open_style_link,
    open_url_async,
)

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import App


def _move_event(
    *, link: str | None = None, meta: dict | None = None
) -> SimpleNamespace:
    """Build a minimal mouse-move-like event for hover tests."""
    return SimpleNamespace(style=SimpleNamespace(link=link, meta=meta or {}))


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


async def test_open_checked_url_async_blocks_suspicious_url() -> None:
    """Async checked opening should block suspicious URLs before the browser."""
    notify = MagicMock()
    app = cast("App[None]", SimpleNamespace(notify=notify))

    with patch("deepagents_code.tui.widgets._links.webbrowser.open") as mock_open:
        opened = await open_checked_url_async(
            "https://example.com/\u200b[admin]",
            app=app,
            notify_on_success=True,
        )

    assert opened is False
    mock_open.assert_not_called()
    notify.assert_called_once()
    args, kwargs = notify.call_args
    assert "Blocked suspicious URL" in args[0]
    assert "https://example.com/[admin]" in args[0]
    assert kwargs["severity"] == "warning"
    assert kwargs["markup"] is False


_UNSUPPORTED_URLS = [
    "mailto:test@example.com",
    "javascript:alert(1)",
    "file:///etc/passwd",
    "vscode://file/tmp/x",
    "custom-app://open?payload=1",
    "MAILTO:test@example.com",
    "example.com",
]


@pytest.mark.parametrize("url", _UNSUPPORTED_URLS)
async def test_open_checked_url_async_blocks_unsupported_scheme(url: str) -> None:
    """Links with a scheme other than http or https never reach the browser."""
    notify = MagicMock()
    app = cast("App[None]", SimpleNamespace(notify=notify))

    with patch("deepagents_code.tui.widgets._links.webbrowser.open") as mock_open:
        opened = await open_checked_url_async(url, app=app, notify_on_success=True)

    assert opened is False
    mock_open.assert_not_called()
    notify.assert_called_once()
    args, kwargs = notify.call_args
    assert "unsupported scheme" in args[0]
    assert kwargs["severity"] == "warning"
    assert kwargs["markup"] is False


@pytest.mark.parametrize("url", _UNSUPPORTED_URLS)
def test_open_style_link_blocks_unsupported_scheme(url: str) -> None:
    """Style links with a scheme other than http or https are not opened."""
    event = _event_with_link(url)

    with patch("deepagents_code.tui.widgets._links.webbrowser.open") as mock_open:
        open_style_link(event)  # ty: ignore

    mock_open.assert_not_called()
    event.stop.assert_not_called()
    event.app.notify.assert_called_once()
    assert "unsupported scheme" in event.app.notify.call_args.args[0]


async def test_open_url_async_blocks_unparseable_url() -> None:
    """A URL whose scheme cannot be parsed is blocked, not opened."""
    app = cast("App[None]", SimpleNamespace(notify=MagicMock()))

    with patch("deepagents_code.tui.widgets._links.webbrowser.open") as mock_open:
        opened = await open_url_async("http://[::1", app=app)

    assert opened is False
    mock_open.assert_not_called()


@pytest.mark.parametrize("url", ["http://example.com", "HTTPS://example.com"])
async def test_open_checked_url_async_allows_web_schemes(url: str) -> None:
    """Http and https links still open, regardless of scheme case."""
    app = cast("App[None]", SimpleNamespace(notify=MagicMock()))

    with patch(
        "deepagents_code.tui.widgets._links.webbrowser.open", return_value=True
    ) as mock_open:
        opened = await open_checked_url_async(url, app=app)

    assert opened is True
    mock_open.assert_called_once_with(url)


async def test_open_url_async_can_toast_on_success() -> None:
    """Async link opening can opt into the same success toast."""
    notify = MagicMock()
    app = cast("App[None]", SimpleNamespace(notify=notify))

    with (
        patch.dict(os.environ, {SHOW_URL_OPEN_TOAST: "1"}),
        patch("deepagents_code.tui.widgets._links.webbrowser.open", return_value=True),
    ):
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


async def test_open_url_async_warns_on_failure() -> None:
    """Async link opening warns with the URL when the browser declines."""
    notify = MagicMock()
    app = cast("App[None]", SimpleNamespace(notify=notify))

    with patch(
        "deepagents_code.tui.widgets._links.webbrowser.open",
        return_value=False,
    ):
        opened = await open_url_async("https://example.com", app=app)

    assert opened is False
    notify.assert_called_once()
    args, kwargs = notify.call_args
    assert "https://example.com" in args[0]
    assert kwargs["severity"] == "warning"
    assert kwargs["markup"] is False


def test_event_targets_link_detects_markdown_click_action() -> None:
    """Markdown `@click=link(...)` meta actions count as links."""
    event = _move_event(meta={"@click": "link('https://example.com')"})
    assert event_targets_link(event) is True  # ty: ignore


def test_event_targets_link_detects_osc8_link() -> None:
    """A Rich `Style(link=...)` span counts as a link."""
    assert event_targets_link(_move_event(link="https://example.com")) is True  # ty: ignore


def test_event_targets_link_ignores_other_click_actions() -> None:
    """Non-link `@click` actions are not treated as links."""
    assert event_targets_link(_move_event(meta={"@click": "toggle()"})) is False  # ty: ignore


def test_event_targets_link_ignores_plain_text() -> None:
    """Plain hovered text is not a link."""
    assert event_targets_link(_move_event()) is False  # ty: ignore


def test_open_style_link_config_can_suppress_success_toast(tmp_path: Path) -> None:
    """The config file can disable success toasts when env is unset."""
    event = _event_with_link("https://example.com")
    # The resolver's user tier is the `DEFAULT_CONFIG_PATH` file that the
    # `_isolate_state_dir` fixture redirects under `tmp_path`.
    (tmp_path / "config.toml").write_text(
        "[ui]\nshow_url_open_toast = false\n", encoding="utf-8"
    )

    with (
        patch.dict(os.environ, {SHOW_URL_OPEN_TOAST: ""}),
        patch("deepagents_code.tui.widgets._links.webbrowser.open", return_value=True),
    ):
        open_style_link(event)  # ty: ignore

    event.stop.assert_called_once()
    event.app.notify.assert_not_called()


def test_open_style_link_ignores_malformed_markdown_link_action() -> None:
    """Malformed Markdown link metadata should not reach the browser opener."""
    event = _event_with_meta({"@click": "link(https://example.com)"})

    with patch("deepagents_code.tui.widgets._links.webbrowser.open") as mock_open:
        open_style_link(event)  # ty: ignore

    mock_open.assert_not_called()
    event.stop.assert_not_called()
    event.app.notify.assert_not_called()
