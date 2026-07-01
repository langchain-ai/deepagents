"""Shared link-click handling for Textual widgets."""

from __future__ import annotations

import ast
import asyncio
import logging
import webbrowser
from typing import TYPE_CHECKING

from deepagents_code.unicode_security import check_url_safety, strip_dangerous_unicode

if TYPE_CHECKING:
    from textual.app import App
    from textual.events import Click, MouseMove


def _event_app(event: object, app: App | None = None) -> App | None:
    """Return the app for a click event, including real Textual widgets."""
    if app is not None:
        return app
    widget = getattr(event, "widget", None)
    widget_app = getattr(widget, "app", None)
    if widget_app is not None:
        return widget_app
    event_app = getattr(event, "app", None)
    return event_app if event_app is not None else None


logger = logging.getLogger(__name__)


def _link_action_url(click: object) -> str | None:
    """Extract a URL from Textual's Markdown `link(...)` click action.

    Args:
        click: The `@click` style metadata value to inspect.

    Returns:
        The parsed URL when the metadata is a quoted `link(...)` action.
    """
    if not isinstance(click, str):
        return None
    if not click.startswith("link(") or not click.endswith(")"):
        return None
    try:
        url = ast.literal_eval(click[len("link(") : -1].strip())
    except (SyntaxError, ValueError):
        return None
    return url if isinstance(url, str) and url else None


def _style_url(style: object) -> str | None:
    """Return a URL from either Rich link style or Textual click metadata.

    Args:
        style: The Textual event style to inspect.

    Returns:
        The URL embedded in the style, if one is present.
    """
    url = getattr(style, "link", None)
    if isinstance(url, str) and url:
        return url
    meta = getattr(style, "meta", None)
    if not isinstance(meta, dict):
        return None
    return _link_action_url(meta.get("@click"))


def event_targets_link(event: MouseMove) -> bool:
    """Return whether the style under the mouse points to a clickable link.

    Detects both Rich `Style(link=...)` (OSC 8) hyperlinks and the
    `@click=link(...)` meta actions that Textual's `Markdown` widget attaches
    to rendered links and images.

    Args:
        event: The Textual mouse-move event to inspect.

    Returns:
        `True` when the hovered character belongs to a link span.
    """
    return _style_url(event.style) is not None


async def open_url_async(
    url: str, *, app: App, notify_on_success: bool = False
) -> bool:
    """Open url in a browser and toast on failure.

    Runs `webbrowser.open` in a thread, catches the platform errors
    that can arise when no browser backend is available, and posts a
    warning toast containing the URL so the user can copy it manually
    instead of the failure vanishing into a background worker log.

    Args:
        url: The URL to open.
        app: App used to post browser-open notifications.
        notify_on_success: Whether to post an informational toast when the
            browser accepts the URL.

    Returns:
        `True` when the browser accepted the URL; `False` otherwise
            (in which case a warning toast has already been posted).
    """
    try:
        opened = await asyncio.to_thread(webbrowser.open, url)
    except (webbrowser.Error, OSError) as exc:
        logger.warning("webbrowser.open failed for %s: %s", url, exc, exc_info=True)
        opened = False
    if not opened:
        app.notify(
            f"Could not open a browser. URL: {url}",
            severity="warning",
            timeout=8,
            markup=False,
        )
    elif notify_on_success:
        app.notify(
            f"Opening URL in default browser: {strip_dangerous_unicode(url)}",
            severity="information",
            timeout=4,
            markup=False,
        )
    return opened


def open_style_link(event: Click, *, app: App | None = None) -> None:
    """Open the URL from a Rich link style on click, if present.

    Rich `Style(link=...)` embeds OSC 8 terminal hyperlinks, but Textual's
    mouse capture intercepts normal clicks before the terminal can act on them.
    By handling the Textual click event directly we open the URL with a single
    click, matching the behavior of links in the Markdown widget.

    URLs that fail the safety check (e.g. containing hidden Unicode or
    homograph domains) are blocked and not opened; the event bubbles and a
    warning is logged and displayed as a Textual notification.

    On success the event is stopped so it does not bubble further and an
    informational toast confirms which URL was opened. On failure (e.g. no
    browser available in a headless environment) the error is logged at debug
    level and the event bubbles normally.

    Args:
        event: The Textual click event to inspect.
        app: App used to post browser-open notifications.
    """
    notify_app = _event_app(event, app)
    url = _style_url(event.style)
    if not url:
        return

    safety = check_url_safety(url)
    if not safety.safe:
        detail = safety.warnings[0] if safety.warnings else "Suspicious URL"
        logger.warning("Blocked suspicious URL: %s (%s)", url, detail)
        try:
            notify = getattr(notify_app, "notify", None)
            if callable(notify):
                safe_url = strip_dangerous_unicode(url)
                notify(
                    f"Blocked suspicious URL: {safe_url}\n{detail}",
                    severity="warning",
                    markup=False,
                )
        except (AttributeError, TypeError):
            logger.debug("Could not send URL-blocked notification", exc_info=True)
        return

    try:
        opened = webbrowser.open(url)
    except Exception:
        logger.debug("Could not open browser for URL: %s", url, exc_info=True)
        return
    if not opened:
        logger.debug("Browser backend did not open URL: %s", url)
        return
    try:
        notify = getattr(notify_app, "notify", None)
        if callable(notify):
            notify(
                f"Opening URL in default browser: {strip_dangerous_unicode(url)}",
                severity="information",
                timeout=4,
                markup=False,
            )
    except (AttributeError, TypeError):
        logger.debug("Could not send URL-opened notification", exc_info=True)
    event.stop()
