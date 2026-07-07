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


def _notify(
    app: App | None, message: str, *, severity: str, timeout: int | None = None
) -> None:
    """Post a best-effort Textual toast, tolerating apps without `notify`.

    Centralizes the guard/`markup=False`/exception-swallowing pattern shared by
    every toast in this module so the call sites cannot drift apart. `markup` is
    always disabled so URL content can never be interpreted as Textual markup.

    Args:
        app: App-like object used to post the toast, or `None`.
        message: The toast body. Callers must sanitize any URL with
            `strip_dangerous_unicode` before interpolating it here.
        severity: Textual notification severity (e.g. `information`, `warning`).
        timeout: Optional toast lifetime in seconds; the Textual default is used
            when omitted.
    """
    notify = getattr(app, "notify", None)
    if not callable(notify):
        return
    kwargs: dict[str, object] = {"severity": severity, "markup": False}
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        notify(message, **kwargs)
    except (AttributeError, TypeError):
        logger.debug("Could not send notification", exc_info=True)


def _url_open_toasts_enabled() -> bool:
    """Return whether successful URL-open clicks should show a toast."""
    from deepagents_code.config_manifest import (
        get_option,
        load_config_toml,
        resolve_scalar,
    )

    option = get_option("display.show_url_open_toast")
    if option is None:
        return True
    value, _ = resolve_scalar(option, toml_data=load_config_toml())
    return bool(value)


def _notify_url_opened(app: App | None, url: str) -> None:
    """Show the URL-opened toast when the user has not opted out."""
    if app is None or not _url_open_toasts_enabled():
        return
    _notify(
        app,
        f"Opening URL in default browser: {strip_dangerous_unicode(url)}",
        severity="information",
        timeout=4,
    )


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


async def open_checked_url_async(
    url: str, *, app: App, notify_on_success: bool = False
) -> bool:
    """Open a URL after applying the shared URL safety check.

    Args:
        url: The URL to validate and open.
        app: App used to post browser-open notifications.
        notify_on_success: Whether to post an informational toast when the
            browser accepts the URL.

    Returns:
        `True` when the URL passed safety checks and the browser accepted it;
        `False` when the URL was blocked or the browser could not open it.
    """
    safety = check_url_safety(url)
    if not safety.safe:
        detail = safety.warnings[0] if safety.warnings else "Suspicious URL"
        logger.warning("Blocked suspicious URL: %s (%s)", url, detail)
        _notify(
            app,
            f"Blocked suspicious URL: {strip_dangerous_unicode(url)}\n{detail}",
            severity="warning",
        )
        return False
    return await open_url_async(url, app=app, notify_on_success=notify_on_success)


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
        _notify(
            app,
            f"Could not open a browser. URL: {strip_dangerous_unicode(url)}",
            severity="warning",
            timeout=8,
        )
    elif notify_on_success:
        _notify_url_opened(app, url)
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

    On success the event is stopped so it does not bubble further and, unless
    the user has opted out, a best-effort informational toast reports the URL
    that was opened. If the browser cannot be launched -- either
    `webbrowser.open` raises or the backend declines and returns a falsy value
    -- a warning toast with the URL is shown so it can be copied manually, the
    failure is logged, and the event bubbles normally.

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
        _notify(
            notify_app,
            f"Blocked suspicious URL: {strip_dangerous_unicode(url)}\n{detail}",
            severity="warning",
        )
        return

    try:
        opened = webbrowser.open(url)
    except (webbrowser.Error, OSError) as exc:
        logger.warning("webbrowser.open failed for %s: %s", url, exc, exc_info=True)
        opened = False
    if not opened:
        _notify(
            notify_app,
            f"Could not open a browser. URL: {strip_dangerous_unicode(url)}",
            severity="warning",
            timeout=8,
        )
        return
    _notify_url_opened(notify_app, url)
    event.stop()
