"""Shared link-click handling for Textual widgets."""

from __future__ import annotations

import asyncio
import logging
import webbrowser
from typing import TYPE_CHECKING

from deepagents_cli.unicode_security import check_url_safety, strip_dangerous_unicode

if TYPE_CHECKING:
    from textual.app import App
    from textual.events import Click

logger = logging.getLogger(__name__)


async def open_url_async(url: str, *, app: App) -> bool:
    """Open url in a browser and toast on failure.

    Runs `webbrowser.open` in a thread, catches the platform errors
    that can arise when no browser backend is available, and posts a
    warning toast containing the URL so the user can copy it manually
    instead of the failure vanishing into a background worker log.

    Args:
        url: The URL to open.
        app: App used to post the failure toast.

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
    return opened


def open_style_link(event: Click) -> None:
    """Open the URL from a Rich link style on click, if present.

    Rich `Style(link=...)` embeds OSC 8 terminal hyperlinks, but Textual's
    mouse capture intercepts normal clicks before the terminal can act on them.
    By handling the Textual click event directly we open the URL with a single
    click, matching the behavior of links in the Markdown widget.

    URLs that fail the safety check (e.g. containing hidden Unicode or
    homograph domains) are blocked and not opened; the event bubbles and a
    warning is logged and displayed as a Textual notification.

    On success the event is stopped so it does not bubble further. On failure
    (e.g. no browser available in a headless environment) the error is logged at
    debug level and the event bubbles normally.

    Args:
        event: The Textual click event to inspect.
    """
    url = event.style.link
    if not url:
        return

    safety = check_url_safety(url)
    if not safety.safe:
        detail = safety.warnings[0] if safety.warnings else "Suspicious URL"
        logger.warning("Blocked suspicious URL: %s (%s)", url, detail)
        try:
            app = getattr(event, "app", None)
            notify = getattr(app, "notify", None)
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
        webbrowser.open(url)
    except Exception:
        logger.debug("Could not open browser for URL: %s", url, exc_info=True)
        return
    event.stop()
