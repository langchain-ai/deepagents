"""Shared link-click handling for Textual widgets."""

from __future__ import annotations

import logging
import webbrowser
from contextlib import suppress
from typing import TYPE_CHECKING

from deepagents_cli.unicode_security import check_url_safety

if TYPE_CHECKING:
    from textual.events import Click

logger = logging.getLogger(__name__)


def open_style_link(event: Click) -> None:
    """Open the URL from a Rich link style on click, if present.

    Rich `Style(link=...)` embeds OSC 8 terminal hyperlinks, but Textual's
    mouse capture intercepts normal clicks before the terminal can act on them.
    By handling the Textual click event directly we open the URL with a single
    click, matching the behavior of links in the Markdown widget.

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
        detail = "; ".join(safety.warnings[:2]) or "Suspicious URL"
        logger.warning("Blocked suspicious URL: %s (%s)", url, detail)
        with suppress(Exception):
            app = getattr(event, "app", None)
            notify = getattr(app, "notify", None)
            if callable(notify):
                notify(
                    f"Blocked suspicious URL: {url}\n{detail}",
                    severity="warning",
                )
        return

    try:
        webbrowser.open(url)
    except Exception:
        logger.debug("Could not open browser for URL: %s", url, exc_info=True)
        return
    event.stop()
