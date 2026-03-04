"""Shared link-click handling for Textual widgets."""

from __future__ import annotations

import logging
import webbrowser
from typing import TYPE_CHECKING

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
    try:
        webbrowser.open(url)
    except Exception:
        logger.debug("Could not open browser for URL: %s", url, exc_info=True)
        return
    event.stop()
