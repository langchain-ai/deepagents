"""Optional runtime tools shared by Talon channel agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code.tools import fetch_url, web_search

if TYPE_CHECKING:
    from collections.abc import Callable


def build_web_tools() -> list[Callable[..., object]]:
    """Return web tools available to a Talon runtime.

    Returns:
        Tools for URL fetch and web search.
    """
    return [fetch_url, web_search]
