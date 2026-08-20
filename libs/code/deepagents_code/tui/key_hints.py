"""Shared keyboard hints for terminal UI components."""

from deepagents_code.config import Glyphs


def modal_navigation_hint(glyphs: Glyphs) -> str:
    """Build the standard navigation hint for modal choices.

    Returns:
        The navigation hint for the active glyph set.
    """
    return f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab/Shift+Tab navigate"
