"""Tests for shared terminal UI keyboard hints."""

from deepagents_code.config import ASCII_GLYPHS
from deepagents_code.tui.key_hints import modal_navigation_hint


def test_modal_navigation_hint_lists_forward_and_reverse_tab() -> None:
    """Modal navigation copy advertises both Tab directions."""
    assert modal_navigation_hint(ASCII_GLYPHS) == (
        f"{ASCII_GLYPHS.arrow_up}/{ASCII_GLYPHS.arrow_down} or Tab/Shift+Tab navigate"
    )
