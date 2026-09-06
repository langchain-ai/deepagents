"""Tests for shared terminal UI keyboard hints."""

import pytest

from deepagents_code.config import ASCII_GLYPHS, UNICODE_GLYPHS, Glyphs
from deepagents_code.tui.key_hints import modal_navigation_hint


@pytest.mark.parametrize(
    ("glyphs", "expected"),
    [
        (UNICODE_GLYPHS, "↑/↓ or Tab/Shift+Tab navigate"),
        (ASCII_GLYPHS, "^/v or Tab/Shift+Tab navigate"),
    ],
    ids=["unicode", "ascii"],
)
def test_modal_navigation_hint_copy(glyphs: Glyphs, expected: str) -> None:
    """Both glyph sets render the literal footer copy users read.

    Spelled out rather than re-derived from `glyphs.arrow_up` so a hardcoded
    arrow (the regression the shared helper exists to prevent) fails here
    instead of passing against its own f-string.
    """
    assert modal_navigation_hint(glyphs) == expected
