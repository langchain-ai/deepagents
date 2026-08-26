"""Shared keyboard hints for terminal UI components.

Modal footers advertise the same navigation keys in a dozen screens. Keeping
the wording here means a binding change is edited once instead of being chased
across every modal, and the glyph substitution for ASCII terminals cannot drift
between them.
"""

from deepagents_code.config import Glyphs


def modal_navigation_hint(glyphs: Glyphs) -> str:
    """Build the navigation hint for modals whose Tab keys move the cursor.

    Only for screens where Tab and Shift+Tab step the selection. A modal that
    binds Tab to something else writes its own line rather than using this one
    -- `mcp_viewer` (Tab jumps between servers), `model_selector` (Tab
    autocompletes), and `plugin_manager` (Tab cycles tabs) all do.

    The hint is long enough to wrap once a modal narrows, so the host `Static`
    needs `height: auto` to grow and `dock: bottom` to reserve the extra row.
    Without the dock the wrapped row is laid out past the modal's bottom edge,
    where the compositor never paints it.

    Args:
        glyphs: Glyph set for the active terminal mode.

    Returns:
        The navigation hint for the active glyph set.
    """
    return f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab/Shift+Tab navigate"
