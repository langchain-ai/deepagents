"""Theme system for deepagents-cli.

Simple mapping to Textual's built-in themes plus custom themes.
We use Textual's CSS variables ($primary, $success, etc.) everywhere.

Usage:
    from deepagents_cli.themes import THEMES, get_textual_theme, set_theme

    # Set theme by name
    set_theme("tokyo-night")

    # Get Textual theme name for App.theme
    app.theme = get_textual_theme()
"""

from __future__ import annotations

from textual.theme import Theme

from deepagents_cli.themes.detection import detect_dark_mode

# Custom LangChain theme using official brand colors
# - Te Papa Green: #1d3c3e (dark teal - primary)
# - Lotus: #823a45 (muted maroon - accent)
# - Selago: #ebe6fd (light lavender - for light mode)
LANGCHAIN_THEME = Theme(
    name="langchain",
    primary="#1d3c3e",       # Te Papa Green - main brand color
    secondary="#4a7c7f",     # Lighter teal for secondary elements
    accent="#823a45",        # Lotus - accent/highlight color
    background="#0a1415",    # Very dark teal-black
    surface="#142122",       # Slightly lighter surface
    panel="#1d3c3e",         # Panel matches primary
    foreground="#e8ebe8",    # Off-white text
    success="#4ade80",       # Green for success
    warning="#fbbf24",       # Amber for warnings
    error="#ef4444",         # Red for errors
    dark=True,
    variables={
        "block-cursor-foreground": "#0a1415",
        "footer-key-foreground": "#1d3c3e",
        "input-selection-background": "#1d3c3e 50%",
    },
)

# Available themes - maps our names to Textual theme names
# Custom themes use the same name for both key and value
THEMES: dict[str, str] = {
    "default": "textual-dark",
    "tokyo-night": "tokyo-night",
    "catppuccin": "catppuccin-mocha",
    "langchain": "langchain",
}

# Custom themes that need to be registered with the app
CUSTOM_THEMES: list[Theme] = [LANGCHAIN_THEME]

# Current theme state
_current_theme: str = "default"


def set_theme(name: str) -> None:
    """Set the current theme by name.

    Args:
        name: Theme name (default, tokyo-night, catppuccin)

    Raises:
        ValueError: If theme name is not recognized
    """
    global _current_theme
    if name not in THEMES:
        valid = ", ".join(THEMES.keys())
        msg = f"Unknown theme: {name}. Valid themes: {valid}"
        raise ValueError(msg)
    _current_theme = name


def get_textual_theme() -> str:
    """Get the Textual theme name for the current theme.

    Returns:
        Textual theme name (e.g., "textual-dark", "tokyo-night")
    """
    return THEMES.get(_current_theme, "textual-dark")


def is_dark_mode() -> bool:
    """Check if dark mode is enabled.

    Returns:
        True if dark mode (always True for now, light mode not fully supported)
    """
    return detect_dark_mode()


# For backwards compatibility - will be removed
def get_theme() -> None:
    """Deprecated - use Textual CSS variables instead."""
    return None
