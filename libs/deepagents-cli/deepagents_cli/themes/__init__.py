"""Theme system for deepagents-cli.

Provides centralized color management with dark/light mode support.

Usage:
    from deepagents_cli.themes import theme

    # Get a color for Rich markup
    color = theme.primary  # e.g., "#58a6ff"

    # Use in Rich markup
    f"[{theme.primary}]Hello[/{theme.primary}]"

    # Use in Textual CSS
    f"color: {theme.text};"
"""

from __future__ import annotations

from deepagents_cli.themes.builtin import BUILTIN_THEMES, DEFAULT_THEME
from deepagents_cli.themes.detection import detect_dark_mode
from deepagents_cli.themes.theme import Theme, ThemeColors

__all__ = [
    "BUILTIN_THEMES",
    "Theme",
    "ThemeColors",
    "get_theme",
    "is_dark_mode",
    "set_theme",
    "theme",
]

# Module-level state
_state: dict[str, Theme | bool | None] = {
    "theme": DEFAULT_THEME,
    "dark_mode": None,  # Lazy detection
}


def is_dark_mode() -> bool:
    """Check if dark mode is active (cached after first detection)."""
    if _state["dark_mode"] is None:
        _state["dark_mode"] = detect_dark_mode()
    return bool(_state["dark_mode"])


def get_theme() -> ThemeColors:
    """Get the current theme colors for the active mode."""
    current = _state["theme"]
    if not isinstance(current, Theme):
        current = DEFAULT_THEME
    return current.colors(dark_mode=is_dark_mode())


def set_theme(name: str) -> None:
    """Set the active theme by name.

    Args:
        name: Theme name (e.g., "default", "tokyo-night", "catppuccin")

    Raises:
        ValueError: If theme name is not found.
    """
    if name not in BUILTIN_THEMES:
        available = ", ".join(BUILTIN_THEMES.keys())
        msg = f"Unknown theme: {name}. Available: {available}"
        raise ValueError(msg)
    _state["theme"] = BUILTIN_THEMES[name]


class _ThemeProxy:
    """Proxy object that delegates attribute access to the current theme colors.

    This allows using `theme.primary` instead of `get_theme().primary`,
    and automatically respects dark/light mode.
    """

    def __getattr__(self, name: str) -> str:
        colors = get_theme()
        if hasattr(colors, name):
            return getattr(colors, name)
        msg = f"Theme has no color '{name}'"
        raise AttributeError(msg)


# The main theme object - use this to access colors
theme = _ThemeProxy()
