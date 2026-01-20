"""Theme definitions and color tokens for deepagents-cli."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ThemeColors:
    """Color definitions for a theme.

    All colors are hex strings (e.g., "#ffffff").
    Semantic tokens reference these base colors.
    """

    # Base colors
    text: str = "#c9d1d9"  # Primary text
    text_muted: str = "#8b949e"  # Secondary/muted text
    background: str = "#0d1117"  # Main background
    surface: str = "#161b22"  # Elevated surfaces

    # Accent colors
    primary: str = "#58a6ff"  # Primary accent (links, selections)
    secondary: str = "#8b949e"  # Secondary accent

    # Semantic colors
    success: str = "#3fb950"  # Success states, additions
    error: str = "#f85149"  # Error states, deletions
    warning: str = "#d29922"  # Warning states

    # Diff colors
    diff_add: str = "#3fb950"
    diff_add_bg: str = "#1b4721"
    diff_remove: str = "#f85149"
    diff_remove_bg: str = "#5c2121"

    # Mode indicators
    mode_bash: str = "#f97583"
    mode_command: str = "#d2a8ff"

    # Approval states
    approve: str = "#3fb950"
    reject: str = "#f85149"
    auto_approve_on: str = "#3fb950"
    auto_approve_off: str = "#d29922"


@dataclass
class Theme:
    """A complete theme definition."""

    name: str
    dark: ThemeColors = field(default_factory=ThemeColors)
    light: ThemeColors | None = None  # Optional light variant

    def colors(self, *, dark_mode: bool = True) -> ThemeColors:
        """Get colors for the current mode."""
        if dark_mode or self.light is None:
            return self.dark
        return self.light
