"""Built-in themes for deepagents-cli."""

from __future__ import annotations

from deepagents_cli.themes.theme import Theme, ThemeColors

# Default theme - GitHub-inspired, easy on the eyes
DEFAULT_THEME = Theme(
    name="default",
    dark=ThemeColors(
        text="#c9d1d9",
        text_muted="#8b949e",
        background="#0d1117",
        surface="#161b22",
        primary="#58a6ff",
        secondary="#8b949e",
        success="#3fb950",
        error="#f85149",
        warning="#d29922",
        diff_add="#3fb950",
        diff_add_bg="#1b4721",
        diff_remove="#f85149",
        diff_remove_bg="#5c2121",
        mode_bash="#f97583",
        mode_command="#d2a8ff",
        approve="#3fb950",
        reject="#f85149",
        auto_approve_on="#3fb950",
        auto_approve_off="#d29922",
    ),
    light=ThemeColors(
        text="#24292f",
        text_muted="#57606a",
        background="#ffffff",
        surface="#f6f8fa",
        primary="#0969da",
        secondary="#57606a",
        success="#1a7f37",
        error="#cf222e",
        warning="#9a6700",
        diff_add="#1a7f37",
        diff_add_bg="#dafbe1",
        diff_remove="#cf222e",
        diff_remove_bg="#ffebe9",
        mode_bash="#cf222e",
        mode_command="#8250df",
        approve="#1a7f37",
        reject="#cf222e",
        auto_approve_on="#1a7f37",
        auto_approve_off="#9a6700",
    ),
)

# Tokyo Night theme
TOKYO_NIGHT_THEME = Theme(
    name="tokyo-night",
    dark=ThemeColors(
        text="#a9b1d6",
        text_muted="#565f89",
        background="#1a1b26",
        surface="#24283b",
        primary="#7aa2f7",
        secondary="#9ece6a",
        success="#9ece6a",
        error="#f7768e",
        warning="#e0af68",
        diff_add="#9ece6a",
        diff_add_bg="#1f2e1f",
        diff_remove="#f7768e",
        diff_remove_bg="#2e1f1f",
        mode_bash="#f7768e",
        mode_command="#bb9af7",
        approve="#9ece6a",
        reject="#f7768e",
        auto_approve_on="#9ece6a",
        auto_approve_off="#e0af68",
    ),
)

# Catppuccin Mocha theme
CATPPUCCIN_THEME = Theme(
    name="catppuccin",
    dark=ThemeColors(
        text="#cdd6f4",
        text_muted="#6c7086",
        background="#1e1e2e",
        surface="#313244",
        primary="#89b4fa",
        secondary="#a6adc8",
        success="#a6e3a1",
        error="#f38ba8",
        warning="#fab387",
        diff_add="#a6e3a1",
        diff_add_bg="#1e3a1e",
        diff_remove="#f38ba8",
        diff_remove_bg="#3a1e2a",
        mode_bash="#f38ba8",
        mode_command="#cba6f7",
        approve="#a6e3a1",
        reject="#f38ba8",
        auto_approve_on="#a6e3a1",
        auto_approve_off="#fab387",
    ),
)

# All built-in themes
BUILTIN_THEMES: dict[str, Theme] = {
    "default": DEFAULT_THEME,
    "tokyo-night": TOKYO_NIGHT_THEME,
    "catppuccin": CATPPUCCIN_THEME,
}
