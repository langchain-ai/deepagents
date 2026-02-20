"""LangChain brand colors and semantic constants for the CLI.

This module is the single source of truth for all color values used across the
TUI and Rich console output. It contains only plain string constants so it is
safe to import from the argument-parsing path (no heavy deps).
"""

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
LC_DARK = "#030710"
"""Deep dark-blue background."""

LC_CARD = "#0B1120"
"""Surface / card background."""

LC_BORDER = "#B8DFFF"
"""Light borders on dark backgrounds."""

LC_BORDER_DK = "#1A2740"
"""Borders on dark backgrounds."""

LC_MUTED = "#6B8299"
"""Muted / secondary text."""

LC_BODY = "#C8DDF0"
"""Body text."""

LC_WHITE = "#FFFFFF"
"""Headings."""

LC_BLUE = "#7FC8FF"
"""Primary accent blue."""

LC_BLUE_HVR = "#99D4FF"
"""Hover variant of primary blue."""

LC_LIME = "#E3FF8F"
"""Success / positive indicator."""

LC_ROSE = "#B27D75"
"""Warm accent (dev install indicator)."""

LC_PINK = "#C78EAD"
"""Decorative pink."""

LC_LAVENDER = "#D5C3F7"
"""Badges / labels."""

# ---------------------------------------------------------------------------
# Semantic constants  (Rich markup — cannot use CSS variables)
# ---------------------------------------------------------------------------
PRIMARY = LC_BLUE
PRIMARY_DEV = LC_ROSE
PRIMARY_HOVER = LC_BLUE_HVR
SUCCESS = LC_LIME
WARNING = "#f59e0b"
ERROR = "#ef4444"
MUTED = LC_MUTED

MODE_BASH = LC_PINK
MODE_COMMAND = LC_LAVENDER

# Diff colors
DIFF_ADD_FG = "#8ce99a"
DIFF_ADD_BG = "#0f2818"
DIFF_REMOVE_FG = "#ff8787"
DIFF_REMOVE_BG = "#2a1018"
DIFF_CONTEXT = LC_MUTED

# Tool call widget
TOOL_BORDER = LC_BORDER_DK
TOOL_BORDER_HVR = "#2a3a55"
TOOL_HEADER = WARNING
TOOL_PENDING = WARNING
TOOL_SUCCESS = SUCCESS
TOOL_ERROR = ERROR

# File listing colors
FILE_PYTHON = LC_BLUE
FILE_CONFIG = WARNING
FILE_DIR = SUCCESS

# Misc
ERROR_BG = "#2a1018"
SPINNER = LC_BLUE
