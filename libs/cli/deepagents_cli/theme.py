"""LangChain brand colors and semantic constants for the CLI.

This module is the single source of truth for all color values used across the
TUI and Rich console output. It contains only plain string constants so it is
safe to import from the argument-parsing path (no heavy deps).
"""

# ---------------------------------------------------------------------------
# Brand palette  (tokyonight-inspired, LangChain blue primary)
# ---------------------------------------------------------------------------
LC_DARK = "#11121D"
"""Background — visible blue tint, distinguishable from pure black."""

LC_CARD = "#1A1B2E"
"""Surface / card — clearly elevated above background."""

LC_BORDER_DK = "#25283B"
"""Borders on dark backgrounds."""

LC_BODY = "#C0CAF5"
"""Body text — high contrast on dark backgrounds."""

LC_BLUE = "#7AA2F7"
"""Primary accent blue."""

LC_PURPLE = "#BB9AF7"
"""Secondary accent / badges / labels."""

LC_GREEN = "#9ECE6A"
"""Success / positive indicator."""

LC_AMBER = "#E0AF68"
"""Warning / caution indicator."""

LC_PINK = "#F7768E"
"""Error / destructive actions."""

LC_ORANGE = "#FF9E64"
"""Dev install indicator / warm accent."""

LC_CYAN = "#7DCFFF"
"""Info / decorative accent."""

# ---------------------------------------------------------------------------
# Semantic constants  (Rich markup — cannot use CSS variables)
# ---------------------------------------------------------------------------
PRIMARY = LC_BLUE
PRIMARY_DEV = LC_ORANGE
SUCCESS = LC_GREEN
WARNING = LC_AMBER
ERROR = LC_PINK
MUTED = "#545C7E"

MODE_BASH = LC_PINK
MODE_COMMAND = LC_PURPLE

# Diff colors
DIFF_ADD_FG = "#9ECE6A"
DIFF_ADD_BG = "#1C2A38"
DIFF_REMOVE_FG = "#F7768E"
DIFF_REMOVE_BG = "#2A1F32"
DIFF_CONTEXT = MUTED

# Tool call widget
TOOL_BORDER = LC_BORDER_DK
TOOL_BORDER_HVR = "#3A3E57"
TOOL_HEADER = WARNING
TOOL_PENDING = WARNING
TOOL_SUCCESS = SUCCESS
TOOL_ERROR = ERROR

# File listing colors
FILE_PYTHON = LC_BLUE
FILE_CONFIG = WARNING
FILE_DIR = SUCCESS

# Misc
ERROR_BG = "#2A1F32"
SPINNER = LC_BLUE
