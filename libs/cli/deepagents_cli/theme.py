"""LangChain brand colors and semantic constants for the CLI.

Single source of truth for color values used in Python code (Rich markup,
`Content.styled`, `Content.from_markup`). CSS-side styling should reference
Textual CSS variables (`$primary`, `$muted`, `$tool-border`, etc.) that are
backed by these constants via `App.get_theme_variable_defaults()`.
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

LC_BORDER_LT = "#3A3E57"
"""Borders on lighter / hovered backgrounds."""

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

LC_MUTED = "#545C7E"
"""Muted / secondary text."""

LC_GREEN_BG = "#1C2A38"
"""Subtle green-tinted background for diff additions."""

LC_PINK_BG = "#2A1F32"
"""Subtle pink-tinted background for diff removals / errors."""

# ---------------------------------------------------------------------------
# Semantic constants  (Rich markup + CSS variable defaults)
# ---------------------------------------------------------------------------
PRIMARY = LC_BLUE
"""Default accent for headings, borders, links, and active elements."""

PRIMARY_DEV = LC_ORANGE
"""Accent used when running from an editable (dev) install."""

SUCCESS = LC_GREEN
"""Positive outcomes — tool success, approved actions."""

WARNING = LC_AMBER
"""Caution and notice states — auto-approve off, pending tool calls, notices."""

ERROR = LC_PINK
"""Errors, destructive actions, and failures."""

MUTED = LC_MUTED
"""De-emphasized text — timestamps, secondary labels."""

MODE_BASH = LC_PINK
"""Shell mode indicator — borders, prompts, and message prefixes."""

MODE_COMMAND = LC_PURPLE
"""Command mode indicator — borders, prompts, and message prefixes."""

# Diff colors
DIFF_ADD_FG = LC_GREEN
"""Added-line foreground in inline diffs."""

DIFF_ADD_BG = LC_GREEN_BG
"""Added-line background in inline diffs."""

DIFF_REMOVE_FG = LC_PINK
"""Removed-line foreground in inline diffs."""

DIFF_REMOVE_BG = LC_PINK_BG
"""Removed-line background in inline diffs."""

DIFF_CONTEXT = MUTED
"""Unchanged context lines in inline diffs."""

# Tool call widget
TOOL_BORDER = LC_BORDER_DK
"""Tool call card border."""

TOOL_BORDER_HVR = LC_BORDER_LT
"""Tool call card border on hover."""

TOOL_HEADER = WARNING
"""Tool call headers, slash-command tokens, and approval-menu commands."""

TOOL_PENDING = WARNING
"""Tool call status while awaiting result or after rejection."""

TOOL_SUCCESS = SUCCESS
"""Tool call status on successful completion."""

TOOL_ERROR = ERROR
"""Tool call status on failure."""

# File listing colors
FILE_PYTHON = LC_BLUE
"""Python files in tool-call file listings."""

FILE_CONFIG = WARNING
"""Config / data files in tool-call file listings."""

FILE_DIR = SUCCESS
"""Directories in tool-call file listings."""

# Misc
ERROR_BG = DIFF_REMOVE_BG
"""Background for error message containers."""

SPINNER = LC_BLUE
"""Loading spinner color."""


CSS_VARIABLE_DEFAULTS: dict[str, str] = {
    "muted": MUTED,
    "tool-border": TOOL_BORDER,
    "tool-border-hover": TOOL_BORDER_HVR,
    "mode-bash": MODE_BASH,
    "mode-command": MODE_COMMAND,
    "diff-add-fg": DIFF_ADD_FG,
    "diff-add-bg": DIFF_ADD_BG,
    "diff-remove-fg": DIFF_REMOVE_FG,
    "diff-remove-bg": DIFF_REMOVE_BG,
    "error-bg": ERROR_BG,
}
"""Custom CSS variable defaults, referenced as `$muted`, `$tool-border`, etc.

Textual's built-in theme variables (`$primary`, `$background`, ...) don't cover
app-specific semantic tokens. This dict maps each custom variable name to its
hex value and feeds two consumers:

* `DeepAgentsApp.get_theme_variable_defaults()` — registers the variables so
    `.tcss` stylesheets can resolve `$custom-var` references at parse time.
* Test conftest — patches the same mapping onto plain `App[None]` subclasses
    so unit tests resolve custom variables without instantiating the full
    `DeepAgentsApp`.

Keeping one shared dict avoids duplicate hardcoded mappings that would drift
independently.
"""
