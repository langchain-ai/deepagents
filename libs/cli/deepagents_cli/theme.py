"""LangChain brand colors and semantic constants for the CLI.

Single source of truth for color values used in Python code (Rich markup,
`Content.styled`, `Content.from_markup`). CSS-side styling should reference
Textual CSS variables (`$primary`, `$muted`, `$tool-border`, etc.) that are
backed by these constants via `App.get_theme_variable_defaults()`.

Code that needs custom CSS variable values should call
`get_css_variable_defaults(dark=...)`. For the full semantic color palette,
look up the `ThemeColors` instance via `ThemeEntry.REGISTRY`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping

# ---------------------------------------------------------------------------
# Brand palette — dark  (tokyonight-inspired, LangChain blue primary)
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
# Brand palette — light
# ---------------------------------------------------------------------------
LC_LIGHT_BG = "#F5F5F7"
"""Background — warm neutral white."""

LC_LIGHT_SURFACE = "#EAEAEE"
"""Surface / card — slightly darker than background."""

LC_LIGHT_BORDER = "#C8CAD0"
"""Borders on light backgrounds."""

LC_LIGHT_BORDER_HVR = "#A0A4B0"
"""Borders on hovered / focused surfaces."""

LC_LIGHT_BODY = "#24283B"
"""Body text — high contrast on light backgrounds."""

LC_LIGHT_BLUE = "#2E5EAA"
"""Primary accent blue (darkened for light bg contrast)."""

LC_LIGHT_PURPLE = "#7C3AED"
"""Secondary accent (darkened for light bg contrast)."""

LC_LIGHT_GREEN = "#3A7D0A"
"""Success / positive (darkened for light bg contrast)."""

LC_LIGHT_AMBER = "#B45309"
"""Warning / caution (darkened for light bg contrast)."""

LC_LIGHT_PINK = "#BE185D"
"""Error / destructive (darkened for light bg contrast)."""

LC_LIGHT_ORANGE = "#C2410C"
"""Dev install indicator (darkened for light bg contrast)."""

LC_LIGHT_MUTED = "#6B7280"
"""Muted / secondary text on light backgrounds."""

LC_LIGHT_GREEN_BG = "#DCFCE7"
"""Subtle green-tinted background for diff additions."""

LC_LIGHT_PINK_BG = "#FEE2E2"
"""Subtle pink-tinted background for diff removals / errors."""


# ---------------------------------------------------------------------------
# Semantic constants  (Rich markup + CSS variable defaults)
#
# These are the *dark-mode* values, used directly in Python code that renders
# via Rich's `Console.print()` (non-interactive output, `non_interactive.py`,
# `main.py`). Textual widget code should prefer CSS variables. Python code
# needing theme-aware values should look up the active `ThemeEntry` from
# the registry.
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


# ---------------------------------------------------------------------------
# Theme variant dataclass
# ---------------------------------------------------------------------------


_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
"""Matches a 7-character hex color string like `#7AA2F7`.

Textual's `Color.parse` could also validate, but importing it here would pull
Textual into `theme.py` which is otherwise pure Python with zero framework deps.
"""


@dataclass(frozen=True, slots=True)
class ThemeColors:
    """Complete set of semantic colors for one theme variant.

    Every field must be a 7-character hex color string (e.g., `'#7AA2F7'`).
    """

    primary: str
    primary_dev: str
    secondary: str
    success: str
    warning: str
    error: str
    muted: str
    mode_bash: str
    mode_command: str
    foreground: str
    background: str
    surface: str
    diff_add_fg: str
    diff_add_bg: str
    diff_remove_fg: str
    diff_remove_bg: str
    tool_border: str
    tool_border_hover: str
    error_bg: str
    spinner: str

    def __post_init__(self) -> None:
        """Validate that every field is a valid hex color.

        Raises:
            ValueError: If any field is not a 7-character hex color string.
        """
        for f in fields(self):
            val = getattr(self, f.name)
            if not _HEX_RE.match(val):
                msg = f"ThemeColors.{f.name} must be a hex color, got {val!r}"
                raise ValueError(msg)


DARK_COLORS = ThemeColors(
    primary=LC_BLUE,
    primary_dev=LC_ORANGE,
    secondary=LC_PURPLE,
    success=LC_GREEN,
    warning=LC_AMBER,
    error=LC_PINK,
    muted=LC_MUTED,
    mode_bash=LC_PINK,
    mode_command=LC_PURPLE,
    foreground=LC_BODY,
    background=LC_DARK,
    surface=LC_CARD,
    diff_add_fg=LC_GREEN,
    diff_add_bg=LC_GREEN_BG,
    diff_remove_fg=LC_PINK,
    diff_remove_bg=LC_PINK_BG,
    tool_border=LC_BORDER_DK,
    tool_border_hover=LC_BORDER_LT,
    error_bg=LC_PINK_BG,
    spinner=LC_BLUE,
)
"""Color set for the dark LangChain theme."""

LIGHT_COLORS = ThemeColors(
    primary=LC_LIGHT_BLUE,
    primary_dev=LC_LIGHT_ORANGE,
    secondary=LC_LIGHT_PURPLE,
    success=LC_LIGHT_GREEN,
    warning=LC_LIGHT_AMBER,
    error=LC_LIGHT_PINK,
    muted=LC_LIGHT_MUTED,
    mode_bash=LC_LIGHT_PINK,
    mode_command=LC_LIGHT_PURPLE,
    foreground=LC_LIGHT_BODY,
    background=LC_LIGHT_BG,
    surface=LC_LIGHT_SURFACE,
    diff_add_fg=LC_LIGHT_GREEN,
    diff_add_bg=LC_LIGHT_GREEN_BG,
    diff_remove_fg=LC_LIGHT_PINK,
    diff_remove_bg=LC_LIGHT_PINK_BG,
    tool_border=LC_LIGHT_BORDER,
    tool_border_hover=LC_LIGHT_BORDER_HVR,
    error_bg=LC_LIGHT_PINK_BG,
    spinner=LC_LIGHT_BLUE,
)
"""Color set for the light LangChain theme."""


# ---------------------------------------------------------------------------
# Available themes  (name → display label, dark flag, colors)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThemeEntry:
    """Metadata for a registered theme."""

    label: str
    """Human-readable label shown in the theme picker."""

    dark: bool
    """Whether this is a dark theme variant."""

    colors: ThemeColors
    """Resolved color set."""

    custom: bool = True
    """Whether this theme is registered as a custom `Theme` with Textual.

    `True` for LangChain-branded themes (registered via `register_theme()`).
    `False` for Textual built-in themes that Textual already knows about.
    """

    REGISTRY: ClassVar[Mapping[str, ThemeEntry]]
    """All registered theme entries, keyed by Textual theme name.

    Read-only after module load (`MappingProxyType`).
    """


# Build registry, then freeze to prevent accidental mutation.
_registry: dict[str, ThemeEntry] = {}
"""Mutable staging dict — populated below, then frozen into `ThemeEntry.REGISTRY`."""

_registry["langchain"] = ThemeEntry(
    label="LangChain Dark",
    dark=True,
    colors=DARK_COLORS,
)
_registry["langchain-light"] = ThemeEntry(
    label="LangChain Light",
    dark=False,
    colors=LIGHT_COLORS,
)
# Textual built-in themes — not registered via register_theme() (Textual's own
# $primary, $background, etc. apply), but carry color sets for custom CSS vars.
_registry["textual-dark"] = ThemeEntry(
    label="Textual Dark",
    dark=True,
    colors=DARK_COLORS,
    custom=False,
)
_registry["textual-light"] = ThemeEntry(
    label="Textual Light",
    dark=False,
    colors=LIGHT_COLORS,
    custom=False,
)
_registry["textual-ansi"] = ThemeEntry(
    label="Terminal (ANSI)",
    dark=True,
    colors=DARK_COLORS,
    custom=False,
)

ThemeEntry.REGISTRY = MappingProxyType(_registry)

DEFAULT_THEME = "langchain"
"""Theme name used when no preference is saved."""


def get_css_variable_defaults(
    *, dark: bool = True, colors: ThemeColors | None = None
) -> dict[str, str]:
    """Return custom CSS variable defaults for the given mode.

    Args:
        dark: Selects `DARK_COLORS` or `LIGHT_COLORS` when `colors` is None.
        colors: Explicit color set to use. Takes precedence over `dark`.

    Returns:
        Mapping of CSS variable names to hex color values.
    """
    c = colors if colors is not None else (DARK_COLORS if dark else LIGHT_COLORS)
    return {
        "muted": c.muted,
        "tool-border": c.tool_border,
        "tool-border-hover": c.tool_border_hover,
        "mode-bash": c.mode_bash,
        "mode-command": c.mode_command,
        "diff-add-fg": c.diff_add_fg,
        "diff-add-bg": c.diff_add_bg,
        "diff-remove-fg": c.diff_remove_fg,
        "diff-remove-bg": c.diff_remove_bg,
        "error-bg": c.error_bg,
    }


# Keep backwards-compatible alias used by test conftest
CSS_VARIABLE_DEFAULTS: dict[str, str] = get_css_variable_defaults(dark=True)
"""Dark-mode CSS variable defaults (backwards-compatible alias).

Previously used by `DeepAgentsApp.get_theme_variable_defaults()`; now retained
for the test conftest, which patches this mapping onto plain `App[None]`
subclasses so unit tests resolve custom variables without instantiating the
full `DeepAgentsApp`.

Production code calls `get_css_variable_defaults(dark=...)` instead.
"""
