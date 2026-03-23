"""LangChain brand colors and semantic constants for the CLI.

Single source of truth for color values used in Python code (Rich markup,
`Content.styled`, `Content.from_markup`). CSS-side styling should reference
Textual CSS variables: built-in variables (`$primary`, `$background`, etc.) are
set via `register_theme()` in `DeepAgentsApp.__init__`, while app-specific
variables (`$muted`, `$tool-border`, etc.) are backed by these constants via
`App.get_theme_variable_defaults()`.

Code that needs custom CSS variable values should call
`get_css_variable_defaults(dark=...)`. For the full semantic color palette,
look up the `ThemeColors` instance via `ThemeEntry.REGISTRY`.

Users can define custom themes in `~/.deepagents/config.toml` under
`[themes.<name>]` sections. Each section must include `label` (str) and `dark`
(bool); color fields are optional and fall back to the built-in dark/light
palette based on the `dark` flag. See `_load_user_themes()` for details.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brand palette — dark  (originally tokyonight-inspired, LangChain blue primary)
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

TOOL_HEADER = WARNING
"""Tool call headers, slash-command tokens, and approval-menu commands."""

# File listing colors
FILE_PYTHON = LC_BLUE
"""Python files in tool-call file listings."""

FILE_CONFIG = WARNING
"""Config / data files in tool-call file listings."""

FILE_DIR = SUCCESS
"""Directories in tool-call file listings."""

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
    """Accent for headings, borders, links, and active elements."""

    primary_dev: str
    """Accent used when running from an editable (dev) install."""

    secondary: str
    """Secondary accent for badges, labels, and decorative highlights."""

    success: str
    """Positive outcomes — tool success, approved actions."""

    warning: str
    """Caution and notice states — pending tool calls, notices."""

    error: str
    """Error and destructive-action indicator."""

    muted: str
    """De-emphasized text — timestamps, secondary labels."""

    mode_bash: str
    """Shell mode indicator — borders, prompts, and message prefixes."""

    mode_command: str
    """Command mode indicator — borders, prompts, and message prefixes."""

    foreground: str
    """Primary body text."""

    background: str
    """Base application background."""

    surface: str
    """Elevated card / panel background."""

    diff_add_fg: str
    """Added-line foreground in inline diffs."""

    diff_add_bg: str
    """Added-line background in inline diffs."""

    diff_remove_fg: str
    """Removed-line foreground in inline diffs."""

    diff_remove_bg: str
    """Removed-line background in inline diffs."""

    tool_border: str
    """Tool-call card border."""

    tool_border_hover: str
    """Tool-call card border on hover / focus."""

    error_bg: str
    """Subtle tinted background for error states."""

    spinner: str
    """Loading spinner color."""

    def __post_init__(self) -> None:
        """Validate that every field is a valid hex color.

        Raises:
            ValueError: If any field is not a 7-character hex color string.
        """
        for f in fields(self):
            val = getattr(self, f.name)
            if not _HEX_RE.match(val):
                msg = (
                    f"ThemeColors.{f.name} must be a 7-char hex color"
                    f" (#RRGGBB), got {val!r}"
                )
                raise ValueError(msg)

    @classmethod
    def merged(cls, base: ThemeColors, overrides: dict[str, str]) -> ThemeColors:
        """Create a new `ThemeColors` by overlaying overrides onto a base.

        Fields present in `overrides` replace the corresponding base value;
        missing fields inherit from `base`. This lets users specify only the
        colors they want to customize.

        Args:
            base: Fallback color set for any field not in `overrides`.
            overrides: Field-name to hex-color mapping. Unknown keys are
                silently ignored.

        Returns:
            New `ThemeColors` with merged values.
        """
        valid_names = {f.name for f in fields(cls)}
        kwargs = {f.name: getattr(base, f.name) for f in fields(cls)}
        kwargs.update({k: v for k, v in overrides.items() if k in valid_names})
        return cls(**kwargs)


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
    """Whether this theme must be registered with Textual via `register_theme()`.

    `True` for LangChain-branded themes and user-defined themes.
    `False` for Textual built-in themes that Textual already knows about.
    """

    REGISTRY: ClassVar[Mapping[str, ThemeEntry]]
    """All registered theme entries, keyed by Textual theme name.

    Read-only after module load (`MappingProxyType`).
    """


def _builtin_themes() -> dict[str, ThemeEntry]:
    """Return the built-in theme entries as a mutable dict.

    Returns:
        Dict of built-in theme names to `ThemeEntry` instances.
    """
    r: dict[str, ThemeEntry] = {}
    r["langchain"] = ThemeEntry(
        label="LangChain Dark",
        dark=True,
        colors=DARK_COLORS,
    )
    r["langchain-light"] = ThemeEntry(
        label="LangChain Light",
        dark=False,
        colors=LIGHT_COLORS,
    )
    # Textual built-in themes — not registered via register_theme() (Textual's
    # own $primary, $background, etc. apply), but carry color sets for custom
    # CSS vars.
    r["textual-dark"] = ThemeEntry(
        label="Textual Dark",
        dark=True,
        colors=DARK_COLORS,
        custom=False,
    )
    r["textual-light"] = ThemeEntry(
        label="Textual Light",
        dark=False,
        colors=LIGHT_COLORS,
        custom=False,
    )
    r["textual-ansi"] = ThemeEntry(
        label="Terminal (ANSI)",
        dark=True,
        colors=DARK_COLORS,
        custom=False,
    )
    return r


_BUILTIN_NAMES: frozenset[str] = frozenset(_builtin_themes())
"""Names reserved for built-in themes — user themes cannot shadow these.

Derived from `_builtin_themes()` to stay in sync automatically.
"""


def _load_user_themes(
    builtins: dict[str, ThemeEntry],
    *,
    config_path: Path | None = None,
) -> None:
    """Load user-defined themes from `config.toml` into `builtins` (mutated).

    Each `[themes.<name>]` section must have:

    - `label` (str) — human-readable name shown in the theme picker.
    - `dark` (bool) — whether this is a dark-mode variant.

    All `ThemeColors` fields are optional; omitted fields fall back to the
    built-in dark or light palette based on the `dark` flag.

    Invalid themes (bad hex, missing required keys, name collision with
    built-ins) are logged as warnings and skipped — they never crash startup.

    Example `config.toml` snippet:

    ```toml
    [themes.solarized-dark]
    label = "Solarized Dark"
    dark = true
    primary = "#268BD2"
    warning = "#B58900"
    ```

    Args:
        builtins: Mutable dict to append user themes into.
        config_path: Override for the config file path (testing).
    """
    if config_path is None:
        try:
            config_path = Path.home() / ".deepagents" / "config.toml"
        except RuntimeError:
            return

    try:
        if not config_path.exists():
            return

        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, PermissionError, OSError) as exc:
        logger.warning(
            "Could not read %s for user themes: %s",
            config_path,
            exc,
        )
        return

    themes_section: Any = data.get("themes")
    if not isinstance(themes_section, dict) or not themes_section:
        return

    for name, section in themes_section.items():
        if not isinstance(section, dict):
            logger.warning("Ignoring non-table [themes.%s]", name)
            continue

        if name in _BUILTIN_NAMES:
            logger.warning(
                "User theme '%s' shadows a built-in theme and will be ignored",
                name,
            )
            continue

        label = section.get("label")
        dark = section.get("dark")
        if not isinstance(label, str) or not label.strip():
            logger.warning(
                "User theme '%s' missing required 'label' (str); skipping",
                name,
            )
            continue
        if not isinstance(dark, bool):
            logger.warning(
                "User theme '%s' missing required 'dark' (bool); skipping",
                name,
            )
            continue

        base = DARK_COLORS if dark else LIGHT_COLORS
        valid_color_names = {f.name for f in fields(ThemeColors)}
        reserved = {"label", "dark"}
        color_overrides: dict[str, str] = {}
        for k, v in section.items():
            if k in reserved:
                continue
            if not isinstance(v, str):
                logger.warning(
                    "User theme '%s' field '%s' must be a string, got %s; ignoring",
                    name,
                    k,
                    type(v).__name__,
                )
                continue
            if k in valid_color_names:
                color_overrides[k] = v
            else:
                logger.warning(
                    "User theme '%s' has unknown color field '%s'; ignoring",
                    name,
                    k,
                )

        try:
            colors = ThemeColors.merged(base, color_overrides)
        except ValueError as exc:
            logger.warning(
                "User theme '%s' has invalid colors: %s; skipping",
                name,
                exc,
            )
            continue

        builtins[name] = ThemeEntry(
            label=label,
            dark=dark,
            colors=colors,
            custom=True,
        )


def _build_registry(
    *, config_path: Path | None = None
) -> MappingProxyType[str, ThemeEntry]:
    """Build and freeze the theme registry (built-in + user themes).

    Args:
        config_path: Override for the config file path (testing).

    Returns:
        Read-only mapping of theme names to `ThemeEntry` instances.
    """
    r = _builtin_themes()
    _load_user_themes(r, config_path=config_path)
    return MappingProxyType(r)


ThemeEntry.REGISTRY = _build_registry()
"""Read-only mapping of Textual theme names to `ThemeEntry` instances.

Built via `_build_registry()` so the mutable staging dict is scoped to a
function call and cannot be mutated after freeze. The `ClassVar` declaration on
`ThemeEntry` provides the type; this assignment supplies the value.
"""

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
        Dict of CSS variable names to hex color values.
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
