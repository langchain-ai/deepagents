"""LangChain brand colors and semantic constants for the CLI.

Single source of truth for color values used in Python code (Rich markup,
`Content.styled`, `Content.from_markup`).  CSS-side styling should reference
Textual CSS variables: built-in variables
(`$primary`, `$background`, `$text-muted`, `$error-muted`, etc.) are set via
`register_theme()` in `DeepAgentsApp.__init__`, while the few app-specific
variables (`$mode-bash`, `$mode-command`) are backed by these constants via
`App.get_theme_variable_defaults()`.

Code that needs custom CSS variable values should call
`get_css_variable_defaults(dark=...)`. For the full semantic color palette, look
up the `ThemeColors` instance via `ThemeEntry.REGISTRY`.

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

    from textual.app import App

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

LC_AMBER = "#EB8B46"
"""Warning / caution indicator."""

LC_PINK = "#F7768E"
"""Error / destructive actions."""

LC_MUTED = "#545C7E"
"""Muted / secondary text."""

LC_GREEN_BG = "#1C2A38"
"""Subtle green-tinted background for diff additions."""

LC_PINK_BG = "#2A1F32"
"""Subtle pink-tinted background for diff removals / errors."""

LC_PANEL = "#25283B"
"""Panel — differentiated section background (above surface)."""

LC_SKILL = "#A78BFA"
"""Skill invocation accent — border and header text."""

LC_SKILL_HOVER = "#C4B5FD"
"""Skill invocation hover — lighter variant for interactive feedback."""

LC_TOOL = LC_AMBER
"""Tool call accent — border and header text."""

LC_TOOL_HOVER = "#FFCB91"
"""Tool call hover — lighter variant for interactive feedback."""


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

LC_LIGHT_MUTED = "#6B7280"
"""Muted / secondary text on light backgrounds."""

LC_LIGHT_GREEN_BG = "#DCFCE7"
"""Subtle green-tinted background for diff additions."""

LC_LIGHT_PINK_BG = "#FEE2E2"
"""Subtle pink-tinted background for diff removals / errors."""

LC_LIGHT_PANEL = "#E0E1E6"
"""Panel for light theme — differentiated section background."""

LC_LIGHT_SKILL = "#7C3AED"
"""Skill invocation accent (darkened for light bg contrast)."""

LC_LIGHT_SKILL_HOVER = "#6D28D9"
"""Skill invocation hover (darkened for light bg contrast)."""

LC_LIGHT_TOOL = LC_LIGHT_AMBER
"""Tool call accent (darkened for light bg contrast)."""

LC_LIGHT_TOOL_HOVER = "#78350F"
"""Tool call hover (darkened for light bg contrast)."""


# ---------------------------------------------------------------------------
# Semantic constants  (ANSI color names for Rich console output)
#
# These are ANSI color names resolved by the user's terminal palette, so they
# adapt to both dark and light terminal backgrounds automatically. They are
# used in Rich's `Console.print()` (non-interactive output, help screens,
# `non_interactive.py`, `main.py`).
#
# Textual widget code should NOT use these. Instead, call
# `get_theme_colors(self.app)` to obtain the active theme's `ThemeColors`
# (hex values), or reference CSS variables (`$primary`, `$muted`, etc.).
# ---------------------------------------------------------------------------
PRIMARY = "blue"
"""Default accent for headings, borders, links, and active elements."""

PRIMARY_DEV = "bright_red"
"""Accent used when running from an editable (dev) install."""

SUCCESS = "green"
"""Positive outcomes — tool success, approved actions."""

WARNING = "yellow"
"""Caution and notice states — auto-approve off, pending tool calls, notices."""

MUTED = "bright_black"
"""De-emphasized text — timestamps, secondary labels."""

MODE_BASH = "red"
"""Shell mode indicator — borders, prompts, and message prefixes."""

MODE_COMMAND = "magenta"
"""Command mode indicator — borders, prompts, and message prefixes."""

# Diff colors
DIFF_ADD_FG = "green"
"""Added-line foreground in inline diffs."""

DIFF_ADD_BG = "green"
"""Added-line background in inline diffs."""

DIFF_REMOVE_FG = "red"
"""Removed-line foreground in inline diffs."""

DIFF_REMOVE_BG = "red"
"""Removed-line background in inline diffs."""

DIFF_CONTEXT = "bright_black"
"""Unchanged context lines in inline diffs."""

# Tool call widget
TOOL_BORDER = "bright_black"
"""Tool call card border."""

TOOL_HEADER = "yellow"
"""Tool call headers, slash-command tokens, and approval-menu commands."""

# File listing colors
FILE_PYTHON = "blue"
"""Python files in tool-call file listings."""

FILE_CONFIG = "yellow"
"""Config / data files in tool-call file listings."""

FILE_DIR = "green"
"""Directories in tool-call file listings."""

SPINNER = "blue"
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

    secondary: str
    """Secondary accent for badges, labels, and decorative highlights."""

    accent: str
    """Attention-drawing contrast accent, distinct from primary/secondary."""

    panel: str
    """Differentiated section background (above surface)."""

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

    skill: str
    """Skill invocation accent — border and header text."""

    skill_hover: str
    """Skill invocation hover — lighter variant for interactive feedback."""

    tool: str
    """Tool call accent — border and header text."""

    tool_hover: str
    """Tool call hover — lighter variant for interactive feedback."""

    foreground: str
    """Primary body text."""

    background: str
    """Base application background."""

    surface: str
    """Elevated card / panel background."""

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


# ---------------------------------------------------------------------------
# Built-in theme color sets
# ---------------------------------------------------------------------------

DARK_COLORS = ThemeColors(
    primary=LC_BLUE,
    secondary=LC_PURPLE,
    accent=LC_GREEN,
    panel=LC_PANEL,
    success=LC_GREEN,
    warning=LC_AMBER,
    error=LC_PINK,
    muted=LC_MUTED,
    mode_bash=LC_PINK,
    mode_command=LC_PURPLE,
    skill=LC_SKILL,
    skill_hover=LC_SKILL_HOVER,
    tool=LC_TOOL,
    tool_hover=LC_TOOL_HOVER,
    foreground=LC_BODY,
    background=LC_DARK,
    surface=LC_CARD,
)
"""Color set for the dark LangChain theme."""

LIGHT_COLORS = ThemeColors(
    primary=LC_LIGHT_BLUE,
    secondary=LC_LIGHT_PURPLE,
    accent=LC_LIGHT_GREEN,
    panel=LC_LIGHT_PANEL,
    success=LC_LIGHT_GREEN,
    warning=LC_LIGHT_AMBER,
    error=LC_LIGHT_PINK,
    muted=LC_LIGHT_MUTED,
    mode_bash=LC_LIGHT_PINK,
    mode_command=LC_LIGHT_PURPLE,
    skill=LC_LIGHT_SKILL,
    skill_hover=LC_LIGHT_SKILL_HOVER,
    tool=LC_LIGHT_TOOL,
    tool_hover=LC_LIGHT_TOOL_HOVER,
    foreground=LC_LIGHT_BODY,
    background=LC_LIGHT_BG,
    surface=LC_LIGHT_SURFACE,
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

    def __post_init__(self) -> None:
        """Validate that the label is a non-empty string.

        Raises:
            ValueError: If `label` is empty or whitespace-only.
        """
        if not self.label.strip():
            msg = "ThemeEntry.label must be a non-empty string"
            raise ValueError(msg)


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
    # own $primary, $background, etc. apply). The `colors` field provides
    # fallback values for app-specific CSS vars ($mode-bash, $mode-command) and
    # Python-side styling.  For standard properties (primary, secondary, etc.),
    # get_theme_colors() dynamically resolves from the actual Textual theme at
    # runtime so the Python and CSS color systems stay in sync.

    def _bi(label: str, *, is_dark: bool) -> ThemeEntry:
        return ThemeEntry(
            label=label,
            dark=is_dark,
            colors=DARK_COLORS if is_dark else LIGHT_COLORS,
            custom=False,
        )

    r["textual-dark"] = _bi("Textual Dark", is_dark=True)
    r["textual-light"] = _bi("Textual Light", is_dark=False)
    r["textual-ansi"] = _bi("Terminal (ANSI)", is_dark=False)
    # Popular community themes (all ship with Textual >= 8.0)
    r["atom-one-dark"] = _bi("Atom One Dark", is_dark=True)
    r["atom-one-light"] = _bi("Atom One Light", is_dark=False)
    r["catppuccin-frappe"] = _bi("Catppuccin Frappé", is_dark=True)
    r["catppuccin-latte"] = _bi("Catppuccin Latte", is_dark=False)
    r["catppuccin-macchiato"] = _bi("Catppuccin Macchiato", is_dark=True)
    r["catppuccin-mocha"] = _bi("Catppuccin Mocha", is_dark=True)
    r["dracula"] = _bi("Dracula", is_dark=True)
    r["flexoki"] = _bi("Flexoki", is_dark=True)
    r["gruvbox"] = _bi("Gruvbox", is_dark=True)
    r["monokai"] = _bi("Monokai", is_dark=True)
    r["nord"] = _bi("Nord", is_dark=True)
    r["rose-pine"] = _bi("Rosé Pine", is_dark=True)
    r["rose-pine-dawn"] = _bi("Rosé Pine Dawn", is_dark=False)
    r["rose-pine-moon"] = _bi("Rosé Pine Moon", is_dark=True)
    r["solarized-dark"] = _bi("Solarized Dark", is_dark=True)
    r["solarized-light"] = _bi("Solarized Light", is_dark=False)
    r["tokyo-night"] = _bi("Tokyo Night", is_dark=True)
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
    [themes.my-solarized]
    label = "My Solarized"
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
            logger.debug("Cannot determine home directory; skipping user theme loading")
            return

    import tomllib

    try:
        if not config_path.exists():
            return

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


def reload_registry() -> MappingProxyType[str, ThemeEntry]:
    """Rebuild the theme registry from disk and update `ThemeEntry.REGISTRY`.

    Re-reads `~/.deepagents/config.toml` for user-defined themes so that
    `/reload` can pick up config changes without restarting the app.

    Returns:
        The new frozen registry.
    """
    ThemeEntry.REGISTRY = _build_registry()
    return ThemeEntry.REGISTRY


def get_css_variable_defaults(
    *, dark: bool = True, colors: ThemeColors | None = None
) -> dict[str, str]:
    """Return custom CSS variable defaults for the given mode.

    Most styling is handled by Textual's built-in CSS variables (`$primary`,
    `$text-muted`, `$error-muted`, etc.).  This function only returns
    app-specific semantic variables that have no Textual equivalent.

    Args:
        dark: Selects `DARK_COLORS` or `LIGHT_COLORS` when `colors` is None.
        colors: Explicit color set to use. Takes precedence over `dark`.

    Returns:
        Dict of CSS variable names to hex color values.
    """
    c = colors if colors is not None else (DARK_COLORS if dark else LIGHT_COLORS)
    return {
        "mode-bash": c.mode_bash,
        "mode-command": c.mode_command,
        "skill": c.skill,
        "skill-hover": c.skill_hover,
        "tool": c.tool,
        "tool-hover": c.tool_hover,
    }


def _resolve_app(widget_or_app: object) -> object:
    """Resolve a widget or App to the App instance.

    Args:
        widget_or_app: Textual `App` or a mounted widget.

    Returns:
        The resolved App instance.
    """
    return (
        widget_or_app.app  # type: ignore[attr-defined]
        if hasattr(type(widget_or_app), "app")
        else widget_or_app
    )


def _colors_from_textual_theme(app: object) -> ThemeColors:
    """Construct `ThemeColors` from the app's active Textual theme.

    Reads standard properties (primary, secondary, etc.) from the resolved
    theme so Python-side styling matches CSS.  `muted` falls back to the
    dark/light base unconditionally (no Textual equivalent).
    `mode_bash` is derived from the theme's `error` color, and `mode_command`
    from `secondary`, falling back to the base palette when non-hex.

    Non-hex values (e.g. `ansi_blue` in the ANSI theme) are detected and fall
    back to the base palette automatically.

    Args:
        app: The Textual App instance.

    Returns:
        `ThemeColors` derived from the active theme.
    """
    ct = app.current_theme  # type: ignore[attr-defined]
    dark: bool = ct.dark
    base = DARK_COLORS if dark else LIGHT_COLORS

    def _hex_or(val: str | None, fallback: str) -> str:
        """Use the Textual theme value when it is a hex color, else fall back.

        Textual built-in themes (e.g. ANSI) may return non-hex color names
        like `ansi_blue` that our `ThemeColors` dataclass rejects. This helper
        detects those and substitutes the base-palette fallback so every field
        is always a valid `#RRGGBB` string.

        Args:
            val: Color string from the active Textual theme (may be None or a
                non-hex name like `ansi_blue`).
            fallback: Guaranteed-hex value from our base palette.

        Returns:
            `val` if it matches `#RRGGBB`, otherwise `fallback`.
        """
        if val is not None and _HEX_RE.match(val):
            return val
        return fallback

    return ThemeColors(
        primary=_hex_or(ct.primary, base.primary),
        secondary=_hex_or(ct.secondary, base.secondary),
        accent=_hex_or(ct.accent, base.accent),
        panel=_hex_or(ct.panel, base.panel),
        success=_hex_or(ct.success, base.success),
        warning=_hex_or(ct.warning, base.warning),
        error=_hex_or(ct.error, base.error),
        muted=base.muted,
        mode_bash=_hex_or(ct.error, base.mode_bash),
        mode_command=_hex_or(ct.secondary, base.mode_command),
        # No Textual equivalent — always use base palette.
        skill=base.skill,
        skill_hover=base.skill_hover,
        # Derived from warning (same semantic color).
        tool=_hex_or(ct.warning, base.tool),
        tool_hover=base.tool_hover,
        foreground=_hex_or(ct.foreground, base.foreground),
        background=_hex_or(ct.background, base.background),
        surface=_hex_or(ct.surface, base.surface),
    )


def get_theme_colors(widget_or_app: App | object | None = None) -> ThemeColors:
    """Return the `ThemeColors` for the active Textual theme.

    For custom themes (LangChain-branded and user-defined), the pre-built
    `ThemeColors` from the registry is returned directly.  For Textual built-in
    themes, colors are resolved dynamically from the actual theme properties so
    Python-side styling stays in sync with CSS variables.

    Textual widget code should call this instead of reading the module-level
    ANSI constants, which are intended for Rich console output only.

    Args:
        widget_or_app: Textual `App`, a mounted widget, or `None`.

    Returns:
        `ThemeColors` for the active theme.
    """
    if widget_or_app is None:
        # Fall back to the active Textual app context var when no explicit
        # widget/app is passed (e.g. from @staticmethod helpers).
        try:
            from textual._context import active_app  # noqa: PLC2701

            widget_or_app = active_app.get()
        except (ImportError, LookupError):
            return DARK_COLORS
    app = _resolve_app(widget_or_app)
    entry = ThemeEntry.REGISTRY.get(app.theme)  # type: ignore[attr-defined]
    # Custom themes (LC-branded / user-defined) use pre-built colors.
    if entry is not None and entry.custom:
        return entry.colors
    # Built-in or unrecognized themes — derive from the resolved Textual
    # theme so Python styling matches CSS.
    try:
        return _colors_from_textual_theme(app)
    except Exception:
        logger.warning("Could not resolve theme colors dynamically", exc_info=True)
        if entry is not None:
            return entry.colors
        return DARK_COLORS
