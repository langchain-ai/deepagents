"""Lightweight theme preference coercion shared by config providers and the UI."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, cast

from deepagents_code import theme

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


def resolve_theme_name(value: object) -> str | None:
    """Resolve a user-supplied theme name to a canonical registry key.

    Accepts the registry key or the human-readable label, case-insensitive on
    both, with surrounding whitespace stripped - config values (especially
    `[ui.terminal_themes]`) and the `DEEPAGENTS_CODE_THEME` env var are commonly
    hand-edited. Also applies the legacy `textual-ansi` to `ansi-light`
    migration, which predates Textual 8.2.5 and can be dropped once no
    supported config can still carry the old name.

    Args:
        value: Raw value read from TOML or an environment variable.

    Returns:
        Canonical registry key, or `None` when the value is not a string or
        names no registered theme.
    """
    if not isinstance(value, str):
        return None
    name = value.strip()
    if name == "textual-ansi":
        name = "ansi-light"
    registry = theme.get_registry()
    if name in registry:
        return name
    folded = name.casefold()
    for registered, entry in registry.items():
        if registered.casefold() == folded or entry.label.casefold() == folded:
            return registered
    return None


def as_toml_table(value: object) -> dict[str, object] | None:
    """Return `value` as a TOML table when it has the expected runtime shape.

    `tomllib` parses TOML tables as string-keyed dicts, which `ty` cannot infer
    from a runtime `dict` check. Keep the cast at this boundary so it does not
    become a general-purpose escape hatch - now more important, not less, since
    the helper became importable from a shared module.
    """
    if not isinstance(value, dict):
        return None
    return cast("dict[str, object]", value)


def resolve_terminal_mapping(ui: Mapping[str, object]) -> str | None:
    """Resolve `[ui.terminal_themes][TERM_PROGRAM]` to a registered theme.

    Centralizes both the lookup and the misconfiguration warnings, which are
    logged exactly once per call. Three callers now share that contract: the
    managed and user TOML tiers and the UI.

    Args:
        ui: An `[ui]` table parsed from a managed or user TOML source.

    Returns:
        Canonical registry key, or `None` when no valid mapping applies.
    """
    terminal_themes = ui.get("terminal_themes")
    if terminal_themes is None:
        return None
    terminal_themes_table = as_toml_table(terminal_themes)
    if terminal_themes_table is None:
        logger.warning(
            "[ui.terminal_themes] should be a table mapping TERM_PROGRAM "
            "values to theme names; got %s",
            type(terminal_themes).__name__,
        )
        return None
    term_program = os.environ.get("TERM_PROGRAM", "").strip()
    if not term_program:
        if terminal_themes_table:
            logger.warning(
                "[ui.terminal_themes] is configured but TERM_PROGRAM is unset; "
                "no per-terminal theme will be applied",
            )
        return None
    mapped = terminal_themes_table.get(term_program)
    resolved = resolve_theme_name(mapped)
    if resolved is not None:
        return resolved
    if isinstance(mapped, str):
        logger.warning(
            "Unknown theme '%s' mapped to TERM_PROGRAM='%s' "
            "in [ui.terminal_themes]; ignoring",
            mapped,
            term_program,
        )
    elif mapped is not None:
        logger.warning(
            "Expected string theme name for TERM_PROGRAM='%s' in "
            "[ui.terminal_themes], got %s; ignoring",
            term_program,
            type(mapped).__name__,
        )
    return None
