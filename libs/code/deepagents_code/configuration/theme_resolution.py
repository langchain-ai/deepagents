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

    Args:
        value: Raw value read from TOML or an environment variable.

    Returns:
        Canonical registry key, or `None` when the value is not registered.
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
    """Return `value` as a TOML table when it has the expected runtime shape."""
    if not isinstance(value, dict):
        return None
    return cast("dict[str, object]", value)


def resolve_terminal_mapping(ui: Mapping[str, object]) -> str | None:
    """Resolve `[ui.terminal_themes][TERM_PROGRAM]` to a registered theme.

    Args:
        ui: The `[ui]` table parsed from `config.toml`.

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
