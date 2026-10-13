"""Stream event that tells the user about unreadable skill sources.

The SDK shows skill load errors only to the model. `PluginSkillsMiddleware`
also streams the errors for skill directories that exist but cannot be read,
and the TUI and headless clients show them. Missing optional directories are
expected, so they do not produce the event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

SKILLS_LOAD_ERRORS_EVENT = "skills_load_errors"
"""Custom stream event `type` that carries the `errors` list."""


def skills_load_errors_event(errors: Sequence[str]) -> dict[str, object]:
    """Return the custom stream payload for `errors`."""
    return {"type": SKILLS_LOAD_ERRORS_EVENT, "errors": list(errors)}


def skills_load_errors_from_event(data: object) -> list[str] | None:
    """Return the errors of a skills load-error event.

    Returns:
        The error strings, or `None` when `data` is another event.
    """
    if not isinstance(data, dict) or data.get("type") != SKILLS_LOAD_ERRORS_EVENT:
        return None
    errors = data.get("errors")
    if not isinstance(errors, list):
        return []
    return [error for error in errors if isinstance(error, str)]


def format_skills_load_errors(errors: Sequence[str]) -> str:
    """Return the user notice for unreadable skill sources."""
    lines = "\n".join(f"  - {error}" for error in errors)
    return f"Some skills could not be loaded, so the agent cannot use them:\n{lines}"
