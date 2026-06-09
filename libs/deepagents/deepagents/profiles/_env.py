"""Shared helpers for environment-variable parsing.

Several profile-bootstrap toggles are driven by boolean environment variables.
Centralizing the truthy-value criteria here keeps every toggle's vocabulary
identical and avoids subtly mismatched inline checks.
"""

from __future__ import annotations

import os


def _env_flag(name: str) -> bool:
    """Return whether environment variable `name` holds a truthy value.

    Truthy values are `1`, `true`, `yes`, and `on`, matched case-insensitively
    with surrounding whitespace ignored. An unset variable, or any other value,
    is treated as `False`.

    Args:
        name: Name of the environment variable to read.

    Returns:
        `True` when the variable is set to a recognized truthy value.
    """
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
