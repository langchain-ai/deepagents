"""Terminal dark/light mode detection."""

from __future__ import annotations

import os


def detect_dark_mode() -> bool:
    """Detect if the terminal is in dark mode.

    Only checks explicit override - defaults to dark mode.
    Most developers use dark terminals, and auto-detection is unreliable.

    Set DEEPAGENTS_COLOR_MODE=light to force light mode.

    Returns:
        True if dark mode, False if light mode.
    """
    override = os.environ.get("DEEPAGENTS_COLOR_MODE", "").lower()
    # Default to dark mode - explicit override required for light
    return override != "light"
