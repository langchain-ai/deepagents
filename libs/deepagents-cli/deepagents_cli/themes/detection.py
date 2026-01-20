"""Terminal dark/light mode detection."""

from __future__ import annotations

import os
import subprocess

# ANSI color threshold - colors 0-8 are typically dark backgrounds
_DARK_BG_THRESHOLD = 8


def detect_dark_mode() -> bool:
    """Detect if the terminal is in dark mode.

    Checks multiple sources in order:
    1. DEEPAGENTS_COLOR_MODE env var (explicit override)
    2. COLORFGBG env var (set by many terminals)
    3. macOS appearance setting
    4. Default to dark mode

    Returns:
        True if dark mode, False if light mode.
    """
    # 1. Explicit override
    override = os.environ.get("DEEPAGENTS_COLOR_MODE", "").lower()
    if override == "dark":
        return True
    if override == "light":
        return False

    # 2. COLORFGBG - format is "fg;bg" where bg < 8 typically means dark
    colorfgbg = os.environ.get("COLORFGBG", "")
    if ";" in colorfgbg:
        try:
            bg = int(colorfgbg.split(";")[-1])
            # ANSI colors 0-7 are dark variants, 8-15 are bright
            # A dark background typically uses colors 0 or values <= 8
            is_dark = bg <= _DARK_BG_THRESHOLD
        except (ValueError, IndexError):
            pass
        else:
            return is_dark

    # 3. macOS appearance (fallback)
    if os.uname().sysname == "Darwin":
        try:
            # Using full path to defaults - safe, no user input
            result = subprocess.run(
                ["/usr/bin/defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            # Returns "Dark" if dark mode, error/empty if light
            return result.stdout.strip().lower() == "dark"
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass

    # 4. Default to dark mode (most developer terminals are dark)
    return True
