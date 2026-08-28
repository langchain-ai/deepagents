"""Lightweight terminal character-set detection."""

from __future__ import annotations

import os
import sys
from typing import Literal


def detect_charset_mode() -> Literal["ascii", "unicode"]:
    """Return the effective terminal character-set mode."""
    prefixed = os.environ.get("DEEPAGENTS_CODE_UI_CHARSET_MODE")
    mode = prefixed if prefixed is not None else os.environ.get("UI_CHARSET_MODE")
    mode = (mode or "auto").lower()
    if mode == "unicode":
        return "unicode"
    if mode == "ascii":
        return "ascii"

    encoding = getattr(sys.stdout, "encoding", "") or ""
    if "utf" in encoding.lower():
        return "unicode"
    lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    return "unicode" if "utf" in lang.lower() else "ascii"
