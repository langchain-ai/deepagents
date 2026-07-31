"""Shared fixtures for workflow helper script tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Production helpers live under domain folders beside this `tests/` tree.
# Nested test packages mirror those domains under tests/<domain>/.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_DOMAIN_DIRS = (
    _SCRIPTS_DIR / "checks",
    _SCRIPTS_DIR / "evals",
    _SCRIPTS_DIR / "labeling",
    _SCRIPTS_DIR / "release",
    _SCRIPTS_DIR,
)
for directory in _DOMAIN_DIRS:
    path = str(directory)
    if directory.is_dir() and path not in sys.path:
        sys.path.insert(0, path)
