"""Pytest configuration for the fixture's test suite.

Adds the fixture root to ``sys.path`` so ``webhooks``, ``common``,
``billing``, and ``ratelimit`` can be imported as top-level packages
without requiring an install step. This mirrors how an in-repo test
suite would typically run during development.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FIXTURE_ROOT = Path(__file__).resolve().parent.parent

if str(_FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIXTURE_ROOT))
