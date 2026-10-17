"""Drift-detection tests for the CLI environment variable registry.

These tests ensure that:

1. Every `DEEPAGENTS_CODE_*` constant in `_env_vars.py` has a matching
   value used somewhere in source code (no stale entries).
2. No source file outside `_env_vars.py` uses a bare string literal like
   `"DEEPAGENTS_CODE_FOO"` -- it must import the constant instead.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2] / "deepagents_code"
_REGISTRY_FILE = _SRC_DIR / "_env_vars.py"

# Matches a full DEEPAGENTS_CODE_* env var name inside quote characters.
# The [A-Z] after the prefix avoids matching the bare prefix constant
# (_ENV_PREFIX = "DEEPAGENTS_CODE_") in model_config.py.
_ENV_VAR_RE = re.compile(r"""["'](DEEPAGENTS_CODE_[A-Z][A-Z0-9_]+)["']""")


class TestEnvVarRegistryDrift:
    """Ensure `_env_vars` stays in sync with source code usage."""


class TestIsEnvTruthy:
    """Parsing of on/off boolean env vars via `is_env_truthy`."""
