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

import pytest

import deepagents_code._env_vars as _mod

_SRC_DIR = Path(__file__).resolve().parents[2] / "deepagents_code"
_REGISTRY_FILE = _SRC_DIR / "_env_vars.py"

# Matches a full DEEPAGENTS_CODE_* env var name inside quote characters.
# The [A-Z] after the prefix avoids matching the bare prefix constant
# (_ENV_PREFIX = "DEEPAGENTS_CODE_") in model_config.py.
_ENV_VAR_RE = re.compile(r"""["'](DEEPAGENTS_CODE_[A-Z][A-Z0-9_]+)["']""")


def _public_constants() -> list[str]:
    """Return public constant names from `_env_vars` (alphabetical order)."""
    return [
        k
        for k, v in vars(_mod).items()
        if isinstance(v, str)
        and not k.startswith("_")
        and v.startswith("DEEPAGENTS_CODE_")
    ]


def _registered_values() -> set[str]:
    """Collect all `DEEPAGENTS_CODE_*` string values from `_env_vars`."""
    return {getattr(_mod, k) for k in _public_constants()}


def _collect_bare_literals(*, include_registry: bool = False) -> dict[str, set[str]]:
    """Map source files to bare `DEEPAGENTS_CODE_*` string literals found.

    Args:
        include_registry: When `True`, also scan `_env_vars.py`.

    Returns:
        `{relative_path: {var_name, ...}}` for files with hits.
    """
    hits: dict[str, set[str]] = {}
    for py_file in _SRC_DIR.rglob("*.py"):
        if not include_registry and py_file == _REGISTRY_FILE:
            continue
        matches = set(_ENV_VAR_RE.findall(py_file.read_text()))
        if matches:
            hits[str(py_file.relative_to(_SRC_DIR))] = matches
    return hits


class TestEnvVarRegistryDrift:
    """Ensure `_env_vars` stays in sync with source code usage."""


class TestIsEnvTruthy:
    """Parsing of on/off boolean env vars via `is_env_truthy`."""
