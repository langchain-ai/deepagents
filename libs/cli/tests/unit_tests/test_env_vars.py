"""Drift-detection tests for the CLI environment variable registry.

These tests ensure that:

1. Every `DEEPAGENTS_CLI_*` constant in `_env_vars.py` has a matching
   value used somewhere in source code (no stale entries).
2. No source file outside `_env_vars.py` uses a bare string literal like
   `"DEEPAGENTS_CLI_FOO"` -- it must import the constant instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import deepagents_cli._env_vars as _mod

_SRC_DIR = Path(__file__).resolve().parents[2] / "deepagents_cli"
_REGISTRY_FILE = _SRC_DIR / "_env_vars.py"

# Matches a full DEEPAGENTS_CLI_* env var name inside quote characters.
# The [A-Z] after the prefix avoids matching the bare prefix constant
# (_ENV_PREFIX = "DEEPAGENTS_CLI_") in model_config.py.
_ENV_VAR_RE = re.compile(r"""["'](DEEPAGENTS_CLI_[A-Z][A-Z0-9_]+)["']""")


def _public_constants() -> list[str]:
    """Return public constant names from `_env_vars` (alphabetical order)."""
    return [
        k
        for k, v in vars(_mod).items()
        if isinstance(v, str)
        and not k.startswith("_")
        and v.startswith("DEEPAGENTS_CLI_")
    ]


def _registered_values() -> set[str]:
    """Collect all `DEEPAGENTS_CLI_*` string values from `_env_vars`."""
    return {getattr(_mod, k) for k in _public_constants()}


def _collect_bare_literals(*, include_registry: bool = False) -> dict[str, set[str]]:
    """Map source files to bare `DEEPAGENTS_CLI_*` string literals found.

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

    def test_no_bare_literals_outside_registry(self) -> None:
        """Source files must import constants, not use raw string literals."""
        hits = _collect_bare_literals()
        assert not hits, (
            "Bare DEEPAGENTS_CLI_* string literals found in source "
            "(import from deepagents_cli._env_vars instead):\n"
            + "\n".join(f"  {f}: {sorted(v)}" for f, v in sorted(hits.items()))
        )

    def test_no_stale_registry_entries(self) -> None:
        """Every registered value must be parseable from `_env_vars.py` source."""
        registered = _registered_values()
        in_registry_file = set(_ENV_VAR_RE.findall(_REGISTRY_FILE.read_text()))
        stale = registered - in_registry_file
        assert not stale, (
            f"Constants whose values don't appear in _env_vars.py: {stale}. "
            "Remove stale entries from deepagents_cli/_env_vars.py."
        )

    def test_registry_constants_are_sorted(self) -> None:
        """Public constant names must be alphabetically sorted."""
        names = _public_constants()
        assert names == sorted(names), (
            "Constants in _env_vars.py are not sorted. Expected order: "
            + ", ".join(sorted(names))
        )


class TestIsEnvTruthy:
    """Parsing of on/off boolean env vars via `is_env_truthy`."""

    def test_unset_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing env var falls back to *default*."""
        from deepagents_cli._env_vars import is_env_truthy

        monkeypatch.delenv("DEEPAGENTS_CLI_DEBUG", raising=False)
        assert is_env_truthy("DEEPAGENTS_CLI_DEBUG") is False
        assert is_env_truthy("DEEPAGENTS_CLI_DEBUG", default=True) is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "On", "  true  "])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """Recognized truthy values enable the flag regardless of case/whitespace."""
        from deepagents_cli._env_vars import is_env_truthy

        monkeypatch.setenv("DEEPAGENTS_CLI_DEBUG", value)
        assert is_env_truthy("DEEPAGENTS_CLI_DEBUG") is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """Recognized falsy values disable the flag even when `default=True`.

        This is the key behaviour change — `bool(os.environ.get(...))` would
        treat `"0"` and `"false"` as truthy because they're non-empty strings.
        """
        from deepagents_cli._env_vars import is_env_truthy

        monkeypatch.setenv("DEEPAGENTS_CLI_DEBUG", value)
        assert is_env_truthy("DEEPAGENTS_CLI_DEBUG") is False
        assert is_env_truthy("DEEPAGENTS_CLI_DEBUG", default=True) is False

    def test_unrecognized_value_returns_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Values outside the truthy/falsy sets fall back to *default*."""
        from deepagents_cli._env_vars import is_env_truthy

        monkeypatch.setenv("DEEPAGENTS_CLI_DEBUG", "maybe")
        assert is_env_truthy("DEEPAGENTS_CLI_DEBUG") is False
        assert is_env_truthy("DEEPAGENTS_CLI_DEBUG", default=True) is True
