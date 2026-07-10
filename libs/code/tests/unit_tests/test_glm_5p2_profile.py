"""Tests for the GLM-5.2 harness profile registered by deepagents-code."""

from __future__ import annotations

import tomllib
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from deepagents.profiles._builtin_profiles import _ensure_builtin_profiles_loaded
from deepagents.profiles.harness.harness_profiles import (
    _HARNESS_PROFILES,
    _get_harness_profile,
)

from deepagents_code._glm_5p2_profile import (
    _GLM_5P2_MODEL_KEYS,
    _SYSTEM_PROMPT_SUFFIX,
    register,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"
_ENTRY_POINT_GROUP = "deepagents.harness_profiles"


@pytest.fixture(autouse=True)
def _isolate_harness_registry() -> Iterator[None]:
    """Snapshot and restore the global harness registry around each test.

    `register()` mutates the process-global `_HARNESS_PROFILES`. Bootstrap
    first so the snapshot already contains the built-in and entry-point
    registrations (including this profile), then restore to that snapshot on
    teardown so a test's `register()` call cannot leak into sibling tests.
    """
    _ensure_builtin_profiles_loaded()
    original = dict(_HARNESS_PROFILES)
    try:
        yield
    finally:
        _HARNESS_PROFILES.clear()
        _HARNESS_PROFILES.update(original)


def test_profile_resolves_through_bootstrap() -> None:
    """Each spec resolves via the entry-point bootstrap, without a manual register()."""
    for key in _GLM_5P2_MODEL_KEYS:
        profile = _get_harness_profile(key)
        assert profile is not None, f"bootstrap did not register {key!r}"
        assert profile.system_prompt_suffix == _SYSTEM_PROMPT_SUFFIX


def test_register_registers_every_declared_key() -> None:
    """`register()` makes the profile resolve for each key in `_GLM_5P2_MODEL_KEYS`."""
    register()
    for key in _GLM_5P2_MODEL_KEYS:
        profile = _get_harness_profile(key)
        assert profile is not None, f"no harness profile resolved for {key!r}"
        assert profile.system_prompt_suffix == _SYSTEM_PROMPT_SUFFIX


def test_suffix_covers_the_media_prohibition() -> None:
    """The suffix keeps the non-text `read_file` guard the profile exists for."""
    assert "<media_file_handling>" in _SYSTEM_PROMPT_SUFFIX
    assert "Do not call `read_file`" in _SYSTEM_PROMPT_SUFFIX
    # The rationale clause is load-bearing: it tells the model *why* the guard
    # exists. Guard against it being silently softened out.
    assert "non-text block this model cannot process" in _SYSTEM_PROMPT_SUFFIX


def test_entry_point_is_declared_in_pyproject() -> None:
    """The profile is declared as a `deepagents.harness_profiles` entry point."""
    data = tomllib.loads(_PYPROJECT.read_text())
    entry_points_table = data["project"]["entry-points"][_ENTRY_POINT_GROUP]
    assert entry_points_table["glm-5p2"] == "deepagents_code._glm_5p2_profile:register"


def test_entry_point_loads_to_register() -> None:
    """The installed entry point is discoverable and loads to `register`.

    Complements the pyproject-string check: this exercises the real
    `importlib.metadata` path the SDK bootstrap uses, so it catches stale
    installed metadata or a target path that parses but fails to import.
    """
    eps = {ep.name: ep for ep in entry_points(group=_ENTRY_POINT_GROUP)}
    assert "glm-5p2" in eps, "glm-5p2 entry point not found in installed metadata"
    assert eps["glm-5p2"].load() is register
