"""Tests for the GLM-5.2 harness profile registered by deepagents-code."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from deepagents.profiles.harness.harness_profiles import _get_harness_profile

from deepagents_code._glm_5p2_profile import (
    _GLM_5P2_MODEL_KEYS,
    _SYSTEM_PROMPT_SUFFIX,
    register,
)

_PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def test_register_registers_every_provider_key() -> None:
    """`register()` makes the profile resolve for each provider spec."""
    register()
    for key in _GLM_5P2_MODEL_KEYS:
        profile = _get_harness_profile(key)
        assert profile is not None, f"no harness profile resolved for {key!r}"
        assert profile.system_prompt_suffix == _SYSTEM_PROMPT_SUFFIX


def test_suffix_covers_the_media_prohibition() -> None:
    """The suffix keeps the non-text `read_file` guard the profile exists for."""
    assert "<media_file_handling>" in _SYSTEM_PROMPT_SUFFIX
    assert "Do not call `read_file`" in _SYSTEM_PROMPT_SUFFIX


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib requires 3.11+")
def test_entry_point_is_declared() -> None:
    """The profile is wired as a `deepagents.harness_profiles` entry point."""
    data = tomllib.loads(_PYPROJECT.read_text())
    entry_points = data["project"]["entry-points"]["deepagents.harness_profiles"]
    assert entry_points["glm-5p2"] == "deepagents_code._glm_5p2_profile:register"
