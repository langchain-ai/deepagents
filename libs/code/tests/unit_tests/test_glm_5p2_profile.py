"""Tests for the GLM-5.2 harness profile shipped by Deep Agents Code."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import HarnessProfile, register_harness_profile
from deepagents.profiles._builtin_profiles import _ensure_builtin_profiles_loaded
from deepagents.profiles.harness.harness_profiles import (
    _HARNESS_PROFILES,
    _get_harness_profile,
)

import deepagents_code._glm_5p2_profile as glm_profile
from deepagents_code._glm_5p2_profile import (
    _GLM_5P2_MODEL_KEYS,
    _SYSTEM_PROMPT_SUFFIX,
    _ensure_glm_5p2_profile_registered,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _isolate_harness_registry() -> Iterator[None]:
    """Restore the harness registry and registration guard after each test."""
    _ensure_builtin_profiles_loaded()
    original = dict(_HARNESS_PROFILES)
    original_registered = glm_profile._registered
    glm_profile._registered = False
    for key in _GLM_5P2_MODEL_KEYS:
        _HARNESS_PROFILES.pop(key, None)
    try:
        yield
    finally:
        _HARNESS_PROFILES.clear()
        _HARNESS_PROFILES.update(original)
        glm_profile._registered = original_registered


def test_registration_is_idempotent_for_every_declared_key() -> None:
    """Deep Agents Code registers one profile for every supported GLM-5.2 spec."""
    _ensure_glm_5p2_profile_registered()
    _ensure_glm_5p2_profile_registered()

    assert glm_profile._registered is True
    for key in _GLM_5P2_MODEL_KEYS:
        profile = _get_harness_profile(key)
        assert profile is not None, f"no harness profile resolved for {key!r}"
        assert profile.system_prompt_suffix == _SYSTEM_PROMPT_SUFFIX


def test_registration_preserves_existing_exact_suffix() -> None:
    """App registration must not overwrite an advanced caller's exact suffix."""
    custom_key = _GLM_5P2_MODEL_KEYS[0]
    custom_suffix = "Use the locally registered GLM instructions."
    register_harness_profile(
        custom_key,
        HarnessProfile(system_prompt_suffix=custom_suffix),
    )

    _ensure_glm_5p2_profile_registered()

    custom_profile = _get_harness_profile(custom_key)
    assert custom_profile is not None
    assert custom_profile.system_prompt_suffix == custom_suffix
    for key in _GLM_5P2_MODEL_KEYS[1:]:
        profile = _get_harness_profile(key)
        assert profile is not None
        assert profile.system_prompt_suffix == _SYSTEM_PROMPT_SUFFIX


def test_suffix_covers_the_media_prohibition() -> None:
    """The suffix retains the non-text `read_file` guard."""
    assert "<media_file_handling>" in _SYSTEM_PROMPT_SUFFIX
    assert "Do not call `read_file`" in _SYSTEM_PROMPT_SUFFIX
    # The rationale clause is load-bearing: it tells the model *why* the guard
    # exists. Guard against it being silently softened out.
    assert "non-text block this model cannot process" in _SYSTEM_PROMPT_SUFFIX
