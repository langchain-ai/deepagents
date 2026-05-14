"""Smoke tests for the Pi-style harness profile."""

from __future__ import annotations

import pytest
from deepagents import HarnessProfile
from deepagents.profiles.harness.harness_profiles import (
    _HARNESS_PROFILES,
    _get_harness_profile,
)

from pi_profile import (
    PI_BASE_SYSTEM_PROMPT,
    PI_TOOL_DESCRIPTIONS,
    pi_harness_profile,
    register_pi_harness,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Drop test-only registrations so suites stay independent."""
    snapshot = dict(_HARNESS_PROFILES)
    yield
    _HARNESS_PROFILES.clear()
    _HARNESS_PROFILES.update(snapshot)


class TestPiHarnessProfile:
    """`pi_harness_profile()` returns the expected `HarnessProfile`."""

    def test_returns_harness_profile_instance(self) -> None:
        assert isinstance(pi_harness_profile(), HarnessProfile)

    def test_uses_pi_base_system_prompt(self) -> None:
        assert pi_harness_profile().base_system_prompt == PI_BASE_SYSTEM_PROMPT

    def test_overrides_default_filesystem_tools(self) -> None:
        overrides = pi_harness_profile().tool_description_overrides
        assert set(overrides) == {
            "read_file",
            "write_file",
            "edit_file",
            "ls",
            "glob",
            "grep",
            "execute",
        }
        assert dict(overrides) == dict(PI_TOOL_DESCRIPTIONS)

    def test_no_middleware_or_tool_exclusions(self) -> None:
        profile = pi_harness_profile()
        assert profile.excluded_tools == frozenset()
        assert profile.excluded_middleware == frozenset()

    def test_factory_returns_fresh_instance(self) -> None:
        assert pi_harness_profile() is not pi_harness_profile()

    def test_tool_descriptions_view_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            PI_TOOL_DESCRIPTIONS["read_file"] = "tampered"  # type: ignore[index]


class TestRegisterPiHarness:
    """`register_pi_harness` lands the profile in the harness registry."""

    def test_registers_under_provider_key(self) -> None:
        register_pi_harness("anthropic")
        resolved = _get_harness_profile("anthropic")
        assert resolved is not None
        assert resolved.base_system_prompt == PI_BASE_SYSTEM_PROMPT

    def test_registers_under_model_key(self) -> None:
        register_pi_harness("openai:gpt-5.3")
        resolved = _get_harness_profile("openai:gpt-5.3")
        assert resolved is not None
        assert "read_file" in resolved.tool_description_overrides

    def test_returns_registered_profile(self) -> None:
        profile = register_pi_harness("anthropic:claude-sonnet-4-6")
        assert isinstance(profile, HarnessProfile)
        assert profile.base_system_prompt == PI_BASE_SYSTEM_PROMPT

    def test_rejects_invalid_key(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            register_pi_harness("")
