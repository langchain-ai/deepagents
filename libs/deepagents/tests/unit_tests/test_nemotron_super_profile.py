"""Registration checks for the initial Nemotron 3.5 Super profile."""

from __future__ import annotations

from deepagents.profiles.harness.harness_profiles import _get_harness_profile


def test_super_profile_resolves_for_documented_model() -> None:
    """Both ChatNVIDIA and `init_chat_model` identifiers receive the profile."""
    for provider in ("NVIDIA", "nvidia"):
        profile = _get_harness_profile(
            f"{provider}:nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        )
        assert profile is not None
        assert "<state_changes>" in (profile.system_prompt_suffix or "")
        assert "NemotronToolCallShim" in {
            middleware.name for middleware in profile.materialize_extra_middleware()
        }


def test_super_profile_does_not_apply_to_other_nvidia_models() -> None:
    """The registration must stay scoped to the selected model."""
    assert _get_harness_profile("NVIDIA:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B") is None
