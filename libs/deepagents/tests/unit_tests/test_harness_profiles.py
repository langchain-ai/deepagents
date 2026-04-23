"""Tests for harness-profile config and sub-profile serialization."""

from __future__ import annotations

import pytest

from deepagents import (
    AsyncSubAgentMiddleware,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    HarnessProfileConfig,
)
from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware


class TestGeneralPurposeSubagentProfileSerde:
    """`to_dict` / `from_dict` round-trips for the GP subagent sub-profile."""

    def test_empty_profile_round_trips_to_empty_dict(self) -> None:
        profile = GeneralPurposeSubagentProfile()
        assert profile.to_dict() == {}
        assert GeneralPurposeSubagentProfile.from_dict({}) == profile

    def test_all_fields_round_trip(self) -> None:
        profile = GeneralPurposeSubagentProfile(
            enabled=False,
            description="Custom description.",
            system_prompt="Do the thing.",
        )
        data = profile.to_dict()
        assert data == {
            "enabled": False,
            "description": "Custom description.",
            "system_prompt": "Do the thing.",
        }
        assert GeneralPurposeSubagentProfile.from_dict(data) == profile

    def test_from_dict_rejects_unknown_keys(self) -> None:
        with pytest.raises(TypeError, match="Unknown keys"):
            GeneralPurposeSubagentProfile.from_dict({"enabled": True, "bogus": 1})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("enabled", "yes"),
            ("description", 1),
            ("system_prompt", ["list"]),
        ],
    )
    def test_from_dict_rejects_wrong_types(self, key: str, value: object) -> None:
        with pytest.raises(TypeError, match=key):
            GeneralPurposeSubagentProfile.from_dict({key: value})


class TestHarnessProfileConfigSerde:
    """`to_dict` / `from_dict` round-trips for `HarnessProfileConfig`."""

    def test_empty_config_round_trips_to_empty_dict(self) -> None:
        config = HarnessProfileConfig()
        assert config.to_dict() == {}
        assert HarnessProfileConfig.from_dict({}) == config

    def test_full_config_round_trips(self) -> None:
        config = HarnessProfileConfig(
            base_system_prompt="You are helpful.",
            system_prompt_suffix="Respond briefly.",
            tool_description_overrides={"ls": "List files."},
            excluded_tools=frozenset({"execute", "grep"}),
            excluded_middleware=frozenset({"SummarizationMiddleware", "TodoListMiddleware"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )
        data = config.to_dict()
        assert data == {
            "base_system_prompt": "You are helpful.",
            "system_prompt_suffix": "Respond briefly.",
            "tool_description_overrides": {"ls": "List files."},
            "excluded_tools": ["execute", "grep"],
            "excluded_middleware": [
                "SummarizationMiddleware",
                "TodoListMiddleware",
            ],
            "general_purpose_subagent": {"enabled": False},
        }
        assert HarnessProfileConfig.from_dict(data) == config

    def test_to_harness_profile_returns_runtime_profile(self) -> None:
        config = HarnessProfileConfig(
            system_prompt_suffix="Respond briefly.",
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
        )
        assert config.to_harness_profile() == HarnessProfile(
            system_prompt_suffix="Respond briefly.",
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
        )

    def test_to_harness_profile_resolves_import_ref_entries(self) -> None:
        config = HarnessProfileConfig(excluded_middleware=frozenset({"deepagents.middleware.async_subagents:AsyncSubAgentMiddleware"}))
        assert config.to_harness_profile() == HarnessProfile(excluded_middleware=frozenset({AsyncSubAgentMiddleware}))

    def test_to_harness_profile_rejects_invalid_import_ref_target(self) -> None:
        config = HarnessProfileConfig(excluded_middleware=frozenset({"deepagents.profiles.harness_profiles:HarnessProfileConfig"}))
        with pytest.raises(TypeError, match="AgentMiddleware"):
            config.to_harness_profile()

    def test_to_harness_profile_rejects_malformed_import_ref(self) -> None:
        config = HarnessProfileConfig(excluded_middleware=frozenset({"deepagents:"}))
        with pytest.raises(ValueError, match="module:Class"):
            config.to_harness_profile()

    def test_to_dict_omits_unset_fields(self) -> None:
        """Fields at their default are dropped so the output stays minimal."""
        config = HarnessProfileConfig(system_prompt_suffix="Respond briefly.")
        assert config.to_dict() == {"system_prompt_suffix": "Respond briefly."}

    def test_from_dict_rejects_unknown_keys(self) -> None:
        with pytest.raises(TypeError, match="Unknown keys"):
            HarnessProfileConfig.from_dict({"bogus": 1})

    def test_from_dict_rejects_extra_middleware_key(self) -> None:
        """`extra_middleware` belongs to runtime `HarnessProfile`, not config."""
        with pytest.raises(TypeError, match="extra_middleware"):
            HarnessProfileConfig.from_dict({"extra_middleware": []})

    def test_from_dict_rejects_non_string_excluded_entries(self) -> None:
        with pytest.raises(TypeError, match="excluded_middleware"):
            HarnessProfileConfig.from_dict({"excluded_middleware": [123]})

    def test_from_dict_rejects_non_string_tool_name(self) -> None:
        with pytest.raises(TypeError, match="tool_description_overrides"):
            HarnessProfileConfig.from_dict({"tool_description_overrides": {1: "x"}})

    def test_from_dict_accepts_list_or_set_for_excluded_fields(self) -> None:
        """YAML/JSON produce lists; in-memory dicts may use sets. Both work."""
        config_list = HarnessProfileConfig.from_dict(
            {
                "excluded_tools": ["execute"],
                "excluded_middleware": ["SummarizationMiddleware"],
            }
        )
        config_set = HarnessProfileConfig.from_dict(
            {
                "excluded_tools": {"execute"},
                "excluded_middleware": {"SummarizationMiddleware"},
            }
        )
        assert config_list == config_set

    def test_to_dict_validates_tool_description_value_types(self) -> None:
        """Non-string tool descriptions raise at serialize time."""
        config = HarnessProfileConfig(
            tool_description_overrides={"ls": 42}  # ty: ignore[invalid-argument-type]
        )
        with pytest.raises(TypeError, match="tool_description_overrides"):
            config.to_dict()

    def test_empty_general_purpose_subagent_preserves_identity(self) -> None:
        """An explicit empty sub-profile stays distinct from `None`."""
        config = HarnessProfileConfig(general_purpose_subagent=GeneralPurposeSubagentProfile())
        data = config.to_dict()
        assert data == {"general_purpose_subagent": {}}
        assert HarnessProfileConfig.from_dict(data) == config

    def test_to_dict_output_ordering_is_deterministic(self) -> None:
        """Set-backed config fields emit in sorted order regardless of construction order."""
        config_a = HarnessProfileConfig(
            excluded_tools=frozenset(["z_tool", "a_tool", "m_tool"]),
            excluded_middleware=frozenset(["ZooMiddleware", "AlphaMiddleware", "MiddleMiddleware"]),
        )
        config_b = HarnessProfileConfig(
            excluded_tools=frozenset(["m_tool", "z_tool", "a_tool"]),
            excluded_middleware=frozenset(["MiddleMiddleware", "ZooMiddleware", "AlphaMiddleware"]),
        )
        assert config_a.to_dict() == config_b.to_dict()
        assert config_a.to_dict()["excluded_tools"] == ["a_tool", "m_tool", "z_tool"]
        assert config_a.to_dict()["excluded_middleware"] == [
            "AlphaMiddleware",
            "MiddleMiddleware",
            "ZooMiddleware",
        ]

    def test_mapping_proxy_type_tool_description_overrides_round_trip(self) -> None:
        """Config output converts mapping proxies back to plain dicts."""
        config = HarnessProfileConfig(
            tool_description_overrides={
                "ls": "List files.",
                "grep": "Search files.",
            }
        )
        data = config.to_dict()
        assert isinstance(data["tool_description_overrides"], dict)
        assert HarnessProfileConfig.from_dict(data) == config

    def test_from_harness_profile_preserves_string_entries(self) -> None:
        profile = HarnessProfile(
            system_prompt_suffix="Respond briefly.",
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
        )
        assert HarnessProfileConfig.from_harness_profile(profile) == HarnessProfileConfig(
            system_prompt_suffix="Respond briefly.",
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
        )

    def test_from_harness_profile_serializes_class_entries_as_import_refs(self) -> None:
        profile = HarnessProfile(excluded_middleware=frozenset({AsyncSubAgentMiddleware}))
        assert HarnessProfileConfig.from_harness_profile(profile) == HarnessProfileConfig(
            excluded_middleware=frozenset({"deepagents.middleware.async_subagents:AsyncSubAgentMiddleware"})
        )

    def test_from_harness_profile_prefers_public_alias_for_summarization(self) -> None:
        profile = HarnessProfile(excluded_middleware=frozenset({_DeepAgentsSummarizationMiddleware}))
        assert HarnessProfileConfig.from_harness_profile(profile) == HarnessProfileConfig(excluded_middleware=frozenset({"SummarizationMiddleware"}))

    def test_from_harness_profile_rejects_non_empty_extra_middleware(self) -> None:
        class _Fake:
            pass

        profile = HarnessProfile(extra_middleware=(_Fake.__new__(_Fake),))
        with pytest.raises(ValueError, match="extra_middleware"):
            HarnessProfileConfig.from_harness_profile(profile)

    def test_from_harness_profile_rejects_local_class_entries(self) -> None:
        class LocalMiddleware(AsyncSubAgentMiddleware):
            pass

        profile = HarnessProfile(excluded_middleware=frozenset({LocalMiddleware}))
        with pytest.raises(ValueError, match="module-level"):
            HarnessProfileConfig.from_harness_profile(profile)


class TestHarnessProfileConfigYamlRoundTrip:
    """Exercises the full YAML roundtrip path using `yaml.safe_dump` / `safe_load`."""

    def test_yaml_round_trip(self) -> None:
        yaml = pytest.importorskip("yaml")
        config = HarnessProfileConfig(
            base_system_prompt="You are helpful.",
            system_prompt_suffix="Respond briefly.",
            tool_description_overrides={"ls": "List files."},
            excluded_tools=frozenset({"execute"}),
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )
        serialized = yaml.safe_dump(config.to_dict())
        reconstructed = HarnessProfileConfig.from_dict(yaml.safe_load(serialized))
        assert reconstructed == config

    def test_yaml_safe_dump_accepts_output_without_custom_tags(self) -> None:
        """The serialized config must contain only YAML-safe primitives."""
        yaml = pytest.importorskip("yaml")
        config = HarnessProfileConfig(
            excluded_tools=frozenset({"execute"}),
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
        )
        output = yaml.safe_dump(config.to_dict())
        assert "!!" not in output
