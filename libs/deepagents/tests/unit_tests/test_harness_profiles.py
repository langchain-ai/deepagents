"""Tests for `HarnessProfile` / `GeneralPurposeSubagentProfile` serialization."""

from __future__ import annotations

import pytest

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware
from deepagents.profiles.harness_profiles import _merge_profiles


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


class TestHarnessProfileSerde:
    """`to_dict` / `from_dict` round-trips for `HarnessProfile`.

    Covers the declarative subset: string fields, string sets, the tool
    description mapping, and the nested general-purpose subagent dict.
    Non-serializable state (class entries in `excluded_middleware`,
    `extra_middleware`) raises on `to_dict` — not supported by design.
    """

    def test_empty_profile_round_trips_to_empty_dict(self) -> None:
        profile = HarnessProfile()
        assert profile.to_dict() == {}
        assert HarnessProfile.from_dict({}) == profile

    def test_full_profile_round_trips(self) -> None:
        profile = HarnessProfile(
            base_system_prompt="You are helpful.",
            system_prompt_suffix="Respond briefly.",
            tool_description_overrides={"ls": "List files."},
            excluded_tools=frozenset({"execute", "grep"}),
            excluded_middleware=frozenset({"SummarizationMiddleware", "TodoListMiddleware"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )
        data = profile.to_dict()
        assert data == {
            "base_system_prompt": "You are helpful.",
            "system_prompt_suffix": "Respond briefly.",
            "tool_description_overrides": {"ls": "List files."},
            "excluded_tools": ["execute", "grep"],
            "excluded_middleware": ["SummarizationMiddleware", "TodoListMiddleware"],
            "general_purpose_subagent": {"enabled": False},
        }
        assert HarnessProfile.from_dict(data) == profile

    def test_to_dict_omits_unset_fields(self) -> None:
        """Fields at their default are dropped so the output stays minimal."""
        profile = HarnessProfile(system_prompt_suffix="Respond briefly.")
        assert profile.to_dict() == {"system_prompt_suffix": "Respond briefly."}

    def test_to_dict_rejects_class_form_excluded_middleware(self) -> None:
        profile = HarnessProfile(
            excluded_middleware=frozenset({_DeepAgentsSummarizationMiddleware}),
        )
        with pytest.raises(ValueError, match="class-form entries"):
            profile.to_dict()

    def test_to_dict_rejects_mixed_class_and_string_entries(self) -> None:
        profile = HarnessProfile(
            excluded_middleware=frozenset({_DeepAgentsSummarizationMiddleware, "todo_list"}),
        )
        with pytest.raises(ValueError, match="class-form entries"):
            profile.to_dict()

    def test_to_dict_rejects_non_empty_extra_middleware_sequence(self) -> None:
        class _Fake:
            pass

        profile = HarnessProfile(extra_middleware=(_Fake.__new__(_Fake),))
        with pytest.raises(ValueError, match="extra_middleware"):
            profile.to_dict()

    def test_to_dict_rejects_extra_middleware_factory(self) -> None:
        profile = HarnessProfile(extra_middleware=list)
        with pytest.raises(ValueError, match="extra_middleware"):
            profile.to_dict()

    def test_to_dict_allows_empty_extra_middleware(self) -> None:
        """Default empty tuple is serializable and simply omitted."""
        profile = HarnessProfile(extra_middleware=())
        assert profile.to_dict() == {}

    def test_from_dict_rejects_unknown_keys(self) -> None:
        with pytest.raises(TypeError, match="Unknown keys"):
            HarnessProfile.from_dict({"bogus": 1})

    def test_from_dict_rejects_extra_middleware_key(self) -> None:
        """`extra_middleware` must be set in code, never loaded from config."""
        with pytest.raises(TypeError, match="extra_middleware"):
            HarnessProfile.from_dict({"extra_middleware": []})

    def test_from_dict_rejects_non_string_excluded_entries(self) -> None:
        with pytest.raises(TypeError, match="excluded_middleware"):
            HarnessProfile.from_dict({"excluded_middleware": [123]})

    def test_from_dict_rejects_non_string_tool_name(self) -> None:
        with pytest.raises(TypeError, match="tool_description_overrides"):
            HarnessProfile.from_dict({"tool_description_overrides": {1: "x"}})

    def test_from_dict_accepts_list_or_set_for_excluded_fields(self) -> None:
        """YAML/JSON produce lists; in-memory dicts may use sets. Both work."""
        profile_list = HarnessProfile.from_dict({"excluded_tools": ["execute"], "excluded_middleware": ["SummarizationMiddleware"]})
        profile_set = HarnessProfile.from_dict({"excluded_tools": {"execute"}, "excluded_middleware": {"SummarizationMiddleware"}})
        assert profile_list == profile_set

    def test_to_dict_validates_tool_description_value_types(self) -> None:
        """`tool_description_overrides` with a non-string value raises at serialize time.

        Python doesn't enforce `Mapping[str, str]` at runtime, so a bad value
        can slip in past the dataclass field annotation. Surfacing the error
        at `to_dict` (symmetric with `from_dict`'s validation) prevents a
        confusing failure later in `yaml.safe_dump` or a downstream
        `from_dict` call.
        """
        profile = HarnessProfile(tool_description_overrides={"ls": 42})  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match="tool_description_overrides"):
            profile.to_dict()

    def test_empty_general_purpose_subagent_preserves_identity(self) -> None:
        """`GeneralPurposeSubagentProfile()` round-trips to an equal sub-profile, not `None`.

        An explicit empty sub-profile and no sub-profile are semantically
        different (the first forces the default GP subagent on, the second
        leaves it implicit). `to_dict` must preserve this distinction so
        downstream consumers see the same profile on reload.
        """
        profile = HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile())
        data = profile.to_dict()
        assert data == {"general_purpose_subagent": {}}
        assert HarnessProfile.from_dict(data) == profile

    def test_to_dict_output_ordering_is_deterministic(self) -> None:
        """`excluded_tools` / `excluded_middleware` emit in sorted order regardless of construction order.

        `frozenset` iteration is unordered, so `to_dict` must sort its string
        lists for deterministic output. Diffable config files depend on this
        — a profile written out twice should produce identical YAML.
        """
        profile_a = HarnessProfile(
            excluded_tools=frozenset(["z_tool", "a_tool", "m_tool"]),
            excluded_middleware=frozenset(["ZooMiddleware", "AlphaMiddleware", "MiddleMiddleware"]),
        )
        profile_b = HarnessProfile(
            excluded_tools=frozenset(["m_tool", "z_tool", "a_tool"]),
            excluded_middleware=frozenset(["MiddleMiddleware", "ZooMiddleware", "AlphaMiddleware"]),
        )
        assert profile_a.to_dict() == profile_b.to_dict()
        assert profile_a.to_dict()["excluded_tools"] == ["a_tool", "m_tool", "z_tool"]
        assert profile_a.to_dict()["excluded_middleware"] == ["AlphaMiddleware", "MiddleMiddleware", "ZooMiddleware"]

    def test_mapping_proxy_type_tool_description_overrides_round_trip(self) -> None:
        """`__post_init__` wraps `tool_description_overrides` in `MappingProxyType`.

        The round-trip through `to_dict` must convert it back to a plain dict
        (for portability) without losing keys or values, and `from_dict` must
        rebuild a profile that compares equal.
        """
        profile = HarnessProfile(tool_description_overrides={"ls": "List files.", "grep": "Search files."})
        data = profile.to_dict()
        # to_dict must produce a plain dict, not a MappingProxyType, so
        # downstream encoders (json, yaml) accept it.
        assert isinstance(data["tool_description_overrides"], dict)
        assert HarnessProfile.from_dict(data) == profile

    def test_merged_profile_with_class_entry_fails_to_dict(self) -> None:
        """Merging a class-form provider exclusion with a string-form model exclusion still raises.

        Regression path: provider-level profile registers a class-form
        exclusion, model-level profile registers a string-form exclusion,
        caller does `_merge_profiles(base, override).to_dict()`. The merged
        set has a class entry → `to_dict` must still raise at serialize time.
        """
        base = HarnessProfile(excluded_middleware=frozenset({_DeepAgentsSummarizationMiddleware}))
        override = HarnessProfile(excluded_middleware=frozenset({"OtherMiddleware"}))
        merged = _merge_profiles(base, override)
        with pytest.raises(ValueError, match="class-form entries"):
            merged.to_dict()


class TestHarnessProfileYamlRoundTrip:
    """Exercises the full YAML roundtrip path using `yaml.safe_dump` / `safe_load`."""

    def test_yaml_round_trip(self) -> None:
        yaml = pytest.importorskip("yaml")
        profile = HarnessProfile(
            base_system_prompt="You are helpful.",
            system_prompt_suffix="Respond briefly.",
            tool_description_overrides={"ls": "List files."},
            excluded_tools=frozenset({"execute"}),
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )
        serialized = yaml.safe_dump(profile.to_dict())
        reconstructed = HarnessProfile.from_dict(yaml.safe_load(serialized))
        assert reconstructed == profile

    def test_yaml_safe_dump_accepts_output_without_custom_tags(self) -> None:
        """The `to_dict` output must be primitive-only (no tuples, sets, custom types).

        Custom types force yaml to emit Python-specific tags under `safe_dump`,
        which fails loudly. A clean `safe_dump` run proves the serialized form
        is portable across YAML parsers.
        """
        yaml = pytest.importorskip("yaml")
        profile = HarnessProfile(
            excluded_tools=frozenset({"execute"}),
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
        )
        output = yaml.safe_dump(profile.to_dict())
        assert "!!" not in output
