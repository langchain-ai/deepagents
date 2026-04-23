"""Tests for `HarnessProfile` / `GeneralPurposeSubagentProfile` serialization."""

from __future__ import annotations

import pytest

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
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
            excluded_middleware=frozenset({"summarization", "todo_list"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )
        data = profile.to_dict()
        assert data == {
            "base_system_prompt": "You are helpful.",
            "system_prompt_suffix": "Respond briefly.",
            "tool_description_overrides": {"ls": "List files."},
            "excluded_tools": ["execute", "grep"],
            "excluded_middleware": ["summarization", "todo_list"],
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
        profile_list = HarnessProfile.from_dict({"excluded_tools": ["execute"], "excluded_middleware": ["summarization"]})
        profile_set = HarnessProfile.from_dict({"excluded_tools": {"execute"}, "excluded_middleware": {"summarization"}})
        assert profile_list == profile_set


class TestHarnessProfileYamlRoundTrip:
    """Exercises the full YAML roundtrip path using `yaml.safe_dump` / `safe_load`."""

    def test_yaml_round_trip(self) -> None:
        yaml = pytest.importorskip("yaml")
        profile = HarnessProfile(
            base_system_prompt="You are helpful.",
            system_prompt_suffix="Respond briefly.",
            tool_description_overrides={"ls": "List files."},
            excluded_tools=frozenset({"execute"}),
            excluded_middleware=frozenset({"summarization"}),
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
            excluded_middleware=frozenset({"summarization"}),
        )
        output = yaml.safe_dump(profile.to_dict())
        assert "!!" not in output
