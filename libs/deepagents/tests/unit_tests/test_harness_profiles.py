"""Tests for harness-profile config and sub-profile serialization."""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagents import (
    AsyncSubAgentMiddleware,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    HarnessProfileConfig,
    create_deep_agent,
)
from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware
from deepagents.profiles.harness._glm import _GLM_MODEL_SPECS
from deepagents.profiles.harness._kimi import _KIMI_MODEL_SPECS
from deepagents.profiles.harness._minimax import _MINIMAX_MODEL_SPECS, _SYSTEM_PROMPT_SUFFIX
from deepagents.profiles.harness._precompletion import PreCompletionVerificationMiddleware
from deepagents.profiles.harness._reasoning_gate import ReasoningGateMiddleware
from deepagents.profiles.harness.harness_profiles import (
    _ensure_harness_profiles_loaded,
    _get_harness_profile,
)


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

    def test_rejects_class_path_entries_at_construction(self) -> None:
        """Class-path (`module:Class`) entries are reserved for a future revision."""
        with pytest.raises(ValueError, match="not currently supported"):
            HarnessProfileConfig(excluded_middleware=frozenset({"deepagents.middleware.async_subagents:AsyncSubAgentMiddleware"}))

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

    @pytest.mark.parametrize(
        "overrides",
        [
            {"ls": 42},
            {1: "desc"},
        ],
        ids=["non_string_value", "non_string_key"],
    )
    def test_to_dict_validates_tool_description_types(self, overrides: object) -> None:
        """Both non-string keys and values in `tool_description_overrides` raise at serialize time."""
        config = HarnessProfileConfig(
            tool_description_overrides=overrides  # ty: ignore[invalid-argument-type]  # test fixture types
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

    def test_from_harness_profile_rejects_unaliased_class_entries(self) -> None:
        """Class-form entries without a `serialized_name` alias cannot be serialized."""
        profile = HarnessProfile(excluded_middleware=frozenset({AsyncSubAgentMiddleware}))
        with pytest.raises(ValueError, match="serialized_name"):
            HarnessProfileConfig.from_harness_profile(profile)

    def test_from_harness_profile_prefers_public_alias_for_summarization(self) -> None:
        profile = HarnessProfile(excluded_middleware=frozenset({_DeepAgentsSummarizationMiddleware}))
        assert HarnessProfileConfig.from_harness_profile(profile) == HarnessProfileConfig(excluded_middleware=frozenset({"SummarizationMiddleware"}))

    def test_from_harness_profile_rejects_non_empty_extra_middleware(self) -> None:
        class _Fake:
            pass

        profile = HarnessProfile(extra_middleware=(_Fake.__new__(_Fake),))
        with pytest.raises(ValueError, match="extra_middleware"):
            HarnessProfileConfig.from_harness_profile(profile)


class TestExcludedMiddlewareGrammar:
    """Grammar-level validation runs at `HarnessProfile[Config]` construction."""

    @pytest.mark.parametrize("entry", ["", " ", "\t"])
    def test_rejects_empty_or_whitespace_entries(self, entry: str) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            HarnessProfileConfig(excluded_middleware=frozenset({entry}))
        with pytest.raises(ValueError, match="non-empty"):
            HarnessProfile(excluded_middleware=frozenset({entry}))

    @pytest.mark.parametrize(
        "entry",
        ["a:b", "a:b:c", ":Foo", "deepagents:", "deepagents.middleware.async_subagents:AsyncSubAgentMiddleware"],
    )
    def test_rejects_colon_containing_entries(self, entry: str) -> None:
        """Class-path (`module:Class`) entries are reserved for a future revision."""
        with pytest.raises(ValueError, match="not currently supported"):
            HarnessProfileConfig(excluded_middleware=frozenset({entry}))
        with pytest.raises(ValueError, match="not currently supported"):
            HarnessProfile(excluded_middleware=frozenset({entry}))

    def test_rejects_underscore_prefixed_names(self) -> None:
        with pytest.raises(ValueError, match="cannot start with"):
            HarnessProfileConfig(excluded_middleware=frozenset({"_PrivateMiddleware"}))

    def test_accepts_plain_public_name(self) -> None:
        config = HarnessProfileConfig(excluded_middleware=frozenset({"PublicStubMiddleware"}))
        assert "PublicStubMiddleware" in config.excluded_middleware


class TestFromHarnessProfileRuntimeOnlyRejection:
    """`from_harness_profile` rejects both sequence and factory forms of `extra_middleware`."""

    def test_rejects_extra_middleware_factory(self) -> None:
        """Factory-form `extra_middleware` is runtime-only — serialization must fail loudly."""

        def factory() -> list[AsyncSubAgentMiddleware]:
            return []

        profile = HarnessProfile(extra_middleware=factory)
        with pytest.raises(ValueError, match="extra_middleware"):
            HarnessProfileConfig.from_harness_profile(profile)


class TestRuntimeRoundTrip:
    """Full `HarnessProfile → Config → HarnessProfile` round-trip.

    Class entries are only serializable when they expose a `serialized_name`
    alias; the alias path is what this round-trip exercises.
    """

    def test_round_trip_preserves_string_entries(self) -> None:
        profile = HarnessProfile(
            system_prompt_suffix="Respond briefly.",
            excluded_tools=frozenset({"execute"}),
            excluded_middleware=frozenset({"PublicStubMiddleware"}),
        )
        round_tripped = HarnessProfileConfig.from_harness_profile(profile).to_harness_profile()
        assert round_tripped == profile

    def test_round_trip_public_alias_stays_a_string(self) -> None:
        """`_DeepAgentsSummarizationMiddleware` serializes to its `"SummarizationMiddleware"` alias.

        The string does not re-resolve to a class on the way back — alias
        round-trips are intentionally stable as strings since the alias
        points at a private impl.
        """
        profile = HarnessProfile(excluded_middleware=frozenset({_DeepAgentsSummarizationMiddleware}))
        config = HarnessProfileConfig.from_harness_profile(profile)
        assert config.excluded_middleware == frozenset({"SummarizationMiddleware"})
        round_tripped = config.to_harness_profile()
        assert round_tripped.excluded_middleware == frozenset({"SummarizationMiddleware"})


class TestSerializedNameDrift:
    """Drift guard for the `serialized_name` ClassVar convention."""

    def test_summarization_serialized_name_matches_runtime_name(self) -> None:
        """`serialized_name` must equal `.name` so string-form exclusion matches the alias."""
        instance = _DeepAgentsSummarizationMiddleware.__new__(_DeepAgentsSummarizationMiddleware)
        assert _DeepAgentsSummarizationMiddleware.serialized_name == instance.name


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


class TestApplyProfilePrompt:
    """Tests for `_apply_profile_prompt`.

    The helper drives prompt assembly for the main agent, declarative
    subagents, and the auto-added GP subagent. These tests pin down the
    layering rules so a refactor of any of those call sites can't silently
    diverge from the documented semantics.
    """

    def test_empty_profile_returns_base_unchanged(self) -> None:
        from deepagents.profiles.harness.harness_profiles import _apply_profile_prompt  # noqa: PLC0415

        assert _apply_profile_prompt(HarnessProfile(), "base") == "base"

    def test_base_system_prompt_replaces_base(self) -> None:
        from deepagents.profiles.harness.harness_profiles import _apply_profile_prompt  # noqa: PLC0415

        profile = HarnessProfile(base_system_prompt="custom base")
        assert _apply_profile_prompt(profile, "ignored") == "custom base"

    def test_system_prompt_suffix_appended_to_base(self) -> None:
        from deepagents.profiles.harness.harness_profiles import _apply_profile_prompt  # noqa: PLC0415

        profile = HarnessProfile(system_prompt_suffix="suffix")
        assert _apply_profile_prompt(profile, "base") == "base\n\nsuffix"

    def test_base_and_suffix_combine(self) -> None:
        from deepagents.profiles.harness.harness_profiles import _apply_profile_prompt  # noqa: PLC0415

        profile = HarnessProfile(
            base_system_prompt="custom base",
            system_prompt_suffix="suffix",
        )
        assert _apply_profile_prompt(profile, "ignored") == "custom base\n\nsuffix"

    def test_empty_string_base_replaces_base(self) -> None:
        """`""` is distinct from `None` — explicitly empty replaces."""
        from deepagents.profiles.harness.harness_profiles import _apply_profile_prompt  # noqa: PLC0415

        profile = HarnessProfile(base_system_prompt="")
        assert _apply_profile_prompt(profile, "ignored") == ""

    def test_empty_string_suffix_still_appended(self) -> None:
        """An explicit empty suffix is appended (with separator) — distinct from `None`."""
        from deepagents.profiles.harness.harness_profiles import _apply_profile_prompt  # noqa: PLC0415

        profile = HarnessProfile(system_prompt_suffix="")
        assert _apply_profile_prompt(profile, "base") == "base\n\n"


class TestMaterializeExtraMiddleware:
    """Tests for `HarnessProfile.materialize_extra_middleware`."""

    def test_empty_profile_returns_empty_list(self) -> None:
        assert HarnessProfile().materialize_extra_middleware() == []

    def test_static_sequence_returned_as_list(self) -> None:
        sentinel = MagicMock()
        profile = HarnessProfile(extra_middleware=(sentinel,))
        assert profile.materialize_extra_middleware() == [sentinel]

    def test_callable_factory_is_invoked(self) -> None:
        sentinel = MagicMock()
        factory = MagicMock(return_value=[sentinel])
        profile = HarnessProfile(extra_middleware=factory)
        result = profile.materialize_extra_middleware()
        factory.assert_called_once()
        assert result == [sentinel]

    def test_returns_fresh_list_each_call(self) -> None:
        sentinel = MagicMock()
        profile = HarnessProfile(extra_middleware=(sentinel,))
        a = profile.materialize_extra_middleware()
        b = profile.materialize_extra_middleware()
        assert a == b
        assert a is not b


class TestMiniMaxBuiltinProfile:
    """The built-in MiniMax profile resolves and shapes the stack as intended."""

    def test_all_specs_resolve_with_expected_shape(self) -> None:
        _ensure_harness_profiles_loaded()
        for spec in _MINIMAX_MODEL_SPECS:
            profile = _get_harness_profile(spec)
            assert profile is not None, spec
            # write_todos stays (no middleware stripped).
            assert "TodoListMiddleware" not in profile.excluded_middleware, spec
            # V4: track_and_verify + report_back prompt framework ...
            assert profile.system_prompt_suffix is not None, spec
            assert "<track_and_verify>" in profile.system_prompt_suffix, spec
            assert "<report_back>" in profile.system_prompt_suffix, spec
            # ... paired with the reasoning-gate controller.
            mws = profile.materialize_extra_middleware()
            assert any(
                type(m).__name__ == "ReasoningGateMiddleware" for m in mws
            ), spec

    def test_unprofiled_spec_is_unaffected(self) -> None:
        _ensure_harness_profiles_loaded()
        # A model with no harness profile resolves to no profile at all.
        assert _get_harness_profile("openai:gpt-4.1") is None

    def test_agent_builds_with_profile(self) -> None:
        """Building the agent resolves the controller's state schema (no network).

        Guards against schema-resolution failures (e.g. a type used only under
        TYPE_CHECKING that get_type_hints can't evaluate at build time).
        """

        class _FakeMiniMax(GenericFakeChatModel):
            def _get_ls_params(self, *args: object, **kwargs: object) -> dict[str, str]:
                return {"ls_provider": "openrouter", "ls_model_name": "minimax/minimax-m3"}

        model = _FakeMiniMax(messages=itertools.cycle([AIMessage(content="ok")]))
        object.__setattr__(model, "model_name", "minimax/minimax-m3")
        agent = create_deep_agent(model=model)  # assembles graph -> resolves state schemas
        assert agent is not None


class TestKimiBuiltinProfile:
    """The Kimi profile mirrors MiniMax: same suffix, no tool changes."""

    def test_all_specs_resolve_and_share_minimax_suffix(self) -> None:
        _ensure_harness_profiles_loaded()
        for spec in _KIMI_MODEL_SPECS:
            profile = _get_harness_profile(spec)
            assert profile is not None, spec
            assert "TodoListMiddleware" not in profile.excluded_middleware, spec
            assert profile.system_prompt_suffix == _SYSTEM_PROMPT_SUFFIX, spec


class TestGlmBuiltinProfile:
    """The GLM profile mirrors MiniMax: same suffix, no tool changes."""

    def test_all_specs_resolve_and_share_minimax_suffix(self) -> None:
        _ensure_harness_profiles_loaded()
        for spec in _GLM_MODEL_SPECS:
            profile = _get_harness_profile(spec)
            assert profile is not None, spec
            assert "TodoListMiddleware" not in profile.excluded_middleware, spec
            assert profile.system_prompt_suffix == _SYSTEM_PROMPT_SUFFIX, spec


class TestPreCompletionVerificationMiddleware:
    """Unit-level logic for the pre-completion verification hook."""

    def test_forces_pass_on_tool_using_turn(self) -> None:
        mw = PreCompletionVerificationMiddleware()
        state = {
            "messages": [
                HumanMessage("do X"),
                AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}]),
            ],
            "_precompletion_baseline": 0,
        }
        out = mw.after_agent(state, None)
        assert out is not None
        assert out["jump_to"] == "model"
        assert out["_precompletion_verified"] is True
        assert out["messages"]

    def test_fires_at_most_once(self) -> None:
        mw = PreCompletionVerificationMiddleware()
        state = {
            "messages": [HumanMessage("do X"), AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])],
            "_precompletion_baseline": 0,
            "_precompletion_verified": True,
        }
        assert mw.after_agent(state, None) is None

    def test_skips_pure_conversational_turn(self) -> None:
        mw = PreCompletionVerificationMiddleware()
        state = {
            "messages": [HumanMessage("hi"), AIMessage(content="hello")],
            "_precompletion_baseline": 0,
        }
        out = mw.after_agent(state, None)
        assert out == {"_precompletion_verified": True}


class TestReasoningGateMiddleware:
    """Gate logic for the conditional rubric controller (model stubbed)."""

    def _build(self, verdict: str) -> ReasoningGateMiddleware:
        fake = GenericFakeChatModel(messages=itertools.cycle([AIMessage(content=verdict)]))
        mw = ReasoningGateMiddleware(grader_model=fake)
        mw._ensure_ready()  # resolves _model=fake + builds the real internal rubric
        mw._rubric = MagicMock()  # stub grading; these tests only exercise the gate
        mw._rubric.after_agent.return_value = {}
        return mw

    def test_simple_turn_skips_grading(self) -> None:
        mw = self._build("SIMPLE")
        state = {"messages": [HumanMessage("what's your name?")], "_gate_baseline": 1}
        assert mw.after_agent(state, None) is None
        mw._rubric.after_agent.assert_not_called()

    def test_complex_turn_grades(self) -> None:
        mw = self._build("COMPLEX")
        state = {
            "messages": [
                HumanMessage("cancel my flight and rebook to JFK"),
                AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}]),
            ],
            "_gate_baseline": 1,
        }
        out = mw.after_agent(state, None)
        mw._rubric.after_agent.assert_called_once()
        assert out is not None
        assert "rubric" in out  # fresh-run state merged in

    def test_in_progress_continues_without_classifying(self) -> None:
        # Classifier would say SIMPLE, but an active grade loop must continue.
        mw = self._build("SIMPLE")
        state = {
            "messages": [HumanMessage("hi")],
            "_gate_baseline": 1,
            "_rubric_status": "needs_revision",
        }
        mw.after_agent(state, None)
        mw._rubric.after_agent.assert_called_once()

    def test_build_rubric_embeds_system_prompt(self) -> None:
        mw = ReasoningGateMiddleware(grader_model="unused-no-resolve")
        state = {
            "messages": [
                SystemMessage(content="POLICY: basic economy is non-refundable."),
                HumanMessage("cancel my flight"),
            ]
        }
        rubric = mw._build_rubric(state)
        assert "Requirement coverage" in rubric
        assert "POLICY: basic economy is non-refundable." in rubric
        assert "<operating_rules>" in rubric
