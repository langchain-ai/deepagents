"""Tests for deepagents._models helpers."""

import os
from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from deepagents._models import (
    _string_value,
    get_model_identifier,
    get_model_provider,
    model_matches_spec,
    resolve_model,
)
from deepagents.profiles import (
    _HARNESS_PROFILES,
    _OPENROUTER_APP_TITLE,
    _OPENROUTER_APP_URL,
    OPENROUTER_MIN_VERSION,
    _get_harness_profile,
    _HarnessProfile,
    _merge_profiles,
    _openrouter_attribution_kwargs,
    _register_harness_profile,
    check_openrouter_version,
)


def _make_model(dump: dict) -> MagicMock:
    """Create a mock BaseChatModel with a given model_dump return."""
    model = MagicMock(spec=BaseChatModel)
    model.model_dump.return_value = dump
    return model


class TestResolveModel:
    """Tests for resolve_model."""

    def test_passthrough_when_already_model(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        assert resolve_model(model) is model

    def test_openai_prefix_uses_responses_api(self) -> None:
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            result = resolve_model("openai:gpt-5")

        mock.assert_called_once_with("openai:gpt-5", use_responses_api=True)
        assert result is mock.return_value

    def test_openrouter_prefix_sets_attribution(self) -> None:
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            result = resolve_model("openrouter:anthropic/claude-sonnet-4-6")

        mock.assert_called_once_with(
            "openrouter:anthropic/claude-sonnet-4-6",
            app_url=_OPENROUTER_APP_URL,
            app_title=_OPENROUTER_APP_TITLE,
        )
        assert result is mock.return_value

    def test_openrouter_env_var_overrides_app_url(self) -> None:
        env = {"OPENROUTER_APP_URL": "https://custom.app"}
        with (
            patch("deepagents._models.init_chat_model") as mock,
            patch.dict("os.environ", env),
        ):
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openrouter:anthropic/claude-sonnet-4-6")

        _, kwargs = mock.call_args
        assert "app_url" not in kwargs
        assert kwargs["app_title"] == _OPENROUTER_APP_TITLE

    def test_openrouter_env_var_overrides_app_title(self) -> None:
        env = {"OPENROUTER_APP_TITLE": "My Custom App"}
        with (
            patch("deepagents._models.init_chat_model") as mock,
            patch.dict("os.environ", env),
        ):
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openrouter:anthropic/claude-sonnet-4-6")

        _, kwargs = mock.call_args
        assert kwargs["app_url"] == _OPENROUTER_APP_URL
        assert "app_title" not in kwargs

    def test_openrouter_env_vars_override_both(self) -> None:
        env = {
            "OPENROUTER_APP_URL": "https://custom.app",
            "OPENROUTER_APP_TITLE": "My Custom App",
        }
        with (
            patch("deepagents._models.init_chat_model") as mock,
            patch.dict("os.environ", env),
        ):
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openrouter:anthropic/claude-sonnet-4-6")

        mock.assert_called_once_with("openrouter:anthropic/claude-sonnet-4-6")

    def test_non_openai_string(self) -> None:
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            result = resolve_model("anthropic:claude-sonnet-4-6")

        mock.assert_called_once_with("anthropic:claude-sonnet-4-6")
        assert result is mock.return_value


class TestGetModelIdentifier:
    """Tests for get_model_identifier."""

    def test_returns_model_name(self) -> None:
        model = _make_model({"model_name": "gpt-5", "model": "something-else"})
        assert get_model_identifier(model) == "gpt-5"

    def test_falls_back_to_model(self) -> None:
        model = _make_model({"model": "claude-sonnet-4-6"})
        assert get_model_identifier(model) == "claude-sonnet-4-6"

    def test_returns_none_when_missing(self) -> None:
        model = _make_model({})
        assert get_model_identifier(model) is None

    def test_skips_empty_model_name(self) -> None:
        model = _make_model({"model_name": "", "model": "fallback"})
        assert get_model_identifier(model) == "fallback"

    def test_skips_non_string_model_name(self) -> None:
        model = _make_model({"model_name": 123, "model": "real-name"})
        assert get_model_identifier(model) == "real-name"


class TestGetModelProvider:
    """Tests for get_model_provider."""

    def test_returns_provider_from_ls_params(self) -> None:
        model = _make_model({})
        model._get_ls_params = MagicMock(return_value={"ls_provider": "anthropic"})
        assert get_model_provider(model) == "anthropic"

    def test_returns_none_when_no_ls_provider(self) -> None:
        model = _make_model({})
        model._get_ls_params = MagicMock(return_value={})
        assert get_model_provider(model) is None

    def test_returns_none_when_ls_provider_empty(self) -> None:
        model = _make_model({})
        model._get_ls_params = MagicMock(return_value={"ls_provider": ""})
        assert get_model_provider(model) is None

    def test_returns_none_when_get_ls_params_raises(self) -> None:
        model = _make_model({})
        model._get_ls_params = MagicMock(side_effect=TypeError("unexpected"))
        assert get_model_provider(model) is None


class TestModelMatchesSpec:
    """Tests for model_matches_spec."""

    def test_exact_match(self) -> None:
        model = _make_model({"model_name": "claude-sonnet-4-6"})
        assert model_matches_spec(model, "claude-sonnet-4-6") is True

    def test_provider_prefixed_match(self) -> None:
        model = _make_model({"model_name": "claude-sonnet-4-6"})
        assert model_matches_spec(model, "anthropic:claude-sonnet-4-6") is True

    def test_no_match(self) -> None:
        model = _make_model({"model_name": "claude-sonnet-4-6"})
        assert model_matches_spec(model, "openai:gpt-5") is False

    def test_none_identifier_returns_false(self) -> None:
        model = _make_model({})
        assert model_matches_spec(model, "anything") is False

    def test_bare_spec_without_colon_no_false_positive(self) -> None:
        model = _make_model({"model_name": "gpt-5"})
        assert model_matches_spec(model, "gpt-4o") is False


class TestCheckOpenRouterVersion:
    """Tests for check_openrouter_version."""

    def test_passes_when_not_installed(self) -> None:
        with patch(
            "deepagents.profiles._openrouter.pkg_version",
            side_effect=PackageNotFoundError("langchain-openrouter"),
        ):
            check_openrouter_version()  # should not raise

    def test_passes_when_version_sufficient(self) -> None:
        with patch(
            "deepagents.profiles._openrouter.pkg_version",
            return_value=OPENROUTER_MIN_VERSION,
        ):
            check_openrouter_version()  # should not raise

    def test_passes_when_version_above_minimum(self) -> None:
        with patch("deepagents.profiles._openrouter.pkg_version", return_value="99.0.0"):
            check_openrouter_version()  # should not raise

    def test_raises_when_version_too_old(self) -> None:
        with (
            patch("deepagents.profiles._openrouter.pkg_version", return_value="0.0.1"),
            pytest.raises(ImportError, match="langchain-openrouter>="),
        ):
            check_openrouter_version()

    def test_skips_check_for_invalid_version(self) -> None:
        with patch("deepagents.profiles._openrouter.pkg_version", return_value="not-a-version"):
            check_openrouter_version()  # should not raise

    def test_resolve_model_calls_check(self) -> None:
        with (
            patch("deepagents.profiles._openrouter.check_openrouter_version") as mock_check,
            patch("deepagents._models.init_chat_model") as mock_init,
        ):
            mock_init.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openrouter:anthropic/claude-sonnet-4-6")

        mock_check.assert_called_once()

    def test_resolve_model_skips_check_for_non_openrouter(self) -> None:
        with (
            patch("deepagents.profiles._openrouter.check_openrouter_version") as mock_check,
            patch("deepagents._models.init_chat_model") as mock_init,
        ):
            mock_init.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("anthropic:claude-sonnet-4-6")

        mock_check.assert_not_called()


class TestOpenRouterAttributionKwargs:
    """Tests for _openrouter_attribution_kwargs."""

    def test_defaults_when_no_env(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            # Ensure the env vars are not set
            os.environ.pop("OPENROUTER_APP_URL", None)
            os.environ.pop("OPENROUTER_APP_TITLE", None)
            result = _openrouter_attribution_kwargs()

        assert result == {
            "app_url": _OPENROUTER_APP_URL,
            "app_title": _OPENROUTER_APP_TITLE,
        }

    def test_omits_app_url_when_env_set(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_APP_URL": "https://example.com"}):
            result = _openrouter_attribution_kwargs()

        assert "app_url" not in result
        assert result["app_title"] == _OPENROUTER_APP_TITLE

    def test_omits_app_title_when_env_set(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_APP_TITLE": "Custom"}):
            result = _openrouter_attribution_kwargs()

        assert result["app_url"] == _OPENROUTER_APP_URL
        assert "app_title" not in result

    def test_empty_when_both_env_set(self) -> None:
        env = {
            "OPENROUTER_APP_URL": "https://example.com",
            "OPENROUTER_APP_TITLE": "Custom",
        }
        with patch.dict("os.environ", env):
            result = _openrouter_attribution_kwargs()

        assert result == {}


class TestStringValue:
    """Tests for _string_value."""

    def test_present(self) -> None:
        assert _string_value({"key": "val"}, "key") == "val"

    def test_missing(self) -> None:
        assert _string_value({}, "key") is None

    def test_empty(self) -> None:
        assert _string_value({"key": ""}, "key") is None

    def test_non_string(self) -> None:
        assert _string_value({"key": 42}, "key") is None


class TestHarnessProfile:
    """Tests for the _HarnessProfile dataclass."""

    def test_defaults_are_empty(self) -> None:
        profile = _HarnessProfile()
        assert profile.init_kwargs == {}
        assert profile.pre_init is None
        assert profile.init_kwargs_factory is None
        assert profile.base_system_prompt is None
        assert profile.system_prompt_suffix is None
        assert profile.tool_description_overrides == {}
        assert profile.tool_aliases == {}
        assert profile.excluded_tools == frozenset()
        assert profile.extra_middleware == ()

    def test_frozen(self) -> None:
        profile = _HarnessProfile()
        with pytest.raises(AttributeError):
            profile.system_prompt_suffix = "nope"  # type: ignore[misc]


class TestHarnessProfileRegistry:
    """Tests for _register_harness_profile / _get_harness_profile."""

    def test_register_and_retrieve_by_provider(self) -> None:
        profile = _HarnessProfile(init_kwargs={"temperature": 0})
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile("test_provider", profile)
            assert _get_harness_profile("test_provider:some-model") is profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_exact_model_match_merges_with_provider(self) -> None:
        base_profile = _HarnessProfile(init_kwargs={"a": 1})
        model_profile = _HarnessProfile(init_kwargs={"b": 2})
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile("test_prov", base_profile)
            _register_harness_profile("test_prov:special-model", model_profile)
            merged = _get_harness_profile("test_prov:special-model")
            # Both provider and model kwargs are present
            assert merged.init_kwargs == {"a": 1, "b": 2}
            # Provider-only lookup still returns the base profile directly
            assert _get_harness_profile("test_prov:other-model") is base_profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_returns_empty_default_for_unknown(self) -> None:
        profile = _get_harness_profile("nonexistent:model")
        assert profile == _HarnessProfile()

    def test_bare_model_name_without_colon(self) -> None:
        profile = _get_harness_profile("claude-sonnet-4-6")
        assert profile == _HarnessProfile()


class TestMergeProfiles:
    """Tests for _merge_profiles layering behavior."""

    def test_init_kwargs_merged(self) -> None:
        base = _HarnessProfile(init_kwargs={"a": 1, "shared": "base"})
        override = _HarnessProfile(init_kwargs={"b": 2, "shared": "override"})
        merged = _merge_profiles(base, override)
        assert merged.init_kwargs == {"a": 1, "b": 2, "shared": "override"}

    def test_pre_init_chained(self) -> None:
        calls: list[str] = []
        base = _HarnessProfile(pre_init=lambda s: calls.append(f"base:{s}"))
        override = _HarnessProfile(pre_init=lambda s: calls.append(f"override:{s}"))
        merged = _merge_profiles(base, override)
        assert merged.pre_init is not None
        merged.pre_init("spec")
        assert calls == ["base:spec", "override:spec"]

    def test_pre_init_base_only(self) -> None:
        called = False

        def base_fn(_s: str) -> None:
            nonlocal called
            called = True

        base = _HarnessProfile(pre_init=base_fn)
        override = _HarnessProfile()
        merged = _merge_profiles(base, override)
        assert merged.pre_init is not None
        merged.pre_init("x")
        assert called

    def test_pre_init_override_only(self) -> None:
        called = False

        def over_fn(_s: str) -> None:
            nonlocal called
            called = True

        base = _HarnessProfile()
        override = _HarnessProfile(pre_init=over_fn)
        merged = _merge_profiles(base, override)
        assert merged.pre_init is not None
        merged.pre_init("x")
        assert called

    def test_init_kwargs_factory_chained(self) -> None:
        base = _HarnessProfile(init_kwargs_factory=lambda: {"a": 1, "shared": "base"})
        override = _HarnessProfile(init_kwargs_factory=lambda: {"b": 2, "shared": "override"})
        merged = _merge_profiles(base, override)
        assert merged.init_kwargs_factory is not None
        assert merged.init_kwargs_factory() == {
            "a": 1,
            "b": 2,
            "shared": "override",
        }

    def test_base_system_prompt_override_wins(self) -> None:
        base = _HarnessProfile(base_system_prompt="base prompt")
        override = _HarnessProfile(base_system_prompt="override prompt")
        merged = _merge_profiles(base, override)
        assert merged.base_system_prompt == "override prompt"

    def test_base_system_prompt_inherits_from_base(self) -> None:
        base = _HarnessProfile(base_system_prompt="base prompt")
        override = _HarnessProfile()
        merged = _merge_profiles(base, override)
        assert merged.base_system_prompt == "base prompt"

    def test_base_system_prompt_neither_set_produces_none(self) -> None:
        merged = _merge_profiles(_HarnessProfile(), _HarnessProfile())
        assert merged.base_system_prompt is None

    def test_system_prompt_suffix_override_wins(self) -> None:
        base = _HarnessProfile(system_prompt_suffix="base suffix")
        override = _HarnessProfile(system_prompt_suffix="override suffix")
        merged = _merge_profiles(base, override)
        assert merged.system_prompt_suffix == "override suffix"

    def test_system_prompt_suffix_inherits_from_base(self) -> None:
        base = _HarnessProfile(system_prompt_suffix="base suffix")
        override = _HarnessProfile()
        merged = _merge_profiles(base, override)
        assert merged.system_prompt_suffix == "base suffix"

    def test_base_system_prompt_and_suffix_both_merge(self) -> None:
        base = _HarnessProfile(base_system_prompt="base prompt", system_prompt_suffix="base suffix")
        override = _HarnessProfile(base_system_prompt="override prompt")
        merged = _merge_profiles(base, override)
        assert merged.base_system_prompt == "override prompt"
        assert merged.system_prompt_suffix == "base suffix"

    def test_tool_description_overrides_merged(self) -> None:
        base = _HarnessProfile(tool_description_overrides={"t1": "base", "t2": "base"})
        override = _HarnessProfile(tool_description_overrides={"t2": "override"})
        merged = _merge_profiles(base, override)
        assert merged.tool_description_overrides == {
            "t1": "base",
            "t2": "override",
        }

    def test_tool_aliases_merged(self) -> None:
        base = _HarnessProfile(tool_aliases={"execute": "run", "ls": "list_dir"})
        override = _HarnessProfile(tool_aliases={"execute": "shell_command"})
        merged = _merge_profiles(base, override)
        assert merged.tool_aliases == {
            "execute": "shell_command",
            "ls": "list_dir",
        }

    def test_tool_aliases_inherits_from_base(self) -> None:
        base = _HarnessProfile(tool_aliases={"execute": "shell_command"})
        override = _HarnessProfile()
        merged = _merge_profiles(base, override)
        assert merged.tool_aliases == {"execute": "shell_command"}

    def test_excluded_tools_union(self) -> None:
        base = _HarnessProfile(excluded_tools=frozenset({"execute", "write_file"}))
        override = _HarnessProfile(excluded_tools=frozenset({"execute", "task"}))
        merged = _merge_profiles(base, override)
        assert merged.excluded_tools == frozenset({"execute", "write_file", "task"})

    def test_excluded_tools_base_only(self) -> None:
        base = _HarnessProfile(excluded_tools=frozenset({"execute"}))
        override = _HarnessProfile()
        merged = _merge_profiles(base, override)
        assert merged.excluded_tools == frozenset({"execute"})

    def test_excluded_tools_override_only(self) -> None:
        base = _HarnessProfile()
        override = _HarnessProfile(excluded_tools=frozenset({"task"}))
        merged = _merge_profiles(base, override)
        assert merged.excluded_tools == frozenset({"task"})

    def test_excluded_tools_both_empty(self) -> None:
        merged = _merge_profiles(_HarnessProfile(), _HarnessProfile())
        assert merged.excluded_tools == frozenset()

    def test_extra_middleware_concatenated(self) -> None:
        mw_a, mw_b = MagicMock(), MagicMock()
        base = _HarnessProfile(extra_middleware=[mw_a])
        override = _HarnessProfile(extra_middleware=[mw_b])
        merged = _merge_profiles(base, override)
        # Merged middleware is a factory since both sides had entries
        assert callable(merged.extra_middleware)
        result = merged.extra_middleware()
        assert list(result) == [mw_a, mw_b]

    def test_extra_middleware_callable_and_sequence(self) -> None:
        mw_a, mw_b = MagicMock(), MagicMock()
        base = _HarnessProfile(extra_middleware=lambda: [mw_a])
        override = _HarnessProfile(extra_middleware=[mw_b])
        merged = _merge_profiles(base, override)
        assert callable(merged.extra_middleware)
        result = merged.extra_middleware()
        assert list(result) == [mw_a, mw_b]

    def test_extra_middleware_inherits_from_base(self) -> None:
        mw = MagicMock()
        base = _HarnessProfile(extra_middleware=[mw])
        override = _HarnessProfile()
        merged = _merge_profiles(base, override)
        assert list(merged.extra_middleware) == [mw]


class TestProfileMergingEndToEnd:
    """End-to-end tests: exact-model profiles inherit provider defaults."""

    def test_openai_exact_model_inherits_responses_api(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile(
                "openai:o3-pro",
                _HarnessProfile(system_prompt_suffix="think harder"),
            )
            profile = _get_harness_profile("openai:o3-pro")
            assert profile.init_kwargs == {"use_responses_api": True}
            assert profile.system_prompt_suffix == "think harder"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_anthropic_exact_model_inherits_provider_profile(self) -> None:
        """Per-model Anthropic profile merges with the provider-level profile."""
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile(
                "anthropic:claude-sonnet-4-6-20250514",
                _HarnessProfile(system_prompt_suffix="be concise"),
            )
            profile = _get_harness_profile("anthropic:claude-sonnet-4-6-20250514")
            assert profile.system_prompt_suffix == "be concise"
            # AnthropicPromptCachingMiddleware is applied unconditionally in
            # graph.py, not via the profile, so extra_middleware should be empty.
            assert profile.extra_middleware == ()
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_exact_model_override_wins_for_init_kwargs(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile(
                "openai:o3-pro",
                _HarnessProfile(init_kwargs={"use_responses_api": False}),
            )
            profile = _get_harness_profile("openai:o3-pro")
            assert profile.init_kwargs == {"use_responses_api": False}
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_no_base_profile_returns_exact_unchanged(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            model_profile = _HarnessProfile(init_kwargs={"x": 1})
            _register_harness_profile("noprov:special", model_profile)
            assert _get_harness_profile("noprov:special") is model_profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestBuiltInProfiles:
    """Tests for the built-in provider profile registrations."""

    def test_openai_profile_sets_responses_api(self) -> None:
        profile = _get_harness_profile("openai:gpt-5")
        assert profile.init_kwargs == {"use_responses_api": True}

    def test_openrouter_profile_has_pre_init_and_factory(self) -> None:
        profile = _get_harness_profile("openrouter:anthropic/claude-sonnet-4-6")
        assert profile.pre_init is not None
        assert profile.init_kwargs_factory is not None

    def test_anthropic_returns_default_profile(self) -> None:
        """Anthropic has no registered profile; caching is unconditional in graph.py."""
        profile = _get_harness_profile("anthropic:claude-sonnet-4-6")
        assert profile == _HarnessProfile()

    @pytest.mark.parametrize("spec", ["openai:gpt-5.2-codex", "openai:gpt-5.3-codex"])
    def test_codex_profile_registered(self, spec: str) -> None:
        assert spec in _HARNESS_PROFILES

    @pytest.mark.parametrize("spec", ["openai:gpt-5.2-codex", "openai:gpt-5.3-codex"])
    def test_codex_profile_merges_with_openai_provider(self, spec: str) -> None:
        """Codex profile inherits use_responses_api from the OpenAI provider."""
        profile = _get_harness_profile(spec)
        assert profile.init_kwargs["use_responses_api"] is True
        assert profile.init_kwargs["reasoning_effort"] == "medium"

    @pytest.mark.parametrize("spec", ["openai:gpt-5.2-codex", "openai:gpt-5.3-codex"])
    def test_codex_profile_has_system_prompt_suffix(self, spec: str) -> None:
        profile = _get_harness_profile(spec)
        assert profile.system_prompt_suffix is not None
        assert "autonomous" in profile.system_prompt_suffix.lower()
        assert "parallel" in profile.system_prompt_suffix.lower()
        assert "bias to action" in profile.system_prompt_suffix.lower()

    @pytest.mark.parametrize("spec", ["openai:gpt-5.2-codex", "openai:gpt-5.3-codex"])
    def test_codex_profile_has_execute_tool_override(self, spec: str) -> None:
        profile = _get_harness_profile(spec)
        assert "execute" in profile.tool_description_overrides

    @pytest.mark.parametrize("spec", ["openai:gpt-5.2-codex", "openai:gpt-5.3-codex"])
    def test_codex_profile_does_not_replace_base_system_prompt(self, spec: str) -> None:
        profile = _get_harness_profile(spec)
        assert profile.base_system_prompt is None

    @pytest.mark.parametrize("spec", ["openai:gpt-5.2-codex", "openai:gpt-5.3-codex"])
    def test_codex_profile_has_tool_aliases(self, spec: str) -> None:
        profile = _get_harness_profile(spec)
        assert "execute" in profile.tool_aliases
        assert profile.tool_aliases["execute"] == "shell_command"
        assert "ls" in profile.tool_aliases
        assert profile.tool_aliases["ls"] == "list_dir"

    def test_other_openai_models_unaffected_by_codex_profile(self) -> None:
        """Non-Codex OpenAI models get the provider profile, not the Codex one."""
        profile = _get_harness_profile("openai:gpt-5")
        assert profile.init_kwargs == {"use_responses_api": True}
        assert profile.system_prompt_suffix is None


class TestResolveModelWithProfiles:
    """Tests for resolve_model using the profile registry."""

    def test_openai_uses_profile_init_kwargs(self) -> None:
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openai:gpt-5")

        mock.assert_called_once_with("openai:gpt-5", use_responses_api=True)

    def test_codex_merges_provider_and_model_kwargs(self) -> None:
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openai:gpt-5.3-codex")

        mock.assert_called_once_with(
            "openai:gpt-5.3-codex",
            use_responses_api=True,
            reasoning_effort="medium",
        )

    def test_openrouter_runs_pre_init_and_factory(self) -> None:
        with (
            patch("deepagents._models.init_chat_model") as mock,
            patch("deepagents.profiles._openrouter.check_openrouter_version") as mock_check,
        ):
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openrouter:anthropic/claude-sonnet-4-6")

        mock_check.assert_called_once()
        _, kwargs = mock.call_args
        assert "app_url" in kwargs or "app_title" in kwargs

    def test_unknown_provider_passes_no_extra_kwargs(self) -> None:
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("some_provider:some-model")

        mock.assert_called_once_with("some_provider:some-model")

    def test_custom_profile_kwargs_forwarded(self) -> None:
        profile = _HarnessProfile(init_kwargs={"custom_key": "custom_val"})
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile("customprov", profile)
            with patch("deepagents._models.init_chat_model") as mock:
                mock.return_value = MagicMock(spec=BaseChatModel)
                resolve_model("customprov:my-model")

            mock.assert_called_once_with("customprov:my-model", custom_key="custom_val")
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)
