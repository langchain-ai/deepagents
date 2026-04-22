"""Tests for deepagents._models helpers and internal profile registries."""

import logging
import os
from collections.abc import Iterator
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
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    ProviderProfile,
    register_harness_profile,
    register_provider_profile,
)
from deepagents.profiles.harness_profiles import (
    _HARNESS_PROFILES,
    _get_harness_profile,
    _merge_middleware,
    _merge_profiles,
)
from deepagents.profiles.provider._google_genai import (
    GOOGLE_GENAI_MIN_VERSION,
    check_google_genai_version,
)
from deepagents.profiles.provider._openrouter import (
    _OPENROUTER_APP_TITLE,
    _OPENROUTER_APP_URL,
    OPENROUTER_MIN_VERSION,
    _openrouter_attribution_kwargs,
    check_openrouter_version,
)
from deepagents.profiles.provider.provider_profiles import (
    _PROVIDER_PROFILES,
    _get_provider_profile,
    _merge_provider_profiles,
)


def _make_model(dump: dict) -> MagicMock:
    """Create a mock BaseChatModel with a given `model_dump` return value."""
    model = MagicMock(spec=BaseChatModel)
    model.model_dump.return_value = dump
    return model


class TestResolveModel:
    """Tests for `resolve_model`."""

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

    def test_unknown_provider_passes_no_extra_kwargs(self) -> None:
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            result = resolve_model("anthropic:claude-sonnet-4-6")

        mock.assert_called_once_with("anthropic:claude-sonnet-4-6")
        assert result is mock.return_value


class TestGetModelIdentifier:
    """Tests for `get_model_identifier`."""

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
    """Tests for `get_model_provider`."""

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
    """Tests for `model_matches_spec`."""

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
    """Tests for `check_openrouter_version`."""

    def test_passes_when_not_installed(self) -> None:
        with patch(
            "deepagents.profiles.provider._openrouter.pkg_version",
            side_effect=PackageNotFoundError("langchain-openrouter"),
        ):
            check_openrouter_version()

    def test_passes_when_version_sufficient(self) -> None:
        with patch(
            "deepagents.profiles.provider._openrouter.pkg_version",
            return_value=OPENROUTER_MIN_VERSION,
        ):
            check_openrouter_version()

    def test_passes_when_version_above_minimum(self) -> None:
        with patch("deepagents.profiles.provider._openrouter.pkg_version", return_value="99.0.0"):
            check_openrouter_version()

    def test_raises_when_version_too_old(self) -> None:
        with (
            patch("deepagents.profiles.provider._openrouter.pkg_version", return_value="0.0.1"),
            pytest.raises(ImportError, match="langchain-openrouter>="),
        ):
            check_openrouter_version()

    def test_skips_check_for_invalid_version(self) -> None:
        with patch("deepagents.profiles.provider._openrouter.pkg_version", return_value="not-a-version"):
            check_openrouter_version()

    def test_resolve_model_calls_check(self) -> None:
        with (
            patch("deepagents.profiles.provider._openrouter.check_openrouter_version") as mock_check,
            patch("deepagents._models.init_chat_model") as mock_init,
        ):
            mock_init.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openrouter:anthropic/claude-sonnet-4-6")

        mock_check.assert_called_once()

    def test_resolve_model_skips_check_for_non_openrouter(self) -> None:
        with (
            patch("deepagents.profiles.provider._openrouter.check_openrouter_version") as mock_check,
            patch("deepagents._models.init_chat_model") as mock_init,
        ):
            mock_init.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("anthropic:claude-sonnet-4-6")

        mock_check.assert_not_called()


class TestCheckGoogleGenaiVersion:
    """Tests for `check_google_genai_version`."""

    def test_passes_when_not_installed(self) -> None:
        with patch(
            "deepagents.profiles.provider._google_genai.pkg_version",
            side_effect=PackageNotFoundError("langchain-google-genai"),
        ):
            check_google_genai_version()

    def test_passes_when_version_sufficient(self) -> None:
        with patch(
            "deepagents.profiles.provider._google_genai.pkg_version",
            return_value=GOOGLE_GENAI_MIN_VERSION,
        ):
            check_google_genai_version()

    def test_passes_when_version_above_minimum(self) -> None:
        with patch("deepagents.profiles.provider._google_genai.pkg_version", return_value="99.0.0"):
            check_google_genai_version()

    def test_raises_when_version_too_old(self) -> None:
        with (
            patch("deepagents.profiles.provider._google_genai.pkg_version", return_value="0.0.1"),
            pytest.raises(ImportError, match="langchain-google-genai>="),
        ):
            check_google_genai_version()

    def test_skips_check_for_invalid_version(self) -> None:
        with patch("deepagents.profiles.provider._google_genai.pkg_version", return_value="not-a-version"):
            check_google_genai_version()

    def test_resolve_model_calls_check(self) -> None:
        with (
            patch("deepagents.profiles.provider._google_genai.check_google_genai_version") as mock_check,
            patch("deepagents._models.init_chat_model") as mock_init,
        ):
            mock_init.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("google_genai:gemini-3.1-pro")

        mock_check.assert_called_once()

    def test_resolve_model_skips_check_for_non_google_genai(self) -> None:
        with (
            patch("deepagents.profiles.provider._google_genai.check_google_genai_version") as mock_check,
            patch("deepagents._models.init_chat_model") as mock_init,
        ):
            mock_init.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("anthropic:claude-sonnet-4-6")

        mock_check.assert_not_called()


class TestOpenRouterAttributionKwargs:
    """Tests for `_openrouter_attribution_kwargs`."""

    def test_defaults_when_no_env(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
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
    """Tests for `_string_value`."""

    def test_present(self) -> None:
        assert _string_value({"key": "val"}, "key") == "val"

    def test_missing(self) -> None:
        assert _string_value({}, "key") is None

    def test_empty(self) -> None:
        assert _string_value({"key": ""}, "key") is None

    def test_non_string(self) -> None:
        assert _string_value({"key": 42}, "key") is None


class TestProviderProfile:
    """Tests for `ProviderProfile`."""

    def test_defaults_are_empty(self) -> None:
        profile = ProviderProfile()
        assert profile.init_kwargs == {}
        assert profile.pre_init is None
        assert profile.init_kwargs_factory is None

    def test_frozen(self) -> None:
        profile = ProviderProfile()
        with pytest.raises(AttributeError):
            profile.pre_init = lambda _spec: None  # type: ignore[misc]


class TestProviderProfileRegistry:
    """Tests for provider-profile lookup and registration."""

    def test_register_and_retrieve_by_provider(self) -> None:
        profile = ProviderProfile(init_kwargs={"temperature": 0})
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("test_provider", profile)
            assert _get_provider_profile("test_provider:some-model") is profile
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_exact_model_match_merges_with_provider(self) -> None:
        base_profile = ProviderProfile(init_kwargs={"a": 1})
        model_profile = ProviderProfile(init_kwargs={"b": 2})
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("test_prov", base_profile)
            register_provider_profile("test_prov:special-model", model_profile)
            merged = _get_provider_profile("test_prov:special-model")
            assert merged.init_kwargs == {"a": 1, "b": 2}
            assert _get_provider_profile("test_prov:other-model") is base_profile
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_returns_none_for_unknown(self) -> None:
        assert _get_provider_profile("nonexistent:model") is None

    def test_bare_model_name_without_colon_returns_none(self) -> None:
        assert _get_provider_profile("claude-sonnet-4-6") is None

    def test_empty_spec_returns_none(self) -> None:
        """Empty spec has no colon and no exact match."""
        assert _get_provider_profile("") is None

    def test_exact_miss_falls_back_to_provider(self) -> None:
        """A typo'd model spec should fall back to the provider profile, not None."""
        base = ProviderProfile(init_kwargs={"a": 1})
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("fbprov", base)
            assert _get_provider_profile("fbprov:missing-model") is base
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)


class TestRegisterProviderProfileAdditive:
    """Tests for additive-merge semantics of `register_provider_profile`.

    Re-registering under an existing key must merge the incoming profile on top
    of the existing one, not replace it. User kwargs are additive with built-in
    defaults; explicit conflicts let the user win.
    """

    def test_layers_onto_existing_registration(self) -> None:
        """Registering twice under the same key merges without clobbering."""
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("layered_prov", ProviderProfile(init_kwargs={"a": 1}))
            register_provider_profile("layered_prov", ProviderProfile(init_kwargs={"b": 2}))
            profile = _get_provider_profile("layered_prov")
            assert profile.init_kwargs == {"a": 1, "b": 2}
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_later_registration_wins_on_key_collision(self) -> None:
        """When both registrations set the same key, the later value wins."""
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("coll_prov", ProviderProfile(init_kwargs={"shared": "first"}))
            register_provider_profile("coll_prov", ProviderProfile(init_kwargs={"shared": "second"}))
            profile = _get_provider_profile("coll_prov")
            assert profile.init_kwargs == {"shared": "second"}
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_user_layering_preserves_built_in_openai_default(self) -> None:
        """User layering onto `"openai"` keeps the built-in `use_responses_api=True`."""
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("openai", ProviderProfile(init_kwargs={"temperature": 0}))
            profile = _get_provider_profile("openai:gpt-5")
            assert profile.init_kwargs == {"use_responses_api": True, "temperature": 0}
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_user_can_override_built_in_openai_default(self) -> None:
        """Explicit user value for a built-in key wins."""
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile(
                "openai",
                ProviderProfile(init_kwargs={"use_responses_api": False}),
            )
            profile = _get_provider_profile("openai:gpt-5")
            assert profile.init_kwargs == {"use_responses_api": False}
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_pre_init_chains_on_re_registration(self) -> None:
        """Re-registering a profile with a `pre_init` chains with the existing one."""
        original = dict(_PROVIDER_PROFILES)
        calls: list[str] = []
        try:
            register_provider_profile(
                "chain_prov",
                ProviderProfile(pre_init=lambda spec: calls.append(f"first:{spec}")),
            )
            register_provider_profile(
                "chain_prov",
                ProviderProfile(pre_init=lambda spec: calls.append(f"second:{spec}")),
            )
            profile = _get_provider_profile("chain_prov")
            assert profile.pre_init is not None
            profile.pre_init("spec")
            assert calls == ["first:spec", "second:spec"]
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_novel_provider_registers_cleanly(self) -> None:
        """A provider key that has no existing registration is stored as-is."""
        original = dict(_PROVIDER_PROFILES)
        try:
            profile = ProviderProfile(init_kwargs={"unique_key": True})
            register_provider_profile("novelprov", profile)
            assert _get_provider_profile("novelprov") is profile
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)


class TestMergeProviderProfiles:
    """Tests for `_merge_provider_profiles`."""

    def test_init_kwargs_merged(self) -> None:
        base = ProviderProfile(init_kwargs={"a": 1, "shared": "base"})
        override = ProviderProfile(init_kwargs={"b": 2, "shared": "override"})
        merged = _merge_provider_profiles(base, override)
        assert merged.init_kwargs == {"a": 1, "b": 2, "shared": "override"}

    def test_pre_init_chained(self) -> None:
        calls: list[str] = []
        base = ProviderProfile(pre_init=lambda s: calls.append(f"base:{s}"))
        override = ProviderProfile(pre_init=lambda s: calls.append(f"override:{s}"))
        merged = _merge_provider_profiles(base, override)
        assert merged.pre_init is not None
        merged.pre_init("spec")
        assert calls == ["base:spec", "override:spec"]

    def test_pre_init_base_only(self) -> None:
        called = False

        def base_fn(_spec: str) -> None:
            nonlocal called
            called = True

        merged = _merge_provider_profiles(ProviderProfile(pre_init=base_fn), ProviderProfile())
        assert merged.pre_init is not None
        merged.pre_init("x")
        assert called

    def test_pre_init_override_only(self) -> None:
        called = False

        def override_fn(_spec: str) -> None:
            nonlocal called
            called = True

        merged = _merge_provider_profiles(ProviderProfile(), ProviderProfile(pre_init=override_fn))
        assert merged.pre_init is not None
        merged.pre_init("x")
        assert called

    def test_init_kwargs_factory_chained(self) -> None:
        base = ProviderProfile(init_kwargs_factory=lambda: {"a": 1, "shared": "base"})
        override = ProviderProfile(init_kwargs_factory=lambda: {"b": 2, "shared": "override"})
        merged = _merge_provider_profiles(base, override)
        assert merged.init_kwargs_factory is not None
        assert merged.init_kwargs_factory() == {
            "a": 1,
            "b": 2,
            "shared": "override",
        }


class TestHarnessProfile:
    """Tests for `HarnessProfile`."""

    def test_defaults_are_empty(self) -> None:
        profile = HarnessProfile()
        assert profile.base_system_prompt is None
        assert profile.system_prompt_suffix is None
        assert profile.tool_description_overrides == {}
        assert profile.excluded_tools == frozenset()
        assert profile.extra_middleware == ()
        assert profile.general_purpose_subagent is None

    def test_frozen(self) -> None:
        profile = HarnessProfile()
        with pytest.raises(AttributeError):
            profile.system_prompt_suffix = "nope"  # type: ignore[misc]


class TestHarnessProfileRegistry:
    """Tests for harness-profile lookup and registration."""

    def test_register_and_retrieve_by_provider(self) -> None:
        profile = HarnessProfile(system_prompt_suffix="provider suffix")
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile("test_provider", profile)
            assert _get_harness_profile("test_provider:some-model") is profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_exact_model_match_merges_with_provider(self) -> None:
        base_profile = HarnessProfile(system_prompt_suffix="provider suffix")
        model_profile = HarnessProfile(base_system_prompt="model base")
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile("test_prov", base_profile)
            register_harness_profile("test_prov:special-model", model_profile)
            merged = _get_harness_profile("test_prov:special-model")
            assert merged.base_system_prompt == "model base"
            assert merged.system_prompt_suffix == "provider suffix"
            assert _get_harness_profile("test_prov:other-model") is base_profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_returns_none_for_unknown(self) -> None:
        assert _get_harness_profile("nonexistent:model") is None

    def test_bare_model_name_without_colon_returns_none(self) -> None:
        assert _get_harness_profile("claude-sonnet-4-6") is None

    def test_exact_miss_falls_back_to_provider(self) -> None:
        """A typo'd spec should fall back to the provider profile, not None."""
        base = HarnessProfile(system_prompt_suffix="provider suffix")
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile("fbharness", base)
            assert _get_harness_profile("fbharness:missing-model") is base
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestRegisterHarnessProfileAdditive:
    """Tests for additive-merge semantics of `register_harness_profile`.

    Re-registering under an existing key merges the incoming profile on top of
    the existing one via `_merge_profiles`. This lets users layer settings
    onto built-ins without clobbering them.
    """

    def test_layers_onto_existing_registration(self) -> None:
        """Two registrations under the same key merge non-conflicting fields."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "layered_harness",
                HarnessProfile(system_prompt_suffix="first suffix"),
            )
            register_harness_profile(
                "layered_harness",
                HarnessProfile(tool_description_overrides={"task": "layered"}),
            )
            profile = _get_harness_profile("layered_harness")
            assert profile.system_prompt_suffix == "first suffix"
            assert profile.tool_description_overrides == {"task": "layered"}
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_later_registration_wins_on_key_collision(self) -> None:
        """Conflicting scalar fields resolve to the later registration."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "coll_harness",
                HarnessProfile(system_prompt_suffix="first"),
            )
            register_harness_profile(
                "coll_harness",
                HarnessProfile(system_prompt_suffix="second"),
            )
            profile = _get_harness_profile("coll_harness")
            assert profile.system_prompt_suffix == "second"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_excluded_tools_union_across_registrations(self) -> None:
        """Re-registering with new excluded tools unions with the existing set."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "union_harness",
                HarnessProfile(excluded_tools=frozenset({"execute"})),
            )
            register_harness_profile(
                "union_harness",
                HarnessProfile(excluded_tools=frozenset({"grep"})),
            )
            profile = _get_harness_profile("union_harness")
            assert profile.excluded_tools == frozenset({"execute", "grep"})
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_general_purpose_subagent_merges_fieldwise(self) -> None:
        """Re-registering with a partial `general_purpose_subagent` preserves unset fields."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "gp_harness",
                HarnessProfile(
                    general_purpose_subagent=GeneralPurposeSubagentProfile(description="original desc"),
                ),
            )
            register_harness_profile(
                "gp_harness",
                HarnessProfile(
                    general_purpose_subagent=GeneralPurposeSubagentProfile(system_prompt="new prompt"),
                ),
            )
            profile = _get_harness_profile("gp_harness")
            assert profile.general_purpose_subagent == GeneralPurposeSubagentProfile(
                description="original desc",
                system_prompt="new prompt",
            )
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_novel_key_registers_cleanly(self) -> None:
        """Registering under a new key stores the profile by identity."""
        original = dict(_HARNESS_PROFILES)
        try:
            profile = HarnessProfile(system_prompt_suffix="only one")
            register_harness_profile("novel_harness", profile)
            assert _get_harness_profile("novel_harness") is profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestMergeHarnessProfiles:
    """Tests for `_merge_profiles`."""

    def test_base_system_prompt_override_wins(self) -> None:
        base = HarnessProfile(base_system_prompt="base prompt")
        override = HarnessProfile(base_system_prompt="override prompt")
        merged = _merge_profiles(base, override)
        assert merged.base_system_prompt == "override prompt"

    def test_base_system_prompt_inherits_from_base(self) -> None:
        base = HarnessProfile(base_system_prompt="base prompt")
        merged = _merge_profiles(base, HarnessProfile())
        assert merged.base_system_prompt == "base prompt"

    def test_base_system_prompt_neither_set_produces_none(self) -> None:
        assert _merge_profiles(HarnessProfile(), HarnessProfile()).base_system_prompt is None

    def test_system_prompt_suffix_override_wins(self) -> None:
        base = HarnessProfile(system_prompt_suffix="base suffix")
        override = HarnessProfile(system_prompt_suffix="override suffix")
        merged = _merge_profiles(base, override)
        assert merged.system_prompt_suffix == "override suffix"

    def test_system_prompt_suffix_inherits_from_base(self) -> None:
        base = HarnessProfile(system_prompt_suffix="base suffix")
        merged = _merge_profiles(base, HarnessProfile())
        assert merged.system_prompt_suffix == "base suffix"

    def test_base_system_prompt_and_suffix_both_merge(self) -> None:
        base = HarnessProfile(
            base_system_prompt="base prompt",
            system_prompt_suffix="base suffix",
        )
        override = HarnessProfile(base_system_prompt="override prompt")
        merged = _merge_profiles(base, override)
        assert merged.base_system_prompt == "override prompt"
        assert merged.system_prompt_suffix == "base suffix"

    def test_tool_description_overrides_merged(self) -> None:
        base = HarnessProfile(tool_description_overrides={"t1": "base", "t2": "base"})
        override = HarnessProfile(tool_description_overrides={"t2": "override"})
        merged = _merge_profiles(base, override)
        assert merged.tool_description_overrides == {
            "t1": "base",
            "t2": "override",
        }

    def test_excluded_tools_union(self) -> None:
        base = HarnessProfile(excluded_tools=frozenset({"execute", "write_file"}))
        override = HarnessProfile(excluded_tools=frozenset({"execute", "task"}))
        merged = _merge_profiles(base, override)
        assert merged.excluded_tools == frozenset({"execute", "write_file", "task"})

    def test_extra_middleware_concatenated(self) -> None:
        mw_a, mw_b = MagicMock(), MagicMock()
        base = HarnessProfile(extra_middleware=[mw_a])
        override = HarnessProfile(extra_middleware=[mw_b])
        merged = _merge_profiles(base, override)
        assert callable(merged.extra_middleware)
        assert list(merged.extra_middleware()) == [mw_a, mw_b]

    def test_extra_middleware_callable_and_sequence(self) -> None:
        mw_a, mw_b = MagicMock(), MagicMock()
        base = HarnessProfile(extra_middleware=lambda: [mw_a])
        override = HarnessProfile(extra_middleware=[mw_b])
        merged = _merge_profiles(base, override)
        assert callable(merged.extra_middleware)
        assert list(merged.extra_middleware()) == [mw_a, mw_b]

    def test_extra_middleware_inherits_from_base(self) -> None:
        mw = MagicMock()
        base = HarnessProfile(extra_middleware=[mw])
        merged = _merge_profiles(base, HarnessProfile())
        assert list(merged.extra_middleware) == [mw]

    def test_general_purpose_subagent_merge_combines_fields(self) -> None:
        base = HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(
                description="base description",
            )
        )
        override = HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(
                system_prompt="override prompt",
            )
        )
        merged = _merge_profiles(base, override)
        assert merged.general_purpose_subagent == GeneralPurposeSubagentProfile(
            description="base description",
            system_prompt="override prompt",
        )

    def test_general_purpose_subagent_enabled_override_wins(self) -> None:
        base = HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=True))
        override = HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False))
        merged = _merge_profiles(base, override)
        assert merged.general_purpose_subagent == GeneralPurposeSubagentProfile(enabled=False)


class TestProfileMergingEndToEnd:
    """End-to-end tests for the split registries."""

    def test_openai_exact_model_inherits_provider_defaults(self) -> None:
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile(
                "openai:o3-pro",
                ProviderProfile(init_kwargs={"reasoning_effort": "high"}),
            )
            profile = _get_provider_profile("openai:o3-pro")
            assert profile.init_kwargs == {
                "use_responses_api": True,
                "reasoning_effort": "high",
            }
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_exact_harness_override_inherits_provider_harness_defaults(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "testprov",
                HarnessProfile(system_prompt_suffix="provider suffix"),
            )
            register_harness_profile(
                "testprov:special",
                HarnessProfile(base_system_prompt="model base"),
            )
            profile = _get_harness_profile("testprov:special")
            assert profile.base_system_prompt == "model base"
            assert profile.system_prompt_suffix == "provider suffix"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_no_base_profile_returns_exact_unchanged(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            model_profile = HarnessProfile(system_prompt_suffix="exact only")
            register_harness_profile("noprov:special", model_profile)
            assert _get_harness_profile("noprov:special") is model_profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestBuiltInProfiles:
    """Tests for built-in provider and harness registrations."""

    def test_openai_provider_profile_sets_responses_api(self) -> None:
        profile = _get_provider_profile("openai:gpt-5")
        assert profile.init_kwargs == {"use_responses_api": True}

    def test_openrouter_provider_profile_has_pre_init_and_factory(self) -> None:
        profile = _get_provider_profile("openrouter:anthropic/claude-sonnet-4-6")
        assert profile.pre_init is not None
        assert profile.init_kwargs_factory is not None

    def test_openai_has_no_built_in_harness_profile(self) -> None:
        assert _get_harness_profile("openai:gpt-5") is None


class TestProfilePluginLoader:
    """Tests for the `importlib.metadata` entry-point loader."""

    @pytest.fixture(autouse=True)
    def _isolate_loader_state(self) -> Iterator[None]:
        """Snapshot and restore loader globals plus both registries around every test.

        The real bootstrap runs once at `deepagents.profiles` import, so by
        the time this class executes, `_loaded` is `True` and the registries
        hold built-in entries. Tests here reset `_loaded` to re-exercise the
        loader with patched entry points; without this fixture they would
        leak frozen snapshots and mutated registry state into sibling tests
        — including `test_graph.TestHasAnyHarnessProfile`, which asserts
        `_has_any_harness_profile() is False` at start.
        """
        from deepagents.profiles import _builtin_profiles  # noqa: PLC0415

        saved_loaded = _builtin_profiles._loaded
        saved_snapshot = _builtin_profiles._BOOTSTRAP_HARNESS_KEYS
        saved_harness = dict(_HARNESS_PROFILES)
        saved_provider = dict(_PROVIDER_PROFILES)
        try:
            _builtin_profiles._loaded = False
            _builtin_profiles._BOOTSTRAP_HARNESS_KEYS = frozenset()
            yield
        finally:
            _builtin_profiles._loaded = saved_loaded
            _builtin_profiles._BOOTSTRAP_HARNESS_KEYS = saved_snapshot
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(saved_harness)
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(saved_provider)

    def test_iterates_both_entry_point_groups(self) -> None:
        """Loader must query both provider and harness entry-point groups."""
        from deepagents.profiles._builtin_profiles import (  # noqa: PLC0415
            _HARNESS_PROFILE_GROUP,
            _PROVIDER_PROFILE_GROUP,
            _ensure_builtin_profiles_loaded,
        )

        with patch(
            "deepagents.profiles._builtin_profiles.entry_points",
            return_value=[],
        ) as mock:
            _ensure_builtin_profiles_loaded()

        groups = {call.kwargs["group"] for call in mock.call_args_list}
        assert groups == {_PROVIDER_PROFILE_GROUP, _HARNESS_PROFILE_GROUP}

    def test_broken_plugin_logged_and_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """A plugin whose `load()` raises must not prevent sibling plugins from running."""
        from deepagents.profiles._builtin_profiles import (  # noqa: PLC0415
            _ensure_builtin_profiles_loaded,
        )

        good_called = MagicMock()
        broken = MagicMock()
        broken.name = "broken"
        broken.value = "nope:nope"
        broken.load.side_effect = ImportError("boom")
        good = MagicMock()
        good.name = "good"
        good.value = "mod:register"
        good.load.return_value = good_called

        def fake_entry_points(*, group: str) -> list[MagicMock]:
            if group == "deepagents.provider_profiles":
                return [broken, good]
            return []

        with (
            caplog.at_level(logging.WARNING, logger="deepagents.profiles._builtin_profiles"),
            patch(
                "deepagents.profiles._builtin_profiles.entry_points",
                side_effect=fake_entry_points,
            ),
        ):
            _ensure_builtin_profiles_loaded()

        good_called.assert_called_once_with()
        assert any("broken" in rec.message and "nope:nope" in rec.message for rec in caplog.records)

    def test_non_callable_entry_point_logged_and_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """An entry point resolving to a non-callable must be skipped, not invoked."""
        from deepagents.profiles._builtin_profiles import (  # noqa: PLC0415
            _ensure_builtin_profiles_loaded,
        )

        ep = MagicMock()
        ep.name = "weird"
        ep.value = "mod:CONST"
        ep.load.return_value = "not callable"

        with (
            caplog.at_level(logging.WARNING, logger="deepagents.profiles._builtin_profiles"),
            patch(
                "deepagents.profiles._builtin_profiles.entry_points",
                return_value=[ep],
            ),
        ):
            _ensure_builtin_profiles_loaded()

        assert any("did not resolve to a callable" in rec.message for rec in caplog.records)

    def test_registration_raises_logged_and_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """A registration callable that raises must be isolated to itself."""
        from deepagents.profiles._builtin_profiles import (  # noqa: PLC0415
            _ensure_builtin_profiles_loaded,
        )

        ep = MagicMock()
        ep.name = "angry"
        ep.value = "mod:register"
        ep.load.return_value = MagicMock(side_effect=RuntimeError("bad"))

        with (
            caplog.at_level(logging.WARNING, logger="deepagents.profiles._builtin_profiles"),
            patch(
                "deepagents.profiles._builtin_profiles.entry_points",
                return_value=[ep],
            ),
        ):
            _ensure_builtin_profiles_loaded()

        assert any("registration callable" in rec.message and "raised" in rec.message for rec in caplog.records)

    def test_entry_points_call_itself_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        """If `entry_points(group=...)` raises, the loader must log and continue."""
        from deepagents.profiles._builtin_profiles import (  # noqa: PLC0415
            _ensure_builtin_profiles_loaded,
        )

        with (
            caplog.at_level(logging.WARNING, logger="deepagents.profiles._builtin_profiles"),
            patch(
                "deepagents.profiles._builtin_profiles.entry_points",
                side_effect=RuntimeError("malformed dist-info"),
            ),
        ):
            _ensure_builtin_profiles_loaded()  # must not raise

        assert any("Failed to enumerate" in rec.message for rec in caplog.records)

    def test_loader_is_idempotent(self) -> None:
        """A second call must be a no-op; plugin callables must not fire twice."""
        from deepagents.profiles._builtin_profiles import (  # noqa: PLC0415
            _ensure_builtin_profiles_loaded,
        )

        plugin = MagicMock()
        ep = MagicMock()
        ep.name = "idem"
        ep.value = "mod:register"
        ep.load.return_value = plugin

        def fake_entry_points(*, group: str) -> list[MagicMock]:
            if group == "deepagents.provider_profiles":
                return [ep]
            return []

        with patch(
            "deepagents.profiles._builtin_profiles.entry_points",
            side_effect=fake_entry_points,
        ):
            _ensure_builtin_profiles_loaded()
            _ensure_builtin_profiles_loaded()

        plugin.assert_called_once_with()

    def test_two_plugins_on_same_key_merge(self) -> None:
        """Additive merge semantics must hold across entry-point plugins."""
        from deepagents.profiles._builtin_profiles import (  # noqa: PLC0415
            _ensure_builtin_profiles_loaded,
        )

        def plugin_a() -> None:
            register_provider_profile(
                "collidetest",
                ProviderProfile(init_kwargs={"first": 1}),
            )

        def plugin_b() -> None:
            register_provider_profile(
                "collidetest",
                ProviderProfile(init_kwargs={"second": 2}),
            )

        ep_a = MagicMock()
        ep_a.name = "a"
        ep_a.value = "mod:a"
        ep_a.load.return_value = plugin_a
        ep_b = MagicMock()
        ep_b.name = "b"
        ep_b.value = "mod:b"
        ep_b.load.return_value = plugin_b

        def fake_entry_points(*, group: str) -> list[MagicMock]:
            if group == "deepagents.provider_profiles":
                return [ep_a, ep_b]
            return []

        with patch(
            "deepagents.profiles._builtin_profiles.entry_points",
            side_effect=fake_entry_points,
        ):
            _ensure_builtin_profiles_loaded()

        merged = _PROVIDER_PROFILES["collidetest"]
        assert dict(merged.init_kwargs) == {"first": 1, "second": 2}

    def test_bootstrap_harness_keys_snapshot_after_load(self) -> None:
        """`_BOOTSTRAP_HARNESS_KEYS` must capture bootstrap-registered keys only."""
        from deepagents.profiles import _builtin_profiles  # noqa: PLC0415
        from deepagents.profiles._builtin_profiles import (  # noqa: PLC0415
            _ensure_builtin_profiles_loaded,
        )

        def plugin_registers_harness() -> None:
            register_harness_profile("plugintest:model", HarnessProfile())

        ep = MagicMock()
        ep.name = "harness_plugin"
        ep.value = "mod:register"
        ep.load.return_value = plugin_registers_harness

        def fake_entry_points(*, group: str) -> list[MagicMock]:
            if group == "deepagents.harness_profiles":
                return [ep]
            return []

        with patch(
            "deepagents.profiles._builtin_profiles.entry_points",
            side_effect=fake_entry_points,
        ):
            _ensure_builtin_profiles_loaded()

        snapshot = _builtin_profiles._BOOTSTRAP_HARNESS_KEYS
        assert "plugintest:model" in snapshot

        # Keys registered after bootstrap must NOT appear in the snapshot —
        # that invariant is what lets `_has_any_harness_profile` tell
        # user-registered profiles apart from bootstrap defaults.
        register_harness_profile("postbootstrap:model", HarnessProfile())
        assert "postbootstrap:model" not in _builtin_profiles._BOOTSTRAP_HARNESS_KEYS


class TestResolveModelWithProviderProfiles:
    """Tests for `resolve_model` using provider profiles."""

    def test_openai_uses_provider_profile_init_kwargs(self) -> None:
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openai:gpt-5")

        mock.assert_called_once_with("openai:gpt-5", use_responses_api=True)

    def test_openrouter_runs_pre_init_and_factory(self) -> None:
        with (
            patch("deepagents._models.init_chat_model") as mock,
            patch("deepagents.profiles.provider._openrouter.check_openrouter_version") as mock_check,
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

    def test_custom_provider_profile_kwargs_forwarded(self) -> None:
        profile = ProviderProfile(init_kwargs={"custom_key": "custom_val"})
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("customprov", profile)
            with patch("deepagents._models.init_chat_model") as mock:
                mock.return_value = MagicMock(spec=BaseChatModel)
                resolve_model("customprov:my-model")

            mock.assert_called_once_with("customprov:my-model", custom_key="custom_val")
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_factory_overrides_static_init_kwargs_on_collision(self) -> None:
        """Factory output wins over static `init_kwargs` on key collision.

        When a single profile sets both fields, overlapping keys resolve to
        the factory's value.
        """
        profile = ProviderProfile(
            init_kwargs={"shared": "static", "static_only": "s"},
            init_kwargs_factory=lambda: {"shared": "factory", "factory_only": "f"},
        )
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("mixedprov", profile)
            with patch("deepagents._models.init_chat_model") as mock:
                mock.return_value = MagicMock(spec=BaseChatModel)
                resolve_model("mixedprov:my-model")

            mock.assert_called_once_with(
                "mixedprov:my-model",
                shared="factory",
                static_only="s",
                factory_only="f",
            )
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)


class TestProfileImmutability:
    """Tests that `frozen=True` profile fields can't be mutated post-construction."""

    def test_provider_init_kwargs_unaliased_after_construction(self) -> None:
        """Mutating the source dict after construction must not mutate the profile."""
        source = {"a": 1}
        profile = ProviderProfile(init_kwargs=source)
        source["a"] = 999
        source["b"] = 2
        assert profile.init_kwargs == {"a": 1}

    def test_provider_init_kwargs_cannot_be_mutated_in_place(self) -> None:
        profile = ProviderProfile(init_kwargs={"a": 1})
        with pytest.raises(TypeError):
            profile.init_kwargs["a"] = 999  # type: ignore[index]

    def test_harness_tool_description_overrides_unaliased(self) -> None:
        source = {"ls": "original"}
        profile = HarnessProfile(tool_description_overrides=source)
        source["ls"] = "mutated"
        source["new"] = "added"
        assert dict(profile.tool_description_overrides) == {"ls": "original"}

    def test_harness_tool_description_overrides_cannot_be_mutated_in_place(self) -> None:
        profile = HarnessProfile(tool_description_overrides={"ls": "original"})
        with pytest.raises(TypeError):
            profile.tool_description_overrides["ls"] = "mutated"  # type: ignore[index]


class TestRegisterProfileKeyValidation:
    """Tests for key-shape validation in `register_provider_profile` and `register_harness_profile`."""

    def test_empty_key_rejected_provider(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            register_provider_profile("", ProviderProfile())

    def test_empty_key_rejected_harness(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            register_harness_profile("", HarnessProfile())

    def test_multiple_colons_rejected_provider(self) -> None:
        with pytest.raises(ValueError, match="more than one"):
            register_provider_profile("a:b:c", ProviderProfile())

    def test_multiple_colons_rejected_harness(self) -> None:
        with pytest.raises(ValueError, match="more than one"):
            register_harness_profile("a:b:c", HarnessProfile())

    def test_empty_provider_half_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty provider"):
            register_provider_profile(":model", ProviderProfile())

    def test_empty_model_half_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty provider"):
            register_harness_profile("openai:", HarnessProfile())

    def test_valid_provider_key_accepted(self) -> None:
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("validprov", ProviderProfile())
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_valid_provider_model_key_accepted(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile("validprov:model-name", HarnessProfile())
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestMergeMiddlewareDuplicateTypes:
    """Regression tests for `_merge_middleware` when either side has multiple middleware of the same concrete type."""

    def test_base_with_duplicate_types_drops_trailing_after_replace(self) -> None:
        """With two base entries of one type, the override replaces only the first; the duplicate is dropped."""

        class MW:
            def __init__(self, name: str) -> None:
                self.name = name

        a1, a2 = MW("a1"), MW("a2")
        b1 = MW("b1")
        merged_factory = _merge_middleware([a1, a2], [b1])
        assert callable(merged_factory)
        merged = list(merged_factory())
        assert merged == [b1]

    def test_base_with_duplicate_types_preserves_other_base_entries(self) -> None:
        """Novel base types are kept; only duplicates of replaced types are dropped."""

        class MWA:
            pass

        class MWB:
            pass

        a1, a2 = MWA(), MWA()
        b_novel = MWB()
        override_a = MWA()
        merged_factory = _merge_middleware([a1, a2, b_novel], [override_a])
        assert callable(merged_factory)
        merged = list(merged_factory())
        assert merged == [override_a, b_novel]

    def test_novel_override_types_appended_after_replacement(self) -> None:
        class MWA:
            pass

        class MWB:
            pass

        base_a = MWA()
        override_a = MWA()
        override_b = MWB()
        merged_factory = _merge_middleware([base_a], [override_a, override_b])
        assert callable(merged_factory)
        merged = list(merged_factory())
        assert merged == [override_a, override_b]


class TestResolveModelFullPath:
    """Tests that exercise the full string-spec path through `resolve_model` → profile lookup → `init_chat_model`."""

    def test_openai_responses_api_end_to_end(self) -> None:
        """String spec for OpenAI should reach init_chat_model with the built-in kwarg."""
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openai:gpt-5.4")
        mock.assert_called_once_with("openai:gpt-5.4", use_responses_api=True)

    def test_no_profile_registered_calls_init_chat_model_without_kwargs(self) -> None:
        """If no profile matches, init_chat_model is called with just the spec."""
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("noprofile:some-model")
        mock.assert_called_once_with("noprofile:some-model")

    def test_exact_miss_falls_back_to_provider_profile_kwargs(self) -> None:
        """A typo'd model spec should still get the provider-level init kwargs."""
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile(
                "fallprov",
                ProviderProfile(init_kwargs={"temperature": 0.5}),
            )
            with patch("deepagents._models.init_chat_model") as mock:
                mock.return_value = MagicMock(spec=BaseChatModel)
                resolve_model("fallprov:typo-model-name")
            mock.assert_called_once_with("fallprov:typo-model-name", temperature=0.5)
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)


class TestProfileLookupBreadcrumb:
    """Tests that exact-miss profile fallback emits a debug breadcrumb."""

    def test_harness_exact_miss_logs_breadcrumb(self, caplog: pytest.LogCaptureFixture) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile("crumbprov", HarnessProfile(system_prompt_suffix="s"))
            with caplog.at_level(logging.DEBUG, logger="deepagents.profiles.harness_profiles"):
                _get_harness_profile("crumbprov:typo-model")
            messages = [r.getMessage() for r in caplog.records]
            assert any("No exact HarnessProfile" in m and "crumbprov" in m for m in messages)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_provider_exact_miss_logs_breadcrumb(self, caplog: pytest.LogCaptureFixture) -> None:
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("crumbprov", ProviderProfile(init_kwargs={"a": 1}))
            with caplog.at_level(logging.DEBUG, logger="deepagents.profiles.provider.provider_profiles"):
                _get_provider_profile("crumbprov:typo-model")
            messages = [r.getMessage() for r in caplog.records]
            assert any("No exact ProviderProfile" in m and "crumbprov" in m for m in messages)
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)


class TestGetModelProviderLogging:
    """Tests that `get_model_provider` logs a debug breadcrumb when `_get_ls_params` raises."""

    def test_logs_exception_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        model = _make_model({})
        model._get_ls_params = MagicMock(side_effect=AttributeError("boom"))
        with caplog.at_level(logging.DEBUG, logger="deepagents._models"):
            assert get_model_provider(model) is None
        messages = [r.getMessage() for r in caplog.records]
        assert any("_get_ls_params" in m and "boom" in m for m in messages)


class TestProfileLookupKeyValidation:
    """Tests that malformed lookup specs return `None` without matching bare-provider entries."""

    def test_harness_lookup_rejects_empty_model_half(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile("partprov", HarnessProfile(system_prompt_suffix="x"))
            assert _get_harness_profile("partprov:") is None
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_harness_lookup_rejects_empty_provider_half(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile("partprov", HarnessProfile(system_prompt_suffix="x"))
            assert _get_harness_profile(":some-model") is None
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_harness_lookup_rejects_double_colon(self) -> None:
        assert _get_harness_profile("a:b:c") is None

    def test_harness_lookup_rejects_empty_string(self) -> None:
        assert _get_harness_profile("") is None

    def test_provider_lookup_rejects_empty_model_half(self) -> None:
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("partprov", ProviderProfile(init_kwargs={"a": 1}))
            assert _get_provider_profile("partprov:") is None
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_provider_lookup_rejects_empty_provider_half(self) -> None:
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("partprov", ProviderProfile(init_kwargs={"a": 1}))
            assert _get_provider_profile(":some-model") is None
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_provider_lookup_rejects_double_colon(self) -> None:
        assert _get_provider_profile("a:b:c") is None


class TestOpenRouterEmptyEnvVar:
    """Tests that explicitly empty OpenRouter env vars suppress the SDK default."""

    def test_empty_app_url_suppresses_default(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_APP_URL": ""}):
            result = _openrouter_attribution_kwargs()
        assert "app_url" not in result
        assert result["app_title"] == _OPENROUTER_APP_TITLE

    def test_empty_app_title_suppresses_default(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_APP_TITLE": ""}):
            result = _openrouter_attribution_kwargs()
        assert result["app_url"] == _OPENROUTER_APP_URL
        assert "app_title" not in result


class TestChainedPreInitAndFactoryErrorLogging:
    """Tests that chained `pre_init` and `init_kwargs_factory` log context on failure."""

    def test_chained_pre_init_logs_base_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        def base_pre(_spec: str) -> None:
            msg = "base boom"
            raise RuntimeError(msg)

        over_called: list[str] = []

        def over_pre(spec: str) -> None:
            over_called.append(spec)

        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("chainprov", ProviderProfile(pre_init=base_pre))
            register_provider_profile("chainprov", ProviderProfile(pre_init=over_pre))
            merged = _get_provider_profile("chainprov")
            assert merged is not None
            assert merged.pre_init is not None
            with (
                caplog.at_level(logging.ERROR, logger="deepagents.profiles.provider.provider_profiles"),
                pytest.raises(RuntimeError, match="base boom"),
            ):
                merged.pre_init("chainprov:some-model")
            assert over_called == [], "Override pre_init must not run after base raised"
            messages = [r.getMessage() for r in caplog.records]
            assert any("Base pre_init" in m and "chainprov:some-model" in m for m in messages)
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_chained_pre_init_logs_override_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        base_called: list[str] = []

        def base_pre(spec: str) -> None:
            base_called.append(spec)

        def over_pre(_spec: str) -> None:
            msg = "over boom"
            raise RuntimeError(msg)

        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("chainprov2", ProviderProfile(pre_init=base_pre))
            register_provider_profile("chainprov2", ProviderProfile(pre_init=over_pre))
            merged = _get_provider_profile("chainprov2")
            assert merged is not None
            assert merged.pre_init is not None
            with (
                caplog.at_level(logging.ERROR, logger="deepagents.profiles.provider.provider_profiles"),
                pytest.raises(RuntimeError, match="over boom"),
            ):
                merged.pre_init("chainprov2:some-model")
            assert base_called == ["chainprov2:some-model"]
            messages = [r.getMessage() for r in caplog.records]
            assert any("Override pre_init" in m and "chainprov2:some-model" in m for m in messages)
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_chained_factory_logs_base_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        def base_factory() -> dict:
            msg = "base factory boom"
            raise RuntimeError(msg)

        override_called: list[int] = []

        def override_factory() -> dict:
            override_called.append(1)
            return {"x": 1}

        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile("factprov", ProviderProfile(init_kwargs_factory=base_factory))
            register_provider_profile("factprov", ProviderProfile(init_kwargs_factory=override_factory))
            merged = _get_provider_profile("factprov")
            assert merged is not None
            assert merged.init_kwargs_factory is not None
            with (
                caplog.at_level(logging.ERROR, logger="deepagents.profiles.provider.provider_profiles"),
                pytest.raises(RuntimeError, match="base factory boom"),
            ):
                merged.init_kwargs_factory()
            assert override_called == [], "Override factory must not run after base raised"
            messages = [r.getMessage() for r in caplog.records]
            assert any("Base init_kwargs_factory" in m for m in messages)
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)


class TestHarnessProfileExtraMiddlewareFreeze:
    """Tests that `extra_middleware` sequences are defensively frozen in `__post_init__`."""

    def test_sequence_stored_as_tuple(self) -> None:
        class MW:
            pass

        mw = MW()
        source: list[MW] = [mw]
        profile = HarnessProfile(extra_middleware=source)
        assert isinstance(profile.extra_middleware, tuple)
        assert profile.extra_middleware == (mw,)
        source.append(MW())
        assert profile.extra_middleware == (mw,)

    def test_factory_stored_as_is(self) -> None:
        class MW:
            pass

        mw = MW()

        def factory() -> list[MW]:
            return [mw]

        profile = HarnessProfile(extra_middleware=factory)
        assert profile.extra_middleware is factory
