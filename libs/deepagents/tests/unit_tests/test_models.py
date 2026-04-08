"""Tests for deepagents._models helpers."""

import os
from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from deepagents._models import (
    _BASETEN_TIMEOUT_S,
    _OPENROUTER_APP_TITLE,
    _OPENROUTER_APP_URL,
    OPENROUTER_MIN_VERSION,
    _baseten_auth_kwargs,
    _openrouter_attribution_kwargs,
    _string_value,
    check_openrouter_version,
    get_model_identifier,
    model_matches_spec,
    resolve_model,
)
from deepagents.graph import (
    BASE_AGENT_PROMPT,
    GLM5_PROMPT_SUPPLEMENT,
    LEGACY_BASE_AGENT_PROMPT,
    _get_model_prompt_supplement,
    _resolve_base_agent_prompt,
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

    def test_baseten_prefix_passes_explicit_api_key_when_available(self) -> None:
        env = {"BASETEN_API_KEY": "secret-key"}
        with (
            patch("deepagents._models.init_chat_model") as mock,
            patch.dict("os.environ", env),
        ):
            mock.return_value = MagicMock(spec=BaseChatModel)
            result = resolve_model("baseten:zai-org/GLM-5")

        mock.assert_called_once_with(
            "baseten:zai-org/GLM-5",
            baseten_api_key="secret-key",
            timeout=_BASETEN_TIMEOUT_S,
        )
        assert result is mock.return_value

    def test_baseten_prefix_omits_api_key_when_unavailable(self) -> None:
        with (
            patch("deepagents._models.init_chat_model") as mock,
            patch.dict("os.environ", {}, clear=True),
        ):
            mock.return_value = MagicMock(spec=BaseChatModel)
            result = resolve_model("baseten:zai-org/GLM-5")

        mock.assert_called_once_with("baseten:zai-org/GLM-5", timeout=_BASETEN_TIMEOUT_S)
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


class TestModelPromptSupplements:
    """Tests for model-specific prompt supplements."""

    def test_glm5_gets_prompt_supplement(self) -> None:
        model = _make_model({"model_name": "zai-org/GLM-5"})
        assert _get_model_prompt_supplement(model) == GLM5_PROMPT_SUPPLEMENT

    def test_other_models_do_not_get_prompt_supplement(self) -> None:
        model = _make_model({"model_name": "claude-sonnet-4-6"})
        assert _get_model_prompt_supplement(model) == ""

    def test_resolve_base_prompt_uses_current_prompt_by_default(self) -> None:
        model = _make_model({"model_name": "claude-sonnet-4-6"})
        with patch.dict(os.environ, {}, clear=True):
            assert _resolve_base_agent_prompt(model) == BASE_AGENT_PROMPT

    def test_resolve_base_prompt_uses_legacy_prompt_when_requested(self) -> None:
        model = _make_model({"model_name": "zai-org/GLM-5"})
        with patch.dict("os.environ", {"DEEPAGENTS_BASE_PROMPT_VARIANT": "legacy"}):
            assert _resolve_base_agent_prompt(model) == LEGACY_BASE_AGENT_PROMPT

    def test_resolve_base_prompt_appends_model_supplement(self) -> None:
        model = _make_model({"model_name": "zai-org/GLM-5"})
        with patch.dict(os.environ, {}, clear=True):
            assert _resolve_base_agent_prompt(model) == (
                BASE_AGENT_PROMPT + "\n\n" + GLM5_PROMPT_SUPPLEMENT
            )


class TestCheckOpenRouterVersion:
    """Tests for check_openrouter_version."""

    def test_passes_when_not_installed(self) -> None:
        with patch(
            "deepagents._models.pkg_version",
            side_effect=PackageNotFoundError("langchain-openrouter"),
        ):
            check_openrouter_version()  # should not raise

    def test_passes_when_version_sufficient(self) -> None:
        with patch(
            "deepagents._models.pkg_version",
            return_value=OPENROUTER_MIN_VERSION,
        ):
            check_openrouter_version()  # should not raise

    def test_passes_when_version_above_minimum(self) -> None:
        with patch("deepagents._models.pkg_version", return_value="99.0.0"):
            check_openrouter_version()  # should not raise

    def test_raises_when_version_too_old(self) -> None:
        with (
            patch("deepagents._models.pkg_version", return_value="0.0.1"),
            pytest.raises(ImportError, match="langchain-openrouter>="),
        ):
            check_openrouter_version()

    def test_resolve_model_calls_check(self) -> None:
        with (
            patch("deepagents._models.check_openrouter_version") as mock_check,
            patch("deepagents._models.init_chat_model") as mock_init,
        ):
            mock_init.return_value = MagicMock(spec=BaseChatModel)
            resolve_model("openrouter:anthropic/claude-sonnet-4-6")

        mock_check.assert_called_once()

    def test_resolve_model_skips_check_for_non_openrouter(self) -> None:
        with (
            patch("deepagents._models.check_openrouter_version") as mock_check,
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


class TestBasetenAuthKwargs:
    """Tests for _baseten_auth_kwargs."""

    def test_empty_when_env_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert _baseten_auth_kwargs() == {"timeout": _BASETEN_TIMEOUT_S}

    def test_returns_api_key_when_env_present(self) -> None:
        with patch.dict("os.environ", {"BASETEN_API_KEY": "secret-key"}, clear=True):
            assert _baseten_auth_kwargs() == {
                "baseten_api_key": "secret-key",
                "timeout": _BASETEN_TIMEOUT_S,
            }


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
