"""Tests for deepagents._models helpers."""

from unittest.mock import MagicMock, patch

from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import BaseChatModel

from deepagents._models import (
    _string_value,
    get_model_identifier,
    model_matches_spec,
    resolve_model,
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

    def test_non_openai_string(self) -> None:
        with patch("deepagents._models.init_chat_model") as mock:
            mock.return_value = MagicMock(spec=BaseChatModel)
            result = resolve_model("anthropic:claude-sonnet-4-6")

        mock.assert_called_once_with("anthropic:claude-sonnet-4-6")
        assert result is mock.return_value

    def test_passthrough_configurable_model(self) -> None:
        """_ConfigurableModel instances should pass through unchanged."""
        model = MagicMock(spec=_ConfigurableModel)
        assert resolve_model(model) is model


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
