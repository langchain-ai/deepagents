"""Tests for ConfigurableModelMiddleware."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_cli.configurable_model import (
    CLIContext,
    ConfigurableModelMiddleware,
    _is_anthropic_model,
)


def _make_model(name: str) -> MagicMock:
    """Create a mock BaseChatModel with model_name set."""
    model = MagicMock(spec=BaseChatModel)
    model.model_name = name
    model.model_dump.return_value = {"model_name": name}
    return model


def _make_request(
    model: BaseChatModel,
    context: CLIContext | None = None,
    model_settings: dict[str, Any] | None = None,
) -> ModelRequest:
    """Create a ModelRequest with a runtime that carries CLIContext."""
    runtime = SimpleNamespace(context=context)
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hi")],
        tools=[],
        runtime=cast("Any", runtime),
        model_settings=model_settings,
    )


def _make_response() -> ModelResponse[Any]:
    """Create a minimal model response for handler mocks."""
    return ModelResponse(result=[AIMessage(content="response")])


def _make_model_result(model: MagicMock) -> SimpleNamespace:
    """Create a mock ModelResult with a .model attribute."""
    return SimpleNamespace(model=model)


_PATCH_CREATE = "deepagents_cli.config.create_model"

_mw = ConfigurableModelMiddleware()


class TestNoOverride:
    """Cases where the middleware should pass the request through unchanged."""

    def test_no_context(self) -> None:
        request = _make_request(_make_model("claude-sonnet-4-6"), context=None)
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0].model is request.model

    def test_empty_context(self) -> None:
        request = _make_request(_make_model("claude-sonnet-4-6"), context=CLIContext())
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request

    def test_same_model_spec(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="claude-sonnet-4-6"),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request

    def test_provider_prefixed_spec_matches(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="anthropic:claude-sonnet-4-6"),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request

    def test_none_runtime(self) -> None:
        request = ModelRequest(
            model=_make_model("claude-sonnet-4-6"),
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=None,
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0].model is request.model

    def test_non_dict_context_ignored(self) -> None:
        runtime = SimpleNamespace(context="not-a-dict")
        request = ModelRequest(
            model=_make_model("claude-sonnet-4-6"),
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=cast("Any", runtime),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0].model is request.model

    def test_empty_model_params(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model_params={}),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request


class TestModelSwap:
    """Cases where the middleware should swap the model."""

    def test_different_model_swapped(self) -> None:
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(original, context=CLIContext(model="openai:gpt-4o"))

        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=_make_model_result(override)):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is override
        assert request.model is original  # original unchanged

    async def test_async_model_swapped(self) -> None:
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(original, context=CLIContext(model="openai:gpt-4o"))

        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        with patch(_PATCH_CREATE, return_value=_make_model_result(override)):
            await _mw.awrap_model_call(request, handler)

        assert captured[0].model is override

    def test_class_path_provider_swapped(self) -> None:
        """Config-defined class_path provider resolves through create_model."""
        original = _make_model("claude-sonnet-4-6")
        custom = _make_model("my-model")
        request = _make_request(original, context=CLIContext(model="custom:my-model"))

        captured: list[ModelRequest] = []
        with patch(
            _PATCH_CREATE, return_value=_make_model_result(custom)
        ) as mock_create:
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is custom
        mock_create.assert_called_once_with("custom:my-model")

    def test_create_model_error_propagates(self) -> None:
        """ModelConfigError from create_model propagates to caller."""
        from deepagents_cli.model_config import ModelConfigError

        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="unknown:bad-model"),
        )
        with (
            patch(_PATCH_CREATE, side_effect=ModelConfigError("no such provider")),
            pytest.raises(ModelConfigError, match="no such provider"),
        ):
            _mw.wrap_model_call(request, lambda _r: _make_response())


class TestAnthropicSettingsStripped:
    """Anthropic-specific model_settings stripped on cross-provider swap.

    When swapping from Anthropic to a non-Anthropic model, provider-specific
    settings like `cache_control` must be stripped to avoid TypeError on the
    target provider's API (e.g. OpenAI/Groq).
    """

    def test_cache_control_stripped_on_swap(self) -> None:
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
            model_settings={"cache_control": {"type": "ephemeral", "ttl": "5m"}},
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert "cache_control" not in captured[0].model_settings

    def test_cache_control_preserved_for_anthropic_swap(self) -> None:
        override = _make_model("claude-opus-4-6")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="anthropic:claude-opus-4-6"),
            model_settings={"cache_control": {"type": "ephemeral", "ttl": "5m"}},
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
                return_value=True,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings["cache_control"] == {
            "type": "ephemeral",
            "ttl": "5m",
        }

    def test_other_settings_preserved_on_swap(self) -> None:
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
            model_settings={
                "cache_control": {"type": "ephemeral"},
                "max_tokens": 2048,
            },
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings == {"max_tokens": 2048}

    async def test_async_cache_control_stripped(self) -> None:
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
            model_settings={"cache_control": {"type": "ephemeral"}},
        )
        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            await _mw.awrap_model_call(request, handler)

        assert "cache_control" not in captured[0].model_settings

    def test_swap_with_model_params_and_cache_control(self) -> None:
        """Stripping operates on the merged settings, not the original."""
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(
                model="openai:gpt-4o",
                model_params={"temperature": 0.7},
            ),
            model_settings={
                "cache_control": {"type": "ephemeral"},
                "max_tokens": 2048,
            },
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings == {
            "max_tokens": 2048,
            "temperature": 0.7,
        }

    def test_only_cache_control_results_in_empty_settings(self) -> None:
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
            model_settings={"cache_control": {"type": "ephemeral"}},
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings == {}


class TestIsAnthropicModel:
    """Direct tests for the `_is_anthropic_model` helper."""

    def test_returns_true_for_anthropic(self) -> None:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model_name="claude-sonnet-4-6")
        assert _is_anthropic_model(model) is True

    def test_returns_false_for_non_anthropic(self) -> None:
        assert _is_anthropic_model(_make_model("gpt-4o")) is False

    def test_returns_false_when_import_missing(self) -> None:
        with patch.dict("sys.modules", {"langchain_anthropic": None}):
            assert _is_anthropic_model(_make_model("anything")) is False


class TestModelParams:
    """Cases where model_params are merged into model_settings."""

    def test_params_merged(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model_params={"temperature": 0.7}),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model is request.model
        assert captured[0].model_settings == {"temperature": 0.7}

    def test_params_merge_preserves_existing(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model_params={"temperature": 0.5}),
            model_settings={"max_tokens": 2048},
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {"max_tokens": 2048, "temperature": 0.5}

    def test_params_with_model_swap(self) -> None:
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(
                model="openai:gpt-4o", model_params={"max_tokens": 1024}
            ),
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=_make_model_result(override)):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is override
        assert captured[0].model_settings == {"max_tokens": 1024}

    async def test_async_params(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model_params={"temperature": 0.3}),
        )
        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        await _mw.awrap_model_call(request, handler)
        assert captured[0].model_settings == {"temperature": 0.3}
