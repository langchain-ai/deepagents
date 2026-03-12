"""Tests for ConfigurableModelMiddleware."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_cli.configurable_model import CLIContext, ConfigurableModelMiddleware


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

    def test_non_cli_context_ignored(self) -> None:
        runtime = SimpleNamespace(context={"model": "openai:gpt-4o"})
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
        with patch(
            "deepagents_cli.configurable_model.resolve_model", return_value=override
        ):
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

        with patch(
            "deepagents_cli.configurable_model.resolve_model", return_value=override
        ):
            await _mw.awrap_model_call(request, handler)

        assert captured[0].model is override


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
        with patch(
            "deepagents_cli.configurable_model.resolve_model", return_value=override
        ):
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
