"""Tests for ConfigurableModelMiddleware."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
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
) -> ModelRequest:
    """Create a ModelRequest with a runtime that carries CLIContext."""
    runtime = SimpleNamespace(context=context)
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hi")],
        tools=[],
        runtime=cast("Any", runtime),
    )


def _make_response() -> ModelResponse[Any]:
    """Create a minimal model response for handler mocks."""
    return ModelResponse(result=[AIMessage(content="response")])


class TestGetOverrideModel:
    """Tests for `_get_override_model` logic."""

    def test_no_context_returns_none(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(model, context=None)
        assert mw._get_override_model(request) is None

    def test_empty_context_returns_none(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(model, context=CLIContext())
        assert mw._get_override_model(request) is None

    def test_same_model_spec_returns_none(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            context=CLIContext(model="claude-sonnet-4-6"),
        )
        assert mw._get_override_model(request) is None

    def test_provider_prefixed_spec_matches_model_name(self) -> None:
        """`anthropic:claude-sonnet-4-6` should match the resolved model name."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            context=CLIContext(model="anthropic:claude-sonnet-4-6"),
        )
        assert mw._get_override_model(request) is None

    def test_different_model_resolves_override(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            context=CLIContext(model="openai:gpt-4o"),
        )
        with patch("deepagents_cli.configurable_model.resolve_model") as mock_resolve:
            mock_resolve.return_value = _make_model("gpt-4o")
            result = mw._get_override_model(request)

        assert result is not None
        assert result is mock_resolve.return_value
        mock_resolve.assert_called_once_with("openai:gpt-4o")

    def test_none_runtime_returns_none(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=None,
        )
        assert mw._get_override_model(request) is None

    def test_non_cli_context_returns_none(self) -> None:
        """Runtime with a non-CLIContext context should be ignored."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        runtime = SimpleNamespace(context={"model": "openai:gpt-4o"})
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=cast("Any", runtime),
        )
        assert mw._get_override_model(request) is None


class TestWrapModelCall:
    """Tests for sync `wrap_model_call`."""

    def test_override_swaps_model_on_request(self) -> None:
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(
            original,
            context=CLIContext(model="openai:gpt-4o"),
        )

        captured: list[ModelRequest] = []
        response = _make_response()

        def handler(req: ModelRequest) -> ModelResponse[Any]:
            captured.append(req)
            return response

        with patch(
            "deepagents_cli.configurable_model.resolve_model",
            return_value=override,
        ):
            result = mw.wrap_model_call(request, handler)

        assert request.model is original
        assert len(captured) == 1
        assert captured[0].model is override
        assert result is response

    def test_no_override_preserves_model(self) -> None:
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        request = _make_request(original, context=CLIContext())

        captured: list[ModelRequest] = []
        response = _make_response()

        def handler(req: ModelRequest) -> ModelResponse[Any]:
            captured.append(req)
            return response

        result = mw.wrap_model_call(request, handler)

        assert captured[0].model is original
        assert result is response


class TestAsyncWrapModelCall:
    """Tests for async `awrap_model_call`."""

    async def test_override_swaps_model_on_request(self) -> None:
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(
            original,
            context=CLIContext(model="openai:gpt-4o"),
        )

        captured: list[ModelRequest] = []
        response = _make_response()

        async def handler(req: ModelRequest) -> ModelResponse[Any]:
            await asyncio.sleep(0)
            captured.append(req)
            return response

        with patch(
            "deepagents_cli.configurable_model.resolve_model",
            return_value=override,
        ):
            result = await mw.awrap_model_call(request, handler)

        assert request.model is original
        assert captured[0].model is override
        assert result is response

    async def test_no_override_preserves_model(self) -> None:
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        request = _make_request(original, context=CLIContext())

        response = _make_response()

        async def handler(_req: ModelRequest) -> ModelResponse[Any]:
            await asyncio.sleep(0)
            return response

        result = await mw.awrap_model_call(request, handler)

        assert request.model is original
        assert result is response


class TestModelParams:
    """Tests for `model_params` override via CLIContext."""

    def test_params_only_merges_model_settings(self) -> None:
        """`model_params` without a model swap merges into `model_settings`."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            context=CLIContext(model_params={"temperature": 0.7}),
        )

        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> ModelResponse[Any]:
            captured.append(req)
            return _make_response()

        mw.wrap_model_call(request, handler)

        assert len(captured) == 1
        assert captured[0].model is model
        assert captured[0].model_settings == {"temperature": 0.7}

    def test_params_with_model_swap(self) -> None:
        """`model_params` and a model swap should both apply."""
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(
            original,
            context=CLIContext(
                model="openai:gpt-4o",
                model_params={"max_tokens": 1024},
            ),
        )

        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> ModelResponse[Any]:
            captured.append(req)
            return _make_response()

        with patch(
            "deepagents_cli.configurable_model.resolve_model",
            return_value=override,
        ):
            mw.wrap_model_call(request, handler)

        assert captured[0].model is override
        assert captured[0].model_settings == {"max_tokens": 1024}

    def test_empty_params_no_op(self) -> None:
        """An empty `model_params` dict should not trigger an override."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            context=CLIContext(model_params={}),
        )

        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> ModelResponse[Any]:
            captured.append(req)
            return _make_response()

        mw.wrap_model_call(request, handler)

        assert captured[0] is request

    def test_params_merge_preserves_existing_settings(self) -> None:
        """`model_params` should merge on top of existing model settings."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        runtime = SimpleNamespace(
            context=CLIContext(model_params={"temperature": 0.5})
        )
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=cast("Any", runtime),
            model_settings={"max_tokens": 2048},
        )

        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> ModelResponse[Any]:
            captured.append(req)
            return _make_response()

        mw.wrap_model_call(request, handler)

        assert captured[0].model_settings == {
            "max_tokens": 2048,
            "temperature": 0.5,
        }

    async def test_async_params_override(self) -> None:
        """`model_params` should work in the async path."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            context=CLIContext(model_params={"temperature": 0.3}),
        )

        captured: list[ModelRequest] = []
        response = _make_response()

        async def handler(req: ModelRequest) -> ModelResponse[Any]:
            await asyncio.sleep(0)
            captured.append(req)
            return response

        result = await mw.awrap_model_call(request, handler)

        assert captured[0].model_settings == {"temperature": 0.3}
        assert result is response
