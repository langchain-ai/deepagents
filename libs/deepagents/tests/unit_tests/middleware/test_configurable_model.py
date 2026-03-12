"""Tests for ConfigurableModelMiddleware."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from deepagents.middleware.configurable_model import ConfigurableModelMiddleware


def _make_model(name: str) -> MagicMock:
    """Create a mock BaseChatModel with model_name set."""
    model = MagicMock(spec=BaseChatModel)
    model.model_name = name
    model.model_dump.return_value = {"model_name": name}
    return model


def _make_request(
    model: BaseChatModel,
    config: dict | None = None,
) -> ModelRequest:
    """Create a ModelRequest with a runtime that carries config."""
    runtime = SimpleNamespace(config=config or {})
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hi")],
        tools=[],
        runtime=runtime,
    )


class TestGetOverrideModel:
    """Tests for _get_override_model logic."""

    def test_no_configurable_key_returns_none(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(model, config={})
        assert mw._get_override_model(request) is None

    def test_same_model_spec_returns_none(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            config={"configurable": {"model": "claude-sonnet-4-6"}},
        )
        assert mw._get_override_model(request) is None

    def test_provider_prefixed_spec_matches_model_name(self) -> None:
        """'anthropic:claude-sonnet-4-6' should match model_name 'claude-sonnet-4-6'."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            config={"configurable": {"model": "anthropic:claude-sonnet-4-6"}},
        )
        assert mw._get_override_model(request) is None

    def test_different_model_resolves_override(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            config={"configurable": {"model": "openai:gpt-4o"}},
        )
        with patch("deepagents.middleware.configurable_model.resolve_model") as mock_resolve:
            mock_resolve.return_value = _make_model("gpt-4o")
            result = mw._get_override_model(request)

        assert result is not None
        assert result.model_name == "gpt-4o"
        mock_resolve.assert_called_once_with("openai:gpt-4o")

    def test_uses_langgraph_config_when_available(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=SimpleNamespace(),
        )

        with (
            patch(
                "deepagents.middleware.configurable_model.get_config",
                return_value={"configurable": {"model": "openai:gpt-4o"}},
            ),
            patch("deepagents.middleware.configurable_model.resolve_model") as mock_resolve,
        ):
            mock_resolve.return_value = _make_model("gpt-4o")
            result = mw._get_override_model(request)

        assert result is not None
        mock_resolve.assert_called_once_with("openai:gpt-4o")

    def test_none_runtime_config_returns_none(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        runtime = SimpleNamespace(config=None)
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=runtime,
        )
        assert mw._get_override_model(request) is None

    def test_invalid_runtime_config_type_raises_type_error(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=SimpleNamespace(config="not-a-dict"),
        )

        with pytest.raises(TypeError, match="`config` must be a dictionary"):
            mw._get_override_model(request)

    def test_invalid_configurable_type_raises_type_error(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(model, config={"configurable": "openai:gpt-4o"})

        with pytest.raises(TypeError, match="config\\['configurable'\\]"):
            mw._get_override_model(request)

    def test_invalid_model_spec_type_raises_type_error(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(model, config={"configurable": {"model": 123}})

        with pytest.raises(TypeError, match="config\\['configurable'\\]\\['model'\\]"):
            mw._get_override_model(request)

    def test_missing_runtime_attr_returns_none(self) -> None:
        """Runtime without config attr should not crash."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        runtime = SimpleNamespace()  # no config attr
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=runtime,
        )
        assert mw._get_override_model(request) is None


class TestWrapModelCall:
    """Tests for sync wrap_model_call."""

    def test_override_swaps_model_on_request(self) -> None:
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(
            original,
            config={"configurable": {"model": "openai:gpt-4o"}},
        )

        # Capture the request passed to handler
        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> str:
            captured.append(req)
            return "response"

        with patch(
            "deepagents.middleware.configurable_model.resolve_model",
            return_value=override,
        ):
            result = mw.wrap_model_call(request, handler)

        # Original request is unchanged (immutable override)
        assert request.model is original
        # Handler received new request with overridden model
        assert len(captured) == 1
        assert captured[0].model is override
        assert result == "response"

    def test_no_override_preserves_model(self) -> None:
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        request = _make_request(original, config={})

        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> str:
            captured.append(req)
            return "response"

        result = mw.wrap_model_call(request, handler)

        assert captured[0].model is original
        assert result == "response"


class TestAsyncWrapModelCall:
    """Tests for async awrap_model_call."""

    async def test_override_swaps_model_on_request(self) -> None:
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(
            original,
            config={"configurable": {"model": "openai:gpt-4o"}},
        )

        captured: list[ModelRequest] = []

        async def handler(req: ModelRequest) -> str:
            captured.append(req)
            return "async_response"

        with patch(
            "deepagents.middleware.configurable_model.resolve_model",
            return_value=override,
        ):
            result = await mw.awrap_model_call(request, handler)

        assert request.model is original
        assert captured[0].model is override
        assert result == "async_response"

    async def test_no_override_preserves_model(self) -> None:
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        request = _make_request(original, config={})

        async def handler(_req: ModelRequest) -> str:
            return "async_response"

        result = await mw.awrap_model_call(request, handler)

        assert request.model is original
        assert result == "async_response"


class TestModelParams:
    """Tests for model_params override via configurable."""

    def test_params_only_merges_model_settings(self) -> None:
        """model_params without model swap merges into model_settings."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            config={"configurable": {"model_params": {"temperature": 0.7}}},
        )

        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> str:
            captured.append(req)
            return "response"

        mw.wrap_model_call(request, handler)

        assert len(captured) == 1
        assert captured[0].model is model  # model unchanged
        assert captured[0].model_settings == {"temperature": 0.7}

    def test_params_with_model_swap(self) -> None:
        """model_params + model swap both applied."""
        mw = ConfigurableModelMiddleware()
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(
            original,
            config={
                "configurable": {
                    "model": "openai:gpt-4o",
                    "model_params": {"max_tokens": 1024},
                }
            },
        )

        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> str:
            captured.append(req)
            return "response"

        with patch(
            "deepagents.middleware.configurable_model.resolve_model",
            return_value=override,
        ):
            mw.wrap_model_call(request, handler)

        assert captured[0].model is override
        assert captured[0].model_settings == {"max_tokens": 1024}

    def test_empty_params_no_op(self) -> None:
        """Empty model_params dict does not trigger override."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            config={"configurable": {"model_params": {}}},
        )

        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> str:
            captured.append(req)
            return "response"

        mw.wrap_model_call(request, handler)

        # Handler receives the original request object (no override call)
        assert captured[0] is request

    def test_params_merge_preserves_existing_settings(self) -> None:
        """model_params merges on top of existing model_settings."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        runtime = SimpleNamespace(
            config={
                "configurable": {
                    "model_params": {"temperature": 0.5},
                }
            }
        )
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=runtime,
            model_settings={"max_tokens": 2048},
        )

        captured: list[ModelRequest] = []

        def handler(req: ModelRequest) -> str:
            captured.append(req)
            return "response"

        mw.wrap_model_call(request, handler)

        assert captured[0].model_settings == {
            "max_tokens": 2048,
            "temperature": 0.5,
        }

    async def test_async_params_override(self) -> None:
        """model_params works in async path."""
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            config={"configurable": {"model_params": {"temperature": 0.3}}},
        )

        captured: list[ModelRequest] = []

        async def handler(req: ModelRequest) -> str:
            captured.append(req)
            return "async_response"

        result = await mw.awrap_model_call(request, handler)

        assert captured[0].model_settings == {"temperature": 0.3}
        assert result == "async_response"

    def test_invalid_model_params_type_raises_type_error(self) -> None:
        mw = ConfigurableModelMiddleware()
        model = _make_model("claude-sonnet-4-6")
        request = _make_request(
            model,
            config={"configurable": {"model_params": "temperature=0.3"}},
        )

        with pytest.raises(TypeError, match="config\\['configurable'\\]\\['model_params'\\]"):
            mw.wrap_model_call(request, lambda _req: "response")
