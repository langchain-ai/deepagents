"""Tests for ConfigurableModelMiddleware."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from deepagents.middleware.configurable_model import ConfigurableModelMiddleware


def _make_model(name: str) -> MagicMock:
    """Create a mock BaseChatModel with model_name set."""
    model = MagicMock(spec=BaseChatModel)
    model.model_name = name
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
        with patch("deepagents.middleware.configurable_model._resolve_model_from_spec") as mock_resolve:
            mock_resolve.return_value = _make_model("gpt-4o")
            result = mw._get_override_model(request)

        assert result is not None
        assert result.model_name == "gpt-4o"
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
            "deepagents.middleware.configurable_model._resolve_model_from_spec",
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
            "deepagents.middleware.configurable_model._resolve_model_from_spec",
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
