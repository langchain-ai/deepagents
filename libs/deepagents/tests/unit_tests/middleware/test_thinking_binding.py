"""Tests for the Anthropic thinking-binding middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from deepagents.middleware._thinking_binding import THINKING_BINDING_BETA, _AnthropicThinkingBindingMiddleware

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_DROP_BLOCK = {"prefix_mismatch_behavior": "drop_block"}


class _Answer(BaseModel):
    text: str


class _BedrockLikeChatAnthropic(ChatAnthropic):
    """Stands in for a `ChatAnthropic` subclass that reaches Claude through another platform."""

    @property
    def _llm_type(self) -> str:
        return "anthropic-bedrock-chat"


def _claude(model: str = "claude-opus-5-5", **kwargs: Any) -> ChatAnthropic:
    return ChatAnthropic(model=model, api_key="fake", **kwargs)  # type: ignore[call-arg]


def _sent(request: ModelRequest) -> ModelRequest:
    """Run `request` through the middleware and return the request the model receives."""
    seen: list[ModelRequest] = []

    def handler(req: ModelRequest) -> ModelResponse:
        seen.append(req)
        return ModelResponse(result=[AIMessage(content="ok")])

    _AnthropicThinkingBindingMiddleware().wrap_model_call(request, handler)
    return seen[0]


def _payload(request: ModelRequest) -> dict[str, Any]:
    """Build the API payload `ChatAnthropic` would send for `request`."""
    model = request.model
    assert isinstance(model, ChatAnthropic)
    return model._get_request_payload([HumanMessage("hi")], **request.model_settings)


def _request(model: BaseChatModel, **kwargs: Any) -> ModelRequest:
    return ModelRequest(model=model, messages=[HumanMessage("hi")], state={"messages": []}, **kwargs)


class TestAppliesBinding:
    def test_default_thinking_gets_drop_block(self) -> None:
        payload = _payload(_sent(_request(_claude())))

        assert payload["thinking"] == {"type": "adaptive", "block_binding": _DROP_BLOCK}
        assert payload["betas"] == [THINKING_BINDING_BETA]

    def test_reasoning_effort_keeps_summarized_display(self) -> None:
        """The binding only adds `block_binding` to the thinking config `ChatAnthropic` would send."""
        request = _request(_claude(reasoning_effort="high"))
        before = _payload(request)["thinking"]

        after = _payload(_sent(request))["thinking"]

        assert after == {**before, "block_binding": _DROP_BLOCK}

    def test_model_thinking_is_kept(self) -> None:
        model = _claude(thinking={"type": "adaptive", "display": "summarized"})

        payload = _payload(_sent(_request(model)))

        assert payload["thinking"] == {"type": "adaptive", "display": "summarized", "block_binding": _DROP_BLOCK}
        assert model.thinking == {"type": "adaptive", "display": "summarized"}

    def test_call_time_thinking_is_kept(self) -> None:
        request = _request(_claude(), model_settings={"thinking": {"type": "adaptive", "display": "updates"}})

        payload = _payload(_sent(request))

        assert payload["thinking"] == {"type": "adaptive", "display": "updates", "block_binding": _DROP_BLOCK}

    def test_explicit_mismatch_behavior_is_kept(self) -> None:
        model = _claude(thinking={"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}})

        payload = _payload(_sent(_request(model)))

        assert payload["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "error"}
        assert THINKING_BINDING_BETA in payload["betas"]

    @pytest.mark.parametrize(
        ("model_kwargs", "model_settings", "expected"),
        [
            ({"betas": ["model-beta"]}, {}, ["model-beta", THINKING_BINDING_BETA]),
            ({}, {"betas": ["call-beta"]}, ["call-beta", THINKING_BINDING_BETA]),
            ({"betas": [THINKING_BINDING_BETA]}, {}, [THINKING_BINDING_BETA]),
        ],
    )
    def test_existing_betas_are_kept(self, model_kwargs: dict[str, Any], model_settings: dict[str, Any], expected: list[str]) -> None:
        request = _request(_claude(**model_kwargs), model_settings=model_settings)

        assert _payload(_sent(request))["betas"] == expected

    async def test_async(self) -> None:
        seen: list[ModelRequest] = []

        async def handler(req: ModelRequest) -> ModelResponse:
            seen.append(req)
            return ModelResponse(result=[AIMessage(content="ok")])

        await _AnthropicThinkingBindingMiddleware().awrap_model_call(_request(_claude()), handler)

        assert _payload(seen[0])["thinking"]["block_binding"] == _DROP_BLOCK


class TestSkipsBinding:
    @pytest.mark.parametrize("model_name", ["claude-sonnet-4-6", "claude-3-5-haiku-latest"])
    def test_older_models(self, model_name: str) -> None:
        request = _request(_claude(model_name))

        assert _sent(request) is request

    @pytest.mark.parametrize("thinking_type", ["disabled", "between_tools", "enabled"])
    def test_thinking_that_cannot_carry_a_binding(self, thinking_type: str) -> None:
        request = _request(_claude("claude-sonnet-5-5", thinking={"type": thinking_type}))

        assert _sent(request) is request

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tool_choice": "any"},
            {"tool_choice": {"type": "tool", "name": "answer"}},
            {"response_format": ToolStrategy(_Answer)},
        ],
    )
    def test_forced_tool_call_without_thinking(self, kwargs: dict[str, Any]) -> None:
        """Naming thinking explicitly would push a forced tool call off its forced path."""
        request = _request(_claude(), **kwargs)

        assert _sent(request) is request

    def test_other_platform_subclass(self) -> None:
        request = _request(_BedrockLikeChatAnthropic(model="claude-opus-5-5", api_key="fake"))  # type: ignore[call-arg]

        assert _sent(request) is request

    def test_other_providers(self) -> None:
        request = _request(GenericFakeChatModel(messages=iter([])))

        assert _sent(request) is request

    def test_malformed_betas(self) -> None:
        request = _request(_claude(), model_settings={"betas": "not-a-list"})

        assert _sent(request) is request
