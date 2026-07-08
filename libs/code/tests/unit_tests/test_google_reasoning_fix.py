"""Tests for `GoogleReasoningFixMiddleware`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.google_reasoning_fix import GoogleReasoningFixMiddleware


def _make_model(provider: str) -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model._get_ls_params.return_value = {"ls_provider": provider}
    return model


def _make_request(model: BaseChatModel, messages: list[Any]) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=messages,
        tools=[],
        runtime=cast("Any", SimpleNamespace(context=None)),
        model_settings=None,
    )


def _make_response() -> ModelResponse[Any]:
    return ModelResponse(result=[AIMessage(content="ok")])


def test_non_google_provider_is_pass_through() -> None:
    middleware = GoogleReasoningFixMiddleware()
    messages = [
        AIMessage(content=[{"type": "reasoning", "text": "thought"}]),
    ]
    request = _make_request(_make_model("anthropic"), messages)

    captured: list[ModelRequest] = []
    middleware.wrap_model_call(
        request, lambda r: (captured.append(r), _make_response())[1]
    )

    assert captured[0] is request


def test_rewrites_reasoning_block_with_text_key() -> None:
    middleware = GoogleReasoningFixMiddleware()
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content=[{"type": "reasoning", "text": "thought"}]),
    ]
    request = _make_request(_make_model("google_genai"), messages)

    captured: list[ModelRequest] = []
    middleware.wrap_model_call(
        request, lambda r: (captured.append(r), _make_response())[1]
    )

    forwarded = captured[0].messages
    assert forwarded[0] is messages[0]
    ai = forwarded[1]
    assert isinstance(ai, AIMessage)
    assert ai.content == [{"type": "thinking", "thinking": "thought"}]


def test_drops_reasoning_block_without_text() -> None:
    middleware = GoogleReasoningFixMiddleware()
    messages = [
        AIMessage(
            content=[
                {"type": "reasoning"},
                {"type": "text", "text": "reply"},
            ]
        ),
    ]
    request = _make_request(_make_model("google_genai"), messages)

    captured: list[ModelRequest] = []
    middleware.wrap_model_call(
        request, lambda r: (captured.append(r), _make_response())[1]
    )

    ai = captured[0].messages[0]
    assert isinstance(ai, AIMessage)
    assert ai.content == [{"type": "text", "text": "reply"}]


def test_preserves_already_valid_thinking_block() -> None:
    middleware = GoogleReasoningFixMiddleware()
    messages = [
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "already valid", "signature": "s"},
            ]
        ),
    ]
    request = _make_request(_make_model("google_genai"), messages)

    captured: list[ModelRequest] = []
    middleware.wrap_model_call(
        request, lambda r: (captured.append(r), _make_response())[1]
    )

    assert captured[0] is request


def test_call_reaches_provider_without_keyerror() -> None:
    """The middleware must forward to the handler without raising `KeyError`."""
    middleware = GoogleReasoningFixMiddleware()
    messages = [
        AIMessage(
            content=[
                {"type": "reasoning", "text": "step one"},
                {"type": "reasoning"},
            ]
        ),
    ]
    request = _make_request(_make_model("google_genai"), messages)

    def handler(r: ModelRequest) -> ModelResponse[Any]:
        for msg in r.messages:
            if isinstance(msg, AIMessage) and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") in {
                        "reasoning",
                        "thinking",
                    }:
                        _ = block["thinking"]
        return _make_response()

    middleware.wrap_model_call(request, handler)
