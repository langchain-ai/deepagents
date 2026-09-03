from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

from deepagents.middleware._prompt_caching import _AnthropicPromptCachingMiddleware, _is_anthropic_vertex
from tests.unit_tests.chat_model import GenericFakeChatModel


class AnthropicVertexFakeChatModel(GenericFakeChatModel):
    @property
    def _llm_type(self) -> str:
        return "anthropic-chat-vertexai"


def test_anthropic_vertex_receives_prompt_caching() -> None:
    model = AnthropicVertexFakeChatModel(messages=iter([]))
    request = ModelRequest(model=model, messages=[HumanMessage("Hello")])
    middleware = _AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")

    captured: list[ModelRequest] = []

    def handler(cached: ModelRequest) -> ModelResponse:
        captured.append(cached)
        return ModelResponse(result=[])

    module = SimpleNamespace(ChatAnthropicVertex=AnthropicVertexFakeChatModel)
    with patch("deepagents.middleware._prompt_caching.import_module", return_value=module):
        middleware.wrap_model_call(request, handler)

    assert captured[0].model_settings["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_other_models_do_not_receive_anthropic_prompt_caching() -> None:
    model = GenericFakeChatModel(messages=iter([]))
    request = ModelRequest(model=model, messages=[HumanMessage("Hello")])
    middleware = _AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")

    captured: list[ModelRequest] = []

    def handler(cached: ModelRequest) -> ModelResponse:
        captured.append(cached)
        return ModelResponse(result=[])

    module = SimpleNamespace(ChatAnthropicVertex=AnthropicVertexFakeChatModel)
    with patch("deepagents.middleware._prompt_caching.import_module", return_value=module):
        middleware.wrap_model_call(request, handler)

    assert captured[0] is request
    assert "cache_control" not in captured[0].model_settings


def test_anthropic_vertex_optional_dependency_can_be_absent() -> None:
    model = AnthropicVertexFakeChatModel(messages=iter([]))
    error = ModuleNotFoundError(name="langchain_google_vertexai")

    with patch("deepagents.middleware._prompt_caching.import_module", side_effect=error):
        assert not _is_anthropic_vertex(model)


def test_anthropic_vertex_preserves_unrelated_import_errors() -> None:
    model = AnthropicVertexFakeChatModel(messages=iter([]))

    with (
        patch(
            "deepagents.middleware._prompt_caching.import_module",
            side_effect=ImportError(name="missing_transitive"),
        ),
        pytest.raises(ImportError),
    ):
        _is_anthropic_vertex(model)
