from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

from deepagents.middleware._prompt_caching import _AnthropicPromptCachingMiddleware
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

    middleware.wrap_model_call(request, handler)

    assert captured[0] is request
    assert "cache_control" not in captured[0].model_settings
