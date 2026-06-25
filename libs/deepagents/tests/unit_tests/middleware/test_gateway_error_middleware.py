"""Unit tests for `LLMGatewayErrorMiddleware`."""

from __future__ import annotations

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from deepagents.errors import LLMGatewayError
from deepagents.middleware.gateway_error import LLMGatewayErrorMiddleware


async def test_raises_on_empty_message_with_4xx_status() -> None:
    """Empty AIMessage + 4xx response_metadata.status_code surfaces as `LLMGatewayError`."""
    message = AIMessage(
        content="",
        response_metadata={"status_code": 403, "body": "missing permission gateway:invoke\n"},
    )

    async def handler(_request: object) -> ModelResponse[object]:
        return ModelResponse(result=[message])

    middleware = LLMGatewayErrorMiddleware()

    with pytest.raises(LLMGatewayError) as exc_info:
        await middleware.awrap_model_call(None, handler)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 403
    assert "missing permission gateway:invoke" in exc_info.value.body


async def test_passthrough_when_status_ok() -> None:
    """Non-error responses are returned unchanged."""
    message = AIMessage(content="hello", response_metadata={"status_code": 200})

    async def handler(_request: object) -> ModelResponse[object]:
        return ModelResponse(result=[message])

    middleware = LLMGatewayErrorMiddleware()

    response = await middleware.awrap_model_call(None, handler)  # type: ignore[arg-type]

    assert isinstance(response, ModelResponse)
    assert response.result == [message]


async def test_passthrough_when_content_non_empty() -> None:
    """A 4xx response with non-empty content is not treated as a swallowed error."""
    message = AIMessage(
        content="partial reply",
        response_metadata={"status_code": 503, "body": "transient"},
    )

    async def handler(_request: object) -> ModelResponse[object]:
        return ModelResponse(result=[message])

    middleware = LLMGatewayErrorMiddleware()

    response = await middleware.awrap_model_call(None, handler)  # type: ignore[arg-type]

    assert isinstance(response, ModelResponse)
    assert response.result == [message]


async def test_raises_on_empty_list_content_with_5xx_status() -> None:
    """List-of-blocks content with no text triggers the error path."""
    message = AIMessage(
        content=[{"type": "text", "text": "   "}],
        response_metadata={"status_code": 502, "body": "bad gateway"},
    )

    async def handler(_request: object) -> ModelResponse[object]:
        return ModelResponse(result=[message])

    middleware = LLMGatewayErrorMiddleware()

    with pytest.raises(LLMGatewayError) as exc_info:
        await middleware.awrap_model_call(None, handler)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 502
    assert exc_info.value.body == "bad gateway"
