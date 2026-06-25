"""Middleware that surfaces swallowed LLM gateway errors.

When the chat-model client receives a non-2xx response from the LLM gateway,
some providers (e.g. `langchain_anthropic`, `langchain_openai`) still hand the
agent loop an empty `AIMessage` with the failure details stashed on
`response_metadata`. Without intervention the empty turn is forwarded as the
final reply and the user sees a blank response. This middleware inspects the
result of the wrapped model call and raises a typed `LLMGatewayError` whenever
the message is empty AND `response_metadata.status_code` is in the 4xx/5xx
range, turning the silent failure into a visible one.

TODO(deepagents): the long-term fix is to make the chat-model clients
themselves re-raise non-2xx gateway responses instead of returning an empty
generation; this middleware is a harness-side defensive layer until that lands
upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage

from deepagents.errors import LLMGatewayError

_HTTP_ERROR_MIN = 400
_HTTP_ERROR_MAX = 600

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
        ResponseT,
    )
    from langchain_core.messages import BaseMessage


def _message_text_is_empty(message: BaseMessage) -> bool:
    """Return True when `message.content` is empty/whitespace across string and list-of-blocks shapes."""
    content = message.content
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    return False
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return False
        return True
    return not bool(content)


def _check_message(message: BaseMessage) -> None:
    """Raise `LLMGatewayError` when `message` is an empty AIMessage with a 4xx/5xx gateway status."""
    if not isinstance(message, AIMessage):
        return
    status_code = message.response_metadata.get("status_code")
    if not isinstance(status_code, int) or status_code < _HTTP_ERROR_MIN or status_code >= _HTTP_ERROR_MAX:
        return
    if not _message_text_is_empty(message):
        return
    body = message.response_metadata.get("body", "")
    if not isinstance(body, str):
        body = str(body)
    raise LLMGatewayError(status_code=status_code, body=body)


def _check_response(response: object) -> None:
    """Apply `_check_message` against the messages produced by a wrapped model call."""
    if isinstance(response, AIMessage):
        _check_message(response)
        return
    messages = getattr(response, "result", None)
    if messages is None:
        inner = getattr(response, "model_response", None)
        if inner is not None:
            messages = getattr(inner, "result", None)
    if messages is None:
        return
    for message in messages:
        _check_message(message)


class LLMGatewayErrorMiddleware(AgentMiddleware[Any, Any, Any]):
    """Raise `LLMGatewayError` when the wrapped model call returns an empty `AIMessage` carrying a 4xx/5xx gateway status."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Inspect the response and raise on swallowed gateway errors; otherwise return unchanged."""
        response = handler(request)
        _check_response(response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Async variant of `wrap_model_call`."""
        response = await handler(request)
        _check_response(response)
        return response
