"""Middleware for notifying agent when they provide empty responses."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage


class ToolSafetyMiddleware(AgentMiddleware):
    """Notify agent when they make tool calls without text content.

    Injects a warning when agent provides AIMessage with tool_calls but no content,
    allowing them to course-correct before entering a loop.
    """

    _WARNING_MESSAGE = "SYSTEM WARNING: Your last response had only tool calls with no text. Always include text explaining your reasoning before making tool calls."

    __slots__ = ("_empty_response_detected",)

    def __init__(self):
        super().__init__()
        self._empty_response_detected = False

    def on_model_request(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject warning if flag is set."""
        if not self._empty_response_detected:
            return handler(request)

        self._empty_response_detected = False
        return handler(
            request.override(
                messages=request.messages
                + [ToolMessage(content=self._WARNING_MESSAGE, tool_call_id="tool_safety_warning")]
            )
        )

    def on_model_response(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Detect tool calls without text content."""
        response = handler(request)
        self._detect_tool_calls_without_text(response)
        return response

    async def on_model_request_async(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async: Inject warning if flag is set."""
        if not self._empty_response_detected:
            return await handler(request)

        self._empty_response_detected = False
        return await handler(
            request.override(
                messages=request.messages
                + [ToolMessage(content=self._WARNING_MESSAGE, tool_call_id="tool_safety_warning")]
            )
        )

    async def on_model_response_async(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async: Detect tool calls without text content."""
        response = await handler(request)
        self._detect_tool_calls_without_text(response)
        return response

    def _detect_tool_calls_without_text(self, response: ModelResponse) -> None:
        """Detect if response has tool calls but no text content."""
        if not response.result:
            return

        last_message = response.result[-1]

        if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
            return

        # Extract text content (handles both string and multimodal list)
        content = last_message.content
        if isinstance(content, list):
            content = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)

        if not content or not str(content).strip():
            self._empty_response_detected = True
