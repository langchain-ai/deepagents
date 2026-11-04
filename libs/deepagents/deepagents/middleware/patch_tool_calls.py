"""Middleware to patch dangling tool calls in the messages sent to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.messages.tool import InvalidToolCall, ToolCall


class PatchToolCallsMiddleware(AgentMiddleware):
    """Answer dangling tool calls in each model request.

    A run that is interrupted, or that ends on a malformed tool call, leaves an
    `AIMessage` whose tool calls have no `ToolMessage`. Providers reject that
    history, so every model request gets an error `ToolMessage` right after each
    dangling call.

    State is left as is: by the time the agent starts, the new input already
    follows the dangling call, so fixing state means rewriting the whole list.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Answer dangling tool calls before the request reaches the model."""
        return handler(_patch_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async variant of `wrap_model_call`."""
        return await handler(_patch_request(request))


def _patch_request(request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
    patched = patch_dangling_tool_calls(request.messages)
    if patched is None:
        return request
    return request.override(messages=patched)


def patch_dangling_tool_calls(messages: Sequence[AnyMessage]) -> list[AnyMessage] | None:
    """Insert an error `ToolMessage` after every tool call that has no result.

    Args:
        messages: Conversation to patch.

    Returns:
        The patched conversation, or `None` when no tool call is dangling.
    """
    answered_ids = {msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)}
    patched: list[AnyMessage] = []
    changed = False
    for msg in messages:
        patched.append(msg)
        if not isinstance(msg, AIMessage):
            continue
        for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls):
            if tool_call["id"] is None or tool_call["id"] in answered_ids:
                continue
            patched.append(_error_tool_message(tool_call))
            answered_ids.add(tool_call["id"])
            changed = True
    return patched if changed else None


def _error_tool_message(tool_call: ToolCall | InvalidToolCall) -> ToolMessage:
    name = tool_call["name"] or "unknown"
    tool_call_id = tool_call["id"]
    if tool_call.get("type") == "invalid_tool_call":
        content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
    else:
        content = f"Tool call {name} with id {tool_call_id} did not complete - no result was recorded. It may have been cancelled or interrupted."
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error")
