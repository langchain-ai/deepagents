"""Middleware to patch dangling tool calls in the messages sent to the model."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, TracePolicy, omit_payload
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage


def _patch_dangling_tool_calls(messages: list[AnyMessage]) -> list[AnyMessage] | None:
    """Return `messages` with a synthetic error `ToolMessage` after each unanswered tool call, or `None` if none dangle."""
    answered_ids = {msg.tool_call_id for msg in messages if msg.type == "tool"}

    if not any(
        tool_call["id"] is not None and tool_call["id"] not in answered_ids
        for msg in messages
        if isinstance(msg, AIMessage)
        for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls)
    ):
        return None

    patched_messages: list[AnyMessage] = []
    for msg in messages:
        patched_messages.append(msg)
        if not isinstance(msg, AIMessage):
            continue
        for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls):
            tool_call_id = tool_call["id"]
            if tool_call_id is None or tool_call_id in answered_ids:
                continue
            name = tool_call["name"] or "unknown"
            if tool_call.get("type") == "invalid_tool_call":
                content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
            else:
                content = (
                    f"Tool call {name} with id {tool_call_id} did not complete - no result was recorded. It may have been cancelled or interrupted."
                )
            patched_messages.append(ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error"))
    return patched_messages


class PatchToolCallsMiddleware(AgentMiddleware):
    """Pair dangling tool calls with synthetic error `ToolMessage`s in each model request.

    The patch is applied to the request only. State is never rewritten, so the
    thread's history stays append-only.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Patch dangling tool calls in the request's messages, then call the model."""
        patched = _patch_dangling_tool_calls(request.messages)
        return handler(request if patched is None else request.override(messages=patched))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Patch dangling tool calls in the request's messages, then call the model."""
        patched = _patch_dangling_tool_calls(request.messages)
        return await handler(request if patched is None else request.override(messages=patched))
