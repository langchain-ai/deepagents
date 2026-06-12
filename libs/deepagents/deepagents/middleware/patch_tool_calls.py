"""Middleware to patch dangling tool calls in the messages history."""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

if TYPE_CHECKING:
    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
        ResponseT,
    )


def _normalize_thinking_blocks(messages: list[AnyMessage]) -> tuple[list[AnyMessage], bool]:
    """Rewrite bare `thinking` content parts to Anthropic's nested schema.

    Anthropic's extended-thinking API rejects bare `{"type": "thinking",
    "text": "..."}` parts with `messages.N.content.M.thinking.thinking:
    Field required` — the inner key must be `"thinking"`, not `"text"`.
    Some upstream serializers drop the conversation history into the bare
    shape, so we normalize before every outbound call.

    Returns `(messages, changed)` where `changed` is True iff any part was rewritten.
    """
    patched: list[AnyMessage] = []
    any_changed = False
    for msg in messages:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            patched.append(msg)
            continue
        new_content: list[Any] = []
        changed = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "thinking" and "thinking" not in part and "text" in part:
                new_part = {**part, "thinking": part["text"]}
                new_part.pop("text", None)
                new_content.append(new_part)
                changed = True
                continue
            new_content.append(part)
        if changed:
            patched.append(msg.model_copy(update={"content": new_content}))
            any_changed = True
        else:
            patched.append(msg)
    return patched, any_changed


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in the messages history."""

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Before the agent runs, handle dangling tool calls from any AIMessage."""
        messages = state["messages"]
        if not messages:
            return None

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
                    content = f"Tool call {name} with id {tool_call_id} was cancelled - another message came in before it could be completed."
                patched_messages.append(ToolMessage(content=content, name=name, tool_call_id=tool_call_id))

        return {"messages": Overwrite(patched_messages)}

    def wrap_model_call(
        self,
        request: "ModelRequest[Any]",
        handler: Callable[["ModelRequest[Any]"], "ModelResponse[Any]"],
    ) -> "ModelResponse[Any]":
        """Normalize Anthropic `thinking` blocks in the outbound message list."""
        normalized, changed = _normalize_thinking_blocks(list(request.messages))
        if changed:
            request = request.override(messages=normalized)
        return handler(request)

    async def awrap_model_call(
        self,
        request: "ModelRequest[Any]",
        handler: Callable[["ModelRequest[Any]"], Awaitable["ModelResponse[ResponseT]"]],
    ) -> "ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]":
        """Async variant of `wrap_model_call`."""
        normalized, changed = _normalize_thinking_blocks(list(request.messages))
        if changed:
            request = request.override(messages=normalized)
        return await handler(request)
