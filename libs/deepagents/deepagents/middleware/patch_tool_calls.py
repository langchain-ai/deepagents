"""Middleware to record dangling tool calls and keep model requests well paired."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Sequence

    from langchain_core.messages.tool import InvalidToolCall, ToolCall
    from langgraph.runtime import Runtime


class PatchToolCallsMiddleware(AgentMiddleware):
    """Record dangling tool calls in state and pair tool results in each model request.

    A run that is interrupted, or that ends on a malformed tool call, leaves an
    `AIMessage` whose tool calls have no `ToolMessage`. When the next run starts,
    its input already follows that call. This middleware:

    - appends an error `ToolMessage` for each dangling call to the end of state,
      so state says what happened, in the order it was noticed, without
      rewriting earlier messages;
    - moves each tool result directly after its call in the model request, which
      is the order providers require.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Append an error `ToolMessage` for every tool call that has no result."""
        missing = [_error_tool_message(tool_call) for tool_call in _dangling_tool_calls(state["messages"])]
        return {"messages": missing} if missing else None

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Pair tool results with their calls before the request reaches the model."""
        return handler(_repair_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async variant of `wrap_model_call`."""
        return await handler(_repair_request(request))


def _repair_request(request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
    repaired = repair_tool_call_pairing(request.messages)
    if repaired is None:
        return request
    return request.override(messages=repaired)


def repair_tool_call_pairing(messages: Sequence[AnyMessage]) -> list[AnyMessage] | None:
    """Put each tool result directly after its call, in the order providers require.

    Results keep their relative order. A result whose call is not in `messages`
    (for example, its call was summarized away) or that repeats an earlier result
    is left out. A call with no result gets an error `ToolMessage`.

    Args:
        messages: Conversation to repair.

    Returns:
        The repaired conversation, or `None` when it is already well paired.
    """
    call_ids = {tool_call["id"] for tool_call in _tool_calls(messages) if tool_call["id"] is not None}
    results: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id in call_ids:
            results.setdefault(msg.tool_call_id, msg)
    order = {id(result): position for position, result in enumerate(results.values())}

    repaired: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue
        repaired.append(msg)
        if not isinstance(msg, AIMessage):
            continue
        answered: list[ToolMessage] = []
        for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls):
            tool_call_id = tool_call["id"]
            if tool_call_id is None:
                continue
            result = results.pop(tool_call_id, None)
            if result is not None:
                answered.append(result)
            else:
                repaired.append(_error_tool_message(tool_call))
        repaired.extend(sorted(answered, key=lambda result: order[id(result)]))

    unchanged = len(repaired) == len(messages) and all(a is b for a, b in zip(repaired, messages, strict=True))
    return None if unchanged else repaired


def _tool_calls(messages: Sequence[AnyMessage]) -> Iterator[ToolCall | InvalidToolCall]:
    for msg in messages:
        if isinstance(msg, AIMessage):
            yield from msg.tool_calls
            yield from msg.invalid_tool_calls


def _dangling_tool_calls(messages: Sequence[AnyMessage]) -> list[ToolCall | InvalidToolCall]:
    answered_ids = {msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)}
    dangling: list[ToolCall | InvalidToolCall] = []
    for tool_call in _tool_calls(messages):
        if tool_call["id"] is not None and tool_call["id"] not in answered_ids:
            dangling.append(tool_call)
            answered_ids.add(tool_call["id"])
    return dangling


def _error_tool_message(tool_call: ToolCall | InvalidToolCall) -> ToolMessage:
    name = tool_call["name"] or "unknown"
    tool_call_id = tool_call["id"]
    if tool_call.get("type") == "invalid_tool_call":
        content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
    else:
        content = f"Tool call {name} with id {tool_call_id} did not complete - no result was recorded. It may have been cancelled or interrupted."
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error")
