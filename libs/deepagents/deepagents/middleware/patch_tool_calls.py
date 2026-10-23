"""Middleware to repair tool call and tool result pairing in the messages history."""

from collections.abc import Awaitable, Callable, Iterator
from typing import Any

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
from langchain_core.messages import AIMessage, AnyMessage, InvalidToolCall, RemoveMessage, ToolCall, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

_CALL_BLOCK_ID_KEYS = {"tool_use": "id", "tool_call": "id", "function_call": "call_id"}
"""Content block types that carry a tool call, mapped to the key holding its id."""


def _block_call_ids(message: AIMessage) -> Iterator[str]:
    """Yield ids of tool calls carried only in provider-native content blocks."""
    if not isinstance(message.content, list):
        return
    for block in message.content:
        if isinstance(block, dict) and (key := _CALL_BLOCK_ID_KEYS.get(block.get("type"))):
            call_id = block.get(key)
            if isinstance(call_id, str):
                yield call_id


def _synthetic_result(tool_call: ToolCall | InvalidToolCall) -> ToolMessage:
    """Build the error result recorded for a tool call that has none."""
    tool_call_id = tool_call["id"]
    name = tool_call["name"] or "unknown"
    if tool_call.get("type") == "invalid_tool_call":
        content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
    else:
        content = f"Tool call {name} with id {tool_call_id} did not complete - no result was recorded. It may have been cancelled or interrupted."
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error")


def _results_for(message: AIMessage, results: dict[str, ToolMessage]) -> list[ToolMessage]:
    """Claim each of the message's tool call results, synthesizing any that are missing."""
    paired: list[ToolMessage] = []
    for tool_call in (*message.tool_calls, *message.invalid_tool_calls):
        if tool_call["id"] is None:
            continue
        result = results.pop(tool_call["id"], None)
        paired.append(result if result is not None else _synthetic_result(tool_call))
    # Calls the content carries without a parsed `tool_calls` entry keep a recorded result
    # but get no synthetic one, since their name and kind are unknown.
    paired.extend(result for call_id in _block_call_ids(message) if (result := results.pop(call_id, None)) is not None)
    return paired


def _call_ids(message: AIMessage) -> Iterator[str]:
    """Yield parsed and provider-native tool call ids for a single turn."""
    for call in (*message.tool_calls, *message.invalid_tool_calls):
        if call["id"] is not None:
            yield call["id"]
    yield from _block_call_ids(message)


def _collect_results(messages: list[AnyMessage]) -> dict[int, dict[str, ToolMessage]]:
    """Associate results with the nearest preceding call, or the first following call."""
    owners: dict[str, int] = {}
    for index, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            for call_id in _call_ids(msg):
                owners.setdefault(call_id, index)
    results: dict[int, dict[str, ToolMessage]] = {}
    for index, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            owners.update((call_id, index) for call_id in _call_ids(msg))
        elif isinstance(msg, ToolMessage) and (owner := owners.get(msg.tool_call_id)) is not None:
            results.setdefault(owner, {}).setdefault(msg.tool_call_id, msg)
    return results


def _repair(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Place each tool result directly after its call, dropping results that answer no call."""
    results = _collect_results(messages)
    repaired: list[AnyMessage] = []
    for index, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            continue
        repaired.append(msg)
        if isinstance(msg, AIMessage):
            repaired.extend(_results_for(msg, results.get(index, {})))
    return repaired


def _patch_dangling(messages: list[AnyMessage]) -> list[AnyMessage] | None:
    """Return `messages` with an error result after each unanswered call, or `None` if none are unanswered."""
    answered = {msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)}
    patched: list[AnyMessage] = []
    changed = False
    for msg in messages:
        patched.append(msg)
        if not isinstance(msg, AIMessage):
            continue
        for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls):
            if tool_call["id"] is not None and tool_call["id"] not in answered:
                patched.append(_synthetic_result(tool_call))
                changed = True
    return patched if changed else None


class PatchToolCallsMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Middleware to repair tool call and tool result pairing in the messages history.

    Providers reject a history in which a tool call has no result, a result answers no
    call, or a result is separated from its call. Any of these can be persisted by an
    interrupted or concurrently executed run, and every later model call on the thread
    then fails the same way.

    Before the agent runs, this middleware records an error result in the thread for each
    unanswered call. On every model request, it also moves each result directly after its
    call and drops results that answer no call and duplicate results. That repair applies
    only to the request, so the thread keeps every recorded message.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def before_agent(self, state: AgentState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002
        """Before the agent runs, record an error result for each unanswered tool call."""
        patched = _patch_dangling(state["messages"])
        if patched is None:
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *patched]}

    @staticmethod
    def _repair_request(request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Return `request`, or an override whose messages pair every call with its result."""
        repaired = _repair(request.messages)
        if len(repaired) == len(request.messages) and all(left is right for left, right in zip(repaired, request.messages, strict=True)):
            return request
        return request.override(messages=repaired)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Repair tool call and tool result pairing in the request, then invoke the model.

        Args:
            request: Model request to execute.
            handler: Callback that executes the model request.

        Returns:
            The result of invoking the handler.
        """
        return handler(self._repair_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Repair tool call and tool result pairing in the request, then invoke the model.

        Args:
            request: Model request to execute.
            handler: Async callback that executes the model request.

        Returns:
            The result of invoking the handler.
        """
        return await handler(self._repair_request(request))
