"""Middleware to repair tool call and tool result pairing in the messages history."""

import logging
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

logger = logging.getLogger(__name__)

_ID_KEYED_CALL_TYPES = frozenset({"tool_use", "tool_call"})
"""Content block types that hold their call id under `id` (Anthropic and LangChain v1)."""


def _block_call_id(block: object) -> str | None:
    """Return the call id of a client-executed tool call content block, or `None` for other blocks.

    OpenAI Responses blocks for client-executed calls (`function_call`, `computer_call`,
    `custom_tool_call`, and so on) hold the id under `call_id`. Server-side calls hold only
    `id` and are answered inside the AI message, not by a `ToolMessage`. A `non_standard`
    block wraps the provider block in `value`.
    """
    if not isinstance(block, dict):
        return None
    value = block.get("value")
    call = value if block.get("type") == "non_standard" and isinstance(value, dict) else block
    block_type = call.get("type")
    if block_type in _ID_KEYED_CALL_TYPES:
        call_id = call.get("id")
    elif isinstance(block_type, str) and not block_type.endswith("_output"):
        call_id = call.get("call_id")
    else:
        return None
    return call_id if isinstance(call_id, str) else None


def _block_call_ids(message: AIMessage) -> Iterator[str]:
    """Yield the ids of tool calls found in `content` blocks.

    These often repeat ids from `tool_calls`; callers handle the overlap.
    """
    if isinstance(message.content, list):
        for block in message.content:
            if (call_id := _block_call_id(block)) is not None:
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


def _results_for(message: AIMessage, results: dict[str, list[ToolMessage]]) -> list[ToolMessage]:
    """Claim one result for each of the message's tool calls, synthesizing any that are missing."""
    calls = [call for call in (*message.tool_calls, *message.invalid_tool_calls) if call["id"] is not None]
    paired = [candidates.pop(0) if (candidates := results.get(call["id"])) else _synthetic_result(call) for call in calls]
    # A call found only in a content block keeps a recorded result but gets no synthetic
    # one, since the block does not say whether the call was valid or ever executed.
    parsed = {call["id"] for call in calls}
    paired.extend(candidates.pop(0) for call_id in _block_call_ids(message) if call_id not in parsed and (candidates := results.get(call_id)))
    return paired


def _call_ids(message: AIMessage) -> Iterator[str]:
    """Yield parsed, invalid, and content block tool call ids for a single turn."""
    for call in (*message.tool_calls, *message.invalid_tool_calls):
        if call["id"] is not None:
            yield call["id"]
    yield from _block_call_ids(message)


def _first_owners(messages: list[AnyMessage]) -> dict[str, int]:
    """Map each call id to the index of the first AI message that carries it."""
    owners: dict[str, int] = {}
    for index, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            for call_id in _call_ids(msg):
                owners.setdefault(call_id, index)
    return owners


def _collect_results(messages: list[AnyMessage]) -> dict[int, dict[str, list[ToolMessage]]]:
    """Group results by owning AI message index and call id, best candidate first.

    A result belongs to the nearest preceding call with its id, or else to the first
    following one. Results that answer no call are left out. Candidates keep their recorded
    order, except that error results and results recorded before their call rank last, so
    a real result beats a synthetic error or a stale result for the same call.
    """
    owners = _first_owners(messages)
    found: dict[int, dict[str, list[tuple[tuple[bool, bool], ToolMessage]]]] = {}
    for index, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            owners.update((call_id, index) for call_id in _call_ids(msg))
        elif isinstance(msg, ToolMessage) and (owner := owners.get(msg.tool_call_id)) is not None:
            rank = (msg.status == "error", index < owner)
            found.setdefault(owner, {}).setdefault(msg.tool_call_id, []).append((rank, msg))
    return {
        owner: {call_id: [msg for _, msg in sorted(candidates, key=lambda item: item[0])] for call_id, candidates in by_id.items()}
        for owner, by_id in found.items()
    }


def _repair(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Place one result directly after each call, synthesizing missing ones.

    Duplicate results and results that answer no call are left out.
    """
    results = _collect_results(messages)
    repaired: list[AnyMessage] = []
    for index, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            continue
        repaired.append(msg)
        if isinstance(msg, AIMessage):
            repaired.extend(_results_for(msg, results.get(index, {})))
    return repaired


def _log_dropped(messages: list[AnyMessage], repaired: list[AnyMessage]) -> None:
    """Log each recorded result that the repair left out of the model request."""
    kept = {id(msg) for msg in repaired}
    call_ids = {call_id for msg in messages if isinstance(msg, AIMessage) for call_id in _call_ids(msg)}
    for msg in messages:
        if not isinstance(msg, ToolMessage) or id(msg) in kept:
            continue
        if msg.tool_call_id in call_ids:
            logger.debug("Left a duplicate result for tool call %s out of the model request", msg.tool_call_id)
        else:
            logger.warning(
                "Left the result for tool call %s out of the model request: no call in the request has that id. "
                "The call may have been summarized away or use an unrecognized content block type.",
                msg.tool_call_id,
            )


def _patch_dangling(messages: list[AnyMessage]) -> list[AnyMessage] | None:
    """Return a copy of `messages` with an error result after each unanswered call, or `None` if there are none.

    A call counts as answered when any `ToolMessage` in the history has its id, wherever it
    is. Calls found only in content blocks are not patched. `_repair` fixes the order and
    reused ids for each model request.
    """
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
    unanswered call. On every model request, it also moves one result directly after each
    call, prefers a real result over an error or a stale result, drops duplicate results
    and results that answer no call, and adds an error result for each call that is still
    unanswered (for example, when a call id is reused). Dropped results are logged.

    The request repair does not write to the thread, so indices recorded against thread
    state, such as a summarization cutoff, stay valid.
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
        """Return `request` unchanged when its pairing is valid, otherwise an override with repaired messages."""
        repaired = _repair(request.messages)
        if len(repaired) == len(request.messages) and all(left is right for left, right in zip(repaired, request.messages, strict=True)):
            return request
        _log_dropped(request.messages, repaired)
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
