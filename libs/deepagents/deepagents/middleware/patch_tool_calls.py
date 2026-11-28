"""Middleware to patch dangling tool calls in the messages history."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, TracePolicy, omit_payload
from langchain_core.messages import AIMessage, AnyMessage, InvalidToolCall, RemoveMessage, ToolCall, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in the messages history."""

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

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
                if tool_call["id"] is None or tool_call["id"] in answered_ids:
                    continue
                patched_messages.append(_error_tool_message(tool_call))

        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *patched_messages]}

    def on_cancel(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """When a run is cancelled, answer the last AI message's unanswered tool calls."""
        return _answer_trailing_tool_calls(state["messages"])


def _answer_trailing_tool_calls(messages: list[AnyMessage]) -> dict[str, Any] | None:
    # At cancel time the cut-off call is still last, so its answer can be appended in order.
    answered_ids: set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            answered_ids.add(msg.tool_call_id)
            continue
        if not isinstance(msg, AIMessage):
            return None
        missing = [
            _error_tool_message(tool_call)
            for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls)
            if tool_call["id"] is not None and tool_call["id"] not in answered_ids
        ]
        return {"messages": missing} if missing else None
    return None


def _error_tool_message(tool_call: ToolCall | InvalidToolCall) -> ToolMessage:
    tool_call_id = tool_call["id"]
    name = tool_call["name"] or "unknown"
    if tool_call.get("type") == "invalid_tool_call":
        content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
    else:
        content = f"Tool call {name} with id {tool_call_id} did not complete - no result was recorded. It may have been cancelled or interrupted."
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error")
