"""Middleware to patch dangling tool calls in the messages history."""

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    InvalidToolCall,
    RemoveMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in the messages history."""

    def __init__(self, repair_callback: Callable[[ToolCall | InvalidToolCall], ToolMessage] | None = None) -> None:
        """Initialize the middleware with an optional repair callback."""
        super().__init__()
        self.repair_callback = repair_callback

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
                if self.repair_callback:
                    patched_messages.append(self.repair_callback(tool_call))
                    continue

                name = tool_call["name"] or "unknown"
                if tool_call.get("type") == "invalid_tool_call":
                    content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
                else:
                    content = f"Tool call {name} with id {tool_call_id} has no recorded result."
                patched_messages.append(ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error"))

        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *patched_messages]}
