"""Middleware to patch dangling tool calls in the messages history."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in the messages history."""

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Before the agent runs, handle dangling tool calls from any AIMessage."""
        messages = state["messages"]
        if not messages:
            return None

        answered_ids = {msg.tool_call_id for msg in messages if msg.type == "tool"}  # ty: ignore[unresolved-attribute]

        patched_messages = []
        had_patch = False
        for msg in messages:
            patched_messages.append(msg)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["id"] not in answered_ids:
                        had_patch = True
                        patched_messages.append(
                            ToolMessage(
                                content=(
                                    f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                                    "cancelled - another message came in before it could be completed."
                                ),
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        if not had_patch:
            return None

        return {"messages": Overwrite(patched_messages)}
