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
        if not messages or len(messages) == 0:
            return None

        patched_messages = []
        # Iterate over the messages and add any dangling tool calls
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            if isinstance(msg, AIMessage) and (msg.tool_calls or msg.invalid_tool_calls):
                # Both lists always default to [] on AIMessage, so unpacking is safe.
                for tool_call in [*msg.tool_calls, *msg.invalid_tool_calls]:
                    call_id = tool_call["id"]
                    if call_id is None:
                        # No id means no ToolMessage can be correlated; skip.
                        continue
                    corresponding_tool_msg = next(
                        (m for m in messages[i:] if m.type == "tool" and m.tool_call_id == call_id),  # ty: ignore[unresolved-attribute]
                        None,
                    )
                    if corresponding_tool_msg is None:
                        # We have a dangling tool call which needs a ToolMessage
                        call_name = tool_call["name"] or "unknown"
                        is_invalid = tool_call.get("type") == "invalid_tool_call"
                        if is_invalid:
                            tool_msg = f"Tool call {call_name} with id {call_id} could not be executed - arguments were malformed or truncated."
                        else:
                            tool_msg = (
                                f"Tool call {call_name} with id {call_id} was cancelled - another message came in before it could be completed."
                            )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=call_name,
                                tool_call_id=call_id,
                            )
                        )

        return {"messages": Overwrite(patched_messages)}
