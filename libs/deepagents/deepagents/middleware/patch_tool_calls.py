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
        # Iterate over the messages and add any dangling tool calls (or missing AI messages)
        for i, msg in enumerate(messages):
            patched_messages.append(msg)

            # Case 1: AI message with tool_calls but no following ToolMessage
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    corresponding_tool_msg = next(
                        (args for args in messages[i:] if args.type == "tool" and args.tool_call_id == tool_call["id"]),
                        None,
                    )
                    if corresponding_tool_msg is None:
                        # We have a dangling tool call which needs a ToolMessage
                        tool_msg = (
                            f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                            "cancelled - another message came in before it could be completed."
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

            # Case 2: ToolMessage without a following AI message (and not last message, or next is not AI)
            # This happens if execution was interrupted after tool output but before model response
            if msg.type == "tool":
                # Check if this tool message is followed by an AI message
                next_msg = messages[i + 1] if i + 1 < len(messages) else None
                if next_msg is None or next_msg.type != "ai":
                    # We have a ToolMessage but no AI response follow-up.
                    # We must close the loop with a synthetic AI message.
                    patched_messages.append(AIMessage(content="Tool execution completed.", tool_calls=[], id=f"patched_ai_{msg.id}"))

        return {"messages": Overwrite(patched_messages)}
