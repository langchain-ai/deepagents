"""Middleware to patch dangling tool calls in the messages history."""

from collections.abc import Mapping
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    TracePolicy,
    hook_config,
    omit_payload,
)
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to repair incomplete tool calls in the messages history."""

    @staticmethod
    def _error_tool_message(tool_call: Mapping[str, Any]) -> ToolMessage | None:
        """Create an error result for an identified tool call."""
        tool_call_id = tool_call["id"]
        if tool_call_id is None:
            return None
        name = tool_call["name"] or "unknown"
        if tool_call.get("type") == "invalid_tool_call":
            content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
        else:
            content = f"Tool call {name} with id {tool_call_id} was cancelled - another message came in before it could be completed."
        return ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error")

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Before the agent runs, repair dangling tool calls from prior turns."""
        messages = state["messages"]
        if not messages:
            return None

        answered_ids = {msg.tool_call_id for msg in messages if msg.type == "tool"}
        patched_messages = []
        for msg in messages:
            patched_messages.append(msg)
            if not isinstance(msg, AIMessage):
                continue
            for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls):
                error_message = self._error_tool_message(tool_call)
                if error_message is not None and error_message.tool_call_id not in answered_ids:
                    patched_messages.append(error_message)

        if len(patched_messages) == len(messages):
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *patched_messages]}

    def _recover_invalid_tool_calls(self, state: AgentState) -> dict[str, Any] | None:
        """Give the model error feedback for malformed calls and retry this turn."""
        messages = state["messages"]
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        last_message = messages[-1]
        error_messages = []
        for tool_call in last_message.invalid_tool_calls:
            error_message = self._error_tool_message(tool_call)
            if error_message is not None:
                error_messages.append(error_message)

        if not error_messages:
            return None
        if last_message.tool_calls:
            return {"messages": error_messages}
        return {"messages": error_messages, "jump_to": "model"}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Retry the current turn when the model emits malformed tool arguments."""
        return self._recover_invalid_tool_calls(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Async variant of `after_model`."""
        return self._recover_invalid_tool_calls(state)
