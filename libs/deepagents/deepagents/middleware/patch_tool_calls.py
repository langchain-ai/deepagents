"""Middleware to patch dangling tool calls in the messages history."""

from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, TracePolicy, hook_config, omit_payload
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


class PatchToolCallsState(AgentState):
    """State for bounded invalid tool call retries."""

    invalid_tool_call_retry_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


class PatchToolCallsMiddleware(AgentMiddleware[PatchToolCallsState]):
    """Middleware to patch dangling tool calls in the messages history."""

    state_schema = PatchToolCallsState

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
                tool_call_id = tool_call["id"]
                if tool_call_id is None or tool_call_id in answered_ids:
                    continue
                name = tool_call["name"] or "unknown"
                if tool_call.get("type") == "invalid_tool_call":
                    content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
                else:
                    content = f"Tool call {name} with id {tool_call_id} was cancelled - another message came in before it could be completed."
                patched_messages.append(ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error"))

        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *patched_messages]}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: PatchToolCallsState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Retry once when the model emits an invalid tool call."""
        messages = state["messages"]
        if (
            not messages
            or not isinstance(messages[-1], AIMessage)
            or not messages[-1].invalid_tool_calls
            or state.get("invalid_tool_call_retry_count", 0) > 0
        ):
            return None
        return {
            "invalid_tool_call_retry_count": 1,
            "jump_to": "model",
            "messages": [RemoveMessage(id=messages[-1].id)],
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: PatchToolCallsState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Async variant of `after_model`."""
        return self.after_model(state, runtime)
