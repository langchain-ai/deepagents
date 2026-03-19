"""Middleware to patch dangling tool calls in the messages history."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ContextT, ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from deepagents._models import get_model_identifier


class ToolStrictMiddleware(AgentMiddleware):
    """Middleware to enforce strict tool calling for supported models.
    
    Enables 'strict=True' in model.bind_tools() for OpenAI and Google models,
    which improves reliability by forcing the model to adhere to tool schemas.
    """

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject 'strict': True into model_settings for supported providers."""
        identifier = get_model_identifier(request.model) or ""
        model_type = str(type(request.model)).lower()
        
        # OpenAI and Google (Gemini) models generally support strict tool calling
        is_openai = "gpt-" in identifier.lower() or "openai" in model_type
        is_google = "gemini-" in identifier.lower() or "google" in model_type
        
        if is_openai or is_google:
            # Only inject if not already explicitly set
            if "strict" not in request.model_settings:
                new_settings = {**request.model_settings, "strict": True}
                return request.override(model_settings=new_settings)
                
        return request


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
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    corresponding_tool_msg = next(
                        (msg for msg in messages[i:] if msg.type == "tool" and msg.tool_call_id == tool_call["id"]),  # ty: ignore[unresolved-attribute]
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

        return {"messages": Overwrite(patched_messages)}
