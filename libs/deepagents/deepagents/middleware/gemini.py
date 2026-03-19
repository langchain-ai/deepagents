"""Middleware to fix empty responses from Gemini models (ghosting)."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command


class GeminiEmptyResponseMiddleware(AgentMiddleware):
    """Middleware to detect and handle empty responses from Gemini models.
    
    If the model returns an empty AIMessage (no text, no tool calls), this
    middleware returns a Command to add a system reminder and retry.
    """

    def after_agent(self, response: Any, state: AgentState, runtime: Runtime[Any]) -> Any:
        """Check if the response is an empty AIMessage."""
        if not isinstance(response, AIMessage):
            return response
            
        # Check if message is empty (no content and no tool calls)
        # Handle both empty string and empty list content
        has_content = bool(response.content)
        if isinstance(response.content, list) and not response.content:
            has_content = False
            
        is_empty = not has_content and not response.tool_calls
        
        if is_empty:
            # Return a Command to update history and retry
            return Command(
                update={
                    "messages": [
                        response, # Keep the empty message in history for context
                        ToolMessage(
                            content="System: The last response was empty. Please provide a substantive response or call a tool.",
                            tool_call_id="retry", # Dummy ID
                        )
                    ]
                }
            )
            
        return response
