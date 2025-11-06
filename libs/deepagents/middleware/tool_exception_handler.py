"""Middleware to handle tool exceptions gracefully."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
import traceback

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import ToolException
from langgraph.types import Command


class ToolExceptionHandlerMiddleware(AgentMiddleware):
    """Middleware that catches tool exceptions and returns them as normal tool results.
    
    This allows the AI to see error messages and continue the workflow instead of
    having the entire flow interrupted by a ToolException.
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Wrap synchronous tool call to catch exceptions.
        
        Args:
            request: The tool call request containing tool info and arguments
            handler: The original tool handler function
            
        Returns:
            ToolMessage or Command with either the successful result or error message
        """
        try:
            return handler(request)
        except ToolException as e:
            # Convert ToolException to a normal error message that the AI can see
            error_msg = f"Tool execution failed: {str(e)}"
            return ToolMessage(
                content=error_msg,
                name=request.tool.name,
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
        except Exception as e:
            # Catch any other unexpected exceptions to prevent flow interruption
            error_msg = f"Tool execution encountered an unexpected error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return ToolMessage(
                content=error_msg,
                name=request.tool.name,
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Wrap asynchronous tool call to catch exceptions.
        
        Args:
            request: The tool call request containing tool info and arguments
            handler: The original async tool handler function
            
        Returns:
            ToolMessage or Command with either the successful result or error message
        """
        try:
            return await handler(request)
        except ToolException as e:
            # Convert ToolException to a normal error message that the AI can see
            error_msg = f"Tool execution failed: {str(e)}"
            return ToolMessage(
                content=error_msg,
                name=request.tool.name,
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
        except Exception as e:
            # Catch any other unexpected exceptions to prevent flow interruption
            error_msg = f"Tool execution encountered an unexpected error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return ToolMessage(
                content=error_msg,
                name=request.tool.name,
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

