"""Reference extension: middleware around agent tool calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from deepagents_code.extensions import ExtensionAPI


class ToolCounter(AgentMiddleware):
    """Count tool calls without changing their results."""

    name = "ToolCounter"

    def __init__(self) -> None:
        """Initialize the tool-call count."""
        self.count = 0

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Count and forward one tool call.

        Args:
            request: Pending tool call.
            handler: Remaining middleware chain.

        Returns:
            Unchanged downstream tool result.
        """
        self.count += 1
        return handler(request)


def extension(d: ExtensionAPI) -> None:
    """Register tool-counting middleware.

    Args:
        d: The dcode extension API.
    """
    d.register_middleware(ToolCounter)
