"""Middleware for enforcing ToolPermission rules on tool invocations."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

import wcmatch.fnmatch as wcfnmatch
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.permissions import ToolPermission

_WCMATCH_FLAGS = wcfnmatch.BRACE


def _check_tool_permission(
    rules: list[ToolPermission],
    tool_name: str,
    args: dict,
) -> Literal["allow", "deny"]:
    """Evaluate tool permission rules for a tool invocation.

    Iterates rules in declaration order. The first matching rule's mode
    is applied. If no rule matches, returns `"allow"` (permissive default).

    A rule matches when its `name` equals `tool_name` and every entry in
    `args` (if any) glob-matches the corresponding arg value.

    Args:
        rules: Ordered list of `ToolPermission` rules to evaluate.
        tool_name: The name of the tool being invoked.
        args: The arguments passed to the tool call.

    Returns:
        `"allow"` if the call should proceed, `"deny"` if it should be blocked.
    """
    for rule in rules:
        if rule.name != tool_name:
            continue
        if rule.args is not None and not all(
            wcfnmatch.fnmatch(str(args.get(arg_name, "")), pattern, flags=_WCMATCH_FLAGS) for arg_name, pattern in rule.args.items()
        ):
            continue
        return rule.mode
    return "allow"


class ToolPermissionMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware that enforces `ToolPermission` rules on tool invocations.

    Intercepts each tool call via `wrap_tool_call` / `awrap_tool_call`. On
    deny, returns an error `ToolMessage` without invoking the underlying tool.

    This middleware must be placed last in the stack so it sees the final set
    of tools (including those injected by `FilesystemMiddleware`,
    `SubAgentMiddleware`, etc.).

    Args:
        rules: The tool permission rules to enforce. Rules are evaluated in
            declaration order; the first match wins. If no rule matches, the
            call is allowed (permissive default).

    Example:
        ```python
        from deepagents.permissions import ToolPermission
        from deepagents.middleware.tool_permissions import ToolPermissionMiddleware

        # Allow only pytest, deny everything else
        middleware = ToolPermissionMiddleware(
            rules=[
                ToolPermission(name="execute", args={"command": "pytest *"}),
                ToolPermission(name="execute", mode="deny"),
            ]
        )
        ```
    """

    def __init__(self, *, rules: list[ToolPermission]) -> None:
        """Initialize the tool permission middleware.

        Args:
            rules: Ordered list of `ToolPermission` rules to enforce.
        """
        self._rules = rules

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Pass through — permission checks happen in `wrap_tool_call`.

        Args:
            request: The model request being processed.
            handler: The next handler in the chain.

        Returns:
            The model response from the handler.
        """
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Pass through — permission checks happen in `awrap_tool_call`.

        Args:
            request: The model request being processed.
            handler: The next handler in the chain.

        Returns:
            The model response from the handler.
        """
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Enforce tool permission rules before invoking the tool.

        Args:
            request: The tool call request being processed.
            handler: The next handler in the chain.

        Returns:
            An error `ToolMessage` on deny, otherwise the handler result.
        """
        tool_name = request.tool_call["name"]
        args = request.tool_call.get("args", {}) or {}
        if _check_tool_permission(self._rules, tool_name, args) == "deny":
            return ToolMessage(
                content="Error: tool use denied by permission",
                name=tool_name,
                tool_call_id=request.tool_call["id"],
            )
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async) Enforce tool permission rules before invoking the tool.

        Args:
            request: The tool call request being processed.
            handler: The next handler in the chain.

        Returns:
            An error `ToolMessage` on deny, otherwise the handler result.
        """
        tool_name = request.tool_call["name"]
        args = request.tool_call.get("args", {}) or {}
        if _check_tool_permission(self._rules, tool_name, args) == "deny":
            return ToolMessage(
                content="Error: tool use denied by permission",
                name=tool_name,
                tool_call_id=request.tool_call["id"],
            )
        return await handler(request)
