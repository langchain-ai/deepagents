"""Middleware for explicit before/after tool-call hooks.

This middleware makes tool-governance integration straightforward for external
systems by exposing callback hooks around every tool invocation.
"""

from collections.abc import Awaitable, Callable, Sequence

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

ToolCallResult = ToolMessage | Command
"""Result type returned by wrapped tool calls."""

BeforeToolCallHook = Callable[[ToolCallRequest], ToolCallResult | None]
"""Sync hook called before executing a tool.

Return a `ToolCallResult` to short-circuit execution, or `None` to continue.
"""

AfterToolCallHook = Callable[[ToolCallRequest, ToolCallResult], ToolCallResult | None]
"""Sync hook called after executing a tool.

Return a replacement `ToolCallResult` to override, or `None` to keep original.
"""

AsyncBeforeToolCallHook = Callable[[ToolCallRequest], Awaitable[ToolCallResult | None]]
"""Async hook called before executing a tool."""

AsyncAfterToolCallHook = Callable[[ToolCallRequest, ToolCallResult], Awaitable[ToolCallResult | None]]
"""Async hook called after executing a tool."""


class ToolHooksMiddleware(AgentMiddleware):
    """Run explicit callbacks before and after every tool call.

    This is useful for plugging in policy engines, audit sinks, and
    request/response validators without requiring each integration to define a
    custom middleware class.
    """

    def __init__(
        self,
        *,
        before_call_hooks: Sequence[BeforeToolCallHook] = (),
        after_call_hooks: Sequence[AfterToolCallHook] = (),
        async_before_call_hooks: Sequence[AsyncBeforeToolCallHook] = (),
        async_after_call_hooks: Sequence[AsyncAfterToolCallHook] = (),
    ) -> None:
        self._before_call_hooks = list(before_call_hooks)
        self._after_call_hooks = list(after_call_hooks)
        self._async_before_call_hooks = list(async_before_call_hooks)
        self._async_after_call_hooks = list(async_after_call_hooks)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResult],
    ) -> ToolCallResult:
        """Apply sync hooks around a tool call."""
        for hook in self._before_call_hooks:
            maybe_result = hook(request)
            if maybe_result is not None:
                return maybe_result

        tool_result = handler(request)

        for hook in self._after_call_hooks:
            maybe_result = hook(request, tool_result)
            if maybe_result is not None:
                tool_result = maybe_result

        return tool_result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolCallResult]],
    ) -> ToolCallResult:
        """Apply sync and async hooks around a tool call in async contexts."""
        for hook in self._before_call_hooks:
            maybe_result = hook(request)
            if maybe_result is not None:
                return maybe_result

        for hook in self._async_before_call_hooks:
            maybe_result = await hook(request)
            if maybe_result is not None:
                return maybe_result

        tool_result = await handler(request)

        for hook in self._after_call_hooks:
            maybe_result = hook(request, tool_result)
            if maybe_result is not None:
                tool_result = maybe_result

        for hook in self._async_after_call_hooks:
            maybe_result = await hook(request, tool_result)
            if maybe_result is not None:
                tool_result = maybe_result

        return tool_result
