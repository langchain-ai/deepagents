"""Shell tool middleware that survives HITL pauses."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents.middleware.shell_tool import (
    ShellToolMiddleware,
    ShellToolState,
    _PersistentShellTool,
    _SessionResources,
)
from langchain.agents.middleware.types import AgentState
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


class ResumableShellToolMiddleware(ShellToolMiddleware):
    """Shell middleware that recreates session resources after human interrupts.

    ``ShellToolMiddleware`` stores its session handle in middleware state using an
    ``UntrackedValue``. When a run pauses for human approval, that attribute is not
    checkpointed. Upon resuming, LangGraph restores the state without the shell
    resources, so the next tool execution fails with
    ``Shell session resources are unavailable``.

    This subclass lazily recreates the shell session the first time a resumed run
    touches the shell tool again and only performs shutdown when a session is
    actually active. This keeps behaviour identical for uninterrupted runs while
    allowing HITL pauses to succeed.
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if isinstance(request.tool, _PersistentShellTool):
            resources = self._get_or_create_resources(request.state)
            return self._run_shell_tool(
                resources,
                request.tool_call["args"],
                tool_call_id=request.tool_call.get("id"),
            )
        return super().wrap_tool_call(request, handler)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if isinstance(request.tool, _PersistentShellTool):
            resources = self._get_or_create_resources(request.state)
            return self._run_shell_tool(
                resources,
                request.tool_call["args"],
                tool_call_id=request.tool_call.get("id"),
            )
        return await super().awrap_tool_call(request, handler)

    def after_agent(self, state: ShellToolState, runtime) -> None:  # type: ignore[override]
        if self._has_resources(state):
            super().after_agent(state, runtime)

    async def aafter_agent(self, state: ShellToolState, runtime) -> None:  # type: ignore[override]
        if self._has_resources(state):
            await super().aafter_agent(state, runtime)

    @staticmethod
    def _has_resources(state: AgentState) -> bool:
        resources = state.get("shell_session_resources")
        return isinstance(resources, _SessionResources)

    def _get_or_create_resources(self, state: AgentState) -> _SessionResources:
        resources = state.get("shell_session_resources")
        if isinstance(resources, _SessionResources):
            return resources

        new_resources = self._create_resources()
        cast("dict[str, Any]", state)["shell_session_resources"] = new_resources
        return new_resources


__all__ = ["ResumableShellToolMiddleware"]
