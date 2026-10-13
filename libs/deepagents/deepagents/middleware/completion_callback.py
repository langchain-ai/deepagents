"""Middleware that lets a separately deployed async subagent wake the agent that started it."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.errors import GraphInterrupt
from langgraph.runtime import Runtime
from langgraph.types import Command

from deepagents.middleware.async_subagents import (
    _CALLBACK_STATE_KEY,
    _current_run_id,
    _failure_text,
    _final_text,
    _notify_parent,
    _task_event,
    parent_reference,
)


class CompletionCallbackState(AgentState):
    """State marking that this thread's runs report their outcome to the agent that started them."""

    deepagents_callback: NotRequired[bool]


class CompletionCallbackMiddleware(AgentMiddleware[CompletionCallbackState, ContextT, ResponseT]):
    """Wake the agent that started this graph as an async subagent when a run ends.

    !!! warning "Experimental"

        This middleware may change without notice.

    Add it to a separately deployed graph that agents start with
    `start_async_task`. When a run finishes, fails in a model call, or pauses
    with `interrupt()` inside a tool, it notifies the starting agent if that
    agent is on the same deployment; anything else is found by the starting
    agent's scheduled checks. Put it first in `middleware` so it only sees
    model errors once retries have given up.

    Args:
        name: Name used for this subagent in notifications. Defaults to the
            graph ID of the run.

    Example:
        ```python
        from deepagents import create_deep_agent
        from deepagents.middleware import CompletionCallbackMiddleware

        researcher = create_deep_agent(model=..., middleware=[CompletionCallbackMiddleware()])
        ```
    """

    state_schema = CompletionCallbackState

    def __init__(self, *, name: str | None = None) -> None:
        """Initialize the `CompletionCallbackMiddleware`."""
        super().__init__()
        self._name = name
        self._reported: dict[str, None] = {}

    def before_agent(self, state: CompletionCallbackState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Mark this thread as reporting its outcome, so the starting agent checks it less often."""
        return {_CALLBACK_STATE_KEY: True} if parent_reference(get_config()) else None

    async def abefore_agent(self, state: CompletionCallbackState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """(async) Mark this thread as reporting its outcome, so the starting agent checks it less often."""
        return self.before_agent(state, runtime)

    async def aafter_agent(self, state: CompletionCallbackState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Report a finished run with its final answer."""
        await self._areport("success", result=_final_text(state))
        return None

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Pass through: reporting needs the async path, which the Agent Server uses."""
        return handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Pass through: reporting needs the async path, which the Agent Server uses."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Report a model call that fails the run."""
        try:
            return await handler(request)
        except GraphInterrupt:
            raise
        except Exception as e:
            await self._areport("error", error=_failure_text(e))
            raise

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Report a run that pauses with `interrupt()` inside a tool."""
        try:
            return await handler(request)
        except GraphInterrupt as paused:
            interrupts = [{"id": item.id, "value": item.value} for item in (paused.args[0] if paused.args else ())]
            await self._areport("waiting", interrupts=interrupts)
            raise

    async def _areport(self, status: Literal["success", "error", "waiting"], **details: object) -> None:
        """Queue the notification once per run, when an async subagent tool started it."""
        config = get_config()
        if not parent_reference(config):
            return
        run_key = _current_run_id(config)
        if run_key is None or run_key in self._reported:
            return
        self._reported[run_key] = None
        if len(self._reported) > 1000:  # noqa: PLR2004  # bounded memory in long-lived processes
            del self._reported[next(iter(self._reported))]
        name = self._name or str((config.get("metadata") or {}).get("graph_id") or "subagent")
        await _notify_parent(config, _task_event(config, name, status, **details))
