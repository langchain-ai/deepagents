"""Lets a deep agent started as an async subagent wake the agent that started it."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langgraph.config import get_config
from langgraph.errors import GraphInterrupt
from langgraph.runtime import Runtime

from deepagents.middleware.async_subagents import (
    _current_run_id,
    _failure_text,
    _final_content,
    _launching_agent,
    _notify_parent,
    _task_event,
)


class _ParentNotifierMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Wake the agent that started this run with `start_async_task`, when the run ends.

    `create_deep_agent` attaches it to every deep agent. It does nothing unless the
    run carries the parent reference, and skips nested runs.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def __init__(self) -> None:
        super().__init__()
        self._reported: dict[str, None] = {}

    def after_agent(self, state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Pass through: reporting needs the async path, which the Agent Server uses."""
        return None

    async def aafter_agent(self, state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Report a finished run with its final answer."""
        await self._areport("success", result=_final_content(state))
        return None

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
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

    async def _areport(self, status: Literal["success", "error"], **details: object) -> None:
        """Notify once per run, for this graph's own run that an async subagent tool started."""
        config = get_config()
        if not _launching_agent(config) or not _is_own_task_run(config):
            return
        run_key = _current_run_id(config)
        if run_key is None or run_key in self._reported:
            return
        self._reported[run_key] = None
        if len(self._reported) > 1000:  # noqa: PLR2004  # bounded memory in long-lived processes
            del self._reported[next(iter(self._reported))]
        name = str((config.get("metadata") or {}).get("graph_id") or "subagent")
        await _notify_parent(config, _task_event(config, name, status, **details))


def _is_own_task_run(config: Mapping[str, Any]) -> bool:
    """Whether this is the outermost graph of a task run, not a graph nested in it."""
    return "|" not in str((config.get("configurable") or {}).get("checkpoint_ns") or "")
