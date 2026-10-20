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
from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from langgraph.errors import GraphInterrupt
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.async_subagents import (
    PrepareRun,
    _current_run_id,
    _failure_text,
    _final_content,
    _launching_agent,
    _notify_parent,
    _task_event,
)


class _MessageParentSchema(BaseModel):
    message: str = Field(description="A question, blocker, or progress update for the agent that launched this task.")


class _ParentNotifierMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Message the agent that started this run with `start_async_task`, including when it ends.

    `create_deep_agent` attaches it to every deep agent. It does nothing unless the
    run carries the parent reference, and skips nested runs.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def __init__(self, *, prepare_run: PrepareRun | None = None) -> None:
        super().__init__()
        self._prepare_run = prepare_run
        self._reported: dict[str, None] = {}
        self.tools = [
            StructuredTool.from_function(
                name="message_parent",
                coroutine=self._amessage_parent,
                description=(
                    "Send a question, blocker, or progress update to the agent that launched this async task. "
                    "This does not finish or pause your task. A busy parent receives it after its current run."
                ),
                args_schema=_MessageParentSchema,
            )
        ]

    async def _amessage_parent(self, message: str, runtime: ToolRuntime) -> str:
        """Send an update without marking this run finished."""
        config = runtime.config
        if not _launching_agent(config) or not _is_top_level_graph(config):
            return "No launching agent is available for this graph."
        metadata = config.get("metadata") or {}
        name = str(metadata.get("subagent_name") or metadata.get("graph_id") or "subagent")
        delivered = await _notify_parent(
            config,
            _task_event(config, name, "message", message=message),
            runtime.context,
            prepare_run=self._prepare_run,
            tool_call_id=runtime.tool_call_id,
        )
        if not delivered:
            return "Message not delivered: the parent is unavailable or waiting for human input."
        return "Message queued for your parent. Continue working; this task is still running."

    def _model_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Expose parent messaging only to a launched, top-level graph."""
        config = get_config()
        if not _launching_agent(config) or not _is_top_level_graph(config):
            return request.override(tools=[tool for tool in request.tools if tool is not self.tools[0]])
        prompt = (
            "You are running as an async subagent. Use `message_parent` to ask your launching agent a question "
            "or flag a blocker while you continue working. Sending a message does not finish or pause your task."
        )
        return request.override(system_message=append_to_system_message(request.system_message, prompt))

    def after_agent(self, state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Pass through: reporting needs the async path, which the Agent Server uses."""
        return None

    async def aafter_agent(self, state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Report a finished run with its final answer."""
        await self._areport("success", runtime.context, result=_final_content(state))
        return None

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Expose parent messaging only in a launched, top-level graph."""
        return handler(self._model_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Report a model call that fails the run."""
        try:
            return await handler(self._model_request(request))
        except GraphInterrupt:
            raise
        except Exception as e:
            await self._areport("error", request.runtime.context, error=_failure_text(e))
            raise

    async def _areport(self, status: Literal["success", "error"], context: ContextT, **details: object) -> None:
        """Notify once per run, for this graph's own run that an async subagent tool started."""
        config = get_config()
        # Graphs nested in this run (e.g. a sync subagent that's a deep agent) inherit its config,
        # parent reference included; only the top-level graph reports, so the parent gets this agent's answer.
        if not _launching_agent(config) or not _is_top_level_graph(config):
            return
        run_key = _current_run_id(config)
        if run_key is None or run_key in self._reported:
            return
        self._reported[run_key] = None
        if len(self._reported) > 1000:  # noqa: PLR2004  # bounded memory in long-lived processes
            del self._reported[next(iter(self._reported))]
        metadata = config.get("metadata") or {}
        name = str(metadata.get("subagent_name") or metadata.get("graph_id") or "subagent")
        await _notify_parent(config, _task_event(config, name, status, **details), context, prepare_run=self._prepare_run)


def _is_top_level_graph(config: Mapping[str, Any]) -> bool:
    """Whether this is the outermost graph of a task run, not a graph nested in it."""
    return "|" not in str((config.get("configurable") or {}).get("checkpoint_ns") or "")
