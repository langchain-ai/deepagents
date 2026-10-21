"""Lets a deep agent started as an async subagent wake the agent that started it."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal
from uuid import uuid4

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
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from langgraph.errors import GraphInterrupt
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from deepagents.middleware._mailbox import _Mail, _OutboxState, _post, _unannounced
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.async_subagents import (
    _current_run_id,
    _failure_text,
    _final_content,
    _launching_agent,
    _notify_parent,
    _task_event,
)


class _MessageParentSchema(BaseModel):
    message: str = Field(description="A question, blocker, or progress update for the agent that launched this task.")


class _ParentNotifierMiddleware(AgentMiddleware[_OutboxState, ContextT, ResponseT]):
    """Message the agent that started this run with `start_async_task`, including when it ends.

    `create_deep_agent` attaches it to every deep agent. It does nothing unless the
    run carries the parent reference, and skips nested runs.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    state_schema = _OutboxState

    def __init__(self) -> None:
        super().__init__()
        self._reported: dict[str, None] = {}
        self.tools = [
            StructuredTool.from_function(
                name="message_parent",
                coroutine=self._amessage_parent,
                description=(
                    "Send a question, blocker, or progress update to the agent that launched this async task. "
                    "This does not finish or pause your task. The parent receives it before a model call, after any pending human approval."
                ),
                args_schema=_MessageParentSchema,
            )
        ]

    async def _amessage_parent(self, message: str, runtime: ToolRuntime) -> str | Command:
        """Save an update without marking this run finished."""
        config = runtime.config
        if not _launching_agent(config) or not _is_top_level_graph(config):
            return "No launching agent is available for this graph."
        run_id = _current_run_id(config)
        mail = _Mail(id=f"{run_id}:{runtime.tool_call_id or uuid4()}", run_id=run_id, content=message)
        return Command(
            update={
                "deepagents_outbox": _post(mail),
                "messages": [
                    ToolMessage("Message saved for your parent. Continue working; this task is still running.", tool_call_id=runtime.tool_call_id)
                ],
            }
        )

    def before_model(self, state: _OutboxState, runtime: Runtime[ContextT]) -> None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Pass through: parent notifications use the async Agent Server path."""
        return

    async def abefore_model(self, state: _OutboxState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Ring an idle parent's doorbell once new mail has been checkpointed; a busy parent pulls it itself."""
        config = get_config()
        rung = _unannounced(state.get("deepagents_outbox"))
        if rung is None or not _launching_agent(config) or not _is_top_level_graph(config):
            return None
        if not await _notify_parent(config, _task_event(config, _agent_name(config), "message"), runtime.context):
            return None
        return {"deepagents_outbox": {"mail": [], "rung": rung}}

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
        await _notify_parent(config, _task_event(config, _agent_name(config), status, **details), context)


def _agent_name(config: Mapping[str, Any]) -> str:
    metadata = config.get("metadata") or {}
    return str(metadata.get("subagent_name") or metadata.get("graph_id") or "subagent")


def _is_top_level_graph(config: Mapping[str, Any]) -> bool:
    """Whether this is the outermost graph of a task run, not a graph nested in it."""
    return "|" not in str((config.get("configurable") or {}).get("checkpoint_ns") or "")
