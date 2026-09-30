"""Expendable background delegation through the SDK task tool."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from contextlib import aclosing
from copy import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime  # noqa: TC002  # tools inspect injected annotations
from langchain_core.messages import SystemMessage, ToolMessage, convert_to_messages
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph_sdk import get_client

from deepagents_talon.authorization import set_authorization_handler
from deepagents_talon.tool_approvals import APPROVAL_OPERATOR

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Sequence

    from deepagents import CompiledSubAgent, SubAgent
    from deepagents.middleware.async_subagents import AsyncSubAgent
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.runnables import RunnableConfig
    from langgraph_sdk.schema import StreamPart

logger = logging.getLogger(__name__)

_IN_SUBAGENT: contextvars.ContextVar[bool] = contextvars.ContextVar("talon_subagent", default=False)
# Set by the runtime for a scheduled run, which has no user to keep talking to and no later
# turn to deliver into: its delegations finish inside the turn that made them.
_SCHEDULED_TURN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "talon_scheduled_turn", default=False
)
_MAX_TASKS = 128
_MAX_RUNNING = 4
_MAX_DELIVERIES = 3
_TASK_TIMEOUT_SECONDS = 3600
# Deliberately not _TASK_TIMEOUT_SECONDS. That hour was priced for a detached worker, where
# an overrun costs one idle task. Inline it costs the whole fleet: the cron ticker runs due
# jobs one at a time, and a stalled run does not delay later fires so much as delete them,
# because a job's next run time is claimed before the run starts.
_INLINE_TIMEOUT_SECONDS = 600
# Inline delegation never enters the job table, so it escapes _MAX_RUNNING and scheduled
# fan-out needs its own ceiling. A semaphore queues rather than refusing: a scheduled run
# has nobody to retry a refusal, so rejecting degrades its answer where waiting only
# degrades its latency.
_MAX_INLINE_RUNNING = 4
_MAX_RESULT_CHARACTERS = 64_000
_FAILED_RESULT = "Subagent failed before returning a result."
_TIMED_OUT_RESULT = "Subagent ran out of time before returning a result."
# Async task tools the main agent never sees, so approval gates on them can never fire.
HIDDEN_ASYNC_TOOLS = frozenset(
    {
        "check_async_task",
        "list_async_tasks",
        "cancel_async_task",
        "update_async_task",
    }
)
# A scheduled run's delegations finish inside its own tool call, so it never owns a job:
# these two could only ever report nothing and cancel nothing. Hidden per turn at the model
# call, not unregistered, because chat turns on the same process still need them.
_SCHEDULED_HIDDEN_TOOLS = frozenset({"list_subagents", "cancel_subagent"})
_UNDELIVERED_RESULT = (
    f"The conversation failed to process this result {_MAX_DELIVERIES} times, "
    "so it was dropped and never reached the user."
)
_INSTRUCTIONS = (
    "The task and start_async_task tools launch background subagents and return a task ID. "
    "Keep talking to the user while they work. Use list_subagents to inspect progress and "
    "cancel_subagent to stop work. Completed results will arrive as subagent data for you to "
    "process; do not repeatedly poll or wait for them."
)
# Every clause of _INSTRUCTIONS is false on a scheduled run, and believing it loses work: a
# model told the result arrives later ends its turn without one, and a scheduled turn that
# says nothing is suppressed rather than delivered.
_SCHEDULED_INSTRUCTIONS = (
    "This is a scheduled run with nobody to talk to while it works. The task and "
    "start_async_task tools run a subagent to completion and return its result to you "
    "directly, so act on that result in this same turn -- no later turn will."
)


@dataclass
class _Job:
    owner: str
    name: str
    worker: asyncio.Task[None] | None = None
    result: str | None = None
    cancelled: bool = False
    notified: bool = False
    deliveries: int = 0
    tools: list[str] | None = None

    @property
    def status(self) -> str:
        if self.worker is not None and not self.worker.done():
            return "cancelling" if self.cancelled else "running"
        return "cancelled" if self.cancelled else "finished"


class BackgroundSubagents(AgentMiddleware):
    """Keep SDK task invocations alive independently of the main conversation turn."""

    def __init__(self, inline_timeout: float = _INLINE_TIMEOUT_SECONDS) -> None:
        """Keep task handles and results in memory only.

        Args:
            inline_timeout: Seconds a scheduled run may spend in one delegation.
        """
        self._jobs: dict[str, _Job] = {}
        self._lock = asyncio.Lock()
        # Built here rather than per configured copy: `configured` shallow-copies this
        # middleware, so one ceiling covers every graph sharing these workers.
        self._inline_slots = asyncio.Semaphore(_MAX_INLINE_RUNNING)
        self._inline_timeout = inline_timeout
        self._remote: dict[str, AsyncSubAgent] = {}

        @tool
        async def list_subagents(runtime: ToolRuntime) -> list[dict[str, str | list[str] | None]]:
            """Inspect this thread's subagents, including their status and final results."""
            owner = runtime.config.get("configurable", {}).get("thread_id")
            return [
                {
                    "task_id": key,
                    "name": job.name,
                    "status": job.status,
                    "result": job.result,
                    "tools": job.tools,
                }
                for key, job in self._jobs.items()
                if job.owner == owner
            ]

        @tool
        async def cancel_subagent(task_id: str, runtime: ToolRuntime) -> str:
            """Cancel a background subagent belonging to this conversation."""
            owner = runtime.config.get("configurable", {}).get("thread_id")
            job = self._jobs.get(task_id)
            if job is None or job.owner != owner:
                return "Unknown subagent for this conversation."
            await self._cancel_jobs([job])
            return job.status

        self.tools = [list_subagents, cancel_subagent]

    def configured(
        self, subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent]
    ) -> BackgroundSubagents:
        """Bind remote targets to this graph while sharing the in-memory workers."""
        middleware = copy(self)
        middleware._remote = {  # noqa: SLF001  # configure a copy of this middleware
            spec["name"]: cast("AsyncSubAgent", spec) for spec in subagents if "graph_id" in spec
        }
        return middleware

    def owners(self) -> set[str]:
        """Return threads with running work or unprocessed results."""
        return {job.owner for job in self._jobs.values() if not job.cancelled and not job.notified}

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Tell the main agent how far its delegations get before this turn ends."""
        if _IN_SUBAGENT.get():
            return await handler(request)
        scheduled = _SCHEDULED_TURN.get()
        instructions = _SCHEDULED_INSTRUCTIONS if scheduled else _INSTRUCTIONS
        hidden = HIDDEN_ASYNC_TOOLS | _SCHEDULED_HIDDEN_TOOLS if scheduled else HIDDEN_ASYNC_TOOLS
        blocks = request.system_message.content_blocks if request.system_message else []
        message = SystemMessage(content_blocks=[*blocks, {"type": "text", "text": instructions}])
        return await handler(
            request.override(
                system_message=message,
                tools=[tool for tool in request.tools if getattr(tool, "name", "") not in hidden],
            )
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Detach the existing SDK task tool while preserving its compiled subagent graph."""
        if request.tool_call["name"] == "start_async_task" and _IN_SUBAGENT.get():
            return ToolMessage(
                "Delegate remote work from the main agent.", tool_call_id=request.tool_call["id"]
            )
        if request.tool_call["name"] not in {"task", "start_async_task"} or _IN_SUBAGENT.get():
            return await handler(request)
        # A scheduled run has nobody to talk to while a subagent works and no later turn to
        # deliver into, so it waits for the result and answers with it. Resolved after the
        # refusals above, so nested delegation stays banned, and before the lock below,
        # which guards a job table this path never touches: holding it across a whole
        # subagent would serialize a fan-out the tool node otherwise runs concurrently.
        if _SCHEDULED_TURN.get():
            return await self._inline(request, handler)
        owner = request.runtime.config.get("configurable", {}).get("thread_id")
        if not isinstance(owner, str) or not owner:
            return ToolMessage(
                "A conversation thread is required.", tool_call_id=request.tool_call["id"]
            )
        async with self._lock:
            for key in [
                key
                for key, job in self._jobs.items()
                if job.notified or (job.cancelled and job.status == "cancelled")
            ]:
                del self._jobs[key]
            if (
                len(self._jobs) >= _MAX_TASKS
                or sum(
                    job.worker is not None and not job.worker.done() for job in self._jobs.values()
                )
                >= _MAX_RUNNING
            ):
                return ToolMessage(
                    "Background subagent capacity reached.", tool_call_id=request.tool_call["id"]
                )
            task_id = f"subagent-{uuid4().hex}"
            job = _Job(owner, str(request.tool_call["args"].get("subagent_type", "")))
            job.tools = request.tool_call["args"].get("tools")
            self._jobs[task_id] = job
            job.worker = asyncio.create_task(
                self._run(job, request, task_id), name=task_id, context=contextvars.copy_context()
            )
        return ToolMessage(
            f"Started background subagent. task_id: {task_id}", tool_call_id=request.tool_call["id"]
        )

    async def _inline(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Run one delegation to completion inside the calling turn.

        Reached only from a scheduled turn, which already holds the posture `_run` forces
        on a worker: no operator to answer an approval and no authorization handler.

        Nothing may propagate out of here. The tool node re-raises anything that is not a
        tool invocation error, and the runtime treats a timeout as retryable, so an
        escaping deadline would re-invoke the whole graph and relaunch every sibling
        delegation in the same assistant message.

        Args:
            request: Delegating tool call from a scheduled turn.
            handler: Downstream tool invocation, which is the subagent itself.

        Returns:
            The subagent's result, or why it produced none.
        """
        # Held for the same reason `_run` sets it: a subagent must not itself delegate.
        # Inline runs in the caller's context rather than a copy, so this has to be undone.
        token = _IN_SUBAGENT.set(True)
        try:
            # Acquired outside the deadline so a queued call spends its budget on work.
            async with self._inline_slots:
                timeout = asyncio.timeout(self._inline_timeout)
                try:
                    async with timeout:
                        if request.tool_call["name"] == "start_async_task":
                            # Not the SDK tool: that one returns a task id for a poller
                            # this turn does not have, and leaves a running entry behind
                            # on a thread every later fire of this job reuses.
                            return ToolMessage(
                                await self._run_remote(request),
                                tool_call_id=request.tool_call["id"],
                            )
                        return _clamp(await handler(request))
                except Exception:
                    logger.exception("Inline subagent failed")
                    # This result reaches the model and the user: no arguments, no
                    # credentials.
                    return ToolMessage(
                        _TIMED_OUT_RESULT if timeout.expired() else _FAILED_RESULT,
                        tool_call_id=request.tool_call["id"],
                        status="error",
                    )
        finally:
            _IN_SUBAGENT.reset(token)

    async def _run(self, job: _Job, request: ToolCallRequest, task_id: str) -> None:
        _IN_SUBAGENT.set(True)
        APPROVAL_OPERATOR.set(False)
        # The copied context carries the host's history scope and cron origin, which the
        # tools a subagent may hold require. It must not carry the authorization handler:
        # a flow started once the originating turn has ended would outlive the host's
        # `_clear_authorization`, stranding a pending prompt in the conversation. Enabling
        # background authorization needs host-side cleanup first.
        set_authorization_handler(None)
        config: RunnableConfig = {"configurable": {"thread_id": task_id}, "recursion_limit": 500}
        runtime = replace(request.runtime, config=config, state=dict(request.runtime.state))
        call = {**request.tool_call, "args": {**request.tool_call["args"], "runtime": runtime}}
        timeout = asyncio.timeout(_TASK_TIMEOUT_SECONDS)
        try:
            async with timeout:
                if request.tool_call["name"] == "start_async_task":
                    job.result = await self._run_remote(request)
                    return
                if request.tool is None:
                    job.result = "Subagent tool is unavailable"
                    return
                result = await request.tool.ainvoke(call, config)
            if isinstance(result, Command) and isinstance(result.update, dict):
                if result.update.get("__interrupt__"):
                    job.result = "Subagent needs tool approval; the protected action has not run."
                else:
                    messages = result.update.get("messages", [])
                    job.result = (
                        str(messages[-1].content) if messages else "Subagent returned no result."
                    )
            else:
                job.result = str(getattr(result, "content", result))
            job.result = job.result[:_MAX_RESULT_CHARACTERS]
        except asyncio.CancelledError:
            job.cancelled = True
        except Exception:
            logger.exception("Background subagent %s failed", task_id)
            # This result reaches the model and the user: no arguments, no credentials.
            job.result = _TIMED_OUT_RESULT if timeout.expired() else _FAILED_RESULT

    async def _run_remote(self, request: ToolCallRequest) -> str:
        spec = self._remote[request.tool_call["args"]["subagent_type"]]
        client = get_client(
            url=spec.get("url"), headers={"x-auth-scheme": "langsmith", **spec.get("headers", {})}
        )
        stream = client.runs.stream(
            None,
            spec["graph_id"],
            input={
                "messages": [{"role": "user", "content": request.tool_call["args"]["description"]}]
            },
            stream_mode="values",
            on_disconnect="cancel",
        )
        result = "Subagent returned no result."
        async with aclosing(cast("AsyncGenerator[StreamPart, None]", stream)):
            async for part in stream:
                if part.event == "error":
                    msg = "Remote subagent failed"
                    raise RuntimeError(msg)
                if part.event == "values" and isinstance(part.data, dict):
                    if part.data.get("__interrupt__"):
                        result = "Subagent needs tool approval; the protected action has not run."
                    elif messages := part.data.get("messages"):
                        result = str(convert_to_messages(messages)[-1].content)[
                            :_MAX_RESULT_CHARACTERS
                        ]
        return result

    def results(self, owner: str) -> dict[str, str]:
        """Return pending results as data for the owning main agent.

        Args:
            owner: Conversation thread receiving the results.
        """
        return {
            key: f"Background subagent {job.name} ({key}) returned the following data. "
            f"Process it in the context of the user's request.\n"
            f"<subagent_result>\n{job.result}\n</subagent_result>"
            for key, job in self._jobs.items()
            if job.owner == owner
            and job.result is not None
            and not job.cancelled
            and not job.notified
        }

    def record_delivery_failure(self, results: dict[str, str]) -> list[str]:
        """Count one failed delivery and drop results the main agent cannot process.

        Args:
            results: Result IDs handed to a main-agent turn that then failed.

        Returns:
            Result IDs dropped after too many failed delivery attempts.
        """
        dropped = []
        for key in results:
            job = self._jobs.get(key)
            if job is None or job.notified:
                continue
            job.deliveries += 1
            if job.deliveries < _MAX_DELIVERIES:
                continue
            job.notified = True
            job.result = f"{_UNDELIVERED_RESULT}\n{job.result}"
            dropped.append(key)
            logger.warning("Dropped undelivered background subagent result %s", key)
        return dropped

    def requeue(self, results: Iterable[str]) -> None:
        """Return acknowledged results to the pending set after an undelivered turn.

        A turn that completed its model work acknowledges the results it consumed,
        but the host may then discard its reply because a newer turn superseded it.
        The user therefore never heard about work that is already marked delivered.
        Clearing the flag offers it to the next turn instead; re-injection is
        idempotent, because the result is carried as a message keyed by its own id.

        Only the ids handed back are touched, so a result delivered by some earlier
        turn is never resurrected. An id already pruned or cancelled is skipped:
        there is nothing left to offer.

        Args:
            results: Result IDs whose turn produced a reply that was discarded.
        """
        for key in results:
            job = self._jobs.get(key)
            if job is not None and not job.cancelled:
                job.notified = False

    def acknowledge(self, results: dict[str, str]) -> None:
        """Mark results processed only after the main agent completes a turn.

        Args:
            results: Result IDs included in the completed main turn.
        """
        for key in results:
            if key in self._jobs:
                self._jobs[key].notified = True

    async def cancel(self, owner: str | None = None) -> bool:
        """Cancel a thread's workers and discard results, or clear all at shutdown.

        Args:
            owner: Conversation to stop; omit to shut down all workers.

        Returns:
            Whether every selected worker has stopped.
        """
        jobs = [job for job in self._jobs.values() if owner is None or job.owner == owner]
        for job in jobs:
            job.cancelled = True
        return await self._cancel_jobs(jobs)

    async def _cancel_jobs(self, jobs: list[_Job]) -> bool:
        workers = []
        for job in jobs:
            if job.worker is not None and not job.worker.done():
                job.cancelled = True
                job.worker.cancel()
                workers.append(job.worker)
        if not workers:
            return True
        _done, pending = await asyncio.wait(workers, timeout=10)
        return not pending


def _clamp(result: ToolMessage | Command) -> ToolMessage | Command:
    """Bound one inline subagent result.

    `_run` clamps before storing a background result. Inline the subagent's output goes
    straight into the scheduled thread that every later fire of the job reuses, so an
    unclamped result grows that thread without limit.

    Args:
        result: What the delegating tool returned.

    Returns:
        The result, with oversized message text truncated.
    """
    if isinstance(result, Command):
        if not isinstance(result.update, dict):
            return result
        messages = result.update.get("messages")
        if not isinstance(messages, list) or not messages:
            return result
        return replace(
            result,
            update={**result.update, "messages": [_clamped(item) for item in messages]},
        )
    return cast("ToolMessage", _clamped(result))


def _clamped(message: object) -> object:
    """Truncate one message's text, leaving structured content alone.

    Args:
        message: Message carried back from a subagent.

    Returns:
        The message, or a copy of it holding only the leading characters.
    """
    content = getattr(message, "content", None)
    if not isinstance(content, str) or len(content) <= _MAX_RESULT_CHARACTERS:
        return message
    copier = getattr(message, "model_copy", None)
    if not callable(copier):
        return message
    return copier(update={"content": content[:_MAX_RESULT_CHARACTERS]})
