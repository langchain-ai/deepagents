"""Expendable background delegation through the SDK task tool."""

from __future__ import annotations

import asyncio
import contextvars
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

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from deepagents import CompiledSubAgent, SubAgent
    from deepagents.middleware.async_subagents import AsyncSubAgent
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.runnables import RunnableConfig
    from langgraph_sdk.schema import StreamPart

_IN_SUBAGENT: contextvars.ContextVar[bool] = contextvars.ContextVar("talon_subagent", default=False)
_MAX_TASKS = 128
_MAX_RUNNING = 4
_INSTRUCTIONS = (
    "The task and start_async_task tools launch background subagents and return a task ID. "
    "Keep talking to the user while they work. Use list_subagents to inspect progress and "
    "cancel_subagent to stop work. Completed results will arrive as subagent data for you to "
    "process; do not repeatedly poll or wait for them."
)


@dataclass
class _Job:
    owner: str
    name: str
    worker: asyncio.Task[None] | None = None
    result: str | None = None
    cancelled: bool = False
    notified: bool = False
    tools: list[str] | None = None

    @property
    def status(self) -> str:
        if self.worker is not None and not self.worker.done():
            return "cancelling" if self.cancelled else "running"
        return "cancelled" if self.cancelled else "finished"


class BackgroundSubagents(AgentMiddleware):
    """Keep SDK task invocations alive independently of the main conversation turn."""

    def __init__(self) -> None:
        """Keep task handles and results in memory only."""
        self._jobs: dict[str, _Job] = {}
        self._lock = asyncio.Lock()
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
        """Tell the main agent that delegation returns before the work finishes."""
        if _IN_SUBAGENT.get():
            return await handler(request)
        blocks = request.system_message.content_blocks if request.system_message else []
        message = SystemMessage(content_blocks=[*blocks, {"type": "text", "text": _INSTRUCTIONS}])
        return await handler(
            request.override(
                system_message=message,
                tools=[
                    tool
                    for tool in request.tools
                    if getattr(tool, "name", "")
                    not in {
                        "check_async_task",
                        "list_async_tasks",
                        "cancel_async_task",
                        "update_async_task",
                    }
                ],
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
                self._run(job, request, task_id), name=task_id, context=contextvars.Context()
            )
        return ToolMessage(
            f"Started background subagent. task_id: {task_id}", tool_call_id=request.tool_call["id"]
        )

    async def _run(self, job: _Job, request: ToolCallRequest, task_id: str) -> None:
        _IN_SUBAGENT.set(True)
        config: RunnableConfig = {"configurable": {"thread_id": task_id}, "recursion_limit": 500}
        runtime = replace(request.runtime, config=config, state=dict(request.runtime.state))
        call = {**request.tool_call, "args": {**request.tool_call["args"], "runtime": runtime}}
        try:
            async with asyncio.timeout(3600):
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
            job.result = job.result[:64_000]
        except asyncio.CancelledError:
            job.cancelled = True
        except Exception:  # noqa: BLE001  # report failures without exposing tool arguments or credentials
            job.result = "Subagent failed before returning a result."

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
                        result = str(convert_to_messages(messages)[-1].content)[:64_000]
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
