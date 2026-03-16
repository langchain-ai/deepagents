"""Middleware for async subagents running on remote LangGraph servers.

Async subagents use the LangGraph SDK to launch background runs on remote
LangGraph deployments. Unlike synchronous subagents (which block until
completion), async subagents return a job ID immediately, allowing the main
agent to monitor progress and send updates while the subagent works.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ModelResponse, ResponseT
from langchain.tools import ToolRuntime  # noqa: TC002
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from langgraph_sdk import get_client, get_sync_client

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest
    from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
    from langgraph_sdk.schema import Run


class AsyncSubAgent(TypedDict):
    """Specification for an async subagent running on a remote LangGraph server.

    Async subagents connect to LangGraph deployments via the LangGraph SDK.
    They run as background jobs that the main agent can monitor and update.

    Authentication is handled via environment variables (`LANGGRAPH_API_KEY`,
    `LANGSMITH_API_KEY`, or `LANGCHAIN_API_KEY`), which the LangGraph SDK
    reads automatically.
    """

    name: str
    """Unique identifier for the async subagent."""

    description: str
    """What this subagent does.

    The main agent uses this to decide when to delegate.
    """

    graph_id: str
    """The graph name or assistant ID on the remote server."""

    url: NotRequired[str]
    """URL of the LangGraph server (e.g., `"https://my-deployment.langsmith.dev"`).

    Omit to use ASGI transport for local LangGraph servers.
    """

    headers: NotRequired[dict[str, str]]
    """Additional headers to include in requests to the remote server."""


class AsyncSubAgentJob(TypedDict):
    """A tracked async subagent job persisted in agent state."""

    job_id: str
    """Unique identifier for the job (same as `thread_id`)."""

    agent_name: str
    """Name of the async subagent type that is running."""

    thread_id: str
    """LangGraph thread ID for the remote run."""

    run_id: str
    """LangGraph run ID for the current execution on the thread."""

    status: str
    """Current job status (e.g., `'running'`, `'success'`, `'error'`, `'cancelled'`).

    Typed as `str` rather than a `Literal` because the LangGraph SDK's
    `Run.status` is `str` — using a `Literal` here would require `cast` at every
    SDK boundary.
    """


def _jobs_reducer(
    existing: dict[str, AsyncSubAgentJob] | None,
    update: dict[str, AsyncSubAgentJob],
) -> dict[str, AsyncSubAgentJob]:
    """Merge job updates into the existing jobs dict."""
    merged = dict(existing or {})
    merged.update(update)
    return merged


class AsyncSubAgentState(AgentState):
    """State extension for async subagent job tracking."""

    async_subagent_jobs: Annotated[NotRequired[dict[str, AsyncSubAgentJob]], _jobs_reducer]


ASYNC_TASK_TOOL_DESCRIPTION = """Launch an async subagent on a remote LangGraph server. The subagent runs in the background and returns a job ID immediately.

Available async agent types:
{available_agents}

## Usage notes:
1. This tool launches a background job and returns immediately with a job ID. Report the job ID to the user and stop — do NOT immediately check status.
2. Use `check_async_subagent` only when the user asks for a status update or result.
3. Use `update_async_subagent` to send new instructions to a running job.
4. Multiple async subagents can run concurrently — launch several and let them run in the background.
5. The subagent runs on a remote LangGraph server, so it has its own tools and capabilities."""  # noqa: E501

ASYNC_TASK_SYSTEM_PROMPT = """## Async subagents (remote LangGraph servers)

You have access to async subagent tools that launch background jobs on remote LangGraph servers.

### Tools:
- `launch_async_subagent`: Start a new background job. Returns a job ID immediately.
- `check_async_subagent`: Check the status of a running job. Returns status and result if complete.
- `update_async_subagent`: Send an update or new instructions to a running job.
- `cancel_async_subagent`: Cancel a running job that is no longer needed.
- `list_async_subagent_jobs`: List all tracked jobs with live statuses. Use this to check all jobs at once.

### Workflow:
1. **Launch** — Use `launch_async_subagent` to start a job. Report the job ID to the user and stop.
   Do NOT immediately check the status — the job runs in the background while you and the user continue other work.
2. **Check (on request)** — Only use `check_async_subagent` when the user explicitly asks for a status update or
   result. If the status is "running", report that and stop — do not poll in a loop.
3. **Update** (optional) — Use `update_async_subagent` to send new instructions to a running job. This interrupts
   the current run and starts a fresh one on the same thread. The job_id stays the same.
4. **Cancel** (optional) — Use `cancel_async_subagent` to stop a job that is no longer needed.
5. **Collect** — When `check_async_subagent` returns status "success", the result is included in the response.
6. **List** — Use `list_async_subagent_jobs` to see live statuses for all jobs at once, or to recall job IDs after context compaction.

### Critical rules:
- After launching, ALWAYS return control to the user immediately. Never auto-check after launching.
- Never poll `check_async_subagent` in a loop. Check once per user request, then stop.
- If a check returns "running", tell the user and wait for them to ask again.
- Job statuses in conversation history are ALWAYS stale — a job that was "running" may now be done.
  NEVER report a status from a previous tool result. ALWAYS call a tool to get the current status:
  use `list_async_subagent_jobs` when the user asks about multiple jobs or "all jobs",
  use `check_async_subagent` when the user asks about a specific job.
- Always show the full job_id — never truncate or abbreviate it.

### When to use async subagents:
- Long-running tasks that would block the main agent
- Tasks that benefit from running on specialized remote deployments
- When you want to run multiple tasks concurrently and collect results later"""


def _resolve_headers(spec: AsyncSubAgent) -> dict[str, str]:
    """Build headers for a remote LangGraph server, including auth scheme."""
    headers: dict[str, str] = dict(spec.get("headers") or {})
    if "x-auth-scheme" not in headers:
        headers["x-auth-scheme"] = "langsmith"
    return headers


class _ClientCache:
    """Lazily-created, cached LangGraph SDK clients keyed by (url, headers)."""

    def __init__(self, agents: dict[str, AsyncSubAgent]) -> None:
        self._agents = agents
        self._sync: dict[tuple[str | None, frozenset[tuple[str, str]]], SyncLangGraphClient] = {}
        self._async: dict[tuple[str | None, frozenset[tuple[str, str]]], LangGraphClient] = {}

    def _cache_key(self, spec: AsyncSubAgent) -> tuple[str | None, frozenset[tuple[str, str]]]:
        """Build a cache key from the agent spec's url and resolved headers."""
        return (spec.get("url"), frozenset(_resolve_headers(spec).items()))

    def get_sync(self, name: str) -> SyncLangGraphClient:
        """Get or create a sync client for the named agent."""
        spec = self._agents[name]
        if spec.get("url") is None:
            msg = f"Async subagent '{name}' has no url configured. ASGI transport (url=None) requires async invocation."
            raise ValueError(msg)
        key = self._cache_key(spec)
        if key not in self._sync:
            self._sync[key] = get_sync_client(
                url=spec.get("url"),
                headers=_resolve_headers(spec),
            )
        return self._sync[key]

    def get_async(self, name: str) -> LangGraphClient:
        """Get or create an async client for the named agent."""
        spec = self._agents[name]
        key = self._cache_key(spec)
        if key not in self._async:
            self._async[key] = get_client(
                url=spec.get("url"),
                headers=_resolve_headers(spec),
            )
        return self._async[key]


def _validate_agent_type(agent_map: dict[str, AsyncSubAgent], agent_type: str) -> str | None:
    """Return an error message if `agent_type` is not in `agent_map`, or `None` if valid."""
    if agent_type not in agent_map:
        allowed = ", ".join(f"`{k}`" for k in agent_map)
        return f"Unknown async subagent type `{agent_type}`. Available types: {allowed}"
    return None


def _build_launch_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    tool_description: str,
) -> StructuredTool:
    """Build the `launch_async_subagent` tool."""

    def launch_async_subagent(
        description: Annotated[str, "A detailed description of the task for the async subagent to perform."],
        subagent_type: Annotated[str, "The type of async subagent to use. Must be one of the available types listed in the tool description."],
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        try:
            client = clients.get_sync(subagent_type)
            thread = client.threads.create()
            run = client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": description}]},
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, e)
            return f"Failed to launch async subagent '{subagent_type}': {e}"
        job_id = thread["thread_id"]
        job: AsyncSubAgentJob = {
            "job_id": job_id,
            "agent_name": subagent_type,
            "thread_id": job_id,
            "run_id": run["run_id"],
            "status": "running",
        }
        msg = f"Launched async subagent. job_id: {job_id}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_subagent_jobs": {job_id: job},
            }
        )

    async def alaunch_async_subagent(
        description: Annotated[str, "A detailed description of the task for the async subagent to perform."],
        subagent_type: Annotated[str, "The type of async subagent to use. Must be one of the available types listed in the tool description."],
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        try:
            client = clients.get_async(subagent_type)
            thread = await client.threads.create()
            run = await client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": description}]},
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, e)
            return f"Failed to launch async subagent '{subagent_type}': {e}"
        job_id = thread["thread_id"]
        job: AsyncSubAgentJob = {
            "job_id": job_id,
            "agent_name": subagent_type,
            "thread_id": job_id,
            "run_id": run["run_id"],
            "status": "running",
        }
        msg = f"Launched async subagent. job_id: {job_id}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_subagent_jobs": {job_id: job},
            }
        )

    return StructuredTool.from_function(
        name="launch_async_subagent",
        func=launch_async_subagent,
        coroutine=alaunch_async_subagent,
        description=tool_description,
    )


def _build_check_result(
    run: Run,
    thread_id: str,
    thread_values: dict[str, Any],
) -> dict[str, Any]:
    """Build the result dict from a run's current status and its thread values."""
    result: dict[str, Any] = {
        "status": run["status"],
        "thread_id": thread_id,
    }
    if run["status"] == "success":
        messages = thread_values.get("messages", []) if isinstance(thread_values, dict) else []
        if messages:
            last = messages[-1]
            result["result"] = last.get("content", "") if isinstance(last, dict) else str(last)
        else:
            result["result"] = "(completed with no output messages)"
    elif run["status"] == "error":
        error_detail = run.get("error")
        result["error"] = str(error_detail) if error_detail else "The async subagent encountered an error."
    return result


def _build_check_command(
    result: dict[str, Any],
    job: AsyncSubAgentJob,
    tool_call_id: str | None,
) -> Command:
    """Build the `Command` update for a check result."""
    updated_job = AsyncSubAgentJob(
        job_id=job["job_id"],
        agent_name=job["agent_name"],
        thread_id=job["thread_id"],
        run_id=job["run_id"],
        status=result["status"],
    )
    return Command(
        update={
            "messages": [ToolMessage(json.dumps(result), tool_call_id=tool_call_id)],
            "async_subagent_jobs": {job["job_id"]: updated_job},
        }
    )


def _resolve_tracked_job(
    job_id: str,
    runtime: ToolRuntime,
) -> AsyncSubAgentJob | str:
    """Look up a tracked job from state by its `job_id` (`thread_id`).

    Returns:
        The tracked `AsyncSubAgentJob` on success, or an error string.
    """
    jobs: dict[str, AsyncSubAgentJob] = runtime.state.get("async_subagent_jobs") or {}
    tracked = jobs.get(job_id.strip())
    if not tracked:
        return f"No tracked job found for job_id: {job_id!r}"
    return tracked


def _build_check_tool(  # noqa: C901  # complexity from necessary error handling
    clients: _ClientCache,
) -> StructuredTool:
    """Build the `check_async_subagent` tool."""

    def check_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent. Pass it verbatim."],
        runtime: ToolRuntime,
    ) -> str | Command:
        job = _resolve_tracked_job(job_id, runtime)
        if isinstance(job, str):
            return job

        client = clients.get_sync(job["agent_name"])
        try:
            run = client.runs.get(thread_id=job["thread_id"], run_id=job["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to get run status: {e}"

        thread_values: dict[str, Any] = {}
        if run["status"] == "success":
            try:
                thread = client.threads.get(thread_id=job["thread_id"])
                thread_values = thread.get("values") or {}
            except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
                logger.warning("Failed to fetch thread values for job %s: %s", job["job_id"], e)

        result = _build_check_result(run, job["thread_id"], thread_values)
        return _build_check_command(result, job, runtime.tool_call_id)

    async def acheck_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent. Pass it verbatim."],
        runtime: ToolRuntime,
    ) -> str | Command:
        job = _resolve_tracked_job(job_id, runtime)
        if isinstance(job, str):
            return job

        client = clients.get_async(job["agent_name"])
        try:
            run = await client.runs.get(thread_id=job["thread_id"], run_id=job["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to get run status: {e}"

        thread_values: dict[str, Any] = {}
        if run["status"] == "success":
            try:
                thread = await client.threads.get(thread_id=job["thread_id"])
                thread_values = thread.get("values") or {}
            except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
                logger.warning("Failed to fetch thread values for job %s: %s", job["job_id"], e)

        result = _build_check_result(run, job["thread_id"], thread_values)
        return _build_check_command(result, job, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="check_async_subagent",
        func=check_async_subagent,
        coroutine=acheck_async_subagent,
        description="Check the status of an async subagent job. Returns the current status and, if complete, the result.",
    )


def _build_update_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
) -> StructuredTool:
    """Build the `update_async_subagent` tool.

    Sends a follow-up message to an async subagent by creating a new run on the
    same thread. The subagent sees the full conversation history (including the
    original task and any prior results) plus the new message. The `job_id`
    remains the same; only the internal `run_id` is updated.
    """

    def update_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent. Pass it verbatim."],
        message: Annotated[str, "Follow-up instructions or context to send to the subagent."],
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_job(job_id, runtime)
        if isinstance(tracked, str):
            return tracked
        spec = agent_map[tracked["agent_name"]]
        try:
            client = clients.get_sync(tracked["agent_name"])
            run = client.runs.create(
                thread_id=tracked["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": message}]},
                multitask_strategy="interrupt",
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to update async subagent '%s': %s", tracked["agent_name"], e)
            return f"Failed to update async subagent: {e}"
        job: AsyncSubAgentJob = {
            "job_id": tracked["job_id"],
            "agent_name": tracked["agent_name"],
            "thread_id": tracked["thread_id"],
            "run_id": run["run_id"],
            "status": "running",
        }
        msg = f"Updated async subagent. job_id: {tracked['job_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_subagent_jobs": {tracked["job_id"]: job},
            }
        )

    async def aupdate_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent. Pass it verbatim."],
        message: Annotated[str, "Follow-up instructions or context to send to the subagent."],
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_job(job_id, runtime)
        if isinstance(tracked, str):
            return tracked
        spec = agent_map[tracked["agent_name"]]
        try:
            client = clients.get_async(tracked["agent_name"])
            run = await client.runs.create(
                thread_id=tracked["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": message}]},
                multitask_strategy="interrupt",
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to update async subagent '%s': %s", tracked["agent_name"], e)
            return f"Failed to update async subagent: {e}"
        job: AsyncSubAgentJob = {
            "job_id": tracked["job_id"],
            "agent_name": tracked["agent_name"],
            "thread_id": tracked["thread_id"],
            "run_id": run["run_id"],
            "status": "running",
        }
        msg = f"Updated async subagent. job_id: {tracked['job_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_subagent_jobs": {tracked["job_id"]: job},
            }
        )

    return StructuredTool.from_function(
        name="update_async_subagent",
        func=update_async_subagent,
        coroutine=aupdate_async_subagent,
        description=(
            "Send updated instructions to an async subagent. Interrupts the current run and starts "
            "a new one on the same thread, so the subagent sees the full conversation history plus "
            "your new message. The job_id remains the same."
        ),
    )


def _build_cancel_tool(
    clients: _ClientCache,
) -> StructuredTool:
    """Build the `cancel_async_subagent` tool."""

    def cancel_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent. Pass it verbatim."],
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_job(job_id, runtime)
        if isinstance(tracked, str):
            return tracked

        client = clients.get_sync(tracked["agent_name"])
        try:
            client.runs.cancel(thread_id=tracked["thread_id"], run_id=tracked["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to cancel run: {e}"
        updated = AsyncSubAgentJob(
            job_id=tracked["job_id"],
            agent_name=tracked["agent_name"],
            thread_id=tracked["thread_id"],
            run_id=tracked["run_id"],
            status="cancelled",
        )
        msg = f"Cancelled async subagent job: {tracked['job_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_subagent_jobs": {tracked["job_id"]: updated},
            }
        )

    async def acancel_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent. Pass it verbatim."],
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_job(job_id, runtime)
        if isinstance(tracked, str):
            return tracked

        client = clients.get_async(tracked["agent_name"])
        try:
            await client.runs.cancel(thread_id=tracked["thread_id"], run_id=tracked["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to cancel run: {e}"
        updated = AsyncSubAgentJob(
            job_id=tracked["job_id"],
            agent_name=tracked["agent_name"],
            thread_id=tracked["thread_id"],
            run_id=tracked["run_id"],
            status="cancelled",
        )
        msg = f"Cancelled async subagent job: {tracked['job_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_subagent_jobs": {tracked["job_id"]: updated},
            }
        )

    return StructuredTool.from_function(
        name="cancel_async_subagent",
        func=cancel_async_subagent,
        coroutine=acancel_async_subagent,
        description="Cancel a running async subagent job. Use this to stop a job that is no longer needed.",
    )


_TERMINAL_STATUSES = frozenset({"cancelled", "success", "error", "timeout", "interrupted"})
"""Job statuses that will never change, so live-status fetches can be skipped."""


def _fetch_live_status(clients: _ClientCache, job: AsyncSubAgentJob) -> str:
    """Fetch the current run status from the server, falling back to cached status on error."""
    if job["status"] in _TERMINAL_STATUSES:
        return job["status"]
    try:
        client = clients.get_sync(job["agent_name"])
        run = client.runs.get(thread_id=job["thread_id"], run_id=job["run_id"])
        return run["status"]
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        logger.warning(
            "Failed to fetch live status for job %s (agent=%s), returning cached status %r",
            job["job_id"],
            job["agent_name"],
            job["status"],
            exc_info=True,
        )
        return job["status"]


async def _afetch_live_status(clients: _ClientCache, job: AsyncSubAgentJob) -> str:
    """Async version of `_fetch_live_status`."""
    if job["status"] in _TERMINAL_STATUSES:
        return job["status"]
    try:
        client = clients.get_async(job["agent_name"])
        run = await client.runs.get(thread_id=job["thread_id"], run_id=job["run_id"])
        return run["status"]
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        logger.warning(
            "Failed to fetch live status for job %s (agent=%s), returning cached status %r",
            job["job_id"],
            job["agent_name"],
            job["status"],
            exc_info=True,
        )
        return job["status"]


def _format_job_entry(job: AsyncSubAgentJob, status: str) -> str:
    """Format a single job as a display string for list output."""
    return f"- job_id: {job['job_id']}  agent: {job['agent_name']}  status: {status}"


def _filter_jobs(
    jobs: dict[str, AsyncSubAgentJob],
    status_filter: str | None,
) -> list[AsyncSubAgentJob]:
    """Filter jobs by cached status from agent state.

    Filtering happens on the cached status, not live server status. Live
    statuses are fetched after filtering by the calling tool.

    Args:
        jobs: All tracked jobs from state.
        status_filter: If `None` or `'all'`, return all jobs.

            Otherwise return only jobs whose cached status matches.

    Returns:
        Filtered list of jobs.
    """
    if not status_filter or status_filter == "all":
        return list(jobs.values())
    return [job for job in jobs.values() if job["status"] == status_filter]


def _build_list_jobs_tool(clients: _ClientCache) -> StructuredTool:
    """Build the list_async_subagent_jobs tool."""

    def list_async_subagent_jobs(
        runtime: ToolRuntime,
        status_filter: Annotated[
            str | None,
            "Filter jobs by status. One of: 'running', 'success', 'error', 'cancelled', 'all'. Defaults to 'all'.",
        ] = None,
    ) -> str | Command:
        jobs: dict[str, AsyncSubAgentJob] = runtime.state.get("async_subagent_jobs") or {}
        filtered = _filter_jobs(jobs, status_filter)
        if not filtered:
            return "No async subagent jobs tracked."
        updated_jobs: dict[str, AsyncSubAgentJob] = {}
        entries: list[str] = []
        for job in filtered:
            status = _fetch_live_status(clients, job)
            entries.append(_format_job_entry(job, status))
            updated_jobs[job["job_id"]] = AsyncSubAgentJob(
                job_id=job["job_id"],
                agent_name=job["agent_name"],
                thread_id=job["thread_id"],
                run_id=job["run_id"],
                status=status,
            )
        msg = f"{len(entries)} tracked job(s):\n" + "\n".join(entries)
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_subagent_jobs": updated_jobs,
            }
        )

    async def alist_async_subagent_jobs(
        runtime: ToolRuntime,
        status_filter: Annotated[
            str | None,
            "Filter jobs by status. One of: 'running', 'success', 'error', 'cancelled', 'all'. Defaults to 'all'.",
        ] = None,
    ) -> str | Command:
        jobs: dict[str, AsyncSubAgentJob] = runtime.state.get("async_subagent_jobs") or {}
        filtered = _filter_jobs(jobs, status_filter)
        if not filtered:
            return "No async subagent jobs tracked."
        statuses = await asyncio.gather(*(_afetch_live_status(clients, job) for job in filtered))
        updated_jobs: dict[str, AsyncSubAgentJob] = {}
        entries: list[str] = []
        for job, status in zip(filtered, statuses, strict=True):
            entries.append(_format_job_entry(job, status))
            updated_jobs[job["job_id"]] = AsyncSubAgentJob(
                job_id=job["job_id"],
                agent_name=job["agent_name"],
                thread_id=job["thread_id"],
                run_id=job["run_id"],
                status=status,
            )
        msg = f"{len(entries)} tracked job(s):\n" + "\n".join(entries)
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_subagent_jobs": updated_jobs,
            }
        )

    return StructuredTool.from_function(
        name="list_async_subagent_jobs",
        func=list_async_subagent_jobs,
        coroutine=alist_async_subagent_jobs,
        description=(
            "List tracked async subagent jobs with their current live statuses. "
            "By default shows all jobs. Use `status_filter` to narrow by status "
            "(e.g. 'running', 'success', 'error', 'cancelled'). "
            "Use `check_async_subagent` to get the full result of a specific completed job."
        ),
    )


def _build_async_subagent_tools(
    agents: list[AsyncSubAgent],
) -> list[StructuredTool]:
    """Build the async subagent tools from agent specs.

    Args:
        agents: List of async subagent specifications.

    Returns:
        List of StructuredTools for launch, check, update, cancel, and list operations.
    """
    agent_map: dict[str, AsyncSubAgent] = {a["name"]: a for a in agents}
    clients = _ClientCache(agent_map)
    agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in agents)
    launch_desc = ASYNC_TASK_TOOL_DESCRIPTION.format(available_agents=agents_desc)

    return [
        _build_launch_tool(agent_map, clients, launch_desc),
        _build_check_tool(clients),
        _build_update_tool(agent_map, clients),
        _build_cancel_tool(clients),
        _build_list_jobs_tool(clients),
    ]


class AsyncSubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for async subagents running on remote LangGraph servers.

    This middleware adds tools for launching, monitoring, and updating
    background jobs on remote LangGraph deployments. Unlike the synchronous
    `SubAgentMiddleware`, async subagents return immediately with a job ID,
    allowing the main agent to continue working while subagents execute.

    Job IDs are persisted in the agent state under `async_subagent_jobs` so they
    survive context compaction/offloading and can be accessed programmatically.

    Args:
        async_subagents: List of async subagent specifications.

            Each must include `name`, `description`, and `graph_id`. `url` is
            optional — omit it to use ASGI transport for local
            LangGraph servers.
        system_prompt: Instructions appended to the main agent's system prompt
            about how to use the async subagent tools.

    Example:
        ```python
        from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware

        middleware = AsyncSubAgentMiddleware(
            async_subagents=[
                {
                    "name": "researcher",
                    "description": "Research agent for deep analysis",
                    "url": "https://my-deployment.langsmith.dev",
                    "graph_id": "research_agent",
                }
            ],
        )
        ```
    """

    state_schema = AsyncSubAgentState

    def __init__(
        self,
        *,
        async_subagents: list[AsyncSubAgent],
        system_prompt: str | None = ASYNC_TASK_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the `AsyncSubAgentMiddleware`."""
        super().__init__()
        if not async_subagents:
            msg = "At least one async subagent must be specified"
            raise ValueError(msg)

        names = [a["name"] for a in async_subagents]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            msg = f"Duplicate async subagent names: {dupes}"
            raise ValueError(msg)

        self.tools = _build_async_subagent_tools(async_subagents)

        if system_prompt:
            agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in async_subagents)
            self.system_prompt: str | None = system_prompt + "\n\nAvailable async subagent types:\n" + agents_desc
        else:
            self.system_prompt = system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Update the system message to include async subagent instructions."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Update the system message to include async subagent instructions."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
