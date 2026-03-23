"""Middleware for async subagents running on remote LangGraph servers.

Async subagents use the LangGraph SDK to launch background runs on remote
LangGraph deployments. Unlike synchronous subagents (which block until
completion), async subagents return a task ID immediately, allowing the main
agent to monitor progress and send updates while the subagent works.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
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
    They run as background tasks that the main agent can monitor and update.

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


class AsyncTask(TypedDict):
    """A tracked async subagent task persisted in agent state."""

    task_id: str
    """Unique identifier for the task (same as `thread_id`)."""

    agent_name: str
    """Name of the async subagent type that is running."""

    thread_id: str
    """LangGraph thread ID for the remote run."""

    run_id: str
    """LangGraph run ID for the current execution on the thread."""

    status: str
    """Current task status (e.g., `'running'`, `'success'`, `'error'`, `'cancelled'`).

    Typed as `str` rather than a `Literal` because the LangGraph SDK's
    `Run.status` is `str` — using a `Literal` here would require `cast` at every
    SDK boundary.
    """

    created_at: str
    """ISO-8601 timestamp (UTC) when the task was created, with second precision.

    Format: `YYYY-MM-DDTHH:MM:SSZ` (e.g., `2024-01-15T10:30:00Z`).
    """

    last_checked_at: str
    """ISO-8601 timestamp (UTC) when the task status was last checked via SDK.

    Format: `YYYY-MM-DDTHH:MM:SSZ` (e.g., `2024-01-15T10:30:00Z`).
    """

    last_updated_at: str
    """ISO-8601 timestamp (UTC) when the task was last updated with a new message.

    Format: `YYYY-MM-DDTHH:MM:SSZ` (e.g., `2024-01-15T10:30:00Z`).
    """


def _tasks_reducer(
    existing: dict[str, AsyncTask] | None,
    update: dict[str, AsyncTask],
) -> dict[str, AsyncTask]:
    """Merge task updates into the existing tasks dict."""
    merged = dict(existing or {})
    merged.update(update)
    return merged


class AsyncSubAgentState(AgentState):
    """State extension for async subagent task tracking."""

    async_tasks: Annotated[NotRequired[dict[str, AsyncTask]], _tasks_reducer]


ASYNC_TASK_TOOL_DESCRIPTION = """Start an async subagent on a remote LangGraph server. The subagent runs in the background and returns a task ID immediately.

Available async agent types:
{available_agents}

## Usage notes:
1. This tool launches a background task and returns immediately with a task ID. Report the task ID to the user and stop — do NOT immediately check status.
2. Use `check_async_task` only when the user asks for a status update or result.
3. Use `update_async_task` to send new instructions to a running task.
4. Multiple async subagents can run concurrently — launch several and let them run in the background.
5. The subagent runs on a remote LangGraph server, so it has its own tools and capabilities."""  # noqa: E501

ASYNC_TASK_SYSTEM_PROMPT = """## Async subagents (remote LangGraph servers)

You have access to async subagent tools that launch background tasks on remote LangGraph servers.

### Tools:
- `start_async_task`: Start a new background task. Returns a task ID immediately.
- `check_async_task`: Get current status and result of a task. Returns status + result (if complete).
- `update_async_task`: Send new instructions to a running task. Returns confirmation + updated status.
- `cancel_async_task`: Stop a running task. Returns confirmation.
- `list_async_tasks`: List all tracked tasks with live statuses. Returns summary of all tasks.

### Workflow:
1. **Start** — Use `start_async_task` to start a task. Report the task ID to the user and stop.
   Do NOT immediately check the status — the task runs in the background while you and the user continue other work.
2. **Check (on request)** — Only use `check_async_task` when the user explicitly asks for a status update or
   result. If the status is "running", report that and stop — do not poll in a loop.
3. **Update** (optional) — Use `update_async_task` to send new instructions to a running task. This interrupts
   the current run and starts a fresh one on the same thread. The task_id stays the same.
4. **Cancel** (optional) — Use `cancel_async_task` to stop a task that is no longer needed.
5. **Collect** — When `check_async_task` returns status "success", the result is included in the response.
6. **List** — Use `list_async_tasks` to see live statuses for all tasks at once, or to recall task IDs after context compaction.

### Critical rules:
- After launching, ALWAYS return control to the user immediately. Never auto-check after launching.
- Never poll `check_async_task` in a loop. Check once per user request, then stop.
- If a check returns "running", tell the user and wait for them to ask again.
- Task statuses in conversation history are ALWAYS stale — a task that was "running" may now be done.
  NEVER report a status from a previous tool result. ALWAYS call a tool to get the current status:
  use `list_async_tasks` when the user asks about multiple tasks or "all tasks",
  use `check_async_task` when the user asks about a specific task.
- Always show the full task_id — never truncate or abbreviate it.

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


def _extract_callback_context(runtime: ToolRuntime) -> dict[str, str]:
    """Extract the parent thread ID from the tool runtime.

    The thread ID is included in the subagent's input state so the subagent
    can notify the parent when it completes (via
    `CompletionNotifierMiddleware`).

    Returns:
        Dict with `parent_thread_id` if available. Empty dict otherwise.
    """
    config = runtime.config or {}
    configurable = config.get("configurable") or {}

    thread_id = configurable.get("thread_id")
    if thread_id:
        return {"parent_thread_id": str(thread_id)}

    return {}


def _build_start_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    tool_description: str,
) -> StructuredTool:
    """Build the `start_async_task` tool."""

    def start_async_task(
        description: Annotated[str, "A detailed description of the task for the async subagent to perform."],
        subagent_type: Annotated[str, "The type of async subagent to use. Must be one of the available types listed in the tool description."],
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        callback_context = _extract_callback_context(runtime)
        try:
            client = clients.get_sync(subagent_type)
            thread = client.threads.create()
            task_id = thread["thread_id"]
            run = client.runs.create(
                thread_id=task_id,
                assistant_id=spec["graph_id"],
                input={
                    "messages": [{"role": "user", "content": description}],
                    **parent_context,
                },
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, e)
            return f"Failed to launch async subagent '{subagent_type}': {e}"
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": task_id,
            "agent_name": subagent_type,
            "thread_id": task_id,
            "run_id": run["run_id"],
            "status": "running",
            "created_at": now,
            "last_checked_at": now,
            "last_updated_at": now,
        }
        msg = f"Launched async subagent. task_id: {task_id}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {task_id: task},
            }
        )

    async def astart_async_task(
        description: Annotated[str, "A detailed description of the task for the async subagent to perform."],
        subagent_type: Annotated[str, "The type of async subagent to use. Must be one of the available types listed in the tool description."],
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        callback_context = _extract_callback_context(runtime)
        try:
            client = clients.get_async(subagent_type)
            thread = await client.threads.create()
            task_id = thread["thread_id"]
            run = await client.runs.create(
                thread_id=task_id,
                assistant_id=spec["graph_id"],
                input={
                    "messages": [{"role": "user", "content": description}],
                    **parent_context,
                },
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, e)
            return f"Failed to launch async subagent '{subagent_type}': {e}"
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": task_id,
            "agent_name": subagent_type,
            "thread_id": task_id,
            "run_id": run["run_id"],
            "status": "running",
            "created_at": now,
            "last_checked_at": now,
            "last_updated_at": now,
        }
        msg = f"Launched async subagent. task_id: {task_id}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {task_id: task},
            }
        )

    return StructuredTool.from_function(
        name="start_async_task",
        func=start_async_task,
        coroutine=astart_async_task,
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
    task: AsyncTask,
    tool_call_id: str | None,
) -> Command:
    """Build the `Command` update for a check result."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated_task = AsyncTask(
        task_id=task["task_id"],
        agent_name=task["agent_name"],
        thread_id=task["thread_id"],
        run_id=task["run_id"],
        status=result["status"],
        created_at=task["created_at"],
        last_checked_at=now,
        last_updated_at=task["last_updated_at"],
    )
    return Command(
        update={
            "messages": [ToolMessage(json.dumps(result), tool_call_id=tool_call_id)],
            "async_tasks": {task["task_id"]: updated_task},
        }
    )


def _resolve_tracked_task(
    task_id: str,
    runtime: ToolRuntime,
) -> AsyncTask | str:
    """Look up a tracked task from state by its `task_id` (`thread_id`).

    Returns:
        The tracked `AsyncTask` on success, or an error string.
    """
    tasks: dict[str, AsyncTask] = runtime.state.get("async_tasks") or {}
    tracked = tasks.get(task_id.strip())
    if not tracked:
        return f"No tracked task found for task_id: {task_id!r}"
    return tracked


def _build_check_tool(  # noqa: C901  # complexity from necessary error handling
    clients: _ClientCache,
) -> StructuredTool:
    """Build the `check_async_task` tool."""

    def check_async_task(
        task_id: Annotated[str, "The exact task_id string returned by start_async_task. Pass it verbatim."],
        runtime: ToolRuntime,
    ) -> str | Command:
        task = _resolve_tracked_task(task_id, runtime)
        if isinstance(task, str):
            return task

        client = clients.get_sync(task["agent_name"])
        try:
            run = client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to get run status: {e}"

        thread_values: dict[str, Any] = {}
        if run["status"] == "success":
            try:
                thread = client.threads.get(thread_id=task["thread_id"])
                thread_values = thread.get("values") or {}
            except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
                logger.warning("Failed to fetch thread values for task %s: %s", task["task_id"], e)

        result = _build_check_result(run, task["thread_id"], thread_values)
        return _build_check_command(result, task, runtime.tool_call_id)

    async def acheck_async_task(
        task_id: Annotated[str, "The exact task_id string returned by start_async_task. Pass it verbatim."],
        runtime: ToolRuntime,
    ) -> str | Command:
        task = _resolve_tracked_task(task_id, runtime)
        if isinstance(task, str):
            return task

        client = clients.get_async(task["agent_name"])
        try:
            run = await client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to get run status: {e}"

        thread_values: dict[str, Any] = {}
        if run["status"] == "success":
            try:
                thread = await client.threads.get(thread_id=task["thread_id"])
                thread_values = thread.get("values") or {}
            except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
                logger.warning("Failed to fetch thread values for task %s: %s", task["task_id"], e)

        result = _build_check_result(run, task["thread_id"], thread_values)
        return _build_check_command(result, task, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="check_async_task",
        func=check_async_task,
        coroutine=acheck_async_task,
        description="Check the status of an async subagent task. Returns the current status and, if complete, the result.",
    )


def _build_update_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
) -> StructuredTool:
    """Build the `update_async_task` tool.

    Sends a follow-up message to an async subagent by creating a new run on the
    same thread. The subagent sees the full conversation history (including the
    original task and any prior results) plus the new message. The `task_id`
    remains the same; only the internal `run_id` is updated.
    """

    def update_async_task(
        task_id: Annotated[str, "The exact task_id string returned by start_async_task. Pass it verbatim."],
        message: Annotated[str, "Follow-up instructions or context to send to the subagent."],
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_task(task_id, runtime)
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
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": tracked["task_id"],
            "agent_name": tracked["agent_name"],
            "thread_id": tracked["thread_id"],
            "run_id": run["run_id"],
            "status": "running",
            "created_at": tracked["created_at"],
            "last_checked_at": tracked["last_checked_at"],
            "last_updated_at": now,
        }
        msg = f"Updated async subagent. task_id: {tracked['task_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: task},
            }
        )

    async def aupdate_async_task(
        task_id: Annotated[str, "The exact task_id string returned by start_async_task. Pass it verbatim."],
        message: Annotated[str, "Follow-up instructions or context to send to the subagent."],
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_task(task_id, runtime)
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
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": tracked["task_id"],
            "agent_name": tracked["agent_name"],
            "thread_id": tracked["thread_id"],
            "run_id": run["run_id"],
            "status": "running",
            "created_at": tracked["created_at"],
            "last_checked_at": tracked["last_checked_at"],
            "last_updated_at": now,
        }
        msg = f"Updated async subagent. task_id: {tracked['task_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: task},
            }
        )

    return StructuredTool.from_function(
        name="update_async_task",
        func=update_async_task,
        coroutine=aupdate_async_task,
        description=(
            "Send updated instructions to an async subagent. Interrupts the current run and starts "
            "a new one on the same thread, so the subagent sees the full conversation history plus "
            "your new message. The task_id remains the same."
        ),
    )


def _build_cancel_tool(
    clients: _ClientCache,
) -> StructuredTool:
    """Build the `cancel_async_task` tool."""

    def cancel_async_task(
        task_id: Annotated[str, "The exact task_id string returned by start_async_task. Pass it verbatim."],
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked

        client = clients.get_sync(tracked["agent_name"])
        try:
            client.runs.cancel(thread_id=tracked["thread_id"], run_id=tracked["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to cancel run: {e}"
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        updated = AsyncTask(
            task_id=tracked["task_id"],
            agent_name=tracked["agent_name"],
            thread_id=tracked["thread_id"],
            run_id=tracked["run_id"],
            status="cancelled",
            created_at=tracked["created_at"],
            last_checked_at=now,
            last_updated_at=tracked["last_updated_at"],
        )
        msg = f"Cancelled async subagent task: {tracked['task_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: updated},
            }
        )

    async def acancel_async_task(
        task_id: Annotated[str, "The exact task_id string returned by start_async_task. Pass it verbatim."],
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked

        client = clients.get_async(tracked["agent_name"])
        try:
            await client.runs.cancel(thread_id=tracked["thread_id"], run_id=tracked["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to cancel run: {e}"
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        updated = AsyncTask(
            task_id=tracked["task_id"],
            agent_name=tracked["agent_name"],
            thread_id=tracked["thread_id"],
            run_id=tracked["run_id"],
            status="cancelled",
            created_at=tracked["created_at"],
            last_checked_at=now,
            last_updated_at=tracked["last_updated_at"],
        )
        msg = f"Cancelled async subagent task: {tracked['task_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: updated},
            }
        )

    return StructuredTool.from_function(
        name="cancel_async_task",
        func=cancel_async_task,
        coroutine=acancel_async_task,
        description="Cancel a running async subagent task. Use this to stop a task that is no longer needed.",
    )


_TERMINAL_STATUSES = frozenset({"cancelled", "success", "error", "timeout", "interrupted"})
"""Task statuses that will never change, so live-status fetches can be skipped."""


def _fetch_live_status(clients: _ClientCache, task: AsyncTask) -> str:
    """Fetch the current run status from the server, falling back to cached status on error."""
    if task["status"] in _TERMINAL_STATUSES:
        return task["status"]
    try:
        client = clients.get_sync(task["agent_name"])
        run = client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
        return run["status"]
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        logger.warning(
            "Failed to fetch live status for task %s (agent=%s), returning cached status %r",
            task["task_id"],
            task["agent_name"],
            task["status"],
            exc_info=True,
        )
        return task["status"]


async def _afetch_live_status(clients: _ClientCache, task: AsyncTask) -> str:
    """Async version of `_fetch_live_status`."""
    if task["status"] in _TERMINAL_STATUSES:
        return task["status"]
    try:
        client = clients.get_async(task["agent_name"])
        run = await client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
        return run["status"]
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        logger.warning(
            "Failed to fetch live status for task %s (agent=%s), returning cached status %r",
            task["task_id"],
            task["agent_name"],
            task["status"],
            exc_info=True,
        )
        return task["status"]


def _format_task_entry(task: AsyncTask, status: str) -> str:
    """Format a single task as a display string for list output."""
    return f"- task_id: {task['task_id']}  agent: {task['agent_name']}  status: {status}"


def _filter_tasks(
    tasks: dict[str, AsyncTask],
    status_filter: str | None,
) -> list[AsyncTask]:
    """Filter tasks by cached status from agent state.

    Filtering happens on the cached status, not live server status. Live
    statuses are fetched after filtering by the calling tool.

    Args:
        tasks: All tracked tasks from state.
        status_filter: If `None` or `'all'`, return all tasks.

            Otherwise return only tasks whose cached status matches.

    Returns:
        Filtered list of tasks.
    """
    if not status_filter or status_filter == "all":
        return list(tasks.values())
    return [task for task in tasks.values() if task["status"] == status_filter]


def _build_list_tasks_tool(clients: _ClientCache) -> StructuredTool:
    """Build the list_async_tasks tool."""

    def list_async_tasks(
        runtime: ToolRuntime,
        status_filter: Annotated[
            str | None,
            "Filter tasks by status. One of: 'running', 'success', 'error', 'cancelled', 'all'. Defaults to 'all'.",
        ] = None,
    ) -> str | Command:
        tasks: dict[str, AsyncTask] = runtime.state.get("async_tasks") or {}
        filtered = _filter_tasks(tasks, status_filter)
        if not filtered:
            return "No async subagent tasks tracked."
        updated_tasks: dict[str, AsyncTask] = {}
        entries: list[str] = []
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for task in filtered:
            status = _fetch_live_status(clients, task)
            entries.append(_format_task_entry(task, status))
            updated_tasks[task["task_id"]] = AsyncTask(
                task_id=task["task_id"],
                agent_name=task["agent_name"],
                thread_id=task["thread_id"],
                run_id=task["run_id"],
                status=status,
                created_at=task["created_at"],
                last_checked_at=now,
                last_updated_at=task["last_updated_at"],
            )
        msg = f"{len(entries)} tracked task(s):\n" + "\n".join(entries)
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": updated_tasks,
            }
        )

    async def alist_async_tasks(
        runtime: ToolRuntime,
        status_filter: Annotated[
            str | None,
            "Filter tasks by status. One of: 'running', 'success', 'error', 'cancelled', 'all'. Defaults to 'all'.",
        ] = None,
    ) -> str | Command:
        tasks: dict[str, AsyncTask] = runtime.state.get("async_tasks") or {}
        filtered = _filter_tasks(tasks, status_filter)
        if not filtered:
            return "No async subagent tasks tracked."
        statuses = await asyncio.gather(*(_afetch_live_status(clients, task) for task in filtered))
        updated_tasks: dict[str, AsyncTask] = {}
        entries: list[str] = []
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for task, status in zip(filtered, statuses, strict=True):
            entries.append(_format_task_entry(task, status))
            updated_tasks[task["task_id"]] = AsyncTask(
                task_id=task["task_id"],
                agent_name=task["agent_name"],
                thread_id=task["thread_id"],
                run_id=task["run_id"],
                status=status,
                created_at=task["created_at"],
                last_checked_at=now,
                last_updated_at=task["last_updated_at"],
            )
        msg = f"{len(entries)} tracked task(s):\n" + "\n".join(entries)
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": updated_tasks,
            }
        )

    return StructuredTool.from_function(
        name="list_async_tasks",
        func=list_async_tasks,
        coroutine=alist_async_tasks,
        description=(
            "List tracked async subagent tasks with their current live statuses. "
            "By default shows all tasks. Use `status_filter` to narrow by status "
            "(e.g. 'running', 'success', 'error', 'cancelled'). "
            "Use `check_async_task` to get the full result of a specific completed task."
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
        _build_start_tool(agent_map, clients, launch_desc),
        _build_check_tool(clients),
        _build_update_tool(agent_map, clients),
        _build_cancel_tool(clients),
        _build_list_tasks_tool(clients),
    ]


class AsyncSubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for async subagents running on remote LangGraph servers.

    This middleware adds tools for launching, monitoring, and updating
    background tasks on remote LangGraph deployments. Unlike the synchronous
    `SubAgentMiddleware`, async subagents return immediately with a task ID,
    allowing the main agent to continue working while subagents execute.

    Task IDs are persisted in the agent state under `async_tasks` so they
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
