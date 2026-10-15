"""Middleware for async subagents running on remote Agent Protocol servers.

Async subagents use the LangGraph SDK to launch background runs on remote
[Agent Protocol](https://github.com/langchain-ai/agent-protocol) servers.
Unlike synchronous subagents (which block until completion), async subagents
return a task ID immediately, allowing the main agent to monitor progress and
send updates while the subagent works.

Compatible with LangGraph Platform (managed) and self-hosted servers.
"""

import asyncio
import json
import logging
import urllib.parse
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ModelRequest, ModelResponse, ResponseT, TracePolicy, omit_payload
from langchain.tools import ToolRuntime
from langchain_core._api.beta_decorator import warn_beta
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from langgraph_sdk.schema import Interrupt, Run, Thread
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from deepagents.middleware._blob_offload import _BLOB_PAYLOADS_KEY, _ahydrate_tool_messages, _hydrate_tool_messages
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.subagents import _fork_messages
from deepagents.middleware.summarization import SUMMARIZATION_EVENT_KEY

logger = logging.getLogger(__name__)


class AsyncSubAgent(TypedDict):
    """Specification for an async subagent running on a remote [Agent Protocol](https://github.com/langchain-ai/agent-protocol) server.

    Async subagents connect to any Agent Protocol-compliant server via the
    LangGraph SDK. They run as background tasks that the main agent can
    monitor and update.

    Compatible with LangGraph Platform / LangSmith Deployment (managed) and
    self-hosted servers.

    Authentication for LangGraph Platform / LangSmith Deployment is handled
    automatically by the SDK via environment variables (`LANGGRAPH_API_KEY`,
    `LANGSMITH_API_KEY`, or `LANGCHAIN_API_KEY`). For self-hosted servers,
    pass custom auth via `headers`.

    !!! note "Async invocation required for local ASGI transport"

        Omitting `url` uses in-process ASGI transport for a local server. This
        transport is available only through an async parent-agent entrypoint,
        such as `ainvoke`. The synchronous `invoke` path requires a URL for a
        reachable Agent Protocol server.

    !!! warning "Beta"

        When tracing is on, launches and updates of subagents on this deployment
        (no `url`) send the parent's LangSmith trace headers, so a receiver can
        nest the subagent's run under the parent's trace. This may change.

        Each task's thread is created with `metadata={"parent_thread_id": ...}`,
        so apps can find a parent's subagent threads with `threads.search`, and
        a subagent's graph factory can read it from its run's
        `config["metadata"]`. This may change.
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
    """URL of the [Agent Protocol](https://github.com/langchain-ai/agent-protocol) server.

    Defaults to the LangGraph SDK's default endpoint. Omit to use ASGI
    transport for local servers.
    """

    headers: NotRequired[dict[str, str]]
    """Additional headers to include in requests to the remote server."""

    mode: NotRequired[Literal["isolated", "fork"]]
    """When using `mode="fork"`, configure the remote agent to match the parent's model,
    system prompt, tools, and prompt-processing middleware if you want prompt-cache
    reuse. Fork mode copies conversation history; it does not synchronize those
    settings. Matching configuration is necessary but not sufficient: the rendered
    prefix must match, and cache hits depend on the provider.

    - `isolated` (default): send only the delegated task description.
    - `fork`: send a snapshot of the parent's conversation at launch, followed
        by the task description, for background work.

    Different configurations are supported for context sharing, but you should
    not expect parent-to-child cache reuse with them.

    !!! warning "Experimental"

        `mode="fork"` is experimental and may change in a future release.
    """


_WAITING = "waiting"
"""Task status for a subagent waiting for a human: its run finished but its thread is interrupted."""

_THREAD_TO_TASK_STATUS = {"interrupted": _WAITING, "busy": "running", "error": "error"}
"""Task status implied by the thread when the tracked run finished but the thread moved on."""

_TERMINAL_STATUSES = frozenset({"cancelled", "success", "error", "timeout", "interrupted"})
"""Task statuses that will never change, so live-status fetches can be skipped.

`interrupted` is a run cancelled or replaced on the server. A subagent paused
by `interrupt()` is `waiting`, which isn't terminal.
"""


class AsyncTask(TypedDict):
    """A tracked async subagent task persisted in agent state."""

    task_id: str
    """Unique identifier for the task (same as `thread_id`)."""

    agent_name: str
    """Name of the async subagent type that is running."""

    thread_id: str
    """Thread ID on the remote server."""

    run_id: str
    """Run ID for the current execution on the thread."""

    status: str
    """Current task status (e.g., `'running'`, `'waiting'`, `'success'`, `'error'`, `'cancelled'`).

    Typed as `str` rather than a `Literal` because the LangGraph SDK's
    `Run.status` is `str` — using a `Literal` here would require `cast` at every
    SDK boundary.

    !!! warning "Beta"

        `'waiting'` means the subagent paused for its user (an approval or a
        question), who answers it on the subagent's own thread. This may change.
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
    """ISO-8601 timestamp (UTC) when the task status changes or when a follow-up message is sent via the update tool.

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


class StartAsyncTaskSchema(BaseModel):
    """Input schema for the `start_async_task` tool."""

    description: str = Field(description="A detailed description of the task for the async subagent to perform.")
    subagent_type: str = Field(description="The type of async subagent to use. Must be one of the available types listed in the tool description.")


class CheckAsyncTaskSchema(BaseModel):
    """Input schema for the `check_async_task` tool."""

    task_id: str = Field(description="The exact task_id string returned by start_async_task. Pass it verbatim.")


class UpdateAsyncTaskSchema(BaseModel):
    """Input schema for the `update_async_task` tool."""

    task_id: str = Field(description="The exact task_id string returned by start_async_task. Pass it verbatim.")
    message: str = Field(description="Follow-up instructions or context to send to the subagent.")


class CancelAsyncTaskSchema(BaseModel):
    """Input schema for the `cancel_async_task` tool."""

    task_id: str = Field(description="The exact task_id string returned by start_async_task. Pass it verbatim.")


class ListAsyncTasksSchema(BaseModel):
    """Input schema for the `list_async_tasks` tool."""

    status_filter: Literal["running", "waiting", "success", "error", "cancelled", "all"] | None = Field(
        default=None,
        description="Filter tasks by status. One of: 'running', 'waiting', 'success', 'error', 'cancelled', 'all'. Defaults to 'all'.",
    )


ASYNC_TASK_TOOL_DESCRIPTION = """Start an async subagent on a remote server. The subagent runs in the background and returns a task ID immediately.

Available async agent types:
{available_agents}

## Usage notes:
1. This tool launches a background task and returns immediately with a task ID. Report the task ID to the user and stop — do NOT immediately check status.
2. Use `check_async_task` only when the user asks for a status update or result.
3. Use `update_async_task` to send new instructions to a running task.
4. Multiple async subagents can run concurrently — launch several and let them run in the background.
5. The subagent runs on a remote server, so it has its own tools and capabilities.
6. A task with status `waiting` is waiting for a human (an approval or an answer). The user answers it directly; you can't answer or approve it for them. Keep working, and tell the user it's waiting if they ask."""  # noqa: E501


def _resolve_headers(spec: AsyncSubAgent) -> dict[str, str]:
    """Build headers for a remote Agent Protocol server.

    Adds `x-auth-scheme: langsmith` by default unless already provided.
    For self-hosted servers that don't require this header, it is typically
    ignored. Override via the `headers` field on the `AsyncSubAgent` config.
    """
    headers: dict[str, str] = dict(spec.get("headers") or {})
    if "x-auth-scheme" not in headers:
        headers["x-auth-scheme"] = "langsmith"
    return headers


_NEST_UNDER_PARENT = "ls_nest_under_parent"
"""Baggage metadata key (JSON `true`) marking trace headers as a parent to nest under.

Receivers that adopt the trace check for it (Managed Deep Agents does), so keep the name stable.
"""


def _trace_headers(spec: AsyncSubAgent) -> dict[str, str]:
    """Build LangSmith headers pointing at the current span, or `{}` when tracing is off.

    Only sent to subagents on this deployment (no `url`), so the parent's trace
    details don't reach another server. The baggage carries the project and the
    `_NEST_UNDER_PARENT` flag, not the caller's run metadata (such as its
    `thread_id`), so a receiver can't copy that onto the subagent's runs. Any
    `baggage` in the spec's headers is kept, since other tools use it too.
    """
    run_tree = get_current_run_tree()
    if spec.get("url") or run_tree is None or not run_tree.trace_id:
        return {}
    baggage = [value for key, value in (spec.get("headers") or {}).items() if key.lower() == "baggage" and value]
    baggage.append(f"langsmith-metadata={urllib.parse.quote(json.dumps({_NEST_UNDER_PARENT: True}))}")
    if run_tree.session_name:
        baggage.append(f"langsmith-project={urllib.parse.quote(run_tree.session_name)}")
    return {"langsmith-trace": run_tree.dotted_order, "baggage": ",".join(baggage)}


_PARENT_KEY = "deepagents_parent"
"""`configurable` key carrying the launching agent's `_ParentReference` to an async subagent's run.

Apps read it directly (Managed Deep Agents does), so keep the key and shape stable.
"""


class _ParentReference(TypedDict):
    """What an async subagent's run knows about the agent that launched it."""

    thread_id: NotRequired[str]
    """The parent's thread, where notifications for the parent go."""

    assistant_id: NotRequired[str]
    """The parent's assistant, which handles runs queued on its thread."""


def _parent_reference(config: RunnableConfig) -> _ParentReference:
    """Describe the running agent so the async subagent it launches can find it."""
    reference: _ParentReference = {}
    if thread_id := (config.get("configurable") or {}).get("thread_id"):
        reference["thread_id"] = str(thread_id)
    if assistant_id := (config.get("metadata") or {}).get("assistant_id") or (config.get("configurable") or {}).get("assistant_id"):
        reference["assistant_id"] = str(assistant_id)
    return reference


def _thread_metadata(runtime: ToolRuntime) -> dict[str, str]:
    """Metadata for a task's thread, so apps can find a parent's subagent threads with `threads.search`."""
    reference = _parent_reference(runtime.config)
    return {"parent_thread_id": reference["thread_id"]} if "thread_id" in reference else {}


def _run_options(spec: AsyncSubAgent, runtime: ToolRuntime) -> dict[str, Any]:
    """Keyword arguments for `runs.create` that link a launched run to this agent."""
    options: dict[str, Any] = {"headers": _trace_headers(spec)}
    if reference := _parent_reference(runtime.config):
        options["config"] = {"configurable": {_PARENT_KEY: reference}}
    return options


class _ClientCache:
    """Lazily-created, cached Agent Protocol clients keyed by (url, headers)."""

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


_ASYNC_FORK_TASK_PREAMBLE = (
    "[The messages above are prior conversation inherited from the parent agent. "
    "You are the background subagent it has already invoked. Use the established "
    "context to complete the task below, not to repeat the prior delegation. "
    "You have your own tools and runtime; the parent's resources may not be available.]\n\n"
)


def _fork_task_messages(runtime: ToolRuntime, description: str) -> list[AnyMessage]:
    """Snapshot effective history before media downloads can yield to parent edits."""
    messages = _fork_messages(
        runtime.state.get("messages", []),
        runtime.state.get(SUMMARIZATION_EVENT_KEY),
        description,
        preamble=_ASYNC_FORK_TASK_PREAMBLE,
    )
    # Artifacts are local tool data and may contain objects that cannot be copied.
    return deepcopy([message.model_copy(update={"artifact": None}) if isinstance(message, ToolMessage) else message for message in messages])


def _serialize_fork_messages(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Serialize portable conversation content for the remote child."""
    # Eviction tags point to parent-local files; forwarding them would make the
    # child's filesystem middleware hide the full content sent in this snapshot.
    return [message.model_dump(exclude={"artifact": True, "additional_kwargs": {"lc_evicted_to"}}) for message in messages]


def _start_task_messages(spec: AsyncSubAgent, runtime: ToolRuntime, description: str) -> list[dict[str, Any]]:
    """Snapshot the effective conversation for forks, or send only the task."""
    if spec.get("mode") != "fork":
        return [{"role": "user", "content": description}]
    messages = _fork_task_messages(runtime, description)
    # Restore media here so unrelated tools and isolated launches never reload it,
    # including when an approval edits the target before this tool executes.
    messages = _hydrate_tool_messages(messages, runtime.state.get(_BLOB_PAYLOADS_KEY) or {})
    return _serialize_fork_messages(messages)


async def _astart_task_messages(spec: AsyncSubAgent, runtime: ToolRuntime, description: str) -> list[dict[str, Any]]:
    """Async version of `_start_task_messages`."""
    if spec.get("mode") != "fork":
        return [{"role": "user", "content": description}]
    messages = _fork_task_messages(runtime, description)
    messages = await _ahydrate_tool_messages(messages, runtime.state.get(_BLOB_PAYLOADS_KEY) or {})
    return _serialize_fork_messages(messages)


def _describe_async_subagent(spec: AsyncSubAgent) -> str:
    """Describe a remote subagent's context inheritance without promising shared tools."""
    suffix = " (inherits your conversation at launch; uses its own system prompt and tools)" if spec.get("mode") == "fork" else ""
    return f"- {spec['name']}: {spec['description']}{suffix}"


def _build_start_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    tool_description: str,
) -> StructuredTool:
    """Build the `start_async_task` tool."""

    def start_async_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        messages = _start_task_messages(spec, runtime, description)
        try:
            client = clients.get_sync(subagent_type)
            thread = client.threads.create(metadata=_thread_metadata(runtime))
            run = client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": messages},
                **_run_options(spec, runtime),
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, e)
            return f"Failed to launch async subagent '{subagent_type}': {e}"
        task_id = thread["thread_id"]
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
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        messages = await _astart_task_messages(spec, runtime, description)
        try:
            client = clients.get_async(subagent_type)
            thread = await client.threads.create(metadata=_thread_metadata(runtime))
            run = await client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": messages},
                **_run_options(spec, runtime),
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, e)
            return f"Failed to launch async subagent '{subagent_type}': {e}"
        task_id = thread["thread_id"]
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
        infer_schema=False,
        args_schema=StartAsyncTaskSchema,
    )


def _task_status(run: Run, thread: Thread | None) -> str:
    """Return a task's status from its latest run, reading the thread to spot pauses.

    A run that hits `interrupt()` ends as `success`; only its thread shows the pause.
    """
    if run["status"] != "success" or thread is None:
        return run["status"]
    return _THREAD_TO_TASK_STATUS.get(thread.get("status"), "success")


def _needs_latest_run(run: Run, thread: Thread | None) -> bool:
    """Whether a newer run exists than the tracked one, e.g. after a resume from another client."""
    return run["status"] == "success" and thread is not None and thread.get("status") in {"busy", "error"}


def _observe(client: SyncLangGraphClient, task: AsyncTask) -> tuple[Run, Thread | None]:
    """Fetch a task's latest run and, when that run finished, its thread."""
    run = client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
    if run["status"] != "success":
        return run, None
    try:
        thread = client.threads.get(thread_id=task["thread_id"])
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        logger.warning("Failed to fetch thread for task %s", task["task_id"], exc_info=True)
        return run, None
    if _needs_latest_run(run, thread):
        run = next(iter(client.runs.list(thread_id=task["thread_id"], limit=1)), run)
    return run, thread


async def _aobserve(client: LangGraphClient, task: AsyncTask) -> tuple[Run, Thread | None]:
    """Async version of `_observe`."""
    run = await client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
    if run["status"] != "success":
        return run, None
    try:
        thread = await client.threads.get(thread_id=task["thread_id"])
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        logger.warning("Failed to fetch thread for task %s", task["task_id"], exc_info=True)
        return run, None
    if _needs_latest_run(run, thread):
        run = next(iter(await client.runs.list(thread_id=task["thread_id"], limit=1)), run)
    return run, thread


def _pending_interrupts(thread: Thread | None) -> list[Interrupt]:
    """Flatten a thread's pending interrupts, including ones raised inside subgraphs."""
    if not thread:
        return []
    return [item for items in (thread.get("interrupts") or {}).values() for item in items]


_WAITING_NOTE = (
    "The subagent is waiting for a human. The user answers it directly; you can't answer or approve it. "
    "Keep working; check again later for its result."
)


def _build_check_result(run: Run, thread_id: str, thread: Thread | None) -> dict[str, Any]:
    """Build the result dict from a task's latest run and its thread."""
    status = _task_status(run, thread)
    result: dict[str, Any] = {"status": status, "thread_id": thread_id}
    if status == "success":
        thread_values = thread.get("values") if thread else None
        messages = thread_values.get("messages", []) if isinstance(thread_values, dict) else []
        if messages:
            last = messages[-1]
            result["result"] = last.get("content", "") if isinstance(last, dict) else str(last)
        else:
            result["result"] = "(completed with no output messages)"
    elif status == "error":
        error_detail = run.get("error")
        result["error"] = str(error_detail) if error_detail else "The async subagent encountered an error."
    elif status == _WAITING:
        interrupts = _pending_interrupts(thread)
        result["interrupts"] = [{"id": item["id"], "value": item["value"]} for item in interrupts]
        result["note"] = _WAITING_NOTE
    return result


def _build_check_command(
    result: dict[str, Any],
    task: AsyncTask,
    run_id: str,
    tool_call_id: str | None,
) -> Command:
    """Build the `Command` update for a check result."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    changed = result["status"] != task["status"] or run_id != task["run_id"]
    updated_task = AsyncTask(
        task_id=task["task_id"],
        agent_name=task["agent_name"],
        thread_id=task["thread_id"],
        run_id=run_id,
        status=result["status"],
        created_at=task["created_at"],
        last_checked_at=now,
        last_updated_at=now if changed else task["last_updated_at"],
    )
    return Command(
        update={
            "messages": [ToolMessage(json.dumps(result), tool_call_id=tool_call_id)],
            "async_tasks": {task["task_id"]: updated_task},
        }
    )


def _keep_cancelled(task: AsyncTask, result: dict[str, Any]) -> dict[str, Any]:
    """Keep a task cancelled while paused as cancelled; its thread still shows the old pause."""
    if task["status"] == "cancelled" and result["status"] == _WAITING:
        return {"status": "cancelled", "thread_id": result["thread_id"]}
    return result


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


def _build_check_tool(
    clients: _ClientCache,
) -> StructuredTool:
    """Build the `check_async_task` tool."""

    def check_async_task(
        task_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        task = _resolve_tracked_task(task_id, runtime)
        if isinstance(task, str):
            return task
        try:
            run, thread = _observe(clients.get_sync(task["agent_name"]), task)
        except Exception as e:  # noqa: BLE001  # get_sync() may raise ValueError; SDK raises untyped errors
            return f"Failed to get run status: {e}"
        result = _keep_cancelled(task, _build_check_result(run, task["thread_id"], thread))
        return _build_check_command(result, task, run["run_id"], runtime.tool_call_id)

    async def acheck_async_task(
        task_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        task = _resolve_tracked_task(task_id, runtime)
        if isinstance(task, str):
            return task
        try:
            run, thread = await _aobserve(clients.get_async(task["agent_name"]), task)
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to get run status: {e}"
        result = _keep_cancelled(task, _build_check_result(run, task["thread_id"], thread))
        return _build_check_command(result, task, run["run_id"], runtime.tool_call_id)

    return StructuredTool.from_function(
        name="check_async_task",
        func=check_async_task,
        coroutine=acheck_async_task,
        description=(
            "Check the status of an async subagent task. Returns the current status and, if complete, the result. "
            "If the status is `waiting`, it also returns what the subagent is asking the user. "
            "Statuses shown earlier in the conversation are always stale, so call this to get the current status "
            "rather than reporting a status from a previous tool result."
        ),
        infer_schema=False,
        args_schema=CheckAsyncTaskSchema,
    )


def _update_message(tracked: AsyncTask) -> str:
    """Confirm an update, noting when it replaced a pending human request."""
    msg = f"Updated async subagent. task_id: {tracked['task_id']}"
    if tracked["status"] == _WAITING:
        msg += ". It was waiting for a human; that request was dropped."
    return msg


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
        task_id: str,
        message: str,
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
                **_run_options(spec, runtime),
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
        msg = _update_message(tracked)
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: task},
            }
        )

    async def aupdate_async_task(
        task_id: str,
        message: str,
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
                **_run_options(spec, runtime),
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
        msg = _update_message(tracked)
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
        infer_schema=False,
        args_schema=UpdateAsyncTaskSchema,
    )


def _build_cancel_tool(
    clients: _ClientCache,
) -> StructuredTool:
    """Build the `cancel_async_task` tool."""

    def cancel_async_task(
        task_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked

        # A paused subagent has no active run to cancel; marking the task is enough.
        if tracked["status"] != _WAITING:
            try:
                client = clients.get_sync(tracked["agent_name"])
                client.runs.cancel(thread_id=tracked["thread_id"], run_id=tracked["run_id"])
            except Exception as e:  # noqa: BLE001  # get_sync() may raise ValueError; SDK raises untyped errors
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
            last_updated_at=now,
        )
        msg = f"Cancelled async subagent task: {tracked['task_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: updated},
            }
        )

    async def acancel_async_task(
        task_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked

        # A paused subagent has no active run to cancel; marking the task is enough.
        if tracked["status"] != _WAITING:
            try:
                await clients.get_async(tracked["agent_name"]).runs.cancel(thread_id=tracked["thread_id"], run_id=tracked["run_id"])
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
            last_updated_at=now,
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
        description="Cancel a running or waiting async subagent task. Use this to stop a task that is no longer needed.",
        infer_schema=False,
        args_schema=CancelAsyncTaskSchema,
    )


def _fetch_live_status(clients: _ClientCache, task: AsyncTask) -> tuple[str, str]:
    """Fetch a task's current status and run ID, falling back to the cached values on error."""
    if task["status"] in _TERMINAL_STATUSES:
        return task["status"], task["run_id"]
    try:
        run, thread = _observe(clients.get_sync(task["agent_name"]), task)
    except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        _warn_cached_status(task, e)
        return task["status"], task["run_id"]
    return _task_status(run, thread), run["run_id"]


async def _afetch_live_status(clients: _ClientCache, task: AsyncTask) -> tuple[str, str]:
    """Async version of `_fetch_live_status`."""
    if task["status"] in _TERMINAL_STATUSES:
        return task["status"], task["run_id"]
    try:
        run, thread = await _aobserve(clients.get_async(task["agent_name"]), task)
    except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        _warn_cached_status(task, e)
        return task["status"], task["run_id"]
    return _task_status(run, thread), run["run_id"]


def _warn_cached_status(task: AsyncTask, error: Exception) -> None:
    logger.warning(
        "Failed to fetch live status for task %s (agent=%s), returning cached status %r",
        task["task_id"],
        task["agent_name"],
        task["status"],
        exc_info=error,
    )


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


def _build_list_command(filtered: list[AsyncTask], live: list[tuple[str, str]], tool_call_id: str | None) -> Command:
    """Build the `Command` listing tasks with their live statuses and storing them."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries: list[str] = []
    updated: dict[str, AsyncTask] = {}
    for task, (status, run_id) in zip(filtered, live, strict=True):
        entries.append(_format_task_entry(task, status))
        changed = status != task["status"] or run_id != task["run_id"]
        updated[task["task_id"]] = AsyncTask(
            task_id=task["task_id"],
            agent_name=task["agent_name"],
            thread_id=task["thread_id"],
            run_id=run_id,
            status=status,
            created_at=task["created_at"],
            last_checked_at=now,
            last_updated_at=now if changed else task["last_updated_at"],
        )
    msg = f"{len(entries)} tracked task(s):\n" + "\n".join(entries)
    return Command(
        update={
            "messages": [ToolMessage(msg, tool_call_id=tool_call_id)],
            "async_tasks": updated,
        }
    )


def _build_list_tasks_tool(clients: _ClientCache) -> StructuredTool:
    """Build the list_async_tasks tool."""

    def list_async_tasks(
        runtime: ToolRuntime,
        status_filter: Literal["running", "waiting", "success", "error", "cancelled", "all"] | None = None,
    ) -> str | Command:
        tasks: dict[str, AsyncTask] = runtime.state.get("async_tasks") or {}
        filtered = _filter_tasks(tasks, status_filter)
        if not filtered:
            return "No async subagent tasks tracked."
        live = [_fetch_live_status(clients, task) for task in filtered]
        return _build_list_command(filtered, live, runtime.tool_call_id)

    async def alist_async_tasks(
        runtime: ToolRuntime,
        status_filter: Literal["running", "waiting", "success", "error", "cancelled", "all"] | None = None,
    ) -> str | Command:
        tasks: dict[str, AsyncTask] = runtime.state.get("async_tasks") or {}
        filtered = _filter_tasks(tasks, status_filter)
        if not filtered:
            return "No async subagent tasks tracked."
        live = await asyncio.gather(*(_afetch_live_status(clients, task) for task in filtered))
        return _build_list_command(filtered, list(live), runtime.tool_call_id)

    return StructuredTool.from_function(
        name="list_async_tasks",
        func=list_async_tasks,
        coroutine=alist_async_tasks,
        description=(
            "List tracked async subagent tasks with their current live statuses. "
            "By default shows all tasks. Use `status_filter` to narrow by status "
            "(e.g. 'running', 'waiting', 'success', 'error', 'cancelled'). "
            "Use `check_async_task` to get the full result of a specific completed task, "
            "or what a `waiting` task is asking the user. "
            "Statuses shown earlier in the conversation are always stale, so call this to read current "
            "statuses rather than reporting one from a previous tool result."
        ),
        infer_schema=False,
        args_schema=ListAsyncTasksSchema,
    )


def _build_async_subagent_tools(
    agents: list[AsyncSubAgent],
) -> list[StructuredTool]:
    """Build the async subagent tools from agent specs.

    Args:
        agents: List of async subagent specifications.

    Returns:
        List of `StructuredTools` for launch, check, update, cancel, and list operations.
    """
    agent_map: dict[str, AsyncSubAgent] = {a["name"]: a for a in agents}
    clients = _ClientCache(agent_map)
    agents_desc = "\n".join(_describe_async_subagent(a) for a in agents)
    launch_desc = ASYNC_TASK_TOOL_DESCRIPTION.format(available_agents=agents_desc)

    return [
        _build_start_tool(agent_map, clients, launch_desc),
        _build_check_tool(clients),
        _build_update_tool(agent_map, clients),
        _build_cancel_tool(clients),
        _build_list_tasks_tool(clients),
    ]


class AsyncSubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for async subagents running on remote Agent Protocol servers.

    This middleware adds tools for launching, monitoring, and updating
    background tasks on remote Agent Protocol servers. Unlike the synchronous
    `SubAgentMiddleware`, async subagents return immediately with a task ID,
    allowing the main agent to continue working while subagents execute.

    Works with any Agent Protocol-compliant server — LangGraph Platform
    (managed) or self-hosted (e.g. a FastAPI server implementing the Agent
    Protocol spec).

    Task IDs are persisted in the agent state under `async_tasks` so they
    survive context compaction/offloading and can be accessed programmatically.

    Args:
        async_subagents: List of async subagent specifications.

            Each must include `name`, `description`, and `graph_id`. `url` is
            optional — omit it to use ASGI transport for local servers.
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

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    state_schema = AsyncSubAgentState

    def __init__(
        self,
        *,
        async_subagents: list[AsyncSubAgent],
        system_prompt: str | None = None,
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

        for spec in async_subagents:
            mode = spec.get("mode", "isolated")
            if mode not in ("isolated", "fork"):
                msg = f"AsyncSubAgent '{spec['name']}' has invalid mode '{mode}'; expected 'isolated' or 'fork'"
                raise ValueError(msg)
        if any(spec.get("mode") == "fork" for spec in async_subagents):
            warn_beta(name="forked subagents", obj_type="feature")

        self.tools = _build_async_subagent_tools(async_subagents)

        if system_prompt:
            agents_desc = "\n".join(_describe_async_subagent(a) for a in async_subagents)
            self.system_prompt: str | None = system_prompt + "\n\nAvailable async subagent types:\n\n" + agents_desc
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
