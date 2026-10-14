"""Middleware for async subagents running on remote Agent Protocol servers.

Async subagents use the LangGraph SDK to launch background runs on remote
[Agent Protocol](https://github.com/langchain-ai/agent-protocol) servers.
Unlike synchronous subagents (which block until completion), async subagents
return a task ID immediately, allowing the main agent to monitor progress and
send updates while the subagent works.

Compatible with LangGraph Platform (managed) and self-hosted servers.
"""

import asyncio
import html
import json
import logging
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
    TracePolicy,
    hook_config,
    omit_payload,
)
from langchain.tools import ToolRuntime
from langchain_core._api.beta_decorator import warn_beta
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from langgraph.errors import GraphInterrupt
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from langgraph_sdk.schema import Interrupt, Run, Thread
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from deepagents.middleware._blob_offload import _BLOB_PAYLOADS_KEY, _ahydrate_tool_messages, _hydrate_tool_messages
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.filesystem import NUM_CHARS_PER_TOKEN
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

    To run a subagent defined right here in the background instead, give a
    regular subagent spec `background: True` (see `SubAgent.background`).
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

    runnable: NotRequired[Runnable]
    """A background subagent compiled by `create_deep_agent`, which runs on the main agent's own graph."""

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
    async_tasks_extra_turn: Annotated[NotRequired[bool], PrivateStateAttr]
    """Whether this run already took its one extra model turn for news found after its final answer."""


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


ASYNC_TASK_TOOL_DESCRIPTION = """Start an async subagent. The subagent runs in the background and returns a task ID immediately.

Available async agent types:
{available_agents}

## Usage notes:
1. This tool launches a background task and returns immediately with a task ID. Report the task ID to the user and stop — do NOT immediately check status.
2. You'll usually be told when a task finishes, fails or needs input, so don't check just to see if it's done; use `check_async_task` when the user asks or a notification says to.
3. Use `update_async_task` to send new instructions to a running task.
4. Multiple async subagents can run concurrently — launch several and let them run in the background.
5. The subagent runs separately, on its own thread, with its own tools and capabilities.
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


def _trace_headers() -> dict[str, str]:
    """Build LangSmith headers pointing at the current span, or `{}` when tracing is off.

    The baggage carries the project and the `_NEST_UNDER_PARENT` flag, not the
    caller's run metadata (such as its `thread_id`), so a receiver can't copy that
    onto the subagent's runs.
    """
    run_tree = get_current_run_tree()
    if run_tree is None or not run_tree.trace_id:
        return {}
    baggage = [f"langsmith-metadata={urllib.parse.quote(json.dumps({_NEST_UNDER_PARENT: True}))}"]
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


def _run_options(runtime: ToolRuntime, configurable: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Keyword arguments for `runs.create` that link a launched run to this agent.

    `configurable` adds entries to the run's config next to the parent reference.
    """
    options: dict[str, Any] = {"headers": _trace_headers()}
    reference = _parent_reference(runtime.config)
    if reference or configurable:
        options["config"] = {"configurable": {**({_PARENT_KEY: reference} if reference else {}), **(configurable or {})}}
    return options


def _launching_agent(config: RunnableConfig) -> _ParentReference | None:
    """The reference this run was launched with, or `None` if no async subagent tool launched it."""
    reference = (config.get("configurable") or {}).get(_PARENT_KEY)
    return cast("_ParentReference", reference) if isinstance(reference, dict) else None


_WORKER_KEY = "deepagents_worker"
"""Key naming the background subagent a run of the main agent's graph should be.

Set in the run's `configurable` and in the task thread's metadata, which the
Agent Server copies into every run's `metadata`, so a run started elsewhere on
the thread (such as the user answering the subagent) still runs the subagent.
"""


def _thread_options(spec: AsyncSubAgent, runtime: ToolRuntime) -> dict[str, Any]:
    """`threads.create` arguments for a task: its parent, and a background subagent's name."""
    metadata = _thread_metadata(runtime)
    if _is_worker(spec):
        metadata[_WORKER_KEY] = spec["name"]
    return {"metadata": metadata}


def _launch_target(spec: AsyncSubAgent, runtime: ToolRuntime) -> dict[str, Any]:
    """`runs.create` arguments choosing the graph a task runs on and linking it to this agent.

    Raises:
        ValueError: If a background subagent's task can't find this agent's assistant (not on an Agent Server).
    """
    if not _is_worker(spec):
        return {"assistant_id": spec["graph_id"], **_run_options(runtime)}
    options = _run_options(runtime, {_WORKER_KEY: spec["name"]})
    assistant_id = options["config"]["configurable"].get(_PARENT_KEY, {}).get("assistant_id")
    if assistant_id is None:
        msg = "it runs on this agent's own deployment, which needs an Agent Server"
        raise ValueError(msg)
    return {**options, "assistant_id": assistant_id}


_NOT_SHARED_WITH_WORKERS = frozenset({"async_tasks", "jump_to", "structured_response"})
"""Main-graph state a background subagent neither receives nor returns: task tracking and per-run control keys."""


def _is_worker(spec: Mapping[str, Any]) -> bool:
    """Whether a spec handed to the middleware is a compiled background subagent."""
    return "runnable" in spec


_NOTIFICATION_KEY = "deepagents_notification"
"""Message field (`additional_kwargs` key) marking a task notification and carrying its `TaskEvent`."""

_MAX_RESULT_CHARS = NUM_CHARS_PER_TOKEN * 20_000
"""Results longer than this are cut in notifications, matching the limit for large tool results."""


class TaskEvent(TypedDict):
    """How an async subagent's task ended, as reported to the agent that started it.

    !!! warning "Experimental"

        This shape may change without notice.
    """

    task_id: str
    """The task's ID (the subagent's thread)."""

    subagent: str
    """The async subagent's name."""

    status: Literal["success", "error", "timeout", "waiting"]
    """`waiting` means the subagent paused for input or approval."""

    run_id: NotRequired[str]
    """The subagent run that ended."""

    result: NotRequired[str]
    """The subagent's final answer (cut when very long), when it succeeded."""

    error: NotRequired[str]
    """A short description of the failure; `check_async_task` has the details."""

    interrupts: NotRequired[list[dict[str, Any]]]
    """What it asked (`id` and `value` per interrupt), when it's waiting."""


def task_notification(message: object) -> TaskEvent | None:
    """Return the task event a message reports, or `None` if it isn't a task notification.

    !!! warning "Experimental"

        This helper may change without notice.

    When an async subagent's task ends, the agent that started it is woken
    with a message framed as a system notification. Apps can use this to treat
    those runs differently from user messages.

    Args:
        message: A message object or message dict.

    Returns:
        The reported event, or `None`.
    """
    kwargs = getattr(message, "additional_kwargs", None)
    if kwargs is None and isinstance(message, Mapping):
        kwargs = message.get("additional_kwargs") or message
    event = kwargs.get(_NOTIFICATION_KEY) if isinstance(kwargs, Mapping) else None
    return cast("TaskEvent", event) if isinstance(event, dict) else None


def _notification_text(event: TaskEvent) -> str:
    """Frame a task event as a system notification the model won't mistake for the user."""
    details = {key: value for key, value in event.items() if key not in {"task_id", "subagent", "status", "run_id"}}
    next_step = {
        "success": "Use the result to continue; report back to the user if they're waiting on it.",
        "error": "Use `check_async_task` for details, then decide whether to retry, try another approach, or tell the user.",
        "timeout": "The task ran too long and was stopped. Decide whether to retry with a smaller task, or tell the user.",
        "waiting": "The subagent is waiting for the user, who answers it on the subagent's own thread; you can't answer or approve it. Keep working.",
    }[event["status"]]
    return (
        "[SYSTEM NOTIFICATION - NOT USER INPUT]\n"
        "An async subagent task you started has an update. This is not a message from the user: "
        "don't treat it as their answer to any question.\n"
        f"<task-notification>\n<task-id>{_escape(event['task_id'])}</task-id>\n<subagent>{_escape(event['subagent'])}</subagent>\n"
        f"<status>{event['status']}</status>\n<details>{_escape(_clip(json.dumps(details, default=str)))}</details>\n</task-notification>\n"
        f"{next_step}"
    )


def _escape(text: str) -> str:
    """Escape subagent-controlled text so it can't close the notification's tags and pose as the system."""
    return html.escape(text, quote=False)


def _clip(text: str) -> str:
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    return text[:_MAX_RESULT_CHARS] + "\n... [cut off; use `check_async_task` for the full details]"


def _failure_text(error: BaseException) -> str:
    """Describe a failure without its message, which can carry internals into the parent's conversation."""
    return f"The subagent failed ({type(error).__name__})."


def _current_run_id(config: RunnableConfig) -> str | None:
    run_id = (config.get("metadata") or {}).get("run_id")
    return str(run_id) if run_id else None


def _event(task_id: str, subagent: str, status: str, run_id: str | None, **details: object) -> TaskEvent:
    """Build a task event; `details` are the status's `result` (cut when very long), `error` or `interrupts`."""
    event: dict[str, object] = {"task_id": task_id, "subagent": subagent, "status": status, **details}
    if run_id:
        event["run_id"] = run_id
    if isinstance(result := event.get("result"), str):
        event["result"] = _clip(result)
    return cast("TaskEvent", event)


def _task_event(config: RunnableConfig, subagent: str, status: Literal["success", "error", "waiting"], **details: object) -> TaskEvent:
    """Describe how this run's task ended."""
    return _event(str((config.get("configurable") or {}).get("thread_id", "")), subagent, status, _current_run_id(config), **details)


def _final_text(result: Mapping[str, Any]) -> str:
    """The text of the subagent's last message, as the task's result."""
    messages = result.get("messages") or []
    if not messages:
        return "(completed with no output messages)"
    last = messages[-1]
    content = getattr(last, "content", last)
    return content if isinstance(content, str) else json.dumps(content, default=str)


_CHECK_KEY = "deepagents_task_check"
"""`configurable` key marking a stateless run of the parent's graph as a delayed check on one task run."""

_RACE_CHECK_SECONDS = 30
"""How long after a subagent ends the delayed check runs, when the parent couldn't be woken then."""


class _TaskCheck(TypedDict):
    task_id: str
    run_id: str
    parent_thread_id: str
    assistant_id: str


async def _aschedule_check(check: _TaskCheck) -> None:
    """Queue one delayed check on a task run, as a stateless run of the parent's graph.

    It covers a subagent that ends just as the parent finishes its run. It never
    touches the parent's thread or calls the model, and survives restarts.
    """
    try:
        await get_client().runs.create(
            None,
            check["assistant_id"],
            input={"messages": []},
            config={"configurable": {_CHECK_KEY: check}},
            after_seconds=_RACE_CHECK_SECONDS,
        )
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors; the parent still finds out when it next checks
        logger.warning("Failed to schedule a check on task %s", check["task_id"], exc_info=True)


_Delivery = Literal["delivered", "later", "unreachable"]
"""Whether a notification started a parent run, the parent will find out on its own, or the parent isn't on this deployment."""


_DELIVERABLE_THREAD_STATUSES = frozenset({"idle", "error"})
"""Parent thread statuses a notification run can start on: neither running nor waiting for a human."""


async def _awake_parent(parent_thread_id: str, assistant_id: str, event: TaskEvent) -> _Delivery:
    """Start a notification run on the parent's thread only while it's idle (or its last run failed).

    A new run on a thread waiting for a human would cancel that approval; `reject` covers a thread that just became busy.
    A parent thread this deployment doesn't have is `unreachable`: it's on another deployment, or gone.
    """
    client = get_client()
    try:
        if (await client.threads.get(thread_id=parent_thread_id)).get("status") not in _DELIVERABLE_THREAD_STATUSES:
            return "later"
        await client.runs.create(
            thread_id=parent_thread_id,
            assistant_id=assistant_id,
            input={"messages": [_notification_message(event)]},
            multitask_strategy="reject",
        )
    except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404:  # noqa: PLR2004  # HTTP not found
            logger.debug("Parent thread of task %s isn't on this deployment", event["task_id"])
            return "unreachable"
        log = logger.debug if status == 409 else logger.warning  # noqa: PLR2004  # 409: became busy
        log("Didn't deliver the notification for task %s yet", event["task_id"], exc_info=True)
        return "later"
    return "delivered"


async def _notify_parent(config: RunnableConfig, event: TaskEvent) -> None:
    """Wake the parent if it's idle; otherwise it finds out on its own.

    A busy parent sees the news before its next model call, a paused one after
    the user answers its approval. Schedules one delayed check for a parent that
    was just finishing. Never raises.
    """
    parent = _launching_agent(config) or {}
    if "thread_id" not in parent or "assistant_id" not in parent:
        return
    if await _awake_parent(parent["thread_id"], parent["assistant_id"], event) != "later" or "run_id" not in event:
        return
    await _aschedule_check(
        {"task_id": event["task_id"], "run_id": event["run_id"], "parent_thread_id": parent["thread_id"], "assistant_id": parent["assistant_id"]}
    )


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
            thread = client.threads.create(**_thread_options(spec, runtime))
            run = client.runs.create(
                thread_id=thread["thread_id"],
                input={"messages": messages},
                **_launch_target(spec, runtime),
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
            thread = await client.threads.create(**_thread_options(spec, runtime))
            run = await client.runs.create(
                thread_id=thread["thread_id"],
                input={"messages": messages},
                **_launch_target(spec, runtime),
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
                input={"messages": [{"role": "user", "content": message}]},
                multitask_strategy="interrupt",
                **_launch_target(spec, runtime),
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
                input={"messages": [{"role": "user", "content": message}]},
                multitask_strategy="interrupt",
                **_launch_target(spec, runtime),
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


async def _aparent_task(check: _TaskCheck) -> AsyncTask | None:
    """Read the parent's record of the checked task from its thread, without starting a run there."""
    try:
        values = (await get_client().threads.get(thread_id=check["parent_thread_id"])).get("values")
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors; treated as gone
        logger.warning("Failed to read the parent thread of task %s", check["task_id"], exc_info=True)
        return None
    tasks = values.get("async_tasks") if isinstance(values, dict) else None
    return cast("AsyncTask | None", (tasks or {}).get(check["task_id"]))


def _check_event(task: AsyncTask, run: Run, thread: Thread | None, status: str) -> TaskEvent | None:
    """The notification for a task outcome a check found, or `None` for outcomes not worth waking for."""
    if status not in {"success", "error", "timeout", _WAITING}:
        return None
    result = _build_check_result(run, task["thread_id"], thread)
    details = {key: result[key] for key in ("result", "interrupts") if key in result}
    if status == "error":
        details["error"] = "The subagent failed."
    return _event(task["task_id"], task["agent_name"], status, run.get("run_id", task["run_id"]), **details)


_PULL_TIMEOUT_SECONDS = 5
"""How long the parent waits on one task's status before skipping it until the next check."""

_UNFINISHED_STATUSES = frozenset({"running", "pending", _WAITING})
"""Tracked task statuses that can still change, so the parent checks them for news."""


def _notification_message(event: TaskEvent) -> dict[str, Any]:
    """A task event as the message the parent's model sees, marked so apps can recognize it."""
    return {"role": "user", "content": _notification_text(event), _NOTIFICATION_KEY: dict(event)}


def _observed(task: AsyncTask, status: str, run_id: str) -> AsyncTask:
    """`task` updated to what a check found."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    changed = status != task["status"] or run_id != task["run_id"]
    return AsyncTask(
        task_id=task["task_id"],
        agent_name=task["agent_name"],
        thread_id=task["thread_id"],
        run_id=run_id,
        status=status,
        created_at=task["created_at"],
        last_checked_at=now,
        last_updated_at=now if changed else task["last_updated_at"],
    )


class AsyncSubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for async subagents running on Agent Protocol servers or on this agent's own graph.

    This middleware adds tools for launching, monitoring, and updating
    background tasks on remote Agent Protocol servers. Unlike the synchronous
    `SubAgentMiddleware`, async subagents return immediately with a task ID,
    allowing the main agent to continue working while subagents execute.

    Works with any Agent Protocol-compliant server — LangGraph Platform
    (managed) or self-hosted (e.g. a FastAPI server implementing the Agent
    Protocol spec).

    Task IDs are persisted in the agent state under `async_tasks` so they
    survive context compaction/offloading and can be accessed programmatically.

    The agent checks its unfinished tasks at the start of each run, before
    model calls, and once after its final answer, and adds what finished,
    failed or is waiting for the user to its conversation. A subagent that ends
    while the agent is idle wakes it.

    Args:
        async_subagents: List of async subagent specifications.

            Each must include `name`, `description`, and `graph_id`. `url` is
            optional — omit it to use ASGI transport for local servers. A
            spec with a compiled `runnable` (as `create_deep_agent` builds for
            `background` subagents) runs on this agent's own graph.
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

        self._clients = _ClientCache({a["name"]: a for a in async_subagents})
        self.tools = _build_async_subagent_tools(async_subagents)
        self._workers: dict[str, Runnable] = {a["name"]: a["runnable"] for a in async_subagents if _is_worker(a)}
        self.state_keys: frozenset[str] = frozenset({"messages"})
        """State keys of the main graph a background subagent's result may update; `create_deep_agent` sets them."""

        if system_prompt:
            agents_desc = "\n".join(_describe_async_subagent(a) for a in async_subagents)
            self.system_prompt: str | None = system_prompt + "\n\nAvailable async subagent types:\n\n" + agents_desc
        else:
            self.system_prompt = system_prompt

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AsyncSubAgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Route this run: a task check, a background subagent's task, or a normal run.

        Task checks and notifying the parent need the async path; the Agent Server runs graphs that way.
        """
        if (get_config().get("configurable") or {}).get(_CHECK_KEY):
            return {"jump_to": "end"}
        name = self._requested_worker()
        if name is not None:
            return self._worker_update(self._workers[name].invoke(self._worker_input(state)))
        return self._record_notification(state)

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state: AsyncSubAgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """(async) Route this run: a task check, a background subagent's task (which reports how it ended), or a normal run."""
        if check := (get_config().get("configurable") or {}).get(_CHECK_KEY):
            return await self._acheck_task(check)
        name = self._requested_worker()
        if name is not None:
            return await self._arun_worker(name, state)
        recorded = self._record_notification(state) or {}
        tasks = {**(state.get("async_tasks") or {}), **recorded.get("async_tasks", {})}
        news = await self._apull(tasks)
        return {
            **recorded,
            **news,
            "async_tasks": {**recorded.get("async_tasks", {}), **news.get("async_tasks", {})},
            "async_tasks_extra_turn": False,
        }

    def before_model(self, state: AsyncSubAgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Pass through: checking tasks needs the async path, which the Agent Server uses."""
        return None

    async def abefore_model(self, state: AsyncSubAgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """(async) Add news from this agent's tasks before the model call."""
        return await self._apull(state.get("async_tasks") or {}) or None

    def after_model(self, state: AsyncSubAgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Pass through: checking tasks needs the async path, which the Agent Server uses."""
        return None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: AsyncSubAgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """(async) After the final answer, check tasks once more; on news, give the model one more turn."""
        messages = state.get("messages") or []
        if state.get("async_tasks_extra_turn") or not messages or getattr(messages[-1], "tool_calls", None):
            return None
        news = await self._apull(state.get("async_tasks") or {})
        if not news.get("messages"):
            return news or None
        return {**news, "async_tasks_extra_turn": True, "jump_to": "model"}

    async def _apull(self, tasks: Mapping[str, AsyncTask]) -> dict[str, Any]:
        """Check unfinished tasks; return their updated records and a notification per new outcome.

        Outcomes the agent already saw (through a check, a list, or an earlier
        notification) aren't reported again. A task that can't be checked is
        skipped until the next check.
        """
        unfinished = [task for task in tasks.values() if task["status"] in _UNFINISHED_STATUSES]
        observed = await asyncio.gather(*(self._aobserve_task(task) for task in unfinished))
        records: dict[str, AsyncTask] = {}
        messages: list[dict[str, Any]] = []
        for task, found in zip(unfinished, observed, strict=True):
            if found is None:
                continue
            run, thread = found
            status = _task_status(run, thread)
            if status == task["status"] and run["run_id"] == task["run_id"]:
                continue
            records[task["task_id"]] = _observed(task, status, run["run_id"])
            if status != task["status"] and (event := _check_event(task, run, thread, status)):
                messages.append(_notification_message(event))
        update: dict[str, Any] = {"async_tasks": records} if records else {}
        if messages:
            update["messages"] = messages
        return update

    async def _aobserve_task(self, task: AsyncTask) -> tuple[Run, Thread | None] | None:
        try:
            return await asyncio.wait_for(_aobserve(self._clients.get_async(task["agent_name"]), task), _PULL_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors, or it timed out; checked again next time
            logger.warning("Failed to check task %s", task["task_id"], exc_info=True)
            return None

    async def _arun_worker(self, name: str, state: Mapping[str, Any]) -> dict[str, Any]:
        """Run a background subagent as this run, then wake the agent that started it."""
        config = get_config()
        try:
            result = await self._workers[name].ainvoke(self._worker_input(state))
        except GraphInterrupt as paused:
            interrupts = [{"id": item.id, "value": item.value} for item in (paused.args[0] if paused.args else ())]
            await _notify_parent(config, _task_event(config, name, "waiting", interrupts=interrupts))
            raise
        except Exception as e:
            await _notify_parent(config, _task_event(config, name, "error", error=_failure_text(e)))
            raise
        await _notify_parent(config, _task_event(config, name, "success", result=_final_text(result)))
        return self._worker_update(result)

    @staticmethod
    def _record_notification(state: Mapping[str, Any]) -> dict[str, Any] | None:
        """When a task notification started this run, record the task's outcome so checks don't report it again."""
        messages = state.get("messages") or []
        event = task_notification(messages[-1]) if messages else None
        task = (state.get("async_tasks") or {}).get(event["task_id"]) if event else None
        if event is None or task is None:
            return None
        run_id = event.get("run_id", task["run_id"])
        # Answering a pause starts a new run; an update cancels the old run before it can report.
        if run_id != task["run_id"] and task["status"] not in _UNFINISHED_STATUSES:
            return None
        return {"async_tasks": {task["task_id"]: _observed(task, event["status"], run_id)}}

    async def _acheck_task(self, check: _TaskCheck) -> dict[str, Any]:
        """Wake the parent once if the task ended and the parent hasn't recorded it; a busy or paused parent is left alone."""
        task = await _aparent_task(check)
        # A newer launch, or an outcome the parent already recorded, needs nothing.
        if task is None or task["run_id"] != check["run_id"] or task["status"] not in {"running", "pending"}:
            return {"jump_to": "end"}
        try:
            run, thread = await _aobserve(self._clients.get_async(task["agent_name"]), task)
        except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors; the parent finds out when it next checks
            logger.warning("Failed to check task %s", task["task_id"], exc_info=True)
            return {"jump_to": "end"}
        if event := _check_event(task, run, thread, _task_status(run, thread)):
            await _awake_parent(check["parent_thread_id"], check["assistant_id"], event)
        return {"jump_to": "end"}

    @staticmethod
    def _worker_input(state: Mapping[str, Any]) -> dict[str, Any]:
        """The task thread's state (files, todos, summaries, ...) the helper continues from."""
        return {key: value for key, value in state.items() if key not in _NOT_SHARED_WITH_WORKERS}

    def _worker_update(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """Keep the helper's state on the task thread so its next run picks up where this one ended."""
        update = {key: value for key, value in result.items() if key in self.state_keys and key not in _NOT_SHARED_WITH_WORKERS}
        return {**update, "jump_to": "end"}

    def _requested_worker(self) -> str | None:
        """Return the background subagent this run should execute as, or `None` for a normal run.

        Raises:
            ValueError: If the run names a subagent this agent doesn't run in the background.
        """
        config = get_config()
        name = (config.get("configurable") or {}).get(_WORKER_KEY) or (config.get("metadata") or {}).get(_WORKER_KEY)
        if name is None:
            return None
        if name not in self._workers:
            msg = f"This agent has no background subagent named {name!r}"
            raise ValueError(msg)
        return name

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
