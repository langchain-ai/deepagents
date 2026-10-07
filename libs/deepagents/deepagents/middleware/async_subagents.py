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
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict, TypeVar, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ModelRequest, ModelResponse, ResponseT, TracePolicy, omit_payload
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from langgraph_sdk.schema import Run
from langsmith import tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree
from langsmith.utils import tracing_is_enabled
from pydantic import BaseModel, Field

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import BackendProtocol, SandboxBackendProtocol
from deepagents.middleware._utils import append_to_system_message

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

    The subagent's run nests under the parent's trace in LangSmith only if its
    graph is wrapped with `with_parent_trace` (or its factory uses
    `parent_trace_context`). To work in the parent's sandbox, its graph factory
    reconnects to the ID from `parent_sandbox_id`.
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

    status_filter: Literal["running", "success", "error", "cancelled", "all"] | None = Field(
        default=None,
        description="Filter tasks by status. One of: 'running', 'success', 'error', 'cancelled', 'all'. Defaults to 'all'.",
    )


ASYNC_TASK_TOOL_DESCRIPTION = """Start an async subagent on a remote server. The subagent runs in the background and returns a task ID immediately.

Available async agent types:
{available_agents}

## Usage notes:
1. This tool launches a background task and returns immediately with a task ID. Report the task ID to the user and stop — do NOT immediately check status.
2. Use `check_async_task` only when the user asks for a status update or result.
3. Use `update_async_task` to send new instructions to a running task.
4. Multiple async subagents can run concurrently — launch several and let them run in the background.
5. The subagent runs on a remote server, so it has its own tools and capabilities."""  # noqa: E501


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
"""Baggage metadata key (JSON `true`) marking trace headers as a parent to nest under."""


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


@contextmanager
def parent_trace_context(config: RunnableConfig) -> Iterator[None]:
    """Nest this run's trace under the async subagent call that launched it.

    !!! warning "Experimental"

        This helper may change without notice.

    Use it in a subagent's graph factory to adopt the trace `start_async_task`
    sent. It does nothing unless tracing is on and the headers came from
    deepagents. For a plain graph, `with_parent_trace` does this for you.

    Only use it for graphs whose callers you trust: the trace headers come from
    the request, so a caller could choose where these runs are recorded.

    Args:
        config: The run config the Agent Server passes to the graph factory.

    Example:
        ```python
        from contextlib import asynccontextmanager

        from deepagents import create_deep_agent
        from deepagents.middleware import parent_trace_context


        @asynccontextmanager
        async def researcher(config):
            with parent_trace_context(config):
                yield create_deep_agent(model=..., tools=[...])
        ```
    """
    parent = _launching_parent(config.get("configurable") or {})
    if parent is None or not tracing_is_enabled():
        yield
        return
    with tracing_context(parent=parent):
        yield


GraphT = TypeVar("GraphT")


def with_parent_trace(graph: GraphT) -> Callable[[RunnableConfig], AbstractAsyncContextManager[GraphT]]:
    """Wrap an async subagent's graph so its runs nest under the launching call's trace.

    !!! warning "Experimental"

        This helper may change without notice.

    Returns a graph factory that runs `graph` inside `parent_trace_context`.
    Register the result in `langgraph.json` in place of the graph itself. Only
    wrap graphs whose callers you trust (see `parent_trace_context`).

    Args:
        graph: The compiled graph the async subagent's `graph_id` points to.

    Returns:
        A graph factory for the Agent Server.

    Example:
        ```python
        from deepagents import create_deep_agent
        from deepagents.middleware import with_parent_trace

        research_agent = with_parent_trace(create_deep_agent(model=..., tools=[...]))
        ```
    """

    @asynccontextmanager
    async def factory(config: RunnableConfig) -> AsyncIterator[GraphT]:
        with parent_trace_context(config):
            yield graph

    return factory


def _launching_parent(configurable: Mapping[str, Any]) -> RunTree | None:
    """Rebuild the launching span from the trace headers the Agent Server kept, if marked."""
    trace = configurable.get("langsmith-trace")
    metadata = configurable.get("langsmith-metadata")
    if not isinstance(trace, str) or not trace or not isinstance(metadata, dict) or metadata.get(_NEST_UNDER_PARENT) is not True:
        return None
    project = configurable.get("langsmith-project")
    baggage = f"langsmith-project={urllib.parse.quote(project)}" if isinstance(project, str) else ""
    try:
        return RunTree.from_headers({"langsmith-trace": trace, "baggage": baggage})
    except ValueError:
        logger.warning("Ignoring malformed langsmith-trace; the subagent's run keeps its own trace")
        return None


_PARENT_KEY = "deepagents_parent"
"""`configurable` key carrying the launching agent's `ParentReference` to an async subagent's run."""


class ParentReference(TypedDict):
    """What an async subagent's run knows about the agent that launched it.

    !!! warning "Experimental"

        This shape may change without notice.
    """

    thread_id: NotRequired[str]
    """The parent's thread, where notifications for the parent go."""

    assistant_id: NotRequired[str]
    """The parent's assistant, which handles runs queued on its thread."""

    sandbox_id: NotRequired[str]
    """The parent's sandbox, for a child that shares it. Absent when the parent has none."""

    sandbox_provider: NotRequired[str]
    """`__module__.__qualname__` of the parent's sandbox backend class."""


def _provider_name(backend_type: type) -> str:
    return f"{backend_type.__module__}.{backend_type.__qualname__}"


def _parent_sandbox(backend: BackendProtocol | None) -> SandboxBackendProtocol | None:
    """Return the sandbox behind `backend`, looking through a `CompositeBackend`'s default."""
    if isinstance(backend, CompositeBackend):
        backend = backend.default
    return backend if isinstance(backend, SandboxBackendProtocol) else None


def _parent_reference(config: RunnableConfig, backend: BackendProtocol | None) -> ParentReference:
    """Describe the running agent so the async subagent it launches can find it."""
    reference: ParentReference = {}
    if thread_id := (config.get("configurable") or {}).get("thread_id"):
        reference["thread_id"] = str(thread_id)
    if assistant_id := (config.get("metadata") or {}).get("assistant_id") or (config.get("configurable") or {}).get("assistant_id"):
        reference["assistant_id"] = str(assistant_id)
    sandbox = _parent_sandbox(backend)
    if sandbox is not None:
        try:
            reference["sandbox_id"] = sandbox.id
            reference["sandbox_provider"] = _provider_name(type(sandbox))
        except Exception:  # noqa: BLE001  # provider errors; the child then creates its own sandbox
            logger.warning("Could not read the sandbox ID; async subagents get their own sandbox", exc_info=True)
    return reference


def _run_options(runtime: ToolRuntime, backend: BackendProtocol | None) -> dict[str, Any]:
    """Keyword arguments for `runs.create` that link a launched run to this agent."""
    options: dict[str, Any] = {"headers": _trace_headers()}
    if reference := _parent_reference(runtime.config, backend):
        options["config"] = {"configurable": {_PARENT_KEY: reference}}
    return options


def parent_reference(config: RunnableConfig) -> ParentReference | None:
    """Return the reference to the agent that launched this async subagent run.

    !!! warning "Experimental"

        This helper may change without notice.

    Args:
        config: The run config the Agent Server passes to the graph factory.

    Returns:
        The parent reference, or `None` when the run wasn't launched by an async subagent tool.
    """
    reference = (config.get("configurable") or {}).get(_PARENT_KEY)
    return cast("ParentReference", reference) if isinstance(reference, dict) else None


# TODO(review): remove before merge. If reconnecting to the returned ID fails, the  # noqa: FIX002  # reviewer note, removed before merge
# subagent's run errors out. The alternative is to create a new sandbox with a warning;
# we chose failing so users can check the connection and retry instead of a subagent
# silently working in a sandbox the parent never sees.
# https://github.com/langchain-ai/deepagents/issues/6581
def parent_sandbox_id(config: RunnableConfig, provider: type[SandboxBackendProtocol]) -> str | None:
    """Return the launching agent's sandbox ID when it uses the same sandbox provider.

    !!! warning "Experimental"

        This helper may change without notice.

    Use it in an async subagent's graph factory to share the parent's sandbox:
    reconnect to the returned ID with the provider's SDK, and create a new
    sandbox when it returns `None`. The subagent must not stop or delete the
    parent's sandbox.

    Sharing is best effort: if the parent uses another provider, this returns
    `None` (with a warning) and the subagent works in its own sandbox. If
    reconnecting to a returned ID fails, the subagent's run errors out.

    Args:
        config: The run config the Agent Server passes to the graph factory.
        provider: The sandbox backend class the subagent uses, e.g. `ModalSandbox`.

    Returns:
        The parent's sandbox ID, or `None` if the parent has no sandbox or uses another provider.

    Example:
        ```python
        from contextlib import asynccontextmanager

        import modal
        from langchain_modal import ModalSandbox

        from deepagents import create_deep_agent
        from deepagents.middleware import parent_sandbox_id


        @asynccontextmanager
        async def researcher(config):
            sandbox_id = parent_sandbox_id(config, ModalSandbox)
            sandbox = modal.Sandbox.from_id(sandbox_id) if sandbox_id else modal.Sandbox.create(app=...)
            yield create_deep_agent(model=..., backend=ModalSandbox(sandbox=sandbox))
        ```
    """
    reference = parent_reference(config) or {}
    sandbox_id = reference.get("sandbox_id")
    if not sandbox_id:
        return None
    if reference.get("sandbox_provider") != _provider_name(provider):
        logger.warning("Parent sandbox uses %s, not %s; creating a separate sandbox", reference.get("sandbox_provider"), _provider_name(provider))
        return None
    return sandbox_id


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


def _build_start_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    tool_description: str,
    backend: BackendProtocol | None,
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
        try:
            client = clients.get_sync(subagent_type)
            thread = client.threads.create()
            run = client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": description}]},
                **_run_options(runtime, backend),
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
        try:
            client = clients.get_async(subagent_type)
            thread = await client.threads.create()
            run = await client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": description}]},
                **_run_options(runtime, backend),
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
    last_updated_at = now if task["status"] != result["status"] else task["last_updated_at"]
    updated_task = AsyncTask(
        task_id=task["task_id"],
        agent_name=task["agent_name"],
        thread_id=task["thread_id"],
        run_id=task["run_id"],
        status=result["status"],
        created_at=task["created_at"],
        last_checked_at=now,
        last_updated_at=last_updated_at,
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
        task_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        task = _resolve_tracked_task(task_id, runtime)
        if isinstance(task, str):
            return task

        try:
            client = clients.get_sync(task["agent_name"])
            run = client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
        except Exception as e:  # noqa: BLE001  # get_sync() may raise ValueError; SDK raises untyped errors
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
        task_id: str,
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
        description=(
            "Check the status of an async subagent task. Returns the current status and, if complete, the result. "
            "Statuses shown earlier in the conversation are always stale, so call this to get the current status "
            "rather than reporting a status from a previous tool result."
        ),
        infer_schema=False,
        args_schema=CheckAsyncTaskSchema,
    )


def _build_update_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    backend: BackendProtocol | None,
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
                **_run_options(runtime, backend),
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
                **_run_options(runtime, backend),
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
        description="Cancel a running async subagent task. Use this to stop a task that is no longer needed.",
        infer_schema=False,
        args_schema=CancelAsyncTaskSchema,
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
        status_filter: Literal["running", "success", "error", "cancelled", "all"] | None = None,
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
            last_updated_at = now if status != task["status"] else task["last_updated_at"]
            updated_tasks[task["task_id"]] = AsyncTask(
                task_id=task["task_id"],
                agent_name=task["agent_name"],
                thread_id=task["thread_id"],
                run_id=task["run_id"],
                status=status,
                created_at=task["created_at"],
                last_checked_at=now,
                last_updated_at=last_updated_at,
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
        status_filter: Literal["running", "success", "error", "cancelled", "all"] | None = None,
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
            last_updated_at = now if status != task["status"] else task["last_updated_at"]
            updated_tasks[task["task_id"]] = AsyncTask(
                task_id=task["task_id"],
                agent_name=task["agent_name"],
                thread_id=task["thread_id"],
                run_id=task["run_id"],
                status=status,
                created_at=task["created_at"],
                last_checked_at=now,
                last_updated_at=last_updated_at,
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
            "Use `check_async_task` to get the full result of a specific completed task. "
            "Statuses shown earlier in the conversation are always stale, so call this to read current "
            "statuses rather than reporting one from a previous tool result."
        ),
        infer_schema=False,
        args_schema=ListAsyncTasksSchema,
    )


def _build_async_subagent_tools(
    agents: list[AsyncSubAgent],
    backend: BackendProtocol | None = None,
) -> list[StructuredTool]:
    """Build the async subagent tools from agent specs.

    Args:
        agents: List of async subagent specifications.
        backend: The parent agent's backend, whose sandbox launched subagents may share.

    Returns:
        List of `StructuredTools` for launch, check, update, cancel, and list operations.
    """
    agent_map: dict[str, AsyncSubAgent] = {a["name"]: a for a in agents}
    clients = _ClientCache(agent_map)
    agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in agents)
    launch_desc = ASYNC_TASK_TOOL_DESCRIPTION.format(available_agents=agents_desc)

    return [
        _build_start_tool(agent_map, clients, launch_desc, backend),
        _build_check_tool(clients),
        _build_update_tool(agent_map, clients, backend),
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
        backend: The main agent's backend. When it is (or routes by default to)
            a sandbox, launched subagents receive its ID so they can share it
            (see `parent_sandbox_id`). Only a sandbox backend itself or a
            `CompositeBackend`'s default is detected, not one wrapped by another
            backend.

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
        backend: BackendProtocol | None = None,
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

        self.tools = _build_async_subagent_tools(async_subagents, backend)

        if system_prompt:
            agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in async_subagents)
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
