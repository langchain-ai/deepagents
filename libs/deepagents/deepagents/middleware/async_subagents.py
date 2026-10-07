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
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict, TypeVar, cast

from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    hook_config,
    omit_payload,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from langgraph.constants import CONFIG_KEY_CHECKPOINTER
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from langgraph_sdk.schema import Interrupt, Run, Thread
from langsmith import tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree
from langsmith.utils import tracing_is_enabled
from pydantic import BaseModel, Field

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import BackendProtocol, SandboxBackendProtocol
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.filesystem import FilesystemPermission

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

    To run a subagent defined right here instead of a separately deployed
    graph, set `graph_id` to `"self"` and give it a `system_prompt` (plus any
    `tools`, `model`, `middleware`, ... as for a regular subagent). Each task
    then runs on the main agent's own deployment, on its own thread, using the
    main agent's backend. This needs the main agent to run on an Agent Server.

    Example:
        ```python
        create_deep_agent(
            subagents=[
                {
                    "name": "researcher",
                    "description": "Deep research that takes a while",
                    "graph_id": "self",
                    "system_prompt": "You research topics thoroughly.",
                    "tools": [web_search],
                }
            ],
        )
        ```
    """

    name: str
    """Unique identifier for the async subagent."""

    description: str
    """What this subagent does.

    The main agent uses this to decide when to delegate.
    """

    graph_id: str
    """The graph name or assistant ID on the remote server.

    `"self"` with a `system_prompt` runs the subagent defined in this spec on
    the main agent's own graph instead.
    """

    url: NotRequired[str]
    """URL of the [Agent Protocol](https://github.com/langchain-ai/agent-protocol) server.

    Defaults to the LangGraph SDK's default endpoint. Omit to use ASGI
    transport for local servers.
    """

    headers: NotRequired[dict[str, str]]
    """Additional headers to include in requests to the remote server."""

    system_prompt: NotRequired[str]
    """Instructions for a subagent defined here (`graph_id="self"`)."""

    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    """Tools for a subagent defined here. Defaults to the main agent's tools."""

    model: NotRequired[str | BaseChatModel]
    """Model for a subagent defined here. Defaults to the main agent's model."""

    middleware: NotRequired[list[AgentMiddleware]]
    """Extra middleware for a subagent defined here."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """Human-in-the-loop configuration for a subagent defined here."""

    skills: NotRequired[list[str]]
    """Skill sources for a subagent defined here."""

    permissions: NotRequired[list[FilesystemPermission]]
    """Filesystem permissions for a subagent defined here."""

    runnable: NotRequired[Runnable]
    """The compiled subagent that runs on the main agent's graph; `create_deep_agent` builds it."""


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


class ResumeAsyncTaskSchema(BaseModel):
    """Input schema for the `resume_async_task` tool."""

    task_id: str = Field(description="The exact task_id string returned by start_async_task. Pass it verbatim.")
    response: str | dict[str, str] | None = Field(
        default=None,
        description=(
            "Your answer to the subagent's question. With several pending questions, an object mapping each "
            "interrupt id to its answer. Leave empty for approvals (a human decides) and breakpoints."
        ),
    )


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
2. Use `check_async_task` only when the user asks for a status update or result.
3. Use `update_async_task` to send new instructions to a running task.
4. Multiple async subagents can run concurrently — launch several and let them run in the background.
5. The subagent runs separately, on its own thread, with its own tools and capabilities.
6. A task with status `waiting` has paused for input or approval. `check_async_task` shows what it asked; answer with `resume_async_task`."""  # noqa: E501


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


def _run_options(runtime: ToolRuntime, backend: BackendProtocol | None, configurable: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Keyword arguments for `runs.create` that link a launched run to this agent.

    `configurable` adds entries to the run's config next to the parent reference.
    """
    options: dict[str, Any] = {"headers": _trace_headers()}
    reference = _parent_reference(runtime.config, backend)
    if reference or configurable:
        options["config"] = {"configurable": {**({_PARENT_KEY: reference} if reference else {}), **(configurable or {})}}
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


_SELF_GRAPH = "self"
"""`graph_id` for an async subagent defined inline that runs on the main agent's own graph."""

_WORKER_KEY = "deepagents_worker"
"""`configurable` key naming the inline async subagent a run of the main agent's graph should be."""


def _runs_on_parent(spec: Mapping[str, Any]) -> bool:
    """Whether an async subagent spec is defined inline and runs on the main agent's own graph.

    A remote graph that happens to be named `"self"` has no `system_prompt` or
    `runnable`, so it keeps meaning that graph.
    """
    return spec.get("graph_id") == _SELF_GRAPH and ("system_prompt" in spec or "runnable" in spec)


def _launch_target(spec: AsyncSubAgent, runtime: ToolRuntime, backend: BackendProtocol | None) -> dict[str, Any]:
    """`runs.create` arguments choosing the graph a task runs on and linking it to this agent.

    Raises:
        ValueError: If an inline subagent's task can't find this agent's assistant (not on an Agent Server).
    """
    if not _is_worker(spec):
        return {"assistant_id": spec["graph_id"], **_run_options(runtime, backend)}
    options = _run_options(runtime, backend, {_WORKER_KEY: spec["name"]})
    assistant_id = options["config"]["configurable"].get(_PARENT_KEY, {}).get("assistant_id")
    if assistant_id is None:
        msg = "it runs on this agent's own deployment, which needs an Agent Server"
        raise ValueError(msg)
    return {**options, "assistant_id": assistant_id}


def _is_worker(spec: Mapping[str, Any]) -> bool:
    """Whether a spec handed to the middleware is a compiled inline subagent."""
    return "runnable" in spec


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
                input={"messages": [{"role": "user", "content": description}]},
                **_launch_target(spec, runtime, backend),
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
                input={"messages": [{"role": "user", "content": description}]},
                **_launch_target(spec, runtime, backend),
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


_WAITING = "waiting"
"""Task status for a subagent that paused for input: its run finished but its thread is interrupted."""

_THREAD_TO_TASK_STATUS = {"interrupted": _WAITING, "busy": "running", "error": "error"}
"""Task status implied by the thread when the tracked run finished but the thread moved on."""


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


def _is_approval(value: object) -> bool:
    """Whether an interrupt is a `HumanInTheLoopMiddleware` approval request."""
    return isinstance(value, dict) and isinstance(value.get("action_requests"), list) and isinstance(value.get("review_configs"), list)


def _resume_guidance(interrupts: list[Interrupt]) -> str:
    """Tell the model how to answer a paused subagent."""
    if not interrupts:
        return "The subagent stopped at a breakpoint. Call `resume_async_task` with this task_id to continue; no response is needed."
    questions = [item for item in interrupts if not _is_approval(item["value"])]
    hints = []
    if len(questions) < len(interrupts):
        hints.append("Approval requests are decided by a human: call `resume_async_task` and the human is asked; don't decide them yourself.")
    if len(questions) == 1:
        hints.append("Answer the question with `resume_async_task(task_id, response)`.")
    elif questions:
        hints.append("Answer with `resume_async_task`, passing `response` as an object mapping each question's interrupt id to its answer.")
    return " ".join(hints)


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
        result["how_to_resume"] = _resume_guidance(interrupts)
    return result


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _with_status(task: AsyncTask, status: str, run_id: str, *, checked: bool) -> AsyncTask:
    """Copy `task` with a new status and run, stamping the check and change times."""
    now = _now()
    return AsyncTask(
        task_id=task["task_id"],
        agent_name=task["agent_name"],
        thread_id=task["thread_id"],
        run_id=run_id,
        status=status,
        created_at=task["created_at"],
        last_checked_at=now if checked else task["last_checked_at"],
        last_updated_at=now if status != task["status"] or run_id != task["run_id"] else task["last_updated_at"],
    )


def _build_check_command(
    result: dict[str, Any],
    task: AsyncTask,
    run_id: str,
    tool_call_id: str | None,
) -> Command:
    """Build the `Command` update for a check result."""
    updated_task = _with_status(task, result["status"], run_id, checked=True)
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
            "If the status is `waiting`, it also returns what the subagent asked and how to answer with `resume_async_task`. "
            "Statuses shown earlier in the conversation are always stale, so call this to get the current status "
            "rather than reporting a status from a previous tool result."
        ),
        infer_schema=False,
        args_schema=CheckAsyncTaskSchema,
    )


def _update_message(tracked: AsyncTask) -> str:
    """Confirm an update, noting when it replaced a pending question instead of answering it."""
    msg = f"Updated async subagent. task_id: {tracked['task_id']}"
    if tracked["status"] == _WAITING:
        msg += ". It was waiting for input; that question was dropped. Use `resume_async_task` to answer a question instead."
    return msg


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
                input={"messages": [{"role": "user", "content": message}]},
                multitask_strategy="interrupt",
                **_launch_target(spec, runtime, backend),
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
                **_launch_target(spec, runtime, backend),
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


def _labelled_request(subagent: str, request: dict[str, Any]) -> dict[str, Any]:
    """Copy an approval request, naming the subagent that asked in each action's description."""
    label = f"Requested by async subagent '{subagent}'"
    actions = [
        {**action, "description": f"{label}: {action['description']}" if action.get("description") else label}
        for action in request["action_requests"]
    ]
    return {**request, "action_requests": actions}


def _invalid_decisions(request: dict[str, Any], answer: object) -> str | None:
    """Check a human's answer against an approval request, mirroring `HumanInTheLoopMiddleware`."""
    decisions = answer.get("decisions") if isinstance(answer, dict) else None
    actions = request["action_requests"]
    if not isinstance(decisions, list) or len(decisions) != len(actions):
        return f"The approval needs {len(actions)} decision(s) under `decisions`; the subagent is still waiting."
    allowed = {config["action_name"]: config["allowed_decisions"] for config in request["review_configs"]}
    for action, decision in zip(actions, decisions, strict=True):
        if not isinstance(decision, dict) or decision.get("type") not in allowed.get(action["name"], []):
            return f"Decision {decision!r} isn't allowed for `{action['name']}`; the subagent is still waiting."
    return None


def _question_answers(questions: list[Interrupt], response: str | dict[str, str] | None) -> dict[str, Any] | str:
    """Map the model's response to the pending questions' interrupt IDs, or explain what's missing."""
    ids = [item["id"] for item in questions]
    if not ids:
        return {}
    if isinstance(response, dict) and set(ids) <= response.keys():
        return {interrupt_id: response[interrupt_id] for interrupt_id in ids}
    if len(ids) == 1 and response is not None:
        return {ids[0]: response}
    if len(ids) == 1:
        return "The subagent asked a question; pass your answer as `response`."
    return f"The subagent asked {len(ids)} questions; pass `response` as an object mapping each interrupt id ({', '.join(ids)}) to its answer."


def _human_decisions(subagent: str, approvals: list[Interrupt], runtime: ToolRuntime) -> dict[str, Any] | str:
    """Ask this agent's human to decide each approval request, pausing this agent until they answer."""
    if not approvals:
        return {}
    if CONFIG_KEY_CHECKPOINTER not in (runtime.config.get("configurable") or {}):
        return (
            "The subagent is waiting for human approval, but this agent can't pause to ask (it has no "
            "checkpointer). Resume the subagent from a UI that shows approvals, or run this agent with a checkpointer."
        )
    answers: dict[str, Any] = {}
    for item in approvals:
        answer = interrupt(_labelled_request(subagent, item["value"]))
        if error := _invalid_decisions(item["value"], answer):
            return error
        answers[item["id"]] = answer
    return answers


def _resume_input(subagent: str, thread: Thread | None, response: str | dict[str, str] | None, runtime: ToolRuntime) -> dict[str, Any] | str:
    """Return the `runs.create` arguments that continue a paused subagent, or an error for the model.

    Questions are answered with the model's `response`; approval requests go to
    this agent's human, so the model can't approve its subagent's actions.
    A breakpoint (no interrupts) continues with no input.
    """
    interrupts = _pending_interrupts(thread)
    if not interrupts:
        return {}
    answers = _question_answers([item for item in interrupts if not _is_approval(item["value"])], response)
    if isinstance(answers, str):
        return answers
    decisions = _human_decisions(subagent, [item for item in interrupts if _is_approval(item["value"])], runtime)
    if isinstance(decisions, str):
        return decisions
    return {"command": {"resume": {**answers, **decisions}}}


def _not_waiting_message(task: AsyncTask, status: str) -> str:
    return f"Task {task['task_id']} is {status}, not waiting for input, so there is nothing to resume. It may have been resumed elsewhere."


def _resumed_command(tracked: AsyncTask, run_id: str, tool_call_id: str | None) -> Command:
    """Build the `Command` recording the resumed run."""
    return Command(
        update={
            "messages": [ToolMessage(f"Resumed async subagent. task_id: {tracked['task_id']}", tool_call_id=tool_call_id)],
            "async_tasks": {tracked["task_id"]: _with_status(tracked, "running", run_id, checked=False)},
        }
    )


def _build_resume_tool(  # noqa: C901  # complexity from necessary error handling
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    backend: BackendProtocol | None,
) -> StructuredTool:
    """Build the `resume_async_task` tool.

    Starts a new run on the paused subagent's thread that continues from its
    checkpoint, so the subagent picks up exactly where it called `interrupt()`.
    The subagent's state is re-read first, so a subagent resumed elsewhere
    meanwhile isn't resumed twice.
    """

    def resume_async_task(
        task_id: str,
        runtime: ToolRuntime,
        response: str | dict[str, str] | None = None,
    ) -> str | Command:
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked
        try:
            client = clients.get_sync(tracked["agent_name"])
            run, thread = _observe(client, tracked)
        except Exception as e:  # noqa: BLE001  # get_sync() may raise ValueError; SDK raises untyped errors
            return f"Failed to get run status: {e}"
        if (status := _task_status(run, thread)) != _WAITING:
            return _not_waiting_message(tracked, status)
        # Outside the try below: asking the human raises LangGraph's interrupt, which must propagate.
        resume = _resume_input(tracked["agent_name"], thread, response, runtime)
        if isinstance(resume, str):
            return resume
        try:
            new_run = client.runs.create(
                thread_id=tracked["thread_id"],
                **resume,
                **_launch_target(agent_map[tracked["agent_name"]], runtime, backend),
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to resume async subagent '%s': %s", tracked["agent_name"], e)
            return f"Failed to resume async subagent: {e}"
        return _resumed_command(tracked, new_run["run_id"], runtime.tool_call_id)

    async def aresume_async_task(
        task_id: str,
        runtime: ToolRuntime,
        response: str | dict[str, str] | None = None,
    ) -> str | Command:
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked
        client = clients.get_async(tracked["agent_name"])
        try:
            run, thread = await _aobserve(client, tracked)
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to get run status: {e}"
        if (status := _task_status(run, thread)) != _WAITING:
            return _not_waiting_message(tracked, status)
        # Outside the try below: asking the human raises LangGraph's interrupt, which must propagate.
        resume = _resume_input(tracked["agent_name"], thread, response, runtime)
        if isinstance(resume, str):
            return resume
        try:
            new_run = await client.runs.create(
                thread_id=tracked["thread_id"],
                **resume,
                **_launch_target(agent_map[tracked["agent_name"]], runtime, backend),
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to resume async subagent '%s': %s", tracked["agent_name"], e)
            return f"Failed to resume async subagent: {e}"
        return _resumed_command(tracked, new_run["run_id"], runtime.tool_call_id)

    return StructuredTool.from_function(
        name="resume_async_task",
        func=resume_async_task,
        coroutine=aresume_async_task,
        description=(
            "Continue an async subagent whose status is `waiting`, from exactly where it paused. "
            "Answer its questions with `response` (see `how_to_resume` from `check_async_task`). "
            "Approval requests are sent to a human, never decided by you; breakpoints need no response."
        ),
        infer_schema=False,
        args_schema=ResumeAsyncTaskSchema,
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


_TERMINAL_STATUSES = frozenset({"cancelled", "success", "error", "timeout", "interrupted"})
"""Task statuses that will never change, so live-status fetches can be skipped.

`interrupted` is a run cancelled or replaced on the server. A subagent paused
by `interrupt()` is `waiting`, which isn't terminal.
"""


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
    entries = [_format_task_entry(task, status) for task, (status, _) in zip(filtered, live, strict=True)]
    updated = {task["task_id"]: _with_status(task, status, run_id, checked=True) for task, (status, run_id) in zip(filtered, live, strict=True)}
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
            "or what a `waiting` task asked. "
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
        List of `StructuredTools` for launch, check, update, cancel, list, and resume operations.
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
        _build_resume_tool(agent_map, clients, backend),
    ]


class AsyncSubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for async subagents running on Agent Protocol servers or on this agent's own graph.

    This middleware adds tools for launching, monitoring, updating, and
    resuming background tasks on remote Agent Protocol servers. Unlike the synchronous
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
            optional — omit it to use ASGI transport for local servers. A
            spec with a compiled `runnable` (as `create_deep_agent` builds for
            `graph_id="self"`) runs on this agent's own graph.
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
        self._workers: dict[str, Runnable] = {a["name"]: a["runnable"] for a in async_subagents if _is_worker(a)}

        if system_prompt:
            agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in async_subagents)
            self.system_prompt: str | None = system_prompt + "\n\nAvailable async subagent types:\n\n" + agents_desc
        else:
            self.system_prompt = system_prompt

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AsyncSubAgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """Run an inline async subagent instead of this agent when a task asked for one."""
        worker = self._requested_worker()
        if worker is None:
            return None
        result = worker.invoke({"messages": state["messages"]})
        return {"messages": result["messages"], "jump_to": "end"}

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state: AsyncSubAgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # signature set by AgentMiddleware
        """(async) Run an inline async subagent instead of this agent when a task asked for one."""
        worker = self._requested_worker()
        if worker is None:
            return None
        result = await worker.ainvoke({"messages": state["messages"]})
        return {"messages": result["messages"], "jump_to": "end"}

    def _requested_worker(self) -> Runnable | None:
        """Return the inline subagent this run should execute as, or `None` for a normal run.

        Raises:
            ValueError: If the run names a subagent this agent doesn't define inline.
        """
        name = (get_config().get("configurable") or {}).get(_WORKER_KEY)
        if name is None:
            return None
        if name not in self._workers:
            msg = f"This agent has no inline async subagent named {name!r}"
            raise ValueError(msg)
        return self._workers[name]

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
