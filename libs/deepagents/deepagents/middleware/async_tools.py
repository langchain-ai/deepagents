"""Background tools that run as separately addressable graphs on an Agent Server.

A background tool is an ordinary Python function wrapped with `@background_tool`.
The decorator compiles the function into a functional-API graph (an
`@entrypoint`), so the same object can be registered in `langgraph.json` as its
own graph and passed to `create_deep_agent(tools=[...])`.

When an agent is given a background tool, `AsyncToolMiddleware` exposes a
launcher tool with the function's name and schema. Calling it starts a run of
the tool's graph on the Agent Server and returns a task ID immediately; the
agent keeps working while the tool runs. The tracking surface mirrors the
background tasks in Claude Code (`run_in_background`, `TaskOutput` with
`block`/`timeout`, `TaskStop`, and completion reminders) and Codex's unified
exec polling (`write_stdin` with a yield window):

- `check_background_task(task_id, block=False, timeout=30)`: read a task's
    status and result, optionally waiting for it to finish.
- `cancel_background_task(task_id)`: stop a running task.
- `list_background_tasks(status_filter)`: list tracked tasks with live statuses.
- Before each model call, newly finished tasks are announced to the model in a
    `<system-reminder>` message so it doesn't need to poll.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, TypedDict, overload

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT
from langchain.tools import ToolRuntime  # noqa: TC002  # tool runtime injection resolves this annotation at runtime
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.func import entrypoint
from langgraph.pregel import Pregel
from langgraph.types import Command
from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from deepagents.middleware.async_subagents import AsyncSubAgent, _ClientCache, _run_options, _thread_metadata

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.runtime import Runtime
    from langgraph.store.base import BaseStore
    from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
    from langgraph_sdk.schema import Run

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"success", "error", "timeout", "interrupted", "cancelled"})
"""Run statuses that will never change."""

_POLL_INTERVAL_SECONDS = 1.0
"""How often `check_background_task(block=True)` polls the run status."""

_MAX_BLOCK_SECONDS = 600.0
"""Upper bound on how long a single blocking check may wait."""


class BackgroundTool(Pregel):
    """A functional-API graph that an agent can call as a non-blocking tool.

    Create one with [`background_tool`][deepagents.middleware.async_tools.background_tool].
    The object is a regular `Pregel` graph, so it can be registered in
    `langgraph.json` and invoked directly. Its graph input is a dict of the
    wrapped function's arguments, and its output is the function's return value.

    !!! warning "Experimental"

        Background tools are experimental and may change in a future release.
    """

    def __init__(
        self,
        *,
        tool: StructuredTool,
        graph_id: str,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,  # forwarded verbatim to `Pregel.__init__`, including by `Pregel.copy`
    ) -> None:
        """Initialize the graph and attach the tool metadata the launcher needs."""
        super().__init__(**kwargs)
        self.tool = tool
        self.graph_id = graph_id
        self.url = url
        self.headers = headers

    def as_spec(self) -> AsyncSubAgent:
        """Describe where this tool's graph is deployed, in the shape the SDK client cache expects."""
        spec = AsyncSubAgent(name=self.tool.name, description=self.tool.description, graph_id=self.graph_id)
        if self.url is not None:
            spec["url"] = self.url
        if self.headers is not None:
            spec["headers"] = self.headers
        return spec


def _graph_function(tool: StructuredTool, func: Callable[..., Any]) -> Callable[[dict[str, Any]], Any]:
    """Build the entrypoint body that validates the graph input and calls `func`."""
    if inspect.iscoroutinefunction(func):

        async def arun(inputs: dict[str, Any]) -> Any:  # noqa: ANN401  # returns whatever the wrapped tool returns
            return await tool.ainvoke(inputs)

        arun.__name__ = tool.name
        return arun

    def run(inputs: dict[str, Any]) -> Any:  # noqa: ANN401  # returns whatever the wrapped tool returns
        return tool.invoke(inputs)

    run.__name__ = tool.name
    return run


def _build_background_tool(
    func: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    graph_id: str | None,
    url: str | None,
    headers: dict[str, str] | None,
    checkpointer: BaseCheckpointSaver | None,
    store: BaseStore | None,
) -> BackgroundTool:
    """Compile `func` into a `BackgroundTool` graph."""
    kwargs: dict[str, Any] = {"func": func} if not inspect.iscoroutinefunction(func) else {"coroutine": func}
    tool = StructuredTool.from_function(name=name or getattr(func, "__name__", type(func).__name__), description=description, **kwargs)
    graph = entrypoint(checkpointer=checkpointer, store=store)(_graph_function(tool, func))
    attrs = {key: value for key, value in graph.__dict__.items() if key != "__orig_class__"}
    attrs["name"] = tool.name
    return BackgroundTool(tool=tool, graph_id=graph_id or tool.name, url=url, headers=headers, **attrs)


@overload
def background_tool(func: Callable[..., Any], /) -> BackgroundTool: ...


@overload
def background_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    graph_id: str | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> Callable[[Callable[..., Any]], BackgroundTool]: ...


def background_tool(
    func: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    graph_id: str | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> BackgroundTool | Callable[[Callable[..., Any]], BackgroundTool]:
    """Turn a function into a tool that runs in the background on an Agent Server.

    The function is compiled with the functional API into a graph, so its body
    can use `@task`, `interrupt`, and the other functional-API primitives. Register
    the returned object as its own graph in `langgraph.json` (keyed by `graph_id`)
    and pass it to `create_deep_agent(tools=[...])`. The agent then gets a tool
    with the function's name and arguments that starts a run and returns a task
    ID immediately, plus tools to check, cancel, and list running tasks.

    !!! warning "Experimental"

        Background tools are experimental and may change in a future release.

    Args:
        func: The function to wrap. Its docstring becomes the tool description
            and its signature becomes the tool's argument schema.
        name: Tool name shown to the model. Defaults to the function name.
        description: Tool description. Defaults to the function docstring.
        graph_id: The graph name (or assistant ID) the tool is registered
            under on the Agent Server. Defaults to the tool name.
        url: URL of the Agent Server running the tool's graph. Omit to use
            the in-process ASGI transport of the deployment the agent runs on,
            which requires async invocation of the agent.
        headers: Extra headers for requests to the Agent Server.
        checkpointer: Checkpointer for running the graph outside an Agent Server.
        store: Store for running the graph outside an Agent Server.

    Returns:
        A `BackgroundTool` graph, or a decorator that builds one.

    Example:
        ```python
        from deepagents import background_tool, create_deep_agent


        @background_tool
        def crawl_site(url: str) -> str:
            \"\"\"Crawl every page of a website and summarize it.\"\"\"
            ...


        agent = create_deep_agent(tools=[crawl_site])
        ```

        ```json
        {"graphs": {"agent": "./agent.py:agent", "crawl_site": "./agent.py:crawl_site"}}
        ```
    """
    options: dict[str, Any] = {
        "name": name,
        "description": description,
        "graph_id": graph_id,
        "url": url,
        "headers": headers,
        "checkpointer": checkpointer,
        "store": store,
    }
    if func is not None:
        return _build_background_tool(func, **options)
    return lambda f: _build_background_tool(f, **options)


class BackgroundTask(TypedDict):
    """A background tool run tracked in agent state."""

    task_id: str
    """Unique identifier for the task (same as `thread_id`)."""

    tool_name: str
    """Name of the background tool that is running."""

    thread_id: str
    """Thread ID on the Agent Server."""

    run_id: str
    """Run ID of the tool's run on the thread."""

    status: str
    """Last observed run status (e.g. `'pending'`, `'running'`, `'success'`, `'error'`, `'cancelled'`)."""

    created_at: str
    """ISO-8601 timestamp (UTC) when the task was started."""

    finished_at: NotRequired[str]
    """ISO-8601 timestamp (UTC) when the task was first observed in a terminal status."""


def _background_tasks_reducer(
    existing: dict[str, BackgroundTask] | None,
    update: dict[str, BackgroundTask],
) -> dict[str, BackgroundTask]:
    """Merge task updates into the existing tasks dict."""
    return {**(existing or {}), **update}


class AsyncToolState(AgentState):
    """State extension for background tool task tracking."""

    background_tasks: Annotated[NotRequired[dict[str, BackgroundTask]], _background_tasks_reducer]


class CheckBackgroundTaskSchema(BaseModel):
    """Input schema for the `check_background_task` tool."""

    task_id: str = Field(description="The exact task_id returned when the background tool was started.")
    block: bool = Field(default=False, description="Wait for the task to finish before returning.")
    timeout: float = Field(default=30, ge=0, le=_MAX_BLOCK_SECONDS, description="Maximum seconds to wait when `block` is true.")


class CancelBackgroundTaskSchema(BaseModel):
    """Input schema for the `cancel_background_task` tool."""

    task_id: str = Field(description="The exact task_id returned when the background tool was started.")


class ListBackgroundTasksSchema(BaseModel):
    """Input schema for the `list_background_tasks` tool."""

    status_filter: Literal["running", "success", "error", "cancelled", "all"] = Field(
        default="all",
        description="Only list tasks whose last known status matches.",
    )


_LAUNCH_NOTE = (
    "\n\nThis tool runs in the background: it returns a task_id immediately while the work continues. "
    "Keep working; you'll be notified when it finishes. Use `check_background_task` to read its result "
    "(pass `block=true` to wait for it) and `cancel_background_task` to stop it."
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(task: BackgroundTask, run: Run) -> BackgroundTask:
    """Return `task` updated with the run's latest status."""
    updated = BackgroundTask(**task)
    updated["status"] = run["status"]
    if run["status"] in _TERMINAL_STATUSES and "finished_at" not in task:
        updated["finished_at"] = _now()
    return updated


def _resolve_task(task_id: str, runtime: ToolRuntime) -> BackgroundTask | str:
    """Look up a tracked task by ID, or return an error string for the model."""
    tracked = (runtime.state.get("background_tasks") or {}).get(task_id.strip())
    return tracked or f"No tracked background task found for task_id: {task_id!r}"


def _stringify(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(to_jsonable_python(value, fallback=str))


def _build_result(task: BackgroundTask, values: object) -> dict[str, Any]:
    """Describe a task's status, adding its output once it has finished."""
    result: dict[str, Any] = {"task_id": task["task_id"], "tool": task["tool_name"], "status": task["status"]}
    if task["status"] == "success":
        result["result"] = _stringify(values)
    elif task["status"] in {"error", "timeout"}:
        result["error"] = f"The background tool run ended with status {task['status']!r}."
    return result


def _task_command(task: BackgroundTask, content: str, tool_call_id: str | None) -> Command:
    return Command(
        update={
            "messages": [ToolMessage(content, tool_call_id=tool_call_id)],
            "background_tasks": {task["task_id"]: task},
        }
    )


def _new_task(name: str, thread_id: str, run: Run) -> BackgroundTask:
    return BackgroundTask(
        task_id=thread_id,
        tool_name=name,
        thread_id=thread_id,
        run_id=run["run_id"],
        status=run["status"],
        created_at=_now(),
    )


def _launch_message(task: BackgroundTask) -> str:
    return f"Started background tool `{task['tool_name']}`. task_id: {task['task_id']}"


def _build_launcher(spec: BackgroundTool, clients: _ClientCache) -> StructuredTool:
    """Build the model-facing tool that starts a run of `spec`'s graph."""
    name = spec.tool.name
    remote = spec.as_spec()

    def launch(runtime: ToolRuntime, **kwargs: Any) -> str | Command:  # tool args are validated by `args_schema`
        try:
            client = clients.get_sync(name)
            thread = client.threads.create(metadata=_thread_metadata(runtime))
            run = client.runs.create(thread["thread_id"], spec.graph_id, input=to_jsonable_python(kwargs), **_run_options(remote, runtime))
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to start background tool '%s': %s", name, e)
            return f"Failed to start background tool '{name}': {e}"
        task = _new_task(name, thread["thread_id"], run)
        return _task_command(task, _launch_message(task), runtime.tool_call_id)

    async def alaunch(runtime: ToolRuntime, **kwargs: Any) -> str | Command:  # tool args are validated by `args_schema`
        try:
            client = clients.get_async(name)
            thread = await client.threads.create(metadata=_thread_metadata(runtime))
            run = await client.runs.create(thread["thread_id"], spec.graph_id, input=to_jsonable_python(kwargs), **_run_options(remote, runtime))
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            logger.warning("Failed to start background tool '%s': %s", name, e)
            return f"Failed to start background tool '{name}': {e}"
        task = _new_task(name, thread["thread_id"], run)
        return _task_command(task, _launch_message(task), runtime.tool_call_id)

    return StructuredTool.from_function(
        name=name,
        func=launch,
        coroutine=alaunch,
        description=(spec.tool.description or "") + _LAUNCH_NOTE,
        args_schema=spec.tool.args_schema,
        infer_schema=False,
    )


def _poll(client: SyncLangGraphClient, task: BackgroundTask, *, block: bool, timeout: float) -> Run:
    """Fetch the task's run, waiting up to `timeout` seconds for it to finish when `block` is set."""
    deadline = time.monotonic() + timeout
    run = client.runs.get(task["thread_id"], task["run_id"])
    while block and run["status"] not in _TERMINAL_STATUSES and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)
        run = client.runs.get(task["thread_id"], task["run_id"])
    return run


async def _apoll(client: LangGraphClient, task: BackgroundTask, *, block: bool, wait: float) -> Run:
    """Async version of `_poll`."""
    deadline = time.monotonic() + wait
    run = await client.runs.get(task["thread_id"], task["run_id"])
    while block and run["status"] not in _TERMINAL_STATUSES and time.monotonic() < deadline:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        run = await client.runs.get(task["thread_id"], task["run_id"])
    return run


def _build_check_tool(clients: _ClientCache) -> StructuredTool:
    """Build the `check_background_task` tool."""

    def check_background_task(task_id: str, runtime: ToolRuntime, block: bool = False, timeout: float = 30) -> str | Command:  # noqa: FBT001, FBT002  # model-facing tool argument
        task = _resolve_task(task_id, runtime)
        if isinstance(task, str):
            return task
        try:
            client = clients.get_sync(task["tool_name"])
            task = _record(task, _poll(client, task, block=block, timeout=timeout))
            values = client.threads.get(task["thread_id"]).get("values") if task["status"] == "success" else None
        except Exception as e:  # noqa: BLE001  # get_sync() may raise ValueError; SDK raises untyped errors
            return f"Failed to check background task: {e}"
        return _task_command(task, json.dumps(_build_result(task, values)), runtime.tool_call_id)

    async def acheck_background_task(task_id: str, runtime: ToolRuntime, block: bool = False, timeout: float = 30) -> str | Command:  # noqa: FBT001, FBT002, ASYNC109  # model-facing tool arguments
        task = _resolve_task(task_id, runtime)
        if isinstance(task, str):
            return task
        try:
            client = clients.get_async(task["tool_name"])
            task = _record(task, await _apoll(client, task, block=block, wait=timeout))
            values = (await client.threads.get(task["thread_id"])).get("values") if task["status"] == "success" else None
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to check background task: {e}"
        return _task_command(task, json.dumps(_build_result(task, values)), runtime.tool_call_id)

    return StructuredTool.from_function(
        name="check_background_task",
        func=check_background_task,
        coroutine=acheck_background_task,
        description=(
            "Check a background tool task. Returns its status and, once finished, its result. "
            "Set `block` to wait up to `timeout` seconds for it to finish. "
            "Statuses shown earlier in the conversation may be stale."
        ),
        args_schema=CheckBackgroundTaskSchema,
        infer_schema=False,
    )


def _cancelled(task: BackgroundTask) -> BackgroundTask:
    return BackgroundTask(**{**task, "status": "cancelled", "finished_at": task.get("finished_at", _now())})


def _build_cancel_tool(clients: _ClientCache) -> StructuredTool:
    """Build the `cancel_background_task` tool."""

    def cancel_background_task(task_id: str, runtime: ToolRuntime) -> str | Command:
        task = _resolve_task(task_id, runtime)
        if isinstance(task, str):
            return task
        try:
            clients.get_sync(task["tool_name"]).runs.cancel(task["thread_id"], task["run_id"])
        except Exception as e:  # noqa: BLE001  # get_sync() may raise ValueError; SDK raises untyped errors
            return f"Failed to cancel background task: {e}"
        return _task_command(_cancelled(task), f"Cancelled background task: {task['task_id']}", runtime.tool_call_id)

    async def acancel_background_task(task_id: str, runtime: ToolRuntime) -> str | Command:
        task = _resolve_task(task_id, runtime)
        if isinstance(task, str):
            return task
        try:
            await clients.get_async(task["tool_name"]).runs.cancel(task["thread_id"], task["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK raises untyped errors
            return f"Failed to cancel background task: {e}"
        return _task_command(_cancelled(task), f"Cancelled background task: {task['task_id']}", runtime.tool_call_id)

    return StructuredTool.from_function(
        name="cancel_background_task",
        func=cancel_background_task,
        coroutine=acancel_background_task,
        description="Cancel a running background tool task that is no longer needed.",
        args_schema=CancelBackgroundTaskSchema,
        infer_schema=False,
    )


def _refresh(clients: _ClientCache, tasks: list[BackgroundTask]) -> list[BackgroundTask]:
    """Fetch live statuses for unfinished tasks, keeping cached ones on error."""
    refreshed = []
    for task in tasks:
        if task["status"] in _TERMINAL_STATUSES:
            refreshed.append(task)
            continue
        try:
            refreshed.append(_record(task, clients.get_sync(task["tool_name"]).runs.get(task["thread_id"], task["run_id"])))
        except Exception:  # noqa: BLE001  # get_sync() may raise ValueError; SDK raises untyped errors
            logger.warning("Failed to refresh background task %s", task["task_id"], exc_info=True)
            refreshed.append(task)
    return refreshed


async def _arefresh_one(clients: _ClientCache, task: BackgroundTask) -> BackgroundTask:
    if task["status"] in _TERMINAL_STATUSES:
        return task
    try:
        return _record(task, await clients.get_async(task["tool_name"]).runs.get(task["thread_id"], task["run_id"]))
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        logger.warning("Failed to refresh background task %s", task["task_id"], exc_info=True)
        return task


async def _arefresh(clients: _ClientCache, tasks: list[BackgroundTask]) -> list[BackgroundTask]:
    """Async version of `_refresh`, fetching statuses concurrently."""
    return list(await asyncio.gather(*(_arefresh_one(clients, task) for task in tasks)))


def _filter(tasks: dict[str, BackgroundTask], status_filter: str) -> list[BackgroundTask]:
    if status_filter == "all":
        return list(tasks.values())
    return [task for task in tasks.values() if task["status"] == status_filter]


def _list_command(tasks: list[BackgroundTask], tool_call_id: str | None) -> Command:
    lines = [f"- task_id: {t['task_id']}  tool: {t['tool_name']}  status: {t['status']}" for t in tasks]
    return Command(
        update={
            "messages": [ToolMessage(f"{len(lines)} background task(s):\n" + "\n".join(lines), tool_call_id=tool_call_id)],
            "background_tasks": {task["task_id"]: task for task in tasks},
        }
    )


def _build_list_tool(clients: _ClientCache) -> StructuredTool:
    """Build the `list_background_tasks` tool."""

    def list_background_tasks(runtime: ToolRuntime, status_filter: str = "all") -> str | Command:
        tasks = _filter(runtime.state.get("background_tasks") or {}, status_filter)
        if not tasks:
            return "No background tasks tracked."
        return _list_command(_refresh(clients, tasks), runtime.tool_call_id)

    async def alist_background_tasks(runtime: ToolRuntime, status_filter: str = "all") -> str | Command:
        tasks = _filter(runtime.state.get("background_tasks") or {}, status_filter)
        if not tasks:
            return "No background tasks tracked."
        return _list_command(await _arefresh(clients, tasks), runtime.tool_call_id)

    return StructuredTool.from_function(
        name="list_background_tasks",
        func=list_background_tasks,
        coroutine=alist_background_tasks,
        description="List tracked background tool tasks with their live statuses.",
        args_schema=ListBackgroundTasksSchema,
        infer_schema=False,
    )


def _finished_reminder(tasks: list[BackgroundTask]) -> HumanMessage:
    """Tell the model which background tasks finished since it last looked."""
    lines = [f"- `{t['tool_name']}` task {t['task_id']} finished with status: {t['status']}" for t in tasks]
    body = "\n".join(["Background tasks finished:", *lines, "Use `check_background_task` to read their results."])
    return HumanMessage(f"<system-reminder>\n{body}\n</system-reminder>")


def _notification(before: list[BackgroundTask], after: list[BackgroundTask]) -> dict[str, Any] | None:
    """Build the state update for tasks that finished between `before` and `after`."""
    finished = [new for old, new in zip(before, after, strict=True) if new["status"] in _TERMINAL_STATUSES and new != old]
    if not finished:
        return None
    return {"messages": [_finished_reminder(finished)], "background_tasks": {task["task_id"]: task for task in finished}}


def is_background_tool(tool: object) -> bool:
    """Whether `tool` was created with `background_tool`."""
    return isinstance(tool, BackgroundTool)


def split_background_tools(
    tools: Sequence[BaseTool | Callable | dict[str, Any] | BackgroundTool] | None,
) -> tuple[list[BaseTool | Callable | dict[str, Any]] | None, list[BackgroundTool]]:
    """Separate background tools from regular tools in a `tools` argument."""
    if tools is None:
        return None, []
    background = [tool for tool in tools if isinstance(tool, BackgroundTool)]
    regular = [tool for tool in tools if not isinstance(tool, BackgroundTool)]
    return regular, background


class AsyncToolMiddleware(AgentMiddleware[AsyncToolState, ContextT, ResponseT]):
    """Expose background tools to an agent and track their runs.

    For each [`BackgroundTool`][deepagents.middleware.async_tools.BackgroundTool],
    the agent gets a launcher tool with the same name and arguments that starts a
    run of the tool's graph on an Agent Server and returns a task ID. The agent
    also gets `check_background_task`, `cancel_background_task`, and
    `list_background_tasks`. Before each model call, the middleware polls
    unfinished tasks and adds a `<system-reminder>` message for any that finished.

    `create_deep_agent` adds this middleware automatically when `tools` contains
    a background tool.

    !!! warning "Experimental"

        Background tools are experimental and may change in a future release.

    Args:
        tools: Background tools created with `background_tool`.
    """

    state_schema = AsyncToolState

    def __init__(self, *, tools: Sequence[BackgroundTool]) -> None:
        """Initialize the `AsyncToolMiddleware`."""
        super().__init__()
        if not tools:
            msg = "At least one background tool must be specified"
            raise ValueError(msg)
        names = [tool.tool.name for tool in tools]
        if dupes := {name for name in names if names.count(name) > 1}:
            msg = f"Duplicate background tool names: {dupes}"
            raise ValueError(msg)
        self._clients = _ClientCache({tool.tool.name: tool.as_spec() for tool in tools})
        self.tools = [
            *(_build_launcher(tool, self._clients) for tool in tools),
            _build_check_tool(self._clients),
            _build_cancel_tool(self._clients),
            _build_list_tool(self._clients),
        ]

    def before_model(self, state: AsyncToolState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # hook signature
        """Announce background tasks that finished since the last model call."""
        pending = [task for task in (state.get("background_tasks") or {}).values() if task["status"] not in _TERMINAL_STATUSES]
        return _notification(pending, _refresh(self._clients, pending)) if pending else None

    async def abefore_model(self, state: AsyncToolState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002  # hook signature
        """(async) Announce background tasks that finished since the last model call."""
        pending = [task for task in (state.get("background_tasks") or {}).values() if task["status"] not in _TERMINAL_STATUSES]
        return _notification(pending, await _arefresh(self._clients, pending)) if pending else None


__all__ = [
    "AsyncToolMiddleware",
    "AsyncToolState",
    "BackgroundTask",
    "BackgroundTool",
    "background_tool",
]
