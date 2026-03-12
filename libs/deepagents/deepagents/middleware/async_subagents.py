"""Middleware for async subagents running on remote LangGraph servers.

Async subagents use the LangGraph SDK to launch background runs on remote
LangGraph deployments. Unlike synchronous subagents (which block until
completion), async subagents return a job ID immediately, allowing the main
agent to monitor progress and send updates while the subagent works.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelResponse, ResponseT
from langchain_core.tools import StructuredTool
from langgraph_sdk import get_client, get_sync_client

from deepagents.middleware._utils import append_to_system_message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest
    from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient


class AsyncSubAgent(TypedDict):
    """Specification for an async subagent running on a remote LangGraph server.

    Async subagents connect to LangGraph deployments via the LangGraph SDK.
    They run as background jobs that the main agent can monitor and update.

    Authentication is handled via environment variables (`LANGGRAPH_API_KEY`,
    `LANGSMITH_API_KEY`, or `LANGCHAIN_API_KEY`), which the LangGraph SDK
    reads automatically.

    Required fields:
        name: Unique identifier for the async subagent.
        description: What this subagent does. The main agent uses this to decide
            when to delegate.
        url: URL of the LangGraph server (e.g., `"https://my-deployment.langsmith.dev"`).
        graph_id: The graph name or assistant ID on the remote server.

    Optional fields:
        headers: Additional headers to include in requests to the remote server.
    """

    name: str
    description: str
    url: str
    graph_id: str
    headers: NotRequired[dict[str, str]]


ASYNC_TASK_TOOL_DESCRIPTION = """Launch an async subagent on a remote LangGraph server. The subagent runs in the background and returns a job ID immediately.

Available async agent types:
{available_agents}

## Usage notes:
1. This tool launches a background job and returns immediately with a job ID (thread_id + run_id).
2. Use `check_async_subagent` to poll for status and results.
3. Use `update_async_subagent` to send updates or new instructions to a running job.
4. Multiple async subagents can run concurrently — launch several and check them periodically.
5. The subagent runs on a remote LangGraph server, so it has its own tools and capabilities."""  # noqa: E501

ASYNC_TASK_SYSTEM_PROMPT = """## Async subagents (remote LangGraph servers)

You have access to async subagent tools that launch background jobs on remote LangGraph servers.

### Tools:
- `launch_async_subagent`: Start a new background job. Returns a job ID immediately.
- `check_async_subagent`: Check the status of a running job. Returns status and result if complete.
- `update_async_subagent`: Send an update or new instructions to a running job.

### Workflow:
1. **Launch** — Use `launch_async_subagent` to start a job. You get back a job ID.
2. **Monitor** — Use `check_async_subagent` to poll for status. Jobs can be: pending, running, success, error, timeout, or interrupted.
3. **Update** (optional) — Use `update_async_subagent` to send new context or instructions to a running job.
4. **Collect** — When status is "success", the result is included in the check response.

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
    """Lazily-created, cached LangGraph SDK clients keyed by agent name."""

    def __init__(self, agents: dict[str, AsyncSubAgent]) -> None:
        self._agents = agents
        self._sync: dict[str, SyncLangGraphClient] = {}
        self._async: dict[str, LangGraphClient] = {}

    def sync(self, name: str) -> SyncLangGraphClient:
        """Get or create a sync client for the named agent."""
        if name not in self._sync:
            spec = self._agents[name]
            self._sync[name] = get_sync_client(
                url=spec["url"],
                headers=_resolve_headers(spec),
            )
        return self._sync[name]

    def async_(self, name: str) -> LangGraphClient:
        """Get or create an async client for the named agent."""
        if name not in self._async:
            spec = self._agents[name]
            self._async[name] = get_client(
                url=spec["url"],
                headers=_resolve_headers(spec),
            )
        return self._async[name]


_JOB_ID_SEP = "::"


def _format_job_id(agent_name: str, thread_id: str, run_id: str) -> str:
    return f"{agent_name}{_JOB_ID_SEP}{thread_id}{_JOB_ID_SEP}{run_id}"


_JOB_ID_PREFIX = "job_id: "


def _parse_job_id(job_id: str) -> tuple[str, str, str]:
    """Parse a job ID into (agent_name, thread_id, run_id).

    Handles multiple formats the LLM might pass:
    - `agent::thread_abc::run_xyz` (canonical, 3-part)
    - `Launched async subagent. job_id: agent::thread_abc::run_xyz` (full launch output)
    - `thread_abc::run_xyz` (legacy 2-part, agent defaults to empty)
    - `{"thread_id": "...", "run_id": "..."}` (legacy JSON)
    """
    raw = job_id.strip()
    idx = raw.find(_JOB_ID_PREFIX)
    if idx != -1:
        raw = raw[idx + len(_JOB_ID_PREFIX) :].strip()
    if _JOB_ID_SEP in raw:
        parts = raw.split(_JOB_ID_SEP)
        if len(parts) >= 3:  # noqa: PLR2004
            return parts[0], parts[1], parts[2]
        if len(parts) == 2:  # noqa: PLR2004
            return "", parts[0], parts[1]
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return "", data["thread_id"], data["run_id"]
    except (json.JSONDecodeError, KeyError):
        pass
    msg = f"Invalid job_id format: {job_id!r}. Expected the exact job_id string returned by launch_async_subagent."
    raise ValueError(msg)


def _validate_agent_type(agent_map: dict[str, AsyncSubAgent], agent_type: str) -> str | None:
    if agent_type not in agent_map:
        allowed = ", ".join(f"`{k}`" for k in agent_map)
        return f"Unknown async subagent type `{agent_type}`. Available types: {allowed}"
    return None


def _build_launch_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    description: str,
) -> StructuredTool:
    """Build the launch_async_subagent tool."""

    def launch_async_subagent(
        description: Annotated[str, "A detailed description of the task for the async subagent to perform."],
        subagent_type: Annotated[str, "The type of async subagent to use. Must be one of the available types."],
    ) -> str:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        client = clients.sync(subagent_type)
        thread = client.threads.create()
        run = client.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=spec["graph_id"],
            input={"messages": [{"role": "user", "content": description}]},
        )
        job_id = _format_job_id(subagent_type, thread["thread_id"], run["run_id"])
        return f"Launched async subagent. job_id: {job_id}"

    async def alaunch_async_subagent(
        description: Annotated[str, "A detailed description of the task for the async subagent to perform."],
        subagent_type: Annotated[str, "The type of async subagent to use. Must be one of the available types."],
    ) -> str:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        client = clients.async_(subagent_type)
        thread = await client.threads.create()
        run = await client.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=spec["graph_id"],
            input={"messages": [{"role": "user", "content": description}]},
        )
        job_id = _format_job_id(subagent_type, thread["thread_id"], run["run_id"])
        return f"Launched async subagent. job_id: {job_id}"

    return StructuredTool.from_function(
        name="launch_async_subagent",
        func=launch_async_subagent,
        coroutine=alaunch_async_subagent,
        description=description,
    )


def _resolve_client_name(agent_name: str, agent_map: dict[str, AsyncSubAgent]) -> str:
    """Resolve the client name from a parsed job_id agent_name field."""
    if agent_name and agent_name in agent_map:
        return agent_name
    return next(iter(agent_map))


def _build_check_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
) -> StructuredTool:
    """Build the check_async_subagent tool."""

    def check_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent. Pass it verbatim."],
    ) -> str:
        agent_name, thread_id, run_id = _parse_job_id(job_id)
        client = clients.sync(_resolve_client_name(agent_name, agent_map))
        run = client.runs.get(thread_id=thread_id, run_id=run_id)
        result: dict[str, Any] = {"status": run["status"], "run_id": run["run_id"], "thread_id": thread_id}
        if run["status"] == "success":
            state = client.threads.get_state(thread_id=thread_id)
            values = state.get("values", {})
            messages = values.get("messages", []) if isinstance(values, dict) else []
            if messages:
                last = messages[-1]
                result["result"] = last.get("content", "") if isinstance(last, dict) else str(last)
        elif run["status"] == "error":
            result["error"] = "The async subagent encountered an error."
        return json.dumps(result)

    async def acheck_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent. Pass it verbatim."],
    ) -> str:
        agent_name, thread_id, run_id = _parse_job_id(job_id)
        client = clients.async_(_resolve_client_name(agent_name, agent_map))
        run = await client.runs.get(thread_id=thread_id, run_id=run_id)
        result: dict[str, Any] = {"status": run["status"], "run_id": run["run_id"], "thread_id": thread_id}
        if run["status"] == "success":
            state = await client.threads.get_state(thread_id=thread_id)
            values = state.get("values", {})
            messages = values.get("messages", []) if isinstance(values, dict) else []
            if messages:
                last = messages[-1]
                result["result"] = last.get("content", "") if isinstance(last, dict) else str(last)
        elif run["status"] == "error":
            result["error"] = "The async subagent encountered an error."
        return json.dumps(result)

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
    """Build the update_async_subagent tool.

    Sends a follow-up message to an async subagent by creating a new run
    on the same thread. The subagent sees the full conversation history
    (including the original task and any prior results) plus the new message.
    Returns a new job_id for the follow-up run.
    """

    def update_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent or a previous update. Pass it verbatim."],
        message: Annotated[str, "Follow-up instructions or context to send to the subagent."],
    ) -> str:
        agent_name, thread_id, _run_id = _parse_job_id(job_id)
        name = _resolve_client_name(agent_name, agent_map)
        spec = agent_map[name]
        client = clients.sync(name)
        run = client.runs.create(
            thread_id=thread_id,
            assistant_id=spec["graph_id"],
            input={"messages": [{"role": "user", "content": message}]},
        )
        new_job_id = _format_job_id(name, thread_id, run["run_id"])
        return f"Follow-up sent. New job_id: {new_job_id}"

    async def aupdate_async_subagent(
        job_id: Annotated[str, "The exact job_id string returned by launch_async_subagent or a previous update. Pass it verbatim."],
        message: Annotated[str, "Follow-up instructions or context to send to the subagent."],
    ) -> str:
        agent_name, thread_id, _run_id = _parse_job_id(job_id)
        name = _resolve_client_name(agent_name, agent_map)
        spec = agent_map[name]
        client = clients.async_(name)
        run = await client.runs.create(
            thread_id=thread_id,
            assistant_id=spec["graph_id"],
            input={"messages": [{"role": "user", "content": message}]},
        )
        new_job_id = _format_job_id(name, thread_id, run["run_id"])
        return f"Follow-up sent. New job_id: {new_job_id}"

    return StructuredTool.from_function(
        name="update_async_subagent",
        func=update_async_subagent,
        coroutine=aupdate_async_subagent,
        description=(
            "Send a follow-up message to an async subagent. Creates a new run on the same thread "
            "so the subagent sees the full conversation history plus your new message. "
            "Returns a new job_id to track the follow-up."
        ),
    )


def _build_async_subagent_tools(
    agents: list[AsyncSubAgent],
) -> list[StructuredTool]:
    """Build the three async subagent tools from agent specs.

    Args:
        agents: List of async subagent specifications.

    Returns:
        List of StructuredTools for launch, check, and update operations.
    """
    agent_map: dict[str, AsyncSubAgent] = {a["name"]: a for a in agents}
    clients = _ClientCache(agent_map)
    agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in agents)
    launch_desc = ASYNC_TASK_TOOL_DESCRIPTION.format(available_agents=agents_desc)

    return [
        _build_launch_tool(agent_map, clients, launch_desc),
        _build_check_tool(agent_map, clients),
        _build_update_tool(agent_map, clients),
    ]


class AsyncSubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for async subagents running on remote LangGraph servers.

    This middleware adds tools for launching, monitoring, and updating
    background jobs on remote LangGraph deployments. Unlike the synchronous
    `SubAgentMiddleware`, async subagents return immediately with a job ID,
    allowing the main agent to continue working while subagents execute.

    Args:
        async_subagents: List of async subagent specifications. Each must
            include `name`, `description`, `url`, and `graph_id`.
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

        self.tools = _build_async_subagent_tools(async_subagents)

        if system_prompt and async_subagents:
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
