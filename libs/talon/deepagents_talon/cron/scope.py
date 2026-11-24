"""Read-only tool scope for scoped scheduled runs.

Talon is an experimental runtime and is subject to change or removal at any time.

A job with read grants can access the sources its owner specified. The
scope is enforced at the tool boundary, not in the prompt: tools outside it are
hidden from the model, and any call that still names one is refused.
"""

from __future__ import annotations

import contextvars
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest

TOOL_PREFIX = "tool:"
PATH_PREFIX = "path:"
FILE_READ_TOOLS: Mapping[str, str] = {
    "read_file": "file_path",
    "ls": "path",
    "glob": "path",
    "grep": "path",
}
"""Filesystem read tools and the argument that names the path they read."""

ALWAYS_ALLOWED = frozenset({"current_time", "ask_user", "record_job_work"})


@dataclass(frozen=True, slots=True)
class ReadScope:
    """Tools and paths an unattended job run may read.

    Args:
        tools: Tool names the owner named; each must also have a known read-only contract.
        paths: Absolute directories the filesystem read tools may address.
        job_id: Job this job may update or finish.
    """

    tools: frozenset[str] = frozenset()
    paths: tuple[str, ...] = ()
    job_id: str | None = None

    @classmethod
    def from_sources(cls, sources: Iterable[str], *, job_id: str | None = None) -> ReadScope:
        """Derive the job's read scope from its source entries.

        `tool:<name>` names a tool and `path:<absolute dir>` names a directory.
        Other entries grant no tool access.

        Args:
            sources: Read sources confirmed by the owner.
            job_id: The job whose progress this turn may update.

        Returns:
            The read scope.
        """
        tools = {s.removeprefix(TOOL_PREFIX).strip() for s in sources if s.startswith(TOOL_PREFIX)}
        paths = [
            posixpath.normpath(s.removeprefix(PATH_PREFIX).strip())
            for s in sources
            if s.startswith(PATH_PREFIX) and s.removeprefix(PATH_PREFIX).strip().startswith("/")
        ]
        return cls(
            tools=frozenset(filter(None, tools)), paths=tuple(dict.fromkeys(paths)), job_id=job_id
        )

    def allows_tool(self, tool: object) -> bool:
        """Whether a tool may be offered to and called by the job run."""
        name = getattr(tool, "name", None)
        if not isinstance(tool, BaseTool) or not isinstance(name, str):
            return False
        if name in {"edit_job", "remove_job", "get_job_history"}:
            return self.job_id is not None
        if name in ALWAYS_ALLOWED or (name in FILE_READ_TOOLS and self.paths):
            return True
        return name in self.tools and is_read_only(tool)

    def allows_call(self, tool: object, args: Mapping[str, object]) -> bool:
        """Whether one tool call stays inside the scope, including its path."""
        if not self.allows_tool(tool):
            return False
        name = getattr(tool, "name", "")
        if name in {"edit_job", "remove_job", "get_job_history"}:
            return args.get("job_id") == self.job_id
        argument = FILE_READ_TOOLS.get(name)
        if argument is None:
            return True
        return self.covers_path(args.get(argument)) and (
            name != "glob" or _safe_pattern(args.get("pattern"))
        )

    def covers_path(self, path: object) -> bool:
        """Return whether a path is within the owner's read grants."""
        if not isinstance(path, str) or not path.startswith("/"):
            return False
        target = posixpath.normpath(path)
        return any(
            target == root or target.startswith(root.rstrip("/") + "/") for root in self.paths
        )


JOB_READ_SCOPE: contextvars.ContextVar[ReadScope | None] = contextvars.ContextVar(
    "talon_job_read_scope", default=None
)
"""Scope of the job run in progress, or `None` outside job runs."""


def is_read_only(tool: BaseTool) -> bool:
    """Whether the MCP tool has a read-only contract.

    Args:
        tool: Tool to inspect.

    Returns:
        `True` for a read-only annotation or a recognized unannotated service read.
    """
    mcp = (tool.metadata or {}).get("mcp")
    descriptor = mcp.get("tool") if isinstance(mcp, Mapping) else None
    annotations = descriptor.get("annotations") if isinstance(descriptor, Mapping) else None
    if annotations is None:
        return (tool.metadata or {}).get("_deepagents_talon_known_read") is True
    if not isinstance(annotations, Mapping):
        return False
    return (
        annotations.get("read_only_hint") is True
        and annotations.get("destructive_hint") is not True
    )


# Shared across tool tasks in a single isolated check.
JOB_READS: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "talon_job_reads", default=None
)


REQUIRED_READS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "talon_required_reads", default=()
)


class JobToolsUnavailableError(RuntimeError):
    """A scheduled check cannot access its confirmed tool sources."""


class JobScopeMiddleware(AgentMiddleware):
    """Restrict tools during job runs; a no-op for every other turn."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Hide out-of-scope tools from the model."""
        return handler(_scoped_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Hide out-of-scope tools from the model."""
        return await handler(_scoped_request(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Refuse a tool call outside the job's read scope."""
        return _refusal(request) or _observe_read(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        """Refuse a tool call outside the job's read scope."""
        return _refusal(request) or _observe_read(request, await handler(request))


def _scoped_request(request: ModelRequest) -> ModelRequest:
    scope = JOB_READ_SCOPE.get()
    if scope is None:
        if not REQUIRED_READS.get():
            return request
        required = ReadScope.from_sources(REQUIRED_READS.get())
        available = {tool.name for tool in request.tools if isinstance(tool, BaseTool)}
        if missing := required.tools - available:
            msg = f"Required tools are unavailable: {', '.join(sorted(missing))}"
            raise JobToolsUnavailableError(msg)
        return request
    allowed = [
        tool for tool in request.tools if isinstance(tool, BaseTool) and scope.allows_tool(tool)
    ]
    missing = scope.tools.difference(tool.name for tool in allowed)
    if missing:
        msg = f"Objective check blocked by unavailable read tools: {', '.join(sorted(missing))}"
        raise JobToolsUnavailableError(msg)
    return request.override(tools=[*allowed])


def _observe_read(request: ToolCallRequest, result: ToolMessage) -> ToolMessage:
    reads = JOB_READS.get()
    scope = JOB_READ_SCOPE.get()
    if reads is None or not isinstance(result, ToolMessage) or result.status != "success":
        return result
    if isinstance(result.content, str) and result.content.startswith("Error:"):
        return result
    call = request.tool_call
    name = call["name"]
    if name in ALWAYS_ALLOWED | {"edit_job", "remove_job", "get_job_history"}:
        return result
    if scope and (name in scope.tools or name in FILE_READ_TOOLS):
        reads.add(name)
    path = call["args"].get(FILE_READ_TOOLS.get(name, ""))
    for source in REQUIRED_READS.get():
        if source == f"tool:{name}":
            reads.add(source)
        elif (
            source.startswith(PATH_PREFIX)
            and name in {"read_file", "grep"}
            and isinstance(path, str)
        ):
            root = posixpath.normpath(source.removeprefix(PATH_PREFIX))
            target = posixpath.normpath(path)
            if target == root or target.startswith(root.rstrip("/") + "/"):
                reads.add(source)
    return result


def _refusal(request: ToolCallRequest) -> ToolMessage | None:
    scope = JOB_READ_SCOPE.get()
    call = request.tool_call
    if scope is None or scope.allows_call(request.tool, call.get("args") or {}):
        return None
    return ToolMessage(
        content="Refused: this job run may only read the sources its owner confirmed.",
        name=call["name"],
        tool_call_id=call["id"],
        status="error",
    )


def _safe_pattern(pattern: object) -> bool:
    return not isinstance(pattern, str) or (not pattern.startswith("/") and ".." not in pattern)
