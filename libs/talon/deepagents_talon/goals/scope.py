"""Read-only tool scope for unattended goal turns.

Talon is an experimental runtime and is subject to change or removal at any time.

A goal turn may read only what the owner named in the confirmed charter. The
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

ALWAYS_ALLOWED = frozenset({"current_time"})


@dataclass(frozen=True, slots=True)
class ReadScope:
    """Tools and paths an unattended goal turn may read.

    Args:
        tools: Tool names the owner named; each must also be annotated read-only.
        paths: Absolute directories the filesystem read tools may address.
    """

    tools: frozenset[str] = frozenset()
    paths: tuple[str, ...] = ()

    @classmethod
    def from_sources(cls, sources: Iterable[str]) -> ReadScope:
        """Derive the enforceable scope from charter source entries.

        `tool:<name>` names a tool and `path:<absolute dir>` names a directory.
        Other entries guide the model but grant no tool access.

        Args:
            sources: Confirmed charter sources.

        Returns:
            The read scope.
        """
        tools = {s.removeprefix(TOOL_PREFIX).strip() for s in sources if s.startswith(TOOL_PREFIX)}
        paths = [
            posixpath.normpath(s.removeprefix(PATH_PREFIX).strip())
            for s in sources
            if s.startswith(PATH_PREFIX) and s.removeprefix(PATH_PREFIX).strip().startswith("/")
        ]
        return cls(tools=frozenset(filter(None, tools)), paths=tuple(dict.fromkeys(paths)))

    def to_metadata(self) -> dict[str, list[str]]:
        """Serialize for agent request metadata."""
        return {"tools": sorted(self.tools), "paths": list(self.paths)}

    @classmethod
    def from_metadata(cls, value: object) -> ReadScope | None:
        """Parse request metadata; malformed values yield the empty scope.

        Args:
            value: Value of the `goal_read_scope` metadata key.

        Returns:
            The scope, or `None` when the key was absent.
        """
        if value is None:
            return None
        if not isinstance(value, Mapping):
            return cls()
        tools = value.get("tools")
        paths = value.get("paths")
        return cls.from_sources(
            [
                *(f"{TOOL_PREFIX}{t}" for t in _strings(tools)),
                *(f"{PATH_PREFIX}{p}" for p in _strings(paths)),
            ]
        )

    def allows_tool(self, tool: object) -> bool:
        """Whether a tool may be offered to and called by the goal turn."""
        name = getattr(tool, "name", None)
        if not isinstance(tool, BaseTool) or not isinstance(name, str):
            return False
        if name in ALWAYS_ALLOWED or (name in FILE_READ_TOOLS and self.paths):
            return True
        return name in self.tools and is_read_only(tool)

    def allows_call(self, tool: object, args: Mapping[str, object]) -> bool:
        """Whether one tool call stays inside the scope, including its path."""
        if not self.allows_tool(tool):
            return False
        name = getattr(tool, "name", "")
        argument = FILE_READ_TOOLS.get(name)
        if argument is None:
            return True
        return self._covers(args.get(argument)) and (
            name != "glob" or _safe_pattern(args.get("pattern"))
        )

    def _covers(self, path: object) -> bool:
        if not isinstance(path, str) or not path.startswith("/"):
            return False
        target = posixpath.normpath(path)
        return any(
            target == root or target.startswith(root.rstrip("/") + "/") for root in self.paths
        )


GOAL_READ_SCOPE: contextvars.ContextVar[ReadScope | None] = contextvars.ContextVar(
    "talon_goal_read_scope", default=None
)
"""Scope of the goal turn in progress, or `None` outside goal turns."""


def is_read_only(tool: BaseTool) -> bool:
    """Whether a tool's MCP annotations declare it read-only and non-destructive.

    Args:
        tool: Tool to inspect.

    Returns:
        `True` only for a literal read-only hint without a destructive hint.
    """
    mcp = (tool.metadata or {}).get("mcp")
    annotations = mcp.get("tool", {}).get("annotations", {}) if isinstance(mcp, Mapping) else {}
    if not isinstance(annotations, Mapping):
        return False
    return (
        annotations.get("read_only_hint") is True
        and annotations.get("destructive_hint") is not True
    )


class GoalScopeMiddleware(AgentMiddleware):
    """Restrict tools during goal turns; a no-op for every other turn."""

    name = "TalonGoalScope"

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
        """Refuse a tool call outside the goal's read scope."""
        return _refusal(request) or handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        """Refuse a tool call outside the goal's read scope."""
        return _refusal(request) or await handler(request)


def _scoped_request(request: ModelRequest) -> ModelRequest:
    scope = GOAL_READ_SCOPE.get()
    if scope is None:
        return request
    return request.override(tools=[tool for tool in request.tools if scope.allows_tool(tool)])


def _refusal(request: ToolCallRequest) -> ToolMessage | None:
    scope = GOAL_READ_SCOPE.get()
    call = request.tool_call
    if scope is None or scope.allows_call(request.tool, call.get("args") or {}):
        return None
    return ToolMessage(
        content="Refused: this goal turn may only read the sources its owner confirmed.",
        name=call["name"],
        tool_call_id=call["id"],
        status="error",
    )


def _safe_pattern(pattern: object) -> bool:
    return not isinstance(pattern, str) or (not pattern.startswith("/") and ".." not in pattern)


def _strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []
