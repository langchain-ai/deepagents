"""Scope fork snapshots and keep reporting guidance outside cached prompts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, NamedTuple

from langchain.agents.middleware.types import AgentMiddleware, TracePolicy, omit_payload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command


class ForkSnapshot(NamedTuple):
    """The original prompt snapshot and most recent observed facts."""

    context: str
    fingerprint: str | None


_fork_snapshot: ContextVar[ForkSnapshot | None] = ContextVar(
    "fork_snapshot", default=None
)


def inherited_snapshot() -> ForkSnapshot | None:
    """Return the snapshot bound to this task invocation."""
    return _fork_snapshot.get()


@contextmanager
def bind_snapshot(snapshot: ForkSnapshot | None) -> Iterator[None]:
    """Scope inheritance to one invocation, resetting it even on cancellation."""
    token = _fork_snapshot.set(snapshot)
    try:
        yield
    finally:
        _fork_snapshot.reset(token)


class ForkAssignmentMiddleware(AgentMiddleware):
    """Attach generated general-purpose instructions to its task assignment."""

    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(self, instructions: str) -> None:
        """Store the generated fork's assignment instructions."""
        self._instructions = instructions

    def _assignment(self, request: ToolCallRequest) -> ToolCallRequest:
        call = request.tool_call
        args = call["args"]
        if call["name"] != "task" or args.get("subagent_type") != "general-purpose":
            return request
        description = args.get("description")
        if not isinstance(description, str):
            return request
        return request.override(
            tool_call={
                **call,
                "args": {
                    **args,
                    "description": f"{description}\n\n{self._instructions}",
                },
            }
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Add reporting guidance without mutating the parent's tool call.

        Returns:
            The tool handler result.
        """
        return handler(self._assignment(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Add reporting guidance to an asynchronous fork assignment.

        Returns:
            The tool handler result.
        """
        return await handler(self._assignment(request))
