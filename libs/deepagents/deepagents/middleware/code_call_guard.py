"""Refuse approval-gated tool calls that the model did not make itself."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from deepagents.backends.protocol import PERMISSION_DENIED
from deepagents.middleware.filesystem import _error_artifact, _raw_result_request


def _made_by_model(request: ToolCallRequest) -> bool:
    last_ai = next((m for m in reversed(request.state["messages"]) if isinstance(m, AIMessage)), None)
    return last_ai is not None and any(call["id"] == request.tool_call["id"] for call in last_ai.tool_calls)


def _gated(approval: HumanInTheLoopMiddleware, request: ToolCallRequest) -> bool:
    config = approval.interrupt_on.get(request.tool_call["name"])
    if config is None:
        return False
    when = config.get("when")
    return when is None or when(request)


class CodeCallGuardMiddleware(AgentMiddleware[Any, Any, Any]):
    """Fail approval-gated tool calls that code dispatches, instead of running them unreviewed.

    `HumanInTheLoopMiddleware` reviews the tool calls in the model's last message.
    A middleware can also dispatch tool calls of its own through the tool-call chain,
    such as an `fs` call from a code interpreter. Such a call is not in that message,
    so batched review never sees it, and per-call review would pause the run in the
    middle of the code. This guard fails the call with an error that tells the model
    to make the call itself, where a reviewer can approve it.

    A call counts as the model's when its ID matches a tool call in the model's last
    message, so a dispatching middleware must give each call a fresh ID.

    `create_deep_agent` and its declarative subagents place a guard before every
    `HumanInTheLoopMiddleware`. With `create_agent`, place one directly before the
    approval middleware it guards, and after the middleware that dispatches calls:
    a dispatcher listed after the approval middleware sends its calls below both, so
    neither sees them.

    !!! warning "Experimental"

        This guard may change or move into `HumanInTheLoopMiddleware`.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        from deepagents.middleware import CodeCallGuardMiddleware

        approval = HumanInTheLoopMiddleware(interrupt_on={"write_file": True})
        agent = create_agent(
            model,
            tools=tools,
            middleware=[code_interpreter, CodeCallGuardMiddleware(approval), approval],
        )
        ```
    """

    def __init__(self, approval: HumanInTheLoopMiddleware) -> None:
        """Guard the calls `approval` would review.

        Args:
            approval: The approval middleware whose `interrupt_on` rules decide
                which calls from code fail.
        """
        super().__init__()
        self._approval = approval

    @property
    def approval(self) -> HumanInTheLoopMiddleware:
        """The approval middleware this guard stands in front of."""
        return self._approval

    @property
    def name(self) -> str:
        """Unique per guarded approval middleware, since `create_agent` rejects duplicate names."""
        return f"CodeCallGuard[{self._approval.name}]"

    def _refusal(self, request: ToolCallRequest) -> ToolMessage | None:
        if _made_by_model(request) or not _gated(self._approval, request):
            return None
        name = request.tool_call["name"]
        return ToolMessage(
            content=f"Error: `{name}` needs human approval, which a call from code cannot request. Call `{name}` directly instead.",
            name=name,
            tool_call_id=request.tool_call["id"],
            status="error",
            artifact=_error_artifact(_raw_result_request(request.runtime), PERMISSION_DENIED),
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Refuse a gated call from code; pass every other call through."""
        if (refusal := self._refusal(request)) is not None:
            return refusal
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of `wrap_tool_call`."""
        if (refusal := self._refusal(request)) is not None:
            return refusal
        return await handler(request)


def _guard_calls_from_code(middleware: Sequence[AgentMiddleware[Any, Any, Any]]) -> list[AgentMiddleware[Any, Any, Any]]:
    guarded: list[AgentMiddleware[Any, Any, Any]] = []
    for item in middleware:
        previous = guarded[-1] if guarded else None
        already = isinstance(previous, CodeCallGuardMiddleware) and previous.approval is item
        if isinstance(item, HumanInTheLoopMiddleware) and not already:
            guarded.append(CodeCallGuardMiddleware(item))
        guarded.append(item)
    return guarded
