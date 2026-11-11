"""Bounded, cooperative controls for Talon-built background agents."""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.runtime import Runtime
    from langgraph.types import Command

_MAX_EVENTS = 50
_MAX_MESSAGES = 20
_MAX_MESSAGE_CHARACTERS = 2000


@dataclass
class SubagentControl:
    """Keep activity and message receipts outside model-controlled graph state."""

    supported: bool = False
    events: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=_MAX_EVENTS))
    messages: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=_MAX_MESSAGES))

    def record(self, event: str, **metadata: str) -> None:
        """Record bounded activity metadata without arguments, outputs, or reasoning."""
        self.events.append({"timestamp": datetime.now(UTC).isoformat(), "event": event, **metadata})

    def enqueue(self, message: str) -> dict[str, str]:
        """Queue a bounded instruction without evicting unconsumed messages."""
        if not message.strip() or len(message) > _MAX_MESSAGE_CHARACTERS:
            return {"error": f"Message must contain 1-{_MAX_MESSAGE_CHARACTERS} characters."}
        if len(self.messages) == _MAX_MESSAGES and self.messages[0]["status"] == "queued":
            return {"error": "Subagent message inbox is full."}
        receipt = {"message_id": uuid4().hex, "status": "queued", "message": message}
        self.messages.append(receipt)
        self.record("message_queued", message_id=receipt["message_id"])
        return {key: receipt[key] for key in ("message_id", "status")}


INLINE_DEADLINE: ContextVar[float | None] = ContextVar("talon_inline_deadline", default=None)


ACTIVE_CONTROL: ContextVar[SubagentControl | None] = ContextVar(
    "talon_subagent_control", default=None
)


class SubagentControlMiddleware(AgentMiddleware):
    """Apply coordinator messages at model boundaries and expose metadata-only activity."""

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, object] | None:
        """Persist queued coordinator instructions before the next model request."""
        del state, runtime
        control = ACTIVE_CONTROL.get()
        if control is None:
            return None
        messages = []
        for receipt in control.messages:
            if receipt["status"] == "queued":
                messages.append(
                    HumanMessage(
                        content="Coordinator follow-up for this task; retain the original scope "
                        "and all access and approval restrictions:\n" + receipt["message"],
                        id=receipt["message_id"],
                    )
                )
                receipt["status"] = "consumed"
                control.record("message_consumed", message_id=receipt["message_id"])
        return {"messages": messages} if messages else None

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Record model activity without retaining model content."""
        control = ACTIVE_CONTROL.get()
        if control is None:
            return await handler(request)
        control.record("model_started")
        try:
            result = await handler(request)
        except Exception:
            control.record("model_error")
            raise
        control.record("model_finished")
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Record tool lifecycle without retaining arguments or results."""
        control = ACTIVE_CONTROL.get()
        if control is None:
            return await handler(request)
        name = request.tool_call["name"][:128]
        control.record("tool_started", tool=name)
        try:
            result = await handler(request)
        except Exception:
            control.record("tool_error", tool=name)
            raise
        control.record(
            "tool_error" if getattr(result, "status", None) == "error" else "tool_finished",
            tool=name,
        )
        return result
