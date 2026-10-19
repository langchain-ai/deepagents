"""Progress messages scoped to the active Talon invocation."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_talon.background import _IN_SUBAGENT

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langgraph.runtime import Runtime

    from deepagents_talon.interfaces import ProgressMessageHandler


class ProgressMessages(AgentMiddleware):
    """Deliver main-agent narration before its tools execute."""

    async def aafter_model(self, state: AgentState, runtime: Runtime) -> None:
        """Forward visible tool-calling messages to the originating channel."""
        del runtime
        message = next(
            (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
            None,
        )
        if (
            not _IN_SUBAGENT.get()
            and isinstance(message, AIMessage)
            and message.tool_calls
            and message.text.strip()
            and not any(call["name"] == "send_message" for call in message.tool_calls)
        ):
            await _deliver_message(message.text)


logger = logging.getLogger(__name__)
MESSAGE_HANDLER: ContextVar[ProgressMessageHandler | None] = ContextVar(
    "talon_message_handler", default=None
)


@dataclass
class _ProgressBudget:
    remaining: int


_PROGRESS_BUDGET: ContextVar[_ProgressBudget | None] = ContextVar(
    "talon_progress_budget", default=None
)


@contextmanager
def progress_messages(handler: ProgressMessageHandler | None, verbosity: int) -> Iterator[None]:
    """Scope progress delivery and its shared budget to one runtime invocation."""
    limits = {1: 0, 2: 2}
    message_token = MESSAGE_HANDLER.set(handler)
    budget_token = _PROGRESS_BUDGET.set(
        _ProgressBudget(limits[verbosity]) if verbosity in limits else None
    )
    try:
        yield
    finally:
        _PROGRESS_BUDGET.reset(budget_token)
        MESSAGE_HANDLER.reset(message_token)


@tool
async def send_message(text: str) -> str:
    """Send a progress update to the user in the chat that started this turn.

    The agent keeps working after this tool returns. Use only for important updates
    or notifications about changes in working direction, not routine narration.
    Give the final answer normally when finished.
    The destination is fixed by the host and cannot be changed.

    Args:
        text: Message to send to the user.
    """
    return await _deliver_message(text)


async def _deliver_message(text: str) -> str:
    handler = MESSAGE_HANDLER.get()
    if handler is None:
        return "Message unavailable: this run has no originating channel."
    if not text.strip():
        return "Message not sent: text must not be blank."
    budget = _PROGRESS_BUDGET.get()
    if budget is not None:
        if budget.remaining == 0:
            return (
                "Message not sent: interim update limit reached. "
                "Continue working; give your final answer when finished."
            )
        budget.remaining -= 1
    try:
        result = await handler(text)
    except Exception:  # noqa: BLE001  # Keep transport failures out of model context.
        logger.warning("Progress message delivery failed")
        return "Message delivery failed."
    if not result.success:
        logger.warning("Progress message was not delivered")
        return "Message not sent: delivery failed or the turn is no longer active."
    return "Message sent. Continue working; give your final answer when finished."
