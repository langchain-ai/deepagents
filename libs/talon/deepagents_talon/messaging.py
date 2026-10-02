"""Progress messages scoped to the active Talon invocation."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_talon.background import _IN_SUBAGENT

if TYPE_CHECKING:
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


@tool
async def send_message(text: str) -> str:
    """Send a progress update to the user in the chat that started this turn.

    The agent keeps working after this tool returns. Use for brief, useful updates
    during longer tasks; give the final answer normally when finished.
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
    try:
        result = await handler(text)
    except Exception:  # noqa: BLE001  # Keep transport failures out of model context.
        logger.warning("Progress message delivery failed")
        return "Message delivery failed."
    if not result.success:
        logger.warning("Progress message was not delivered")
        return "Message not sent: delivery failed or the turn is no longer active."
    return "Message sent. Continue working; give your final answer when finished."
