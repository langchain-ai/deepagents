"""Progress messages scoped to the active Talon invocation."""

import logging
from contextvars import ContextVar

from langchain_core.tools import tool

from deepagents_talon.interfaces import ProgressMessageHandler

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
