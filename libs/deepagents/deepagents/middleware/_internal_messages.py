"""Internal message markers shared by middleware consumers."""

from __future__ import annotations

from typing import Final

from langchain_core.messages import AnyMessage, HumanMessage

GOAL_CONTROL_MESSAGE_SOURCE: Final = "goal_control"
GOAL_STATE_MESSAGE_SOURCE: Final = "goal_state"
GOAL_INTERNAL_MESSAGE_SOURCES = frozenset({GOAL_CONTROL_MESSAGE_SOURCE, GOAL_STATE_MESSAGE_SOURCE})
_GOAL_INTERNAL_MESSAGE_PREFIXES = (
    "[SYSTEM] Goal set by the user",
    "[SYSTEM] Goal amended by the user.",
    "[SYSTEM] Goal resumed by the user.",
    "[SYSTEM] Goal/rubric state changed.",
)


def message_source(message: AnyMessage) -> str | None:
    """Return a message's `lc_source` value when present."""
    source = message.additional_kwargs.get("lc_source")
    return source if isinstance(source, str) and source else None


def is_goal_internal_message(message: AnyMessage) -> bool:
    """Return whether a human message is goal state or control context."""
    if not isinstance(message, HumanMessage):
        return False
    if message_source(message) in GOAL_INTERNAL_MESSAGE_SOURCES:
        return True
    content = message.content
    return isinstance(content, str) and content.startswith(_GOAL_INTERNAL_MESSAGE_PREFIXES)
