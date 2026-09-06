"""Message normalization shared by conversation archive backends."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from langchain_core.messages import BaseMessage

CHUNK_SIZE = 4000
_ARCHIVE_TOOLS = {"search_conversations", "read_conversation", "list_conversations"}


@dataclass(frozen=True)
class MessageRevision:
    """Normalized text and stable identity for an archived message revision."""

    message_id: str
    revision: str
    role: str
    text: str

    def chunks(self) -> Iterator[tuple[int, str]]:
        """Yield numbered display chunks while retaining complete text for search."""
        for part, start in enumerate(range(0, len(self.text), CHUNK_SIZE)):
            yield part, self.text[start : start + CHUNK_SIZE]


def message_revisions(messages: Sequence[BaseMessage], timestamp: str) -> Iterator[MessageRevision]:
    """Normalize retained messages without recursively archiving history results.

    Args:
        messages: Committed messages, including excluded messages in their original positions.
        timestamp: Checkpoint timestamp used to identify messages without an ID.

    Yields:
        Nonempty revisions with deterministic identities and complete search text.
    """
    for index, message in enumerate(messages):
        if not isinstance(message, (HumanMessage, AIMessage, ToolMessage)):
            continue
        if isinstance(message, ToolMessage) and message.name in _ARCHIVE_TOOLS:
            continue
        text = message.text
        if isinstance(message, AIMessage) and message.tool_calls:
            text += "\nTool calls: " + json.dumps(message.tool_calls, ensure_ascii=False)
        if text:
            yield MessageRevision(
                message.id or f"talon-history:{timestamp}:{index}",
                hashlib.sha256(text.encode()).hexdigest(),
                message.type,
                text,
            )
