"""Message store for virtualized chat history.

This module provides data structures and management for message virtualization,
allowing the CLI to handle large message histories efficiently by keeping only
a window of widgets in the DOM while storing all message data.

Based on patterns from Textual's Log and RichLog widgets.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.widget import Widget


class MessageType(StrEnum):
    """Types of messages in the chat."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ERROR = "error"
    SYSTEM = "system"
    DIFF = "diff"


class ToolStatus(StrEnum):
    """Status of a tool call."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    REJECTED = "rejected"
    SKIPPED = "skipped"


@dataclass
class MessageData:
    """Serialized message data for storage.

    This dataclass holds all information needed to recreate a message widget.
    It's designed to be lightweight and serializable.
    """

    type: MessageType
    content: str
    id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time)

    # Tool-specific metadata
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_status: ToolStatus | None = None
    tool_output: str | None = None
    tool_expanded: bool = False

    # Diff-specific metadata
    diff_file_path: str | None = None

    # Streaming state - True if message is still being streamed
    is_streaming: bool = False

    # Cached height hint for scroll calculations (set after first render)
    height_hint: int | None = None

    def to_widget(self) -> Widget:  # noqa: PLR0911
        """Recreate a widget from this message data.

        Returns:
            The appropriate message widget for this data.
        """
        # Import here to avoid circular imports
        from deepagents_cli.widgets.messages import (
            AssistantMessage,
            DiffMessage,
            ErrorMessage,
            SystemMessage,
            ToolCallMessage,
            UserMessage,
        )

        match self.type:
            case MessageType.USER:
                return UserMessage(self.content, id=self.id)

            case MessageType.ASSISTANT:
                # For assistant messages, we create with content
                # The widget will render it via Markdown
                return AssistantMessage(self.content, id=self.id)

            case MessageType.TOOL:
                widget = ToolCallMessage(
                    self.tool_name or "unknown",
                    self.tool_args,
                    id=self.id,
                )
                # Restore the status and output after mount
                # We'll need to call _restore_tool_state after mounting
                widget._deferred_status = self.tool_status
                widget._deferred_output = self.tool_output
                widget._deferred_expanded = self.tool_expanded
                return widget

            case MessageType.ERROR:
                return ErrorMessage(self.content, id=self.id)

            case MessageType.SYSTEM:
                return SystemMessage(self.content, id=self.id)

            case MessageType.DIFF:
                return DiffMessage(
                    self.content,
                    file_path=self.diff_file_path or "",
                    id=self.id,
                )

            case _:
                # Fallback to system message
                return SystemMessage(self.content, id=self.id)

    @classmethod
    def from_widget(cls, widget: Widget) -> MessageData:  # noqa: PLR0911
        """Create MessageData from an existing widget.

        Args:
            widget: The message widget to serialize.

        Returns:
            MessageData containing all the widget's state.
        """
        from deepagents_cli.widgets.messages import (
            AssistantMessage,
            DiffMessage,
            ErrorMessage,
            SystemMessage,
            ToolCallMessage,
            UserMessage,
        )

        widget_id = widget.id or f"msg-{uuid.uuid4().hex[:8]}"

        if isinstance(widget, UserMessage):
            return cls(
                type=MessageType.USER,
                content=widget._content,
                id=widget_id,
            )

        if isinstance(widget, AssistantMessage):
            return cls(
                type=MessageType.ASSISTANT,
                content=widget._content,
                id=widget_id,
                is_streaming=widget._stream is not None,
            )

        if isinstance(widget, ToolCallMessage):
            return cls(
                type=MessageType.TOOL,
                content="",  # Tool messages don't have simple content
                id=widget_id,
                tool_name=widget._tool_name,
                tool_args=widget._args,
                tool_status=ToolStatus(widget._status) if widget._status else None,
                tool_output=widget._output,
                tool_expanded=widget._expanded,
            )

        if isinstance(widget, ErrorMessage):
            return cls(
                type=MessageType.ERROR,
                content=widget._content,
                id=widget_id,
            )

        if isinstance(widget, SystemMessage):
            return cls(
                type=MessageType.SYSTEM,
                content=widget._content,
                id=widget_id,
            )

        if isinstance(widget, DiffMessage):
            return cls(
                type=MessageType.DIFF,
                content=widget._diff_content,
                id=widget_id,
                diff_file_path=widget._file_path,
            )

        # Unknown widget type - treat as system message
        return cls(
            type=MessageType.SYSTEM,
            content=f"[Unknown widget: {type(widget).__name__}]",
            id=widget_id,
        )


class MessageStore:
    """Manages message data and widget window for virtualization.

    This class stores all messages as data and manages a sliding window
    of widgets that are actually mounted in the DOM. Based on patterns
    from Textual's Log widget.

    Attributes:
        WINDOW_SIZE: Maximum number of widgets to keep in DOM.
        HYDRATE_BUFFER: Number of messages to hydrate when scrolling near edge.
    """

    WINDOW_SIZE: int = 50
    HYDRATE_BUFFER: int = 15

    def __init__(self) -> None:
        """Initialize the message store."""
        self._messages: list[MessageData] = []
        self._visible_start: int = 0
        self._visible_end: int = 0

        # Track active streaming message - never archive this
        self._active_message_id: str | None = None

    @property
    def total_count(self) -> int:
        """Total number of messages stored."""
        return len(self._messages)

    @property
    def visible_count(self) -> int:
        """Number of messages currently visible (as widgets)."""
        return self._visible_end - self._visible_start

    @property
    def has_messages_above(self) -> bool:
        """Check if there are archived messages above the visible window."""
        return self._visible_start > 0

    @property
    def has_messages_below(self) -> bool:
        """Check if there are archived messages below the visible window."""
        return self._visible_end < len(self._messages)

    def append(self, message: MessageData) -> None:
        """Add a new message to the store.

        Args:
            message: The message data to add.
        """
        self._messages.append(message)
        self._visible_end = len(self._messages)

    def get_message(self, message_id: str) -> MessageData | None:
        """Get a message by its ID.

        Args:
            message_id: The ID of the message to find.

        Returns:
            The message data, or None if not found.
        """
        for msg in self._messages:
            if msg.id == message_id:
                return msg
        return None

    def get_message_at_index(self, index: int) -> MessageData | None:
        """Get a message by its index.

        Args:
            index: The index of the message.

        Returns:
            The message data, or None if index is out of bounds.
        """
        if 0 <= index < len(self._messages):
            return self._messages[index]
        return None

    def update_message(self, message_id: str, **updates: Any) -> bool:
        """Update a message's data.

        Args:
            message_id: The ID of the message to update.
            **updates: Fields to update.

        Returns:
            True if the message was found and updated.
        """
        for msg in self._messages:
            if msg.id == message_id:
                for key, value in updates.items():
                    if hasattr(msg, key):
                        setattr(msg, key, value)
                return True
        return False

    def set_active_message(self, message_id: str | None) -> None:
        """Set the currently active (streaming) message.

        Active messages are never archived.

        Args:
            message_id: The ID of the active message, or None to clear.
        """
        self._active_message_id = message_id

    def is_active(self, message_id: str) -> bool:
        """Check if a message is the active streaming message.

        Args:
            message_id: The message ID to check.

        Returns:
            True if this is the active message.
        """
        return message_id == self._active_message_id

    def window_exceeded(self) -> bool:
        """Check if the visible window exceeds the maximum size.

        Returns:
            True if we should prune some widgets.
        """
        return self.visible_count > self.WINDOW_SIZE

    def get_messages_to_prune(self, count: int | None = None) -> list[MessageData]:
        """Get the oldest visible messages that should be pruned.

        This returns messages from the START of the visible window,
        excluding any active streaming message.

        Args:
            count: Number of messages to prune, or None to prune
                   enough to get back to WINDOW_SIZE.

        Returns:
            List of messages to prune (remove widgets for).
        """
        if count is None:
            count = max(0, self.visible_count - self.WINDOW_SIZE)

        if count <= 0:
            return []

        to_prune: list[MessageData] = []
        prune_end = min(self._visible_start + count, self._visible_end)

        for i in range(self._visible_start, prune_end):
            msg = self._messages[i]
            # Never prune the active streaming message
            if msg.id != self._active_message_id:
                to_prune.append(msg)

        return to_prune

    def mark_pruned(self, message_ids: list[str]) -> None:
        """Mark messages as pruned (widgets removed).

        This updates the visible window start index.

        Args:
            message_ids: IDs of messages that were pruned.
        """
        # Find the new start index (first message that wasn't pruned)
        pruned_set = set(message_ids)
        while (
            self._visible_start < self._visible_end
            and self._messages[self._visible_start].id in pruned_set
        ):
            self._visible_start += 1

    def get_messages_to_hydrate(self, count: int | None = None) -> list[MessageData]:
        """Get messages above the visible window to hydrate.

        Args:
            count: Number of messages to hydrate, or None for HYDRATE_BUFFER.

        Returns:
            List of messages to hydrate (create widgets for), in order.
        """
        if count is None:
            count = self.HYDRATE_BUFFER

        if self._visible_start <= 0:
            return []

        hydrate_start = max(0, self._visible_start - count)
        return self._messages[hydrate_start : self._visible_start]

    def mark_hydrated(self, count: int) -> None:
        """Mark that messages above were hydrated.

        Args:
            count: Number of messages that were hydrated.
        """
        self._visible_start = max(0, self._visible_start - count)

    def should_hydrate_above(self, scroll_position: float, viewport_height: int) -> bool:
        """Check if we should hydrate messages above the current view.

        Args:
            scroll_position: Current scroll Y position.
            viewport_height: Height of the viewport.

        Returns:
            True if user is scrolling near the top and we have archived messages.
        """
        if not self.has_messages_above:
            return False

        # Hydrate when within 2x viewport height of the top
        threshold = viewport_height * 2
        return scroll_position < threshold

    def should_prune_below(
        self, scroll_position: float, viewport_height: int, content_height: int
    ) -> bool:
        """Check if we should prune messages below the current view.

        Args:
            scroll_position: Current scroll Y position.
            viewport_height: Height of the viewport.
            content_height: Total height of all content.

        Returns:
            True if we have too many widgets and bottom ones are far from view.
        """
        if self.visible_count <= self.WINDOW_SIZE:
            return False

        # Only prune if user is far from the bottom
        distance_from_bottom = content_height - scroll_position - viewport_height
        threshold = viewport_height * 3
        return distance_from_bottom > threshold

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._visible_start = 0
        self._visible_end = 0
        self._active_message_id = None

    def get_visible_range(self) -> tuple[int, int]:
        """Get the range of visible message indices.

        Returns:
            Tuple of (start_index, end_index).
        """
        return (self._visible_start, self._visible_end)

    def get_all_messages(self) -> list[MessageData]:
        """Get all stored messages.

        Returns:
            List of all message data.
        """
        return list(self._messages)

    def get_visible_messages(self) -> list[MessageData]:
        """Get messages in the visible window.

        Returns:
            List of visible message data.
        """
        return self._messages[self._visible_start : self._visible_end]
