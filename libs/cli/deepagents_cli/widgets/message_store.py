"""Message store for virtualized chat history.

This module provides data structures and management for message virtualization,
allowing the CLI to handle large message histories efficiently by keeping only
a sliding window of widgets in the DOM while storing all message data as
lightweight dataclasses.

The approach is inspired by Textual's `Log` widget, which only keeps `N` lines
in the DOM and recreates older ones on demand.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.widget import Widget

logger = logging.getLogger(__name__)

_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {
        "content",
        "tool_status",
        "tool_output",
        "tool_expanded",
        "skill_expanded",
        "is_streaming",
        "height_hint",
    }
)
"""Fields on `MessageData` that callers are allowed to update via `update_message`.

Prevents accidental overwriting of identity fields like `id`, `type`, or
`timestamp`.
"""


class MessageType(StrEnum):
    """Types of messages in the chat."""

    USER = "user"
    """Input authored by the human, rendered above the agent's response."""

    ASSISTANT = "assistant"
    """Streamed agent response rendered with markdown."""

    TOOL = "tool"
    """Record of a tool invocation, including its args, status, and output."""

    SKILL = "skill"
    """Record of a skill invocation, carrying its SKILL.md body and metadata."""

    ERROR = "error"
    """Error surfaced to the user (e.g., a failed tool call or SDK exception)."""

    APP = "app"
    """App-status note from the CLI itself (version info, command feedback)."""

    SUMMARIZATION = "summarization"
    """Notification that the prior conversation was summarized/offloaded."""

    DIFF = "diff"
    """Unified diff preview attached to a file-modifying tool call."""


class ToolStatus(StrEnum):
    """Status of a tool call."""

    PENDING = "pending"
    """Queued for execution, typically awaiting human approval."""

    RUNNING = "running"
    """Currently executing."""

    SUCCESS = "success"
    """Completed without error."""

    ERROR = "error"
    """Raised an exception or returned a non-zero exit status."""

    REJECTED = "rejected"
    """Human explicitly denied the call at the approval prompt."""

    SKIPPED = "skipped"
    """Bypassed without executing (e.g., the agent canceled the call)."""


@dataclass
class MessageData:
    """In-memory message data for virtualization.

    This dataclass holds all information needed to recreate a message widget.
    It is designed to be lightweight so that thousands of messages can be
    stored without meaningful memory overhead.
    """

    type: MessageType
    """The kind of message (user, assistant, tool, etc.)."""

    content: str
    """Primary text content of the message.

    For most message types this is the display text. For TOOL messages it is
    typically empty because the tool's identity comes from `tool_name` /
    `tool_args` instead.
    """

    id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:8]}")
    """Unique identifier used to match the dataclass to its DOM widget."""

    timestamp: float = field(default_factory=time)
    """Unix epoch timestamp of when the message was created."""

    # TOOL message fields - only populated for TOOL messages
    tool_name: str | None = None
    """Name of the tool that was called."""

    tool_args: dict[str, Any] | None = None
    """Arguments passed to the tool call."""

    tool_status: ToolStatus | None = None
    """Current execution status of the tool call."""

    tool_output: str | None = None
    """Output returned by the tool after execution."""

    tool_expanded: bool = False
    """Whether the tool output section is expanded in the UI."""

    # ---

    diff_file_path: str | None = None
    """File path associated with the diff (DIFF messages only)."""

    # SKILL message fields - only populated for SKILL messages
    skill_name: str | None = None
    """Name of the skill that was invoked."""

    skill_description: str | None = None
    """Short description of the skill."""

    skill_source: str | None = None
    """Origin of the skill (e.g., `'built-in'`, `'user'`, `'project'`)."""

    skill_args: str | None = None
    """User-provided arguments to the skill invocation."""

    skill_body: str | None = None
    """Full SKILL.md content sent to the agent."""

    skill_expanded: bool = False
    """Whether the skill body is expanded in the UI."""

    is_streaming: bool = False
    """Whether the message is still being streamed.

    While `True`, the corresponding widget is actively receiving content
    chunks and should not be pruned or re-hydrated.
    """

    is_markdown: bool = False
    """For APP messages, whether `content` is a markdown source string.

    When `True`, rehydration renders the content via Rich markdown instead of
    the plain dim-italic `AppMessage` styling.
    """

    height_hint: int | None = None
    """Cached widget height in terminal rows for scroll position estimation.

    When `_hydrate_messages_above` inserts widgets above the viewport it needs
    to adjust the scroll offset so the user's view doesn't jump. Currently this
    uses a fixed estimate (5 rows per message). Caching the actual rendered
    height here after first mount would make that estimate accurate, especially
    for tall messages like diffs or long assistant responses.

    Not yet populated — see `_hydrate_messages_above` in `app.py`.
    """

    def __post_init__(self) -> None:
        """Validate type-field coherence after construction.

        Raises:
            ValueError: If a TOOL message is missing `tool_name` or a SKILL
                message is missing `skill_name`.
        """
        if self.type == MessageType.TOOL and not self.tool_name:
            msg = "TOOL messages must have a tool_name"
            raise ValueError(msg)
        if self.type == MessageType.SKILL and not self.skill_name:
            msg = "SKILL messages must have a skill_name"
            raise ValueError(msg)

    def to_widget(self) -> Widget:
        """Recreate a widget from this message data.

        Returns:
            The appropriate message widget for this data.
        """
        # Import here to avoid circular imports
        from deepagents_cli.widgets.messages import (
            AppMessage,
            AssistantMessage,
            DiffMessage,
            ErrorMessage,
            SkillMessage,
            SummarizationMessage,
            ToolCallMessage,
            UserMessage,
        )

        match self.type:
            case MessageType.USER:
                return UserMessage(self.content, id=self.id)

            case MessageType.ASSISTANT:
                return AssistantMessage(self.content, id=self.id)

            case MessageType.TOOL:
                widget = ToolCallMessage(
                    self.tool_name or "unknown",
                    self.tool_args,
                    id=self.id,
                )
                # Deferred state is restored automatically during on_mount
                # via _restore_deferred_state
                widget._deferred_status = self.tool_status
                widget._deferred_output = self.tool_output
                widget._deferred_expanded = self.tool_expanded
                return widget

            case MessageType.SKILL:
                widget = SkillMessage(
                    skill_name=self.skill_name or "unknown",
                    description=self.skill_description or "",
                    source=self.skill_source or "",
                    body=self.skill_body or "",
                    args=self.skill_args or "",
                    id=self.id,
                )
                widget._deferred_expanded = self.skill_expanded
                return widget

            case MessageType.ERROR:
                return ErrorMessage(self.content, id=self.id)

            case MessageType.APP:
                return AppMessage(self.content, markdown=self.is_markdown, id=self.id)

            case MessageType.SUMMARIZATION:
                return SummarizationMessage(self.content, id=self.id)

            case MessageType.DIFF:
                return DiffMessage(
                    self.content,
                    file_path=self.diff_file_path or "",
                    id=self.id,
                )

            case _:
                logger.warning(
                    "Unknown MessageType %r for message %s, falling back to AppMessage",
                    self.type,
                    self.id,
                )
                return AppMessage(self.content, id=self.id)

    @classmethod
    def from_widget(cls, widget: Widget) -> MessageData:
        """Create MessageData from an existing widget.

        Args:
            widget: The message widget to serialize.

        Returns:
            MessageData containing all the widget's state.
        """
        # Deferred: prevents import-order issue — both modules live in the
        # widgets package, and messages is re-exported from widgets/__init__.
        from deepagents_cli.widgets.messages import (
            AppMessage,
            AssistantMessage,
            DiffMessage,
            ErrorMessage,
            SkillMessage,
            SummarizationMessage,
            ToolCallMessage,
            UserMessage,
        )

        widget_id = widget.id or f"msg-{uuid.uuid4().hex[:8]}"

        if isinstance(widget, SkillMessage):
            return cls(
                type=MessageType.SKILL,
                content="",
                id=widget_id,
                skill_name=widget._skill_name,
                skill_description=widget._description,
                skill_source=widget._source,
                skill_body=widget._body,
                skill_args=widget._args,
                skill_expanded=widget._expanded,
            )

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
            tool_status: ToolStatus | None = None
            if widget._status:
                try:
                    tool_status = ToolStatus(widget._status)
                except ValueError:
                    logger.warning(
                        "Unknown tool status %r for widget %s",
                        widget._status,
                        widget_id,
                    )

            return cls(
                type=MessageType.TOOL,
                content="",  # Tool messages don't have simple content
                id=widget_id,
                tool_name=widget._tool_name,
                tool_args=widget._args,
                tool_status=tool_status,
                tool_output=widget._output,
                tool_expanded=widget._expanded,
            )

        if isinstance(widget, ErrorMessage):
            return cls(
                type=MessageType.ERROR,
                content=widget._content,
                id=widget_id,
            )

        # Check specialized subclasses before AppMessage so we keep their type
        # when serializing and can restore their specific styling later.
        if isinstance(widget, DiffMessage):
            return cls(
                type=MessageType.DIFF,
                content=widget._diff_content,
                id=widget_id,
                diff_file_path=widget._file_path,
            )

        if isinstance(widget, SummarizationMessage):
            return cls(
                type=MessageType.SUMMARIZATION,
                content=str(widget._content),
                id=widget_id,
            )

        if isinstance(widget, AppMessage):
            return cls(
                type=MessageType.APP,
                content=str(widget._content),
                id=widget_id,
                is_markdown=widget._is_markdown,
            )

        logger.warning(
            "Unknown widget type %s (id=%s), storing as APP message",
            type(widget).__name__,
            widget_id,
        )
        return cls(
            type=MessageType.APP,
            content=f"[Unknown widget: {type(widget).__name__}]",
            id=widget_id,
        )


class MessageStore:
    """Manages message data and widget window for virtualization.

    This class stores all messages as data and manages a sliding window
    of widgets that are actually mounted in the DOM.

    Attributes:
        WINDOW_SIZE: Maximum number of widgets to keep in DOM.

            Balances DOM performance with smooth scrolling experience.
        HYDRATE_BUFFER: Number of messages to hydrate when scrolling near edge.

            Provides enough buffer to avoid visible loading pauses.
    """

    WINDOW_SIZE: int = 50
    HYDRATE_BUFFER: int = 15

    def __init__(self) -> None:
        """Initialize the message store."""
        self._messages: list[MessageData] = []
        self._index: dict[str, MessageData] = {}
        """ID -> MessageData lookup.

        Must contain exactly one entry per element of `_messages`. Any method
        that adds to or removes from `_messages` must update `_index`
        in lockstep.
        """
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
        if message.id in self._index:
            logger.warning(
                "Duplicate message ID %r appended; previous entry will be "
                "unreachable via get_message()",
                message.id,
            )
        self._messages.append(message)
        self._index[message.id] = message
        self._visible_end = len(self._messages)

    def bulk_load(
        self, messages: list[MessageData]
    ) -> tuple[list[MessageData], list[MessageData]]:
        """Load many messages at once, keeping only the tail visible.

        This is optimized for thread resumption: all messages are stored as
        lightweight data, but only the last `WINDOW_SIZE` entries are marked
        visible (i.e. will need DOM widgets).

        Args:
            messages: Ordered list of message data to load.

        Returns:
            Tuple of (archived, visible) message lists.
        """
        self._messages.extend(messages)
        for msg in messages:
            if msg.id in self._index:
                logger.warning(
                    "Duplicate message ID %r in bulk_load; previous entry "
                    "will be unreachable via get_message()",
                    msg.id,
                )
            self._index[msg.id] = msg
        total = len(self._messages)

        if total <= self.WINDOW_SIZE:
            self._visible_start = 0
        else:
            self._visible_start = total - self.WINDOW_SIZE

        self._visible_end = total

        archived = self._messages[: self._visible_start]
        visible = self._messages[self._visible_start : self._visible_end]
        return archived, visible

    def get_message(self, message_id: str) -> MessageData | None:
        """Get a message by its ID.

        Args:
            message_id: The ID of the message to find.

        Returns:
            The message data, or None if not found.
        """
        return self._index.get(message_id)

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

        Only fields in `_UPDATABLE_FIELDS` may be updated. Unknown field
        names raise `ValueError` to catch typos early.

        Args:
            message_id: The ID of the message to update.
            **updates: Fields to update.

        Returns:
            True if the message was found and updated.

        Raises:
            ValueError: If any key in `updates` is not in the updatable
                allowlist.
        """
        unknown = set(updates) - _UPDATABLE_FIELDS
        if unknown:
            msg = f"Cannot update unknown or protected fields: {unknown}"
            raise ValueError(msg)

        msg_data = self._index.get(message_id)
        if msg_data is None:
            logger.warning(
                "update_message called for unknown ID %r; update discarded",
                message_id,
            )
            return False
        for key, value in updates.items():
            setattr(msg_data, key, value)
        return True

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

        Returns a contiguous run of messages from the START of the visible
        window. Stops at the active streaming message to avoid creating gaps
        in the visible window (which would desync store state from the DOM).

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
        idx = self._visible_start

        while len(to_prune) < count and idx < self._visible_end:
            msg = self._messages[idx]
            # Stop at the active message to keep the window contiguous
            if msg.id == self._active_message_id:
                break
            to_prune.append(msg)
            idx += 1

        return to_prune

    def mark_pruned(self, message_ids: list[str]) -> None:
        """Mark messages as pruned (widgets removed).

        Advances `_visible_start` past consecutive pruned messages at the front
        of the window.

        Args:
            message_ids: IDs of messages that were pruned.
        """
        pruned_set = set(message_ids)
        while (
            self._visible_start < self._visible_end
            and self._messages[self._visible_start].id in pruned_set
        ):
            self._visible_start += 1

    def get_messages_to_hydrate(self, count: int | None = None) -> list[MessageData]:
        """Get messages above the visible window to hydrate.

        Args:
            count: Number of messages to hydrate, or None for `HYDRATE_BUFFER`.

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

    def should_hydrate_above(
        self, scroll_position: float, viewport_height: int
    ) -> bool:
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

        Note:
            Not yet integrated into the scroll handler. Intended for future
            pruning of messages below the viewport when the user scrolls far up.

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
        self._index.clear()
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
            List of all message data (shallow copy).
        """
        return list(self._messages)

    def get_visible_messages(self) -> list[MessageData]:
        """Get messages in the visible window.

        Returns:
            List of visible message data.
        """
        return self._messages[self._visible_start : self._visible_end]
