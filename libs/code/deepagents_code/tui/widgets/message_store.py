"""Message store for virtualized chat history.

This module provides data structures and management for message virtualization,
allowing the TUI to handle large message histories efficiently by keeping only
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

DEFAULT_HEIGHT_HINT = 5
"""Estimated terminal rows for a message whose rendered height is unknown."""

MIN_HEIGHT_HINT = 1
"""Smallest useful row estimate for spacer and range-height math."""

_ACTIVE_REASON = "active"
"""Protection reason for the currently-streaming message."""

_LIVE_REASON = "live"
"""Protection reason for a pending/running tool row (default for protect_message)."""


_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {
        "content",
        "tool_status",
        "tool_output",
        "tool_expanded",
        "tool_reject_reason",
        "skill_expanded",
        "rubric_expanded",
        "is_streaming",
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
    """App-status note from the app itself (version info, command feedback)."""

    RUBRIC = "rubric"
    """Rubric grader result with a compact summary and expandable details."""

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

    id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex}")
    """Unique identifier used to match the dataclass to its DOM widget.

    Uses the full 128-bit `uuid4` hex (not a truncated prefix) so IDs stay
    unique across large histories and long sessions; a widget-ID collision
    raises `DuplicateIds` when the widget is mounted.
    """

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

    tool_reject_reason: str | None = None
    """User-supplied reason attached to a HITL reject decision (if any)."""

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

    rubric_details: str | None = None
    """Complete grader details for RUBRIC messages."""

    rubric_expanded: bool = False
    """Whether the grader details are expanded in the UI."""

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
    """Cached rendered widget height in terminal rows, or None if unmeasured.

    Measured after layout by `_measure_message_height` in `app.py` and stored
    via `set_height_hint`. Consumed by `estimate_height`/`range_height` to size
    the transcript spacers and to keep the scroll anchor stable across
    hydrate-above/below. When None (not yet measured), `estimate_height` falls
    back to `DEFAULT_HEIGHT_HINT`. Always `>= MIN_HEIGHT_HINT` once set.
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
        from deepagents_code.tui.widgets.messages import (
            AppMessage,
            AssistantMessage,
            DiffMessage,
            ErrorMessage,
            RubricResultMessage,
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
                widget._deferred_reject_reason = self.tool_reject_reason
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

            case MessageType.RUBRIC:
                widget = RubricResultMessage(
                    self.content,
                    self.rubric_details or "",
                    id=self.id,
                )
                widget._deferred_expanded = self.rubric_expanded
                return widget

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
        from deepagents_code.tui.widgets.messages import (
            AppMessage,
            AssistantMessage,
            DiffMessage,
            ErrorMessage,
            RubricResultMessage,
            SkillMessage,
            SummarizationMessage,
            ToolCallMessage,
            UserMessage,
        )

        widget_id = widget.id or f"msg-{uuid.uuid4().hex}"

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
                tool_reject_reason=widget._reject_reason,
            )

        if isinstance(widget, ErrorMessage):
            return cls(
                type=MessageType.ERROR,
                # `_content` may be `Content` (link spans drop on resume).
                content=str(widget._content),
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

        if isinstance(widget, RubricResultMessage):
            return cls(
                type=MessageType.RUBRIC,
                content=widget._summary,
                id=widget_id,
                rubric_details=widget._details,
                rubric_expanded=widget._expanded,
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
        WINDOW_SIZE: Maximum number of messages to keep mounted in the DOM.

            Trades DOM cost against scroll smoothness. Note each message may
            also mount a timestamp footer, so the live widget count is up to
            ~2x this value. Spacer rows above/below the window preserve full
            scroll geometry, so this only bounds how much is rendered at once,
            not what the user can scroll to.
        HYDRATE_BUFFER: Number of messages to hydrate when scrolling near edge.

            Provides enough buffer to avoid visible loading pauses.
    """

    WINDOW_SIZE: int = 200
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

        self._protection_reasons: dict[str, set[str]] = {}
        """Message ID -> set of reasons it must stay mounted while live.

        A message is protected from virtualization iff it has at least one
        reason. Reasons are independent (`_ACTIVE_REASON` for the streaming
        message, `_LIVE_REASON` for a pending/running tool), so releasing one
        source never revokes another's protection.
        """

        self._active_message_id: str | None = None
        """The single currently-streaming message, mirrored into
        `_protection_reasons` under `_ACTIVE_REASON`. Retained so the
        `is_active`/`set_active_message` API keeps working."""

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
        was_at_tail = self._visible_end == len(self._messages)
        if message.id in self._index:
            logger.warning(
                "Duplicate message ID %r appended; previous entry will be "
                "unreachable via get_message()",
                message.id,
            )
        self._messages.append(message)
        self._index[message.id] = message
        if was_at_tail:
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

        Active messages are never archived. Only the previous active message's
        `_ACTIVE_REASON` is released, so a message also protected for another
        reason (e.g. a live tool) stays protected.

        Args:
            message_id: The ID of the active message, or None to clear.
        """
        if self._active_message_id is not None:
            self.unprotect_message(self._active_message_id, reason=_ACTIVE_REASON)
        self._active_message_id = message_id
        if message_id is not None:
            self.protect_message(message_id, reason=_ACTIVE_REASON)

    def is_active(self, message_id: str) -> bool:
        """Check if a message is the active streaming message.

        Args:
            message_id: The message ID to check.

        Returns:
            True if this is the active message.
        """
        return message_id == self._active_message_id

    def protect_message(self, message_id: str, *, reason: str = _LIVE_REASON) -> None:
        """Keep a live message mounted during window updates.

        Reasons accumulate independently; a message stays protected until every
        reason is released. Idempotent per reason.

        Args:
            message_id: Message ID to protect.
            reason: Why the message is protected. Defaults to a live tool row.
        """
        self._protection_reasons.setdefault(message_id, set()).add(reason)

    def unprotect_message(self, message_id: str, *, reason: str = _LIVE_REASON) -> None:
        """Release one protection reason from a message.

        The message becomes virtualizable only once it has no remaining
        reasons. Releasing a reason the message does not hold is a no-op.

        Args:
            message_id: Message ID to stop protecting.
            reason: Which reason to release. Defaults to a live tool row.
        """
        reasons = self._protection_reasons.get(message_id)
        if reasons is None:
            return
        reasons.discard(reason)
        if not reasons:
            del self._protection_reasons[message_id]

    def is_protected(self, message_id: str) -> bool:
        """Check whether a message is protected from virtualization.

        Returns:
            Whether the message is protected for at least one reason.
        """
        return message_id in self._protection_reasons

    def window_exceeded(self) -> bool:
        """Check if the visible window exceeds the maximum size.

        Returns:
            True if we should prune some widgets.
        """
        return self.visible_count > self.WINDOW_SIZE

    def get_messages_to_prune(self, count: int | None = None) -> list[MessageData]:
        """Get the oldest visible messages that should be pruned.

        Returns a contiguous run of messages from the START of the visible
        window. Stops at the first protected message (the active stream or a
        live tool run) to avoid creating gaps in the visible window (which
        would desync store state from the DOM).

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
            # Stop at the first protected message to keep the window contiguous
            if self.is_protected(msg.id):
                break
            to_prune.append(msg)
            idx += 1

        return to_prune

    def get_messages_to_prune_below(
        self, count: int | None = None
    ) -> list[MessageData]:
        """Get newest visible messages that should be pruned below the viewport.

        Args:
            count: Number of messages to prune, or enough to return to
                `WINDOW_SIZE` when omitted.

        Returns:
            Messages to remove from the bottom of the visible window.
        """
        if count is None:
            count = max(0, self.visible_count - self.WINDOW_SIZE)
        if count <= 0:
            return []

        to_prune: list[MessageData] = []
        idx = self._visible_end - 1
        while len(to_prune) < count and idx >= self._visible_start:
            msg = self._messages[idx]
            if self.is_protected(msg.id):
                break
            to_prune.append(msg)
            idx -= 1
        to_prune.reverse()
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

    def mark_pruned_below(self, message_ids: list[str]) -> None:
        """Mark bottom-window messages as pruned.

        Args:
            message_ids: IDs removed from the bottom of the mounted window.
        """
        pruned_set = set(message_ids)
        while (
            self._visible_end > self._visible_start
            and self._messages[self._visible_end - 1].id in pruned_set
        ):
            self._visible_end -= 1

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

    def get_messages_to_hydrate_below(
        self, count: int | None = None
    ) -> list[MessageData]:
        """Get messages below the visible window to hydrate.

        Args:
            count: Number of messages to hydrate; defaults to `HYDRATE_BUFFER`
                when omitted.

        Returns:
            Messages below the mounted window, in order.
        """
        if count is None:
            count = self.HYDRATE_BUFFER
        if self._visible_end >= len(self._messages):
            return []
        hydrate_end = min(len(self._messages), self._visible_end + count)
        return self._messages[self._visible_end : hydrate_end]

    def mark_hydrated_below(self, count: int) -> None:
        """Mark that messages below were hydrated.

        Args:
            count: Number of messages that were hydrated below the window.
        """
        self._visible_end = min(len(self._messages), self._visible_end + count)

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

    def should_hydrate_below(
        self,
        scroll_position: float,
        viewport_height: int,
        bottom_spacer_top: int,
    ) -> bool:
        """Check if we should hydrate messages below the current view.

        Args:
            scroll_position: Current scroll Y position.
            viewport_height: Height of the viewport.
            bottom_spacer_top: Estimated row where the bottom spacer begins.

        Returns:
            True if the viewport is near the bottom spacer.
        """
        if not self.has_messages_below:
            return False
        viewport_bottom = scroll_position + viewport_height
        distance_from_bottom_spacer = bottom_spacer_top - viewport_bottom
        threshold = viewport_height * 2
        return distance_from_bottom_spacer < threshold

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._index.clear()
        self._visible_start = 0
        self._visible_end = 0
        self._protection_reasons.clear()
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

    def set_height_hint(self, message_id: str, rows: int) -> bool:
        """Update a measured message height, clamped to `MIN_HEIGHT_HINT`.

        The single write path for `height_hint`; `height_hint` is intentionally
        excluded from `update_message`'s allowlist so every write clamps here.

        Args:
            message_id: Message ID to update.
            rows: Rendered height in terminal rows.

        Returns:
            Whether the message existed and was updated.
        """
        msg_data = self._index.get(message_id)
        if msg_data is None:
            return False
        msg_data.height_hint = max(MIN_HEIGHT_HINT, rows)
        return True

    def invalidate_height_hints(self, *, scale: float | None = None) -> None:
        """Invalidate or scale cached height hints after terminal reflow.

        Args:
            scale: Optional multiplier used when terminal width changes. When
                omitted, all cached hints are cleared.
        """
        for msg in self._messages:
            if msg.height_hint is None:
                continue
            if scale is None:
                msg.height_hint = None
            else:
                msg.height_hint = max(MIN_HEIGHT_HINT, round(msg.height_hint * scale))

    @staticmethod
    def estimate_height(message: MessageData) -> int:
        """Return the best available row estimate for a message."""
        if message.height_hint is None:
            return DEFAULT_HEIGHT_HINT
        return max(MIN_HEIGHT_HINT, message.height_hint)

    def range_height(self, start: int, end: int) -> int:
        """Estimate rows in `[start:end]`.

        Returns:
            Estimated row count in the range.
        """
        bounded_start = max(0, min(start, len(self._messages)))
        bounded_end = max(bounded_start, min(end, len(self._messages)))
        return sum(
            self.estimate_height(msg)
            for msg in self._messages[bounded_start:bounded_end]
        )
