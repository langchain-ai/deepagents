"""Protocol interfaces for Talon host integrations.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ChannelMessage:
    """Inbound message delivered by a channel adapter.

    Args:
        conversation_id: Stable channel-specific conversation identifier.
        text: Plain text message content for the agent.
        sender_id: Channel-specific sender identifier.
        message_id: Optional channel-specific message identifier.
        metadata: Extra channel values that later adapters may need.
    """

    conversation_id: str
    text: str
    sender_id: str | None = None
    message_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChannelReaction:
    """Inbound reaction delivered by a channel adapter.

    Args:
        conversation_id: Stable channel-specific conversation identifier.
        message_id: Channel-specific message identifier that received the reaction.
        emoji: Provider reaction value.
        sender_id: Channel-specific sender identifier.
        metadata: Extra channel values that later adapters may need.
    """

    conversation_id: str
    message_id: str
    emoji: str
    sender_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChannelStatus:
    """Connection status reported by a channel adapter.

    Args:
        provider: Channel provider name.
        connected: Whether the channel is ready to receive and send messages.
        detail: Optional human-readable status detail for logs and diagnostics.
    """

    provider: str
    connected: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ChannelMedia:
    """Outbound media delivered through a channel adapter.

    Args:
        path: Local file path visible to the channel adapter.
        media_type: Channel-level media category.
        caption: Optional text sent with the media payload.
    """

    path: Path
    media_type: Literal["image", "video", "document", "audio", "voice"]
    caption: str | None = None


@dataclass(frozen=True, slots=True)
class SendResult:
    """Result of a channel send operation.

    Args:
        success: Whether the send completed without error.
        message_id: Channel-specific message identifier when available.
        error: Human-readable error description on failure.
        retryable: Whether a transient failure may succeed on retry.
    """

    success: bool
    message_id: str | None = None
    error: str | None = None
    retryable: bool = False


ToolApprovalDecision = Literal["approve", "reject"]


@dataclass(frozen=True, slots=True)
class ToolApprovalRequest:
    """Tool approval request surfaced to a channel operator.

    Args:
        conversation_id: Conversation whose run is waiting for approval.
        interrupt_id: LangGraph interrupt identifier to resume.
        action_requests: Tool calls awaiting one approve/reject decision.
    """

    conversation_id: str
    interrupt_id: str
    action_requests: Sequence[Mapping[str, object]]


ToolApprovalHandler = Callable[[ToolApprovalRequest], Awaitable[ToolApprovalDecision]]


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Agent invocation request from a channel or scheduler.

    Args:
        conversation_id: Conversation whose turns must be serialized.
        text: User or scheduler prompt passed to the agent.
        metadata: Runtime context supplied by the triggering component.
        approval_handler: Optional callback used by runtimes that surface
            tool approval interrupts over the originating channel.
    """

    conversation_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    approval_handler: ToolApprovalHandler | None = field(
        default=None,
        kw_only=True,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Agent invocation result returned to the host.

    Args:
        text: Text to deliver to the triggering channel. Empty text means the
            runtime has no message to send.
        metadata: Runtime metadata for future observability integrations.
    """

    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


MessageHandler = Callable[[ChannelMessage], Awaitable[None]]
ReactionHandler = Callable[[ChannelReaction], Awaitable[None]]


class ChannelAdapter(Protocol):
    """Transport integration managed by the Talon host."""

    async def start(self) -> None:
        """Start the channel connection."""

    async def stop(self) -> None:
        """Stop the channel connection and release resources."""

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Register the host callback for inbound messages.

        Args:
            handler: Coroutine callback invoked for each inbound channel message.
        """

    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        """Send a message to a conversation.

        Args:
            conversation_id: Channel-specific conversation identifier.
            text: Message content to send.

        Returns:
            Result indicating whether the send succeeded.
        """

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> SendResult:
        """Send media to a conversation.

        Args:
            conversation_id: Channel-specific conversation identifier.
            media: Media payload to deliver.

        Returns:
            Result indicating whether the send succeeded.
        """

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> SendResult:
        """Edit a previously sent channel message.

        Args:
            conversation_id: Channel-specific conversation identifier.
            message_id: Channel-specific message identifier.
            text: Replacement message content.

        Returns:
            Result indicating whether the edit succeeded.
        """

    async def send_typing(self, conversation_id: str) -> None:
        """Send a typing indicator to a conversation.

        Args:
            conversation_id: Channel-specific conversation identifier.
        """

    async def status(self) -> ChannelStatus:
        """Report the channel connection status."""


@runtime_checkable
class ReactionChannelAdapter(Protocol):
    """Optional channel surface for inbound reaction events."""

    def set_reaction_handler(self, handler: ReactionHandler) -> None:
        """Register the host callback for inbound reactions.

        Args:
            handler: Coroutine callback invoked for each inbound channel reaction.
        """


class CronScheduler(Protocol):
    """Scheduler integration managed by the Talon host."""

    async def start(self) -> None:
        """Start the scheduler ticker."""

    async def stop(self) -> None:
        """Stop the scheduler ticker and release resources."""


class AgentRuntime(Protocol):
    """Agent runtime invoked by the Talon host."""

    async def start(self) -> None:
        """Initialize the runtime before the host accepts work."""

    async def stop(self) -> None:
        """Release runtime resources."""

    async def invoke(self, request: AgentRequest) -> AgentResult:
        """Invoke the agent for one serialized conversation turn.

        Args:
            request: Agent request supplied by a channel or scheduler.

        Returns:
            Agent output for the host to route back to the trigger.
        """
