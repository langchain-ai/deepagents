"""Protocol interfaces for Talon host integrations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol


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
class AgentRequest:
    """Agent invocation request from a channel or scheduler.

    Args:
        conversation_id: Conversation whose turns must be serialized.
        text: User or scheduler prompt passed to the agent.
        metadata: Runtime context supplied by the triggering component.
    """

    conversation_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


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

    async def send_message(self, conversation_id: str, text: str) -> None:
        """Send a message to a conversation.

        Args:
            conversation_id: Channel-specific conversation identifier.
            text: Message content to send.
        """

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> None:
        """Edit a previously sent channel message.

        Args:
            conversation_id: Channel-specific conversation identifier.
            message_id: Channel-specific message identifier.
            text: Replacement message content.
        """

    async def status(self) -> ChannelStatus:
        """Report the channel connection status."""


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
