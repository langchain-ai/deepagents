from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_talon.interfaces import (
    ChannelMedia,
    ChannelMessage,
    ChannelReaction,
    ChannelStatus,
    SendResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class RecordingChannel:
    """Shared test double for channel adapters.

    Tracks sent messages and media, and supports injecting inbound messages
    via ``receive``.
    """

    def __init__(self, provider: str = "test") -> None:
        self.provider = provider
        self.handler: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self.reaction_handler: Callable[[ChannelReaction], Awaitable[None]] | None = None
        self.next_message_id: str | None = None
        self.started = False
        self.stopped = False
        self.sent: list[tuple[str, str]] = []
        self.media: list[tuple[str, ChannelMedia]] = []
        self.status_report = ChannelStatus(provider=provider, connected=True, detail="connected")

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def set_message_handler(self, handler: Callable[[ChannelMessage], Awaitable[None]]) -> None:
        self.handler = handler

    def set_reaction_handler(
        self,
        handler: Callable[[ChannelReaction], Awaitable[None]],
    ) -> None:
        self.reaction_handler = handler

    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        self.sent.append((conversation_id, text))
        return SendResult(success=True, message_id=self.next_message_id)

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> SendResult:
        self.media.append((conversation_id, media))
        self.sent.append((conversation_id, f"{media.media_type}:{media.path}"))
        return SendResult(success=True)

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> SendResult:
        self.sent.append((conversation_id, f"{message_id}:{text}"))
        return SendResult(success=True)

    async def send_typing(self, conversation_id: str) -> None:
        pass

    async def status(self) -> ChannelStatus:
        return self.status_report

    async def receive(self, text: str, *, conversation_id: str = "chat") -> None:
        """Deliver an inbound message to the registered handler."""
        if self.handler is None:
            msg = "channel handler was not registered"
            raise AssertionError(msg)
        await self.handler(ChannelMessage(conversation_id=conversation_id, text=text))

    async def receive_reaction(
        self,
        emoji: str,
        *,
        conversation_id: str = "chat",
        message_id: str = "message",
        sender_id: str | None = None,
    ) -> None:
        """Deliver an inbound reaction to the registered handler."""
        if self.reaction_handler is None:
            msg = "reaction handler was not registered"
            raise AssertionError(msg)
        await self.reaction_handler(
            ChannelReaction(
                conversation_id=conversation_id,
                message_id=message_id,
                emoji=emoji,
                sender_id=sender_id,
            )
        )
