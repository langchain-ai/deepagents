from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_talon.interfaces import ChannelMedia, ChannelMessage, ChannelStatus

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

    async def send_message(self, conversation_id: str, text: str) -> None:
        self.sent.append((conversation_id, text))

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> None:
        self.media.append((conversation_id, media))
        self.sent.append((conversation_id, f"{media.media_type}:{media.path}"))

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> None:
        self.sent.append((conversation_id, f"{message_id}:{text}"))

    async def status(self) -> ChannelStatus:
        return self.status_report

    async def receive(self, text: str, *, conversation_id: str = "chat") -> None:
        """Deliver an inbound message to the registered handler."""
        if self.handler is None:
            msg = "channel handler was not registered"
            raise AssertionError(msg)
        await self.handler(ChannelMessage(conversation_id=conversation_id, text=text))
