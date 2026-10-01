"""Deferred WhatsApp media is prepared only after Python exposure admission."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.whatsapp import (
    WhatsAppChannel,
    WhatsAppChannelConfig,
    _BridgeTransport,
    _parse_message,
    _WhatsAppBridgeError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.interfaces import ChannelMessage


class AdmissionTransport:
    def __init__(self, envelope: dict[str, object], *, stall: bool = False) -> None:
        self.envelope = envelope
        self.posts: list[str] = []
        self.entered = asyncio.Event()
        self.stall = stall

    async def get(self, path: str) -> object:
        assert path == "/messages"
        return [self.envelope]

    async def post(self, path: str, payload: dict[str, object]) -> object:
        assert payload["preparation_token"] == self.envelope["preparation_token"]
        self.posts.append(path)
        if path == "/prepare":
            self.entered.set()
            if self.stall:
                await asyncio.Event().wait()
            return self.envelope
        return {"success": True}


@pytest.mark.parametrize(
    ("exposure", "sender", "self_chat", "allowed"),
    [
        (ChannelExposure(), "stranger", False, False),
        (ChannelExposure(), "bot", True, True),
        (ChannelExposure(operator_ids=frozenset({"operator"})), "operator", False, True),
        (
            ChannelExposure(mode=ExposureMode.ALLOWLIST, conversations=frozenset({"chat"})),
            "other",
            False,
            True,
        ),
        (
            ChannelExposure(mode=ExposureMode.ALLOWLIST, mention_patterns=("*help*",)),
            "other",
            False,
            True,
        ),
        (ChannelExposure(mode=ExposureMode.OPEN), "other", False, True),
    ],
)
async def test_poll_admits_before_preparing(
    tmp_path: Path, exposure: ChannelExposure, sender: str, *, self_chat: bool, allowed: bool
) -> None:
    envelope: dict[str, object] = {
        "chat_id": "chat",
        "message_id": "input",
        "user_id": sender,
        "text": "help",
        "from_self": self_chat,
        "self_chat": self_chat,
        "has_media": True,
        "preparation_token": "capability",
    }
    transport = AdmissionTransport(envelope)
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(session_dir=tmp_path, exposure=exposure),
        transport=cast("_BridgeTransport", transport),
    )
    delivered: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        delivered.append(message)

    channel.set_message_handler(record)
    polling = asyncio.create_task(channel._poll_messages())
    await asyncio.sleep(0)
    polling.cancel()
    with pytest.raises(asyncio.CancelledError):
        await polling
    assert bool(delivered) is allowed
    assert transport.posts == (["/prepare"] if allowed else ["/discard"])


async def test_cancelled_preparation_discards_capability(tmp_path: Path) -> None:
    envelope: dict[str, object] = {
        "chat_id": "chat",
        "message_id": "input",
        "user_id": "operator",
        "preparation_token": "capability",
        "has_media": True,
    }
    transport = AdmissionTransport(envelope, stall=True)
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path, exposure=ChannelExposure(mode=ExposureMode.OPEN)
        ),
        transport=cast("_BridgeTransport", transport),
    )
    preparing = asyncio.create_task(channel._prepare_message(_parse_message(envelope)))
    await transport.entered.wait()
    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    assert transport.posts == ["/prepare", "/discard"]


async def test_direct_preparation_cannot_bypass_exposure(tmp_path: Path) -> None:
    envelope: dict[str, object] = {
        "chat_id": "chat",
        "message_id": "input",
        "user_id": "stranger",
        "preparation_token": "capability",
        "has_media": True,
    }
    transport = AdmissionTransport(envelope)
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(session_dir=tmp_path),
        transport=cast("_BridgeTransport", transport),
    )
    with pytest.raises(_WhatsAppBridgeError, match="not admitted"):
        await channel._prepare_message(_parse_message(envelope))
    assert transport.posts == ["/discard"]
