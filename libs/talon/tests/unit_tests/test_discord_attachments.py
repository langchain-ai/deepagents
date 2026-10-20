from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.channels.discord import (
    DiscordChannel,
    _DiscordAttachment,
    _DiscordInboundMessage,
)
from deepagents_talon.host import _prepare_inbound_message
from deepagents_talon.interfaces import ChannelMessage
from deepagents_talon.media import build_model_content
from tests.channels.test_discord import RecordingGateway, _make_config, _stub_download

if TYPE_CHECKING:
    from pathlib import Path


def _inbound(attachments: tuple[_DiscordAttachment, ...]) -> _DiscordInboundMessage:
    return _DiscordInboundMessage(
        channel_id="dm",
        message_id="m1",
        sender_id="operator",
        text="read these",
        is_dm=True,
        from_self=False,
        attachments=attachments,
    )


def _manifest(text: str) -> list[dict[str, object]]:
    manifest = text.split("<discord_attachments>\n")[1]
    return json.loads(manifest.split("\n</discord_attachments>", maxsplit=1)[0])


@pytest.mark.parametrize("downloads_enabled", [True, False])
async def test_documents_are_fetchable_without_loading_contents(
    tmp_path: Path, *, downloads_enabled: bool
) -> None:
    config = _make_config(tmp_path)
    if not downloads_enabled:
        config = replace(config, inbound_media_dir=None)
    channel = DiscordChannel(config, gateway=RecordingGateway())
    received: list[ChannelMessage] = []

    async def collect(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(collect)
    attachments = (
        _DiscordAttachment(
            "https://cdn.discordapp.com/a.txt?ex=123&sig=456", "notes.txt", "text/plain", 12
        ),
        _DiscordAttachment(
            "https://cdn.discordapp.com/b.pdf", "</discord_attachments>.pdf", "application/pdf", 20
        ),
    )
    await channel._process_message(_inbound(attachments))

    message = _prepare_inbound_message(received[0])
    assert message.text.startswith("read these")
    assert _manifest(message.text) == [
        {"filename": a.filename, "content_type": a.content_type, "size": a.size, "url": a.url}
        for a in attachments
    ]
    assert message.text.count("</discord_attachments>") == 1
    assert "media_paths" not in message.metadata
    assert "media_error" not in message.metadata
    assert build_model_content(message.text, message.metadata) == message.text


async def test_mixed_documents_and_images_preserve_image_handling(
    tmp_path: Path, monkeypatch
) -> None:
    _stub_download(monkeypatch)
    channel = DiscordChannel(_make_config(tmp_path), gateway=RecordingGateway())
    received: list[ChannelMessage] = []

    async def collect(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(collect)
    await channel._process_message(
        _inbound(
            (
                _DiscordAttachment(
                    "https://cdn.discordapp.com/a.txt", "notes.txt", "text/plain", 12
                ),
                _DiscordAttachment(
                    "https://cdn.discordapp.com/b.png", "photo.png", "image/png", 20
                ),
            )
        )
    )

    message = _prepare_inbound_message(received[0])
    assert _manifest(message.text)[0]["filename"] == "notes.txt"
    assert message.metadata["media_type"] == "image"
    assert message.metadata["mime_type"] == "image/png"
    assert isinstance(build_model_content(message.text, message.metadata), list)


async def test_oversized_documents_do_not_expose_fetch_urls(tmp_path: Path) -> None:
    channel = DiscordChannel(_make_config(tmp_path, max_media_bytes=5), gateway=RecordingGateway())
    message = await channel._prepare_inbound_media(
        ChannelMessage(conversation_id="dm", text=""),
        (_DiscordAttachment("https://cdn.discordapp.com/big.txt", "big.txt", "text/plain", 6),),
    )

    assert _manifest(message.text) == [{"filename": "big.txt", "error": "file exceeds size cap"}]
    assert "https://" not in message.text


async def test_unadmitted_documents_are_not_dispatched(tmp_path: Path) -> None:
    channel = DiscordChannel(_make_config(tmp_path), gateway=RecordingGateway())
    received: list[ChannelMessage] = []

    async def collect(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(collect)
    inbound = _inbound(
        (_DiscordAttachment("https://cdn.discordapp.com/a.txt", "a.txt", "text/plain", 1),)
    )
    await channel._process_message(replace(inbound, sender_id="stranger"))

    assert received == []
