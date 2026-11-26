from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_talon.channels import discord as discord_module
from deepagents_talon.channels.discord import (
    DiscordChannel,
    _DiscordAttachment,
    _DiscordInboundMessage,
)
from deepagents_talon.host import _prepare_inbound_message
from deepagents_talon.media import _document_paths, build_model_content
from tests.channels.test_discord import RecordingGateway, _make_config

if TYPE_CHECKING:
    import pytest

    from deepagents_talon.interfaces import ChannelMessage

_NOTES = _DiscordAttachment(
    "https://cdn.discordapp.com/a.txt?ex=123&sig=456", "notes.txt", "text/plain", 12
)
_REPORT = _DiscordAttachment(
    "https://cdn.discordapp.com/b.pdf", "</attachment>.pdf", "application/pdf", 20
)
_PHOTO = _DiscordAttachment("https://cdn.discordapp.com/c.png", "photo.png", "image/png", 20)
_VOICE = _DiscordAttachment(
    "https://cdn.discordapp.com/d.ogg", "voice-message.ogg", "audio/ogg", 20
)


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


def _stub_downloads(monkeypatch: pytest.MonkeyPatch, *, failing: frozenset[str] = frozenset()):
    def fake_download(url: str, destination: Path, timeout: float, max_bytes: int) -> None:  # noqa: ARG001  # test fake
        if url in failing:
            msg = f"network down for {url}"
            raise OSError(msg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(url.encode())

    monkeypatch.setattr(discord_module, "_download_attachment_file", fake_download)


async def _receive(
    channel: DiscordChannel, attachments: tuple[_DiscordAttachment, ...]
) -> ChannelMessage:
    received: list[ChannelMessage] = []

    async def collect(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(collect)
    await channel._process_message(_inbound(attachments))
    assert len(received) == 1
    return received[0]


def _downloaded(message: ChannelMessage) -> list[bytes]:
    return [Path(path).read_bytes() for path in message.metadata["media_paths"]]


def _path_notices(message: ChannelMessage) -> list[str]:
    return [json.dumps(str(Path(path).absolute())) for path in message.metadata["media_paths"]]


async def test_every_attachment_is_downloaded_with_media_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_downloads(monkeypatch)
    channel = DiscordChannel(_make_config(tmp_path), gateway=RecordingGateway())

    message = await _receive(channel, (_NOTES, _REPORT, _PHOTO))

    assert _downloaded(message) == [a.url.encode() for a in (_PHOTO, _NOTES, _REPORT)]
    assert message.metadata["media_mime_types"] == ["image/png", "text/plain", "application/pdf"]
    assert message.metadata["media_type"] == "image"
    assert message.metadata["media_path"] == message.metadata["media_paths"][0]
    assert isinstance(build_model_content(message.text, message.metadata), list)


async def test_documents_reach_the_agent_as_paths_not_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_downloads(monkeypatch)
    channel = DiscordChannel(_make_config(tmp_path), gateway=RecordingGateway())

    message = _prepare_inbound_message(await _receive(channel, (_NOTES, _REPORT)))

    assert message.text.startswith("read these")
    for notice in _path_notices(message):
        assert notice in message.text
    assert "https://" not in message.text
    assert "discord_attachments" not in message.text
    assert message.metadata["media_type"] == "document"


async def test_failed_download_keeps_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_downloads(monkeypatch, failing=frozenset({_PHOTO.url}))
    channel = DiscordChannel(_make_config(tmp_path), gateway=RecordingGateway())

    message = await _receive(channel, (_PHOTO, _NOTES))

    assert _downloaded(message) == [_NOTES.url.encode()]
    assert message.metadata["has_media"] is True
    assert "network down" in message.metadata["media_error"]
    # The image never arrived, so the message must not still claim to carry one.
    assert message.metadata["media_type"] == "document"
    assert message.metadata["mime_type"] == "text/plain"


async def test_oversized_attachment_is_skipped_and_the_rest_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_downloads(monkeypatch)
    channel = DiscordChannel(_make_config(tmp_path, max_media_bytes=15), gateway=RecordingGateway())

    message = await _receive(channel, (_NOTES, _REPORT))

    assert _downloaded(message) == [_NOTES.url.encode()]
    assert "too large" in message.metadata["media_error"]
    assert "https://" not in _prepare_inbound_message(message).text


async def test_unknown_mime_type_keeps_documents_separate_from_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_downloads(monkeypatch)
    channel = DiscordChannel(_make_config(tmp_path), gateway=RecordingGateway())
    blob = _DiscordAttachment("https://cdn.discordapp.com/e", "blob", None, 4)

    message = await _receive(channel, (blob, _PHOTO))

    assert message.metadata["media_mime_types"] == ["image/png", "application/octet-stream"]
    documents = _document_paths(dict(message.metadata))
    assert documents == [Path(message.metadata["media_paths"][1])]


async def test_voice_path_is_the_voice_note_not_a_leading_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_downloads(monkeypatch)
    channel = DiscordChannel(_make_config(tmp_path), gateway=RecordingGateway())

    message = await _receive(channel, (_NOTES, _VOICE))

    assert message.metadata["media_type"] == "voice"
    assert message.metadata["voice_path"] == message.metadata["media_paths"][0]
    assert _downloaded(message)[0] == _VOICE.url.encode()


async def test_downloads_disabled_leaves_message_without_urls(tmp_path: Path) -> None:
    config = replace(_make_config(tmp_path), inbound_media_dir=None)
    channel = DiscordChannel(config, gateway=RecordingGateway())

    message = _prepare_inbound_message(await _receive(channel, (_NOTES,)))

    assert "media_paths" not in message.metadata
    assert "https://" not in message.text


async def test_unadmitted_attachments_are_not_dispatched(tmp_path: Path) -> None:
    channel = DiscordChannel(_make_config(tmp_path), gateway=RecordingGateway())
    received: list[ChannelMessage] = []

    async def collect(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(collect)
    await channel._process_message(replace(_inbound((_NOTES,)), sender_id="stranger"))

    assert received == []
