from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import pytest

from deepagents_talon.channels.base import (
    ChannelExposure,
    ChannelMediaError,
    ExposureMode,
)
from deepagents_talon.channels.whatsapp import (
    DEFAULT_WHATSAPP_MAX_MEDIA_BYTES,
    WhatsAppChannel,
    WhatsAppChannelConfig,
    _bridge_script_path,
    _BridgeTransport,
    _WhatsAppBridgeError,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import ChannelMedia, ChannelMessage, ChannelReaction

if TYPE_CHECKING:
    import urllib.request


class RecordingTransport:
    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        self.messages = messages or []
        self.posts: list[tuple[str, dict[str, object]]] = []

    async def get(self, path: str) -> object:
        if path == "/messages":
            messages = self.messages
            self.messages = []
            return messages
        if path == "/health":
            return {"status": "connected", "botId": "bot"}
        msg = f"unexpected get path: {path}"
        raise AssertionError(msg)

    async def post(self, path: str, payload: dict[str, object]) -> object:
        self.posts.append((path, payload))
        return {"success": True, "message_id": "sent"}


class DelayedHealthTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def get(self, path: str) -> object:
        assert path == "/health"
        self.calls += 1
        if self.calls == 1:
            msg = "bridge not listening yet"
            raise _WhatsAppBridgeError(msg)
        return {"status": "qr_pending", "botId": None}


class JsonResponse:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"success": True}).encode()


def test_config_from_talon_env_maps_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_WHATSAPP_EXPOSURE": "allowlist",
            "DEEPAGENTS_TALON_WHATSAPP_ALLOWLIST_CHATS": "chat-1, chat-2",
            "DEEPAGENTS_TALON_WHATSAPP_MENTION_PATTERNS": "@agent *",
            "DEEPAGENTS_TALON_WHATSAPP_OPERATOR_ID": "operator",
            "DEEPAGENTS_TALON_WHATSAPP_BOT_HEADER": "test bot",
        },
        base_home=tmp_path,
    )

    whatsapp = WhatsAppChannelConfig.from_talon_config(config)

    assert whatsapp.session_dir == tmp_path / "assistant" / "channels" / "whatsapp"
    assert whatsapp.inbound_media_dir == tmp_path / "assistant" / "media" / "inbound" / "whatsapp"
    assert whatsapp.outbound_media_dir == tmp_path
    assert whatsapp.max_media_bytes == DEFAULT_WHATSAPP_MAX_MEDIA_BYTES
    assert whatsapp.exposure == ChannelExposure(
        mode=ExposureMode.ALLOWLIST,
        operator_ids=frozenset({"operator"}),
        conversations=frozenset({"chat-1", "chat-2"}),
        mention_patterns=("@agent *",),
    )
    assert whatsapp.bot_header == "test bot"


def test_config_from_talon_env_maps_multiple_operator_ids(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_WHATSAPP_OPERATOR_ID": "operator, backup-operator",
        },
        base_home=tmp_path,
    )

    whatsapp = WhatsAppChannelConfig.from_talon_config(config)

    assert whatsapp.exposure == ChannelExposure(
        operator_ids=frozenset({"operator", "backup-operator"}),
    )
    assert whatsapp.exposure.allows(
        ChannelMessage(conversation_id="chat", text="hi", sender_id="operator")
    )
    assert whatsapp.exposure.allows(
        ChannelMessage(conversation_id="chat", text="hi", sender_id="backup-operator")
    )
    assert not whatsapp.exposure.allows(
        ChannelMessage(conversation_id="chat", text="hi", sender_id="other")
    )


def test_config_from_talon_env_maps_max_media_bytes(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_MAX_MEDIA_BYTES": "12345",
        },
        base_home=tmp_path,
    )

    whatsapp = WhatsAppChannelConfig.from_talon_config(config)

    assert whatsapp.max_media_bytes == 12345


def test_config_from_talon_env_clamps_large_max_media_bytes(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_MAX_MEDIA_BYTES": str(DEFAULT_WHATSAPP_MAX_MEDIA_BYTES + 1),
        },
        base_home=tmp_path,
    )

    whatsapp = WhatsAppChannelConfig.from_talon_config(config)

    assert whatsapp.max_media_bytes == DEFAULT_WHATSAPP_MAX_MEDIA_BYTES


def test_config_from_talon_env_accepts_explicit_bridge_token(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_WHATSAPP_BRIDGE_TOKEN": "test-token",
        },
        base_home=tmp_path,
    )

    whatsapp = WhatsAppChannelConfig.from_talon_config(config)

    assert whatsapp.bridge_token == "test-token"  # noqa: S105  # inert test token


def test_bridge_transport_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, *, timeout: float) -> JsonResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return JsonResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transport = _BridgeTransport(
        base_url="http://127.0.0.1:3000",
        timeout=2,
        token="test-token",  # noqa: S106  # inert test token
    )

    result = transport._request("GET", "/health", None)

    request = cast("urllib.request.Request", captured["request"])
    assert result == {"success": True}
    assert captured["timeout"] == 2
    assert request.get_header("Authorization") == "Bearer test-token"


def test_config_rejects_open_exposure_without_acknowledgement(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_WHATSAPP_EXPOSURE": "open",
        },
        base_home=tmp_path,
    )

    with pytest.raises(ValueError, match="allow-arbitrary-senders"):
        WhatsAppChannelConfig.from_talon_config(config)


def test_config_accepts_open_exposure_with_acknowledgement(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_WHATSAPP_EXPOSURE": "open",
            "DEEPAGENTS_TALON_WHATSAPP_OPEN_ACK": "allow-arbitrary-senders",
        },
        base_home=tmp_path,
    )

    whatsapp = WhatsAppChannelConfig.from_talon_config(config)

    assert whatsapp.exposure.mode == ExposureMode.OPEN


async def test_channel_polls_and_dispatches_allowed_messages(tmp_path: Path) -> None:
    transport = RecordingTransport(
        messages=[
            {
                "text": "allowed",
                "chat_id": "chat",
                "user_id": "operator",
                "message_id": "message-1",
                "message_type": "chat",
            },
            {
                "text": "blocked",
                "chat_id": "chat",
                "user_id": "other",
                "message_id": "message-2",
                "message_type": "chat",
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            exposure=ChannelExposure(operator_ids=frozenset({"operator"})),
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert [message.text for message in received] == ["allowed"]
    assert received[0].metadata["provider"] == "whatsapp"


async def test_channel_polls_and_dispatches_reactions(tmp_path: Path) -> None:
    transport = RecordingTransport(
        messages=[
            {
                "event_type": "reaction",
                "chat_id": "chat",
                "message_id": "approval-message",
                "sender_id": "operator",
                "emoji": "👍",
                "raw_reaction": {"orphan": False},
            },
            {
                "event_type": "message",
                "text": "",
                "chat_id": "chat",
                "user_id": "operator",
                "message_id": "should-not-be-reaction",
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            exposure=ChannelExposure(operator_ids=frozenset({"operator"})),
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    reactions: list[ChannelReaction] = []
    messages: list[ChannelMessage] = []

    async def record_reaction(reaction: ChannelReaction) -> None:
        reactions.append(reaction)

    async def record_message(message: ChannelMessage) -> None:
        messages.append(message)

    channel.set_reaction_handler(record_reaction)
    channel.set_message_handler(record_message)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert reactions == [
        ChannelReaction(
            conversation_id="chat",
            message_id="approval-message",
            emoji="👍",
            sender_id="operator",
            metadata={
                "provider": "whatsapp",
                "chat_type": None,
                "chat_name": None,
                "raw_reaction": {"orphan": False},
                "from_self": False,
            },
        )
    ]
    assert [message.message_id for message in messages] == ["should-not-be-reaction"]


async def test_channel_polls_self_reactions_without_operator_id(tmp_path: Path) -> None:
    transport = RecordingTransport(
        messages=[
            {
                "event_type": "reaction",
                "chat_id": "chat",
                "message_id": "approval-message",
                "sender_id": "operator",
                "emoji": "👍",
                "fromSelf": True,
            },
            {
                "event_type": "reaction",
                "chat_id": "chat",
                "message_id": "other-message",
                "sender_id": "other",
                "emoji": "👍",
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    reactions: list[ChannelReaction] = []

    async def record(reaction: ChannelReaction) -> None:
        reactions.append(reaction)

    channel.set_reaction_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert [reaction.message_id for reaction in reactions] == ["approval-message"]
    assert reactions[0].metadata["from_self"] is True


async def test_channel_keeps_legacy_message_entries_compatible(tmp_path: Path) -> None:
    transport = RecordingTransport(
        messages=[
            {
                "text": "legacy",
                "chat_id": "chat",
                "user_id": "operator",
                "message_id": "message-1",
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            exposure=ChannelExposure(operator_ids=frozenset({"operator"})),
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert [message.text for message in received] == ["legacy"]


async def test_channel_drops_unknown_whatsapp_queue_event_type(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = RecordingTransport(
        messages=[
            {
                "event_type": "typing",
                "chat_id": "chat",
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            exposure=ChannelExposure(mode=ExposureMode.OPEN),
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )

    with caplog.at_level(logging.WARNING, logger="deepagents_talon.channels.whatsapp"):
        await channel.start()
        await asyncio.sleep(0)
        await channel.stop()

    assert "unknown event_type" in caplog.text


async def test_channel_ignores_reaction_removal_events(tmp_path: Path) -> None:
    transport = RecordingTransport(
        messages=[
            {
                "event_type": "reaction",
                "chat_id": "chat",
                "message_id": "approval-message",
                "sender_id": "operator",
                "emoji": "👍",
                "action": "remove",
            },
            {
                "event_type": "reaction",
                "chat_id": "chat",
                "message_id": "approval-message",
                "sender_id": "operator",
                "reaction": "",
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            exposure=ChannelExposure(mode=ExposureMode.OPEN),
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    reactions: list[ChannelReaction] = []

    async def record(reaction: ChannelReaction) -> None:
        reactions.append(reaction)

    channel.set_reaction_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert reactions == []


async def test_channel_applies_exposure_to_reactions(tmp_path: Path) -> None:
    transport = RecordingTransport(
        messages=[
            {
                "event_type": "reaction",
                "chat_id": "chat",
                "message_id": "approval-message",
                "sender_id": "other",
                "emoji": "👍",
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            exposure=ChannelExposure(operator_ids=frozenset({"operator"})),
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    reactions: list[ChannelReaction] = []

    async def record(reaction: ChannelReaction) -> None:
        reactions.append(reaction)

    channel.set_reaction_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert reactions == []


async def test_channel_polls_self_messages_without_operator_id(tmp_path: Path) -> None:
    transport = RecordingTransport(
        messages=[
            {
                "body": "self chat",
                "chatId": "chat",
                "senderId": "operator",
                "messageId": "message-1",
                "messageType": "chat",
                "fromSelf": True,
            },
            {
                "body": "other",
                "chatId": "chat",
                "senderId": "other",
                "messageId": "message-2",
                "messageType": "chat",
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert [message.text for message in received] == ["self chat"]
    assert received[0].metadata["from_self"] is True


async def test_channel_parses_inbound_media_payload(tmp_path: Path) -> None:
    media = tmp_path / "voice.ogg"
    media.write_bytes(b"voice")
    transport = RecordingTransport(
        messages=[
            {
                "body": "",
                "chatId": "chat@lid",
                "chatIdFrom": "123@s.whatsapp.net",
                "senderId": "operator",
                "messageId": "message-1",
                "messageType": "ptt",
                "mediaType": "voice",
                "mediaPaths": [str(media)],
                "mediaMimeTypes": ["audio/ogg"],
                "fromSelf": True,
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert received[0].conversation_id == "chat@lid"
    assert received[0].metadata["chat_id_from"] == "123@s.whatsapp.net"
    assert received[0].metadata["media_paths"] == [str(media)]
    assert received[0].metadata["media_mime_types"] == ["audio/ogg"]
    assert received[0].metadata["voice_path"] == str(media)


async def test_channel_filters_oversized_inbound_media_payload(tmp_path: Path) -> None:
    media = tmp_path / "voice.ogg"
    media.write_bytes(b"voice")
    transport = RecordingTransport(
        messages=[
            {
                "body": "oversized",
                "chatId": "chat@lid",
                "senderId": "operator",
                "messageId": "message-1",
                "messageType": "ptt",
                "mediaType": "voice",
                "mediaPaths": [str(media)],
                "mediaMimeTypes": ["audio/ogg"],
                "fromSelf": True,
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            max_media_bytes=1,
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert received[0].text == "oversized"
    assert received[0].metadata["has_media"] is False
    assert "media_error" in received[0].metadata


async def test_channel_normalizes_ptt_payload_as_voice(tmp_path: Path) -> None:
    media = tmp_path / "voice.ogg"
    media.write_bytes(b"voice")
    transport = RecordingTransport(
        messages=[
            {
                "body": "",
                "chatId": "chat@lid",
                "messageType": "ptt",
                "mediaType": "document",
                "mediaPaths": [str(media)],
                "mediaMimeTypes": ["application/octet-stream"],
                "fromSelf": True,
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert received[0].metadata["media_type"] == "voice"
    assert received[0].metadata["voice_path"] == str(media)


async def test_channel_normalizes_audio_mime_payload_as_voice(tmp_path: Path) -> None:
    media = tmp_path / "audio.bin"
    media.write_bytes(b"voice")
    transport = RecordingTransport(
        messages=[
            {
                "body": "",
                "chatId": "chat@lid",
                "messageType": "document",
                "mediaType": "document",
                "mediaPaths": [str(media)],
                "mediaMimeTypes": ["audio/ogg; codecs=opus"],
                "fromSelf": True,
            },
        ],
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            poll_interval_seconds=60,
            health_interval_seconds=60,
        ),
        transport=cast("_BridgeTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert received[0].metadata["media_type"] == "voice"
    assert received[0].metadata["voice_path"] == str(media)


async def test_channel_sends_chunked_formatted_text(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(session_dir=tmp_path),
        transport=cast("_BridgeTransport", transport),
    )

    await channel.send_message("chat", "**bold** " + ("x" * 4096))

    assert transport.posts[0] == (
        "/send",
        {"chatId": "chat", "text": "*deepagents bot*\n*bold*"},
    )
    assert transport.posts[1][0] == "/send"
    assert len(cast("str", transport.posts[1][1]["text"])) <= 4096
    assert cast("str", transport.posts[1][1]["text"]).startswith("*deepagents bot*\n")


async def test_channel_sends_media_and_edits_messages(tmp_path: Path) -> None:
    transport = RecordingTransport()
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    media_dir = tmp_path / "bridge-media"
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(session_dir=tmp_path, inbound_media_dir=media_dir),
        transport=cast("_BridgeTransport", transport),
    )

    await channel.send_media(
        "chat", ChannelMedia(path=image, media_type="image", caption="caption")
    )
    await channel.edit_message("chat", "message", "# Updated")

    staged = Path(cast("str", transport.posts[0][1]["filePath"]))
    assert staged.parent == media_dir
    assert await asyncio.to_thread(staged.read_bytes) == b"image"
    assert transport.posts == [
        (
            "/send-media",
            {
                "chatId": "chat",
                "filePath": str(staged),
                "mediaType": "image",
                "caption": "*deepagents bot*\ncaption",
            },
        ),
        (
            "/edit",
            {
                "chatId": "chat",
                "messageId": "message",
                "content": "*deepagents bot*\nUpdated",
            },
        ),
    ]


async def test_channel_rejects_media_outside_configured_outbound_root(tmp_path: Path) -> None:
    transport = RecordingTransport()
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(session_dir=tmp_path, outbound_media_dir=root),
        transport=cast("_BridgeTransport", transport),
    )

    with pytest.raises(ChannelMediaError, match="escapes outbound root"):
        await channel.send_media("chat", ChannelMedia(path=outside, media_type="image"))

    assert transport.posts == []


async def test_channel_forwards_max_media_bytes_to_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        stdout = None
        stderr = None
        returncode = None

    async def fake_create_subprocess_exec(
        *command: str,
        env: dict[str, str],
        stdout: object,
        stderr: object,
    ) -> FakeProcess:
        captured["command"] = command
        captured["env"] = env
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return FakeProcess()

    async def fake_wait_for_bridge() -> None:
        return None

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    config = WhatsAppChannelConfig(
        session_dir=tmp_path,
        bridge_command=("node", "bridge.js"),
        max_media_bytes=12345,
    )
    channel = WhatsAppChannel(config)
    monkeypatch.setattr(channel, "_wait_for_bridge", fake_wait_for_bridge)

    await channel._start_bridge()

    env = cast("dict[str, str]", captured["env"])
    assert env["WHATSAPP_MAX_MEDIA_BYTES"] == "12345"


async def test_channel_waits_for_bridge_health_before_polling(tmp_path: Path) -> None:
    transport = DelayedHealthTransport()
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(session_dir=tmp_path),
        transport=cast("_BridgeTransport", transport),
    )

    await channel._wait_for_bridge()

    assert transport.calls == 2
    assert (await channel.status()).detail == "qr_pending"


async def test_channel_forwards_bridge_output_to_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = asyncio.StreamReader()
    channel = WhatsAppChannel(WhatsAppChannelConfig(session_dir=tmp_path))

    with caplog.at_level(logging.INFO, logger="deepagents_talon.channels.whatsapp"):
        task = asyncio.create_task(channel._forward_bridge_output(stream, logging.INFO))
        stream.feed_data(b"Scan this QR code to pair WhatsApp:\n")
        stream.feed_eof()
        await task

    assert "WhatsApp bridge: Scan this QR code to pair WhatsApp:" in caplog.text


def test_bridge_script_is_packaged() -> None:
    assert _bridge_script_path().name == "bridge.js"
    assert _bridge_script_path().is_file()


def test_bridge_script_registers_reaction_queue_entries() -> None:
    script = _bridge_script_path().read_text(encoding="utf-8")

    assert 'client.on("message_reaction"' in script
    assert 'event_type: "message"' in script
    assert 'event_type: "reaction"' in script
    assert "raw_reaction" in script
    assert "from_self: fromSelf" in script
    assert "const senderId = fromSelf && botId ? botId : rawSenderId;" in script
    assert "const sentMessageChatIds = new Map();" in script
    assert "sentMessageChatIds.get(messageId) || reactionChatId(reaction)" in script
