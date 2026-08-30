from __future__ import annotations

from pathlib import Path

import pytest

from deepagents_talon.channels import discord as discord_module
from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.discord import (
    DiscordChannel,
    DiscordChannelConfig,
    InboundMessageCallback,
    InboundReactionCallback,
    _DiscordAttachment,
    _DiscordInboundMessage,
    _DiscordInboundReaction,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import ChannelMedia


class RecordingGateway:
    """Fake `_DiscordGateway` used to test `DiscordChannel` without a real Gateway connection."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.sent_text = []
        self.sent_files = []
        self.edits = []
        self.typing = []
        self.next_message_id = "1"
        self.bot_id = "bot-1"
        self._handle_message: InboundMessageCallback | None = None
        self._handle_reaction: InboundReactionCallback | None = None

    async def start(self, *, handle_message, handle_reaction):
        self.started = True
        self._handle_message = handle_message
        self._handle_reaction = handle_reaction

    async def stop(self):
        self.stopped = True

    async def send_message(self, channel_id, text):
        self.sent_text.append((channel_id, text))
        return self.next_message_id

    async def send_file(self, channel_id, file_path, *, content):
        self.sent_files.append((channel_id, file_path, content))
        return self.next_message_id

    async def edit_message(self, channel_id, message_id, text):
        self.edits.append((channel_id, message_id, text))

    async def trigger_typing(self, channel_id):
        self.typing.append(channel_id)

    async def deliver_message(self, inbound):
        assert self._handle_message is not None
        await self._handle_message(inbound)

    async def deliver_reaction(self, inbound):
        assert self._handle_reaction is not None
        await self._handle_reaction(inbound)


class FailingTypingGateway(RecordingGateway):
    async def trigger_typing(self, channel_id):  # noqa: ARG002  # test fake
        msg = "boom"
        raise RuntimeError(msg)


def _make_config(
    tmp_path: Path,
    *,
    exposure: ChannelExposure | None = None,
    allowed_user_ids: frozenset[str] | None = None,
    max_media_bytes: int = 10_000_000,
) -> DiscordChannelConfig:
    return DiscordChannelConfig(
        bot_token="test-token",  # noqa: S106  # inert test token
        inbound_media_dir=tmp_path / "inbound",
        outbound_media_dir=tmp_path,
        exposure=exposure
        or ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"operator"})),
        allowed_user_ids=allowed_user_ids or frozenset(),
        max_media_bytes=max_media_bytes,
    )


def _talon_config(tmp_path: Path, env: dict[str, str]) -> TalonConfig:
    return TalonConfig.from_env({"AGENT_ASSISTANT_ID": "assistant", **env}, base_home=tmp_path)


def _stub_download(monkeypatch, *, content: bytes = b"data"):
    def fake_download(url, destination, timeout, max_bytes):  # noqa: ARG001  # test fake
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    monkeypatch.setattr(discord_module, "_download_attachment_file", fake_download)


def _stub_failing_download(monkeypatch):
    def fake_download(url, destination, timeout, max_bytes):  # noqa: ARG001  # test fake
        msg = "network down"
        raise OSError(msg)

    monkeypatch.setattr(discord_module, "_download_attachment_file", fake_download)


def _collector():
    received = []

    async def handler(value):
        received.append(value)

    return received, handler


# --- config tests -----------------------------------------------------------


def test_from_talon_config_requires_bot_token(tmp_path):
    config = _talon_config(tmp_path, {})

    with pytest.raises(ValueError, match="bot token"):
        DiscordChannelConfig.from_talon_config(config)


def test_from_talon_config_builds_defaults(tmp_path):
    config = _talon_config(
        tmp_path,
        {
            "DEEPAGENTS_TALON_DISCORD_BOT_TOKEN": "abc",
            "DEEPAGENTS_TALON_DISCORD_OPERATOR_ID": "999",
        },
    )

    result = DiscordChannelConfig.from_talon_config(config)

    assert result.bot_token == "abc"  # noqa: S105  # inert test token
    assert result.inbound_media_dir == config.inbound_media_dir / "discord"
    assert result.exposure.mode == ExposureMode.SELF
    assert result.exposure.operator_ids == frozenset({"999"})
    assert result.allowed_user_ids == frozenset()
    assert result.request_timeout_seconds == discord_module.DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_from_talon_config_reads_allowlist_users(tmp_path):
    config = _talon_config(
        tmp_path,
        {
            "DEEPAGENTS_TALON_DISCORD_BOT_TOKEN": "abc",
            "DEEPAGENTS_TALON_DISCORD_OPERATOR_ID": "999",
            "DEEPAGENTS_TALON_DISCORD_ALLOWLIST_USERS": "1, 2 ,3",
        },
    )

    result = DiscordChannelConfig.from_talon_config(config)

    assert result.allowed_user_ids == frozenset({"1", "2", "3"})


def test_from_talon_config_self_exposure_requires_operator_id(tmp_path):
    config = _talon_config(tmp_path, {"DEEPAGENTS_TALON_DISCORD_BOT_TOKEN": "abc"})

    with pytest.raises(ValueError, match="requires"):
        DiscordChannelConfig.from_talon_config(config)


# --- lifecycle tests ---------------------------------------------------------


async def test_start_and_stop_toggle_status(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)

    await channel.start()

    assert gateway.started
    assert (await channel.status()).connected is True

    await channel.stop()

    assert gateway.stopped
    assert (await channel.status()).connected is False


# --- outbound text/media tests ----------------------------------------------


async def test_send_message_splits_long_text_into_multiple_messages(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    text = "a" * 2500

    result = await channel.send_message("channel-1", text)

    assert result.success
    assert len(gateway.sent_text) == 2
    assert all(channel_id == "channel-1" for channel_id, _ in gateway.sent_text)
    assert all(len(chunk) <= discord_module.MAX_TEXT_CHARS for _, chunk in gateway.sent_text)
    assert result.message_id == gateway.next_message_id


async def test_send_message_short_text_sends_a_single_message(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)

    result = await channel.send_message("channel-1", "hello")

    assert result.success
    assert gateway.sent_text == [("channel-1", "hello")]


async def test_send_media_uses_caption_as_message_content(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"")

    result = await channel.send_media(
        "channel-1",
        ChannelMedia(path=image_path, media_type="image", caption="hi"),
    )

    assert result.success
    assert gateway.sent_files == [("channel-1", image_path, "hi")]
    assert gateway.sent_text == []


async def test_send_media_with_long_caption_sends_it_separately(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"")
    caption = "a" * 2500

    result = await channel.send_media(
        "channel-1",
        ChannelMedia(path=image_path, media_type="image", caption=caption),
    )

    assert result.success
    assert gateway.sent_text
    assert gateway.sent_files == [("channel-1", image_path, None)]


async def test_send_media_rejects_files_outside_the_outbound_root(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    outside_dir = tmp_path.parent / "outside"
    outside_dir.mkdir(exist_ok=True)
    image_path = outside_dir / "photo.png"
    image_path.write_bytes(b"")

    with pytest.raises(discord_module.ChannelMediaError):
        await channel.send_media(
            "channel-1",
            ChannelMedia(path=image_path, media_type="image", caption=None),
        )


async def test_edit_message_delegates_to_gateway(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)

    result = await channel.edit_message("channel-1", "msg-1", "updated")

    assert result.success
    assert gateway.edits == [("channel-1", "msg-1", "updated")]


async def test_send_typing_delegates_to_gateway(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)

    await channel.send_typing("channel-1")

    assert gateway.typing == ["channel-1"]


async def test_send_typing_swallows_transport_errors(tmp_path):
    channel = DiscordChannel(_make_config(tmp_path), gateway=FailingTypingGateway())

    await channel.send_typing("channel-1")


# --- inbound message exposure/dispatch tests --------------------------------


async def test_inbound_message_from_operator_dm_is_dispatched(tmp_path):
    gateway = RecordingGateway()
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"op-1"})),
    )
    channel = DiscordChannel(config, gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="op-1",
            text="hi",
            is_dm=True,
            from_self=False,
        ),
    )

    assert len(received) == 1
    assert received[0].text == "hi"
    assert received[0].conversation_id == "chan-1"


async def test_inbound_message_from_non_operator_is_rejected(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="stranger",
            text="hi",
            is_dm=True,
            from_self=False,
        ),
    )

    assert received == []


async def test_self_authored_message_is_dropped(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="bot-1",
            text="echo",
            is_dm=True,
            from_self=True,
        ),
    )

    assert received == []


async def test_allowlist_mode_allows_dm_from_allowlisted_user(tmp_path):
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.ALLOWLIST),
        allowed_user_ids=frozenset({"user-1"}),
    )
    gateway = RecordingGateway()
    channel = DiscordChannel(config, gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="user-1",
            text="hi",
            is_dm=True,
            from_self=False,
        ),
    )

    assert len(received) == 1


async def test_allowlist_mode_allows_guild_channel_in_allowlist(tmp_path):
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(
            mode=ExposureMode.ALLOWLIST,
            conversations=frozenset({"guild-chan"}),
        ),
    )
    gateway = RecordingGateway()
    channel = DiscordChannel(config, gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="guild-chan",
            message_id="m1",
            sender_id="anyone",
            text="hi",
            is_dm=False,
            from_self=False,
        ),
    )

    assert len(received) == 1


async def test_allowlist_mode_rejects_unlisted_guild_channel(tmp_path):
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(
            mode=ExposureMode.ALLOWLIST,
            conversations=frozenset({"guild-chan"}),
        ),
    )
    gateway = RecordingGateway()
    channel = DiscordChannel(config, gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="other-chan",
            message_id="m1",
            sender_id="anyone",
            text="hi",
            is_dm=False,
            from_self=False,
        ),
    )

    assert received == []


# --- inbound reaction tests ---------------------------------------------------


async def test_reaction_from_operator_is_dispatched(tmp_path):
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"op-1"})),
    )
    gateway = RecordingGateway()
    channel = DiscordChannel(config, gateway=gateway)
    received, handler = _collector()
    channel.set_reaction_handler(handler)
    await channel.start()

    await gateway.deliver_reaction(
        _DiscordInboundReaction(channel_id="chan-1", message_id="m1", sender_id="op-1", emoji="👍"),
    )

    assert len(received) == 1
    assert received[0].emoji == "👍"


async def test_reaction_from_non_operator_is_rejected(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    received, handler = _collector()
    channel.set_reaction_handler(handler)
    await channel.start()

    await gateway.deliver_reaction(
        _DiscordInboundReaction(
            channel_id="chan-1",
            message_id="m1",
            sender_id="stranger",
            emoji="👍",
        ),
    )

    assert received == []


async def test_reaction_without_registered_handler_is_dropped(tmp_path):
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"op-1"})),
    )
    gateway = RecordingGateway()
    channel = DiscordChannel(config, gateway=gateway)
    await channel.start()

    await gateway.deliver_reaction(
        _DiscordInboundReaction(channel_id="chan-1", message_id="m1", sender_id="op-1", emoji="👍"),
    )


# --- inbound media tests -----------------------------------------------------


async def test_inbound_message_with_attachment_downloads_media(tmp_path, monkeypatch):
    _stub_download(monkeypatch)
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"op-1"})),
    )
    gateway = RecordingGateway()
    channel = DiscordChannel(config, gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="op-1",
            text="look",
            is_dm=True,
            from_self=False,
            attachments=(
                _DiscordAttachment(
                    url="https://cdn.discord/x.png",
                    filename="x.png",
                    content_type="image/png",
                    size=10,
                ),
            ),
        ),
    )

    assert len(received) == 1
    message = received[0]
    assert message.metadata["has_media"] is True
    assert message.metadata["media_type"] == "image"
    media_path = Path(message.metadata["media_path"])
    assert media_path.exists()  # noqa: ASYNC240  # test assertion, not production I/O
    assert media_path.read_bytes() == b"data"  # noqa: ASYNC240  # test assertion, not production I/O


async def test_inbound_attachment_over_size_cap_is_skipped(tmp_path, monkeypatch):
    _stub_download(monkeypatch)
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"op-1"})),
        max_media_bytes=5,
    )
    gateway = RecordingGateway()
    channel = DiscordChannel(config, gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="op-1",
            text="look",
            is_dm=True,
            from_self=False,
            attachments=(
                _DiscordAttachment(
                    url="https://cdn.discord/x.png",
                    filename="x.png",
                    content_type="image/png",
                    size=999,
                ),
            ),
        ),
    )

    assert len(received) == 1
    message = received[0]
    assert message.metadata["has_media"] is False
    assert "media_error" in message.metadata


async def test_inbound_attachment_download_failure_is_skipped(tmp_path, monkeypatch):
    _stub_failing_download(monkeypatch)
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"op-1"})),
    )
    gateway = RecordingGateway()
    channel = DiscordChannel(config, gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="op-1",
            text="look",
            is_dm=True,
            from_self=False,
            attachments=(
                _DiscordAttachment(
                    url="https://cdn.discord/x.png",
                    filename="x.png",
                    content_type="image/png",
                    size=10,
                ),
            ),
        ),
    )

    assert len(received) == 1
    assert received[0].metadata["has_media"] is False
    assert "media_error" in received[0].metadata


# --- media type inference tests -----------------------------------------------


def test_attachment_media_type_maps_content_types():
    def attachment(filename, content_type):
        return _DiscordAttachment(url="u", filename=filename, content_type=content_type, size=1)

    assert discord_module._attachment_media_type(attachment("clip.mp4", "video/mp4")) == "video"
    assert discord_module._attachment_media_type(attachment("photo.png", "image/png")) == "image"
    assert (
        discord_module._attachment_media_type(attachment("voice-message.ogg", "audio/ogg"))
        == "voice"
    )
    assert discord_module._attachment_media_type(attachment("clip.ogg", "audio/ogg")) == "audio"
    assert (
        discord_module._attachment_media_type(attachment("doc.pdf", "application/pdf"))
        == "document"
    )
