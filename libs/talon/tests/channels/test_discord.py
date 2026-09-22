from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from deepagents_talon.channels import discord as discord_module
from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.discord import (
    DiscordChannel,
    DiscordChannelConfig,
    InboundConnectionCallback,
    InboundInteractionCallback,
    InboundMessageCallback,
    InboundReactionCallback,
    _DiscordAttachment,
    _DiscordConnectionState,
    _DiscordInboundInteraction,
    _DiscordInboundMessage,
    _DiscordInboundReaction,
    _DiscordPyGateway,
)
from deepagents_talon.commands import COMMANDS_BY_NAME, visible_commands
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
        self._handle_connection: InboundConnectionCallback | None = None
        self._handle_interaction: InboundInteractionCallback | None = None

    async def start(
        self,
        *,
        handle_message,
        handle_reaction,
        handle_connection,
        handle_interaction,
    ):
        self.started = True
        self._handle_message = handle_message
        self._handle_reaction = handle_reaction
        self._handle_connection = handle_connection
        self._handle_interaction = handle_interaction

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

    async def deliver_connection(self, state):
        assert self._handle_connection is not None
        await self._handle_connection(state)

    async def deliver_interaction(self, inbound):
        assert self._handle_interaction is not None
        await self._handle_interaction(inbound)


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


class _GatewayFailureError(Exception):
    """Stand-in for an unrecoverable `discord.py` Gateway error such as a 4004 close."""


class FakeClient:
    """Stand-in for `discord.Client` that records handlers and never opens a socket."""

    def __init__(self, *, intents: object) -> None:
        del intents
        self.user = None
        self.closed = asyncio.Event()
        self.handlers = {}
        self.reconnect_values = []
        self.start_error: BaseException | None = None
        self.tree: FakeCommandTree | None = None

    def event(self, callback):
        self.handlers[callback.__name__] = callback
        return callback

    async def start(self, token: str, *, reconnect: bool) -> None:
        del token
        self.reconnect_values.append(reconnect)
        await self.closed.wait()
        if self.start_error is not None:
            raise self.start_error

    async def wait_until_ready(self) -> None:
        return None

    async def close(self) -> None:
        self.closed.set()

    async def fire(self, event: str, *args: object) -> None:
        await self.handlers[event](*args)

    def fail(self, error: BaseException) -> None:
        """End the gateway task with `error`, as an unrecoverable close would."""
        self.start_error = error
        self.closed.set()


class FakeCommandTree:
    """Stand-in for `app_commands.CommandTree`.

    The real tree reaches into `client.http` and `client._connection` in its
    constructor, so it cannot be built against `FakeClient`. It attaches itself to
    the client it was handed, mirroring the real one-tree-per-client relationship.
    """

    def __init__(self, client, *, allowed_contexts=None, allowed_installs=None) -> None:
        self.client = client
        self.allowed_contexts = allowed_contexts
        self.allowed_installs = allowed_installs
        self.commands = {}
        self.syncs = []
        self.sync_error: BaseException | None = None
        client.tree = self

    def add_command(self, command) -> None:
        self.commands[command.name] = command

    async def sync(self, *, guild=None):
        self.syncs.append(guild)
        if self.sync_error is not None:
            raise self.sync_error
        return []


def _install_fake_client(monkeypatch: pytest.MonkeyPatch) -> list[FakeClient]:
    created: list[FakeClient] = []

    def factory(*, intents: object) -> FakeClient:
        client = FakeClient(intents=intents)
        created.append(client)
        return client

    monkeypatch.setattr(discord_module.discord, "Client", factory)
    monkeypatch.setattr(discord_module.app_commands, "CommandTree", FakeCommandTree)
    return created


def _connection_recorder():
    """Record connection states, signalling an event as each one arrives."""
    states = []
    arrived = asyncio.Event()

    async def record(state):
        states.append(state)
        arrived.set()

    return states, arrived, record


async def _start_fake_gateway(
    handle_connection,
    *,
    commands_enabled: bool = True,
    command_guild_id: str | None = None,
    handle_interaction=None,
) -> _DiscordPyGateway:
    gateway = _DiscordPyGateway(
        token="test-token",  # noqa: S106  # inert test token
        connect_timeout_seconds=1,
        commands_enabled=commands_enabled,
        command_guild_id=command_guild_id,
    )
    await gateway.start(
        handle_message=lambda _: asyncio.sleep(0),
        handle_reaction=lambda _: asyncio.sleep(0),
        handle_connection=handle_connection,
        handle_interaction=handle_interaction or (lambda _: asyncio.sleep(0)),
    )
    return gateway


async def test_gateway_enables_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_client(monkeypatch)

    gateway = await _start_fake_gateway(lambda _: asyncio.sleep(0))
    await gateway.stop()

    assert created[0].reconnect_values == [True]


async def test_gateway_reports_connection_events(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_client(monkeypatch)
    states, _arrived, record_state = _connection_recorder()

    gateway = await _start_fake_gateway(record_state)
    client = created[0]
    await client.fire("on_ready")
    await client.fire("on_disconnect")
    await client.fire("on_resumed")
    await gateway.stop()

    assert states == [
        _DiscordConnectionState(connected=True, detail="connected"),
        _DiscordConnectionState(connected=True, detail="reconnecting"),
        _DiscordConnectionState(connected=True, detail="connected"),
    ]


async def test_gateway_reports_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    created = _install_fake_client(monkeypatch)
    states, arrived, record_state = _connection_recorder()

    gateway = await _start_fake_gateway(record_state)
    with caplog.at_level(logging.ERROR, logger="deepagents_talon.channels.discord"):
        created[0].fail(_GatewayFailureError("gateway closed with 4004"))
        await asyncio.wait_for(arrived.wait(), timeout=1)
    await gateway.stop()

    assert states == [
        _DiscordConnectionState(connected=False, detail="gateway stopped: _GatewayFailureError"),
    ]
    assert "Discord Gateway connection ended unexpectedly" in caplog.text


async def test_gateway_watcher_is_quiet_on_a_clean_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_client(monkeypatch)
    states, _arrived, record_state = _connection_recorder()

    gateway = await _start_fake_gateway(record_state)
    assert created[0].reconnect_values == [True]
    await gateway.stop()

    assert states == []


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


async def test_terminal_gateway_failure_marks_the_channel_disconnected(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    await channel.start()

    await gateway.deliver_connection(
        _DiscordConnectionState(connected=False, detail="gateway stopped: ConnectionClosed"),
    )

    status = await channel.status()
    assert status.connected is False
    assert status.detail == "gateway stopped: ConnectionClosed"


async def test_transient_disconnect_keeps_the_channel_connected(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    await channel.start()

    await gateway.deliver_connection(
        _DiscordConnectionState(connected=True, detail="reconnecting"),
    )

    status = await channel.status()
    assert status.connected is True
    assert status.detail == "reconnecting"

    await gateway.deliver_connection(_DiscordConnectionState(connected=True, detail="connected"))

    assert (await channel.status()).detail == "connected"


async def test_connection_events_after_stop_are_ignored(tmp_path):
    gateway = RecordingGateway()
    channel = DiscordChannel(_make_config(tmp_path), gateway=gateway)
    await channel.start()
    await channel.stop()

    await gateway.deliver_connection(_DiscordConnectionState(connected=True, detail="connected"))

    status = await channel.status()
    assert status.connected is False
    assert status.detail == "disconnected"


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


# --- inbound slash command tests ----------------------------------------------


class RecordingResponder:
    """Fake `_InteractionResponder` recording how an interaction was answered."""

    def __init__(self) -> None:
        self.rejects = []
        self.defers = 0
        self.sends = []
        self.next_message_id = "followup-1"

    async def reject(self, text):
        self.rejects.append(text)

    async def defer(self):
        self.defers += 1

    async def send(self, text):
        self.sends.append(text)
        return self.next_message_id


def _interaction(
    command="new",
    *,
    channel_id="chan-1",
    sender_id="op-1",
    responder=None,
    is_dm=True,
):
    return _DiscordInboundInteraction(
        command=command,
        channel_id=channel_id,
        sender_id=sender_id,
        interaction_id="int-1",
        is_dm=is_dm,
        responder=responder if responder is not None else RecordingResponder(),
    )


def _operator_channel(tmp_path, gateway, *, operator="op-1"):
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({operator})),
    )
    return DiscordChannel(config, gateway=gateway)


async def test_slash_command_is_dispatched_as_its_text_equivalent(tmp_path):
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()
    responder = RecordingResponder()

    await gateway.deliver_interaction(_interaction("new", responder=responder))

    assert len(received) == 1
    assert received[0].text == "/new"
    assert received[0].conversation_id == "chan-1"
    assert received[0].sender_id == "op-1"
    assert received[0].message_id == "int-1"


async def test_every_registered_command_dispatches_its_text(tmp_path):
    """A registered command must produce text the host recognizes as that command."""
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    for command in visible_commands():
        await gateway.deliver_interaction(_interaction(command.name))

    assert [message.text for message in received] == [c.text for c in visible_commands()]


async def test_slash_command_metadata_matches_a_typed_command(tmp_path):
    """Authorization reads metadata, so both inbound paths must produce the same keys."""
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_interaction(_interaction("new", is_dm=True))
    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="op-1",
            text="/new",
            is_dm=True,
            from_self=False,
        ),
    )

    assert len(received) == 2
    assert dict(received[0].metadata) == dict(received[1].metadata)
    assert received[0].metadata["from_self"] is False


async def test_slash_command_from_non_operator_is_rejected_without_deferring(tmp_path):
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()
    responder = RecordingResponder()

    await gateway.deliver_interaction(_interaction(sender_id="intruder", responder=responder))

    assert received == []
    assert responder.rejects == ["This assistant does not accept commands from you."]
    assert responder.defers == 0
    assert responder.sends == []


async def test_slash_command_in_an_allowlisted_dm_is_allowed(tmp_path):
    gateway = RecordingGateway()
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.ALLOWLIST),
        allowed_user_ids=frozenset({"guest"}),
    )
    channel = DiscordChannel(config, gateway=gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_interaction(_interaction(sender_id="guest", is_dm=True))

    assert len(received) == 1


async def test_slash_command_without_a_channel_is_rejected(tmp_path):
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()
    responder = RecordingResponder()

    await gateway.deliver_interaction(_interaction(channel_id=None, responder=responder))

    assert received == []
    assert responder.rejects == ["That command is not available here."]
    assert responder.defers == 0


async def test_unknown_slash_command_is_rejected(tmp_path):
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()
    responder = RecordingResponder()

    await gateway.deliver_interaction(_interaction("not-a-command", responder=responder))

    assert received == []
    assert responder.rejects == ["That command is not available here."]


async def test_slash_command_defers_before_dispatching(tmp_path):
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    await channel.start()
    responder = RecordingResponder()
    deferred_when_dispatched = []

    async def handler(_message):
        deferred_when_dispatched.append(responder.defers)

    channel.set_message_handler(handler)

    await gateway.deliver_interaction(_interaction(responder=responder))

    assert deferred_when_dispatched == [1]


async def test_slash_command_reply_answers_the_interaction(tmp_path):
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    await channel.start()
    responder = RecordingResponder()

    async def handler(message):
        result = await channel.send_message(
            message.conversation_id, "Started a fresh conversation."
        )
        assert result.success
        assert result.message_id == "followup-1"

    channel.set_message_handler(handler)

    await gateway.deliver_interaction(_interaction(responder=responder))

    assert responder.sends == ["Started a fresh conversation."]
    assert gateway.sent_text == []


async def test_slash_command_reply_over_the_limit_is_chunked_into_followups(tmp_path):
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    await channel.start()
    responder = RecordingResponder()

    async def handler(message):
        await channel.send_message(message.conversation_id, "x" * 2500)

    channel.set_message_handler(handler)

    await gateway.deliver_interaction(_interaction(responder=responder))

    assert len(responder.sends) == 2
    assert all(len(chunk) <= discord_module.MAX_TEXT_CHARS for chunk in responder.sends)
    assert "".join(responder.sends) == "x" * 2500
    assert gateway.sent_text == []


async def test_slash_command_without_a_reply_still_answers_the_interaction(tmp_path):
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    _received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()
    responder = RecordingResponder()

    await gateway.deliver_interaction(_interaction(responder=responder))

    assert responder.sends == ["Done."]


async def test_slash_command_failure_answers_the_interaction(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    await channel.start()
    responder = RecordingResponder()

    async def handler(_message):
        msg = "boom"
        raise RuntimeError(msg)

    channel.set_message_handler(handler)

    with caplog.at_level(logging.WARNING, logger="deepagents_talon.channels.discord"):
        await gateway.deliver_interaction(_interaction(responder=responder))

    assert responder.sends == ["Something went wrong running that command. Check Talon logs."]
    assert "Discord command new failed" in caplog.text


async def test_slash_command_reply_after_a_partial_send_is_not_duplicated(tmp_path):
    """A followup that fails mid-reply is still an answer; no fallback on top of it."""
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    await channel.start()

    class FailingResponder(RecordingResponder):
        async def send(self, text):
            self.sends.append(text)
            msg = "followup rejected"
            raise RuntimeError(msg)

    responder = FailingResponder()

    async def handler(message):
        await channel.send_message(message.conversation_id, "reply")

    channel.set_message_handler(handler)

    await gateway.deliver_interaction(_interaction(responder=responder))

    assert responder.sends == ["reply"]


async def test_slash_command_does_not_capture_another_conversation(tmp_path):
    """Only the invoking conversation's reply belongs to the interaction."""
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    await channel.start()
    responder = RecordingResponder()

    async def handler(_message):
        await channel.send_message("chan-2", "unrelated")

    channel.set_message_handler(handler)

    await gateway.deliver_interaction(_interaction(responder=responder))

    assert gateway.sent_text == [("chan-2", "unrelated")]
    assert responder.sends == ["Done."]


async def test_concurrent_slash_commands_answer_their_own_interaction(tmp_path):
    """Each interaction runs in its own task, so sinks must not leak between them."""
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    await channel.start()
    first = RecordingResponder()
    second = RecordingResponder()
    released = asyncio.Event()

    async def handler(message):
        # Hold the first command open so both are in flight at the same time.
        if message.message_id == "first":
            await released.wait()
        await channel.send_message(message.conversation_id, f"reply for {message.message_id}")

    channel.set_message_handler(handler)

    def interaction(marker, responder):
        return _DiscordInboundInteraction(
            command="new",
            channel_id="chan-1",
            sender_id="op-1",
            interaction_id=marker,
            is_dm=True,
            responder=responder,
        )

    first_task = asyncio.create_task(gateway.deliver_interaction(interaction("first", first)))
    await asyncio.sleep(0)
    second_task = asyncio.create_task(gateway.deliver_interaction(interaction("second", second)))
    await asyncio.wait_for(second_task, timeout=1)
    released.set()
    await asyncio.wait_for(first_task, timeout=1)

    assert first.sends == ["reply for first"]
    assert second.sends == ["reply for second"]
    assert gateway.sent_text == []


async def test_hidden_commands_still_dispatch_when_typed(tmp_path):
    """`/reset-all-history` is unregistered, not disabled."""
    gateway = RecordingGateway()
    channel = _operator_channel(tmp_path, gateway)
    received, handler = _collector()
    channel.set_message_handler(handler)
    await channel.start()

    await gateway.deliver_message(
        _DiscordInboundMessage(
            channel_id="chan-1",
            message_id="m1",
            sender_id="op-1",
            text="/reset-all-history",
            is_dm=True,
            from_self=False,
        ),
    )

    assert [message.text for message in received] == ["/reset-all-history"]


# --- application command registration tests -----------------------------------


class FakeInteraction:
    """Stand-in for `discord.Interaction` carrying only what the adapter reads."""

    class _User:
        def __init__(self, user_id: int) -> None:
            self.id = user_id

    def __init__(self, *, channel_id, guild_id, user_id, interaction_id) -> None:
        self.channel_id = channel_id
        self.guild_id = guild_id
        self.user = self._User(user_id)
        self.id = interaction_id


async def test_gateway_registers_every_advertised_command(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_client(monkeypatch)

    gateway = await _start_fake_gateway(lambda _: asyncio.sleep(0))
    await gateway.stop()

    tree = created[0].tree
    assert tree is not None
    assert set(tree.commands) == {command.name for command in visible_commands()}
    assert tree.allowed_contexts.dm_channel is True
    assert tree.allowed_installs.guild is True


async def test_gateway_does_not_register_hidden_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_client(monkeypatch)

    gateway = await _start_fake_gateway(lambda _: asyncio.sleep(0))
    await gateway.stop()

    tree = created[0].tree
    assert tree is not None
    assert "reset-all-history" not in tree.commands
    assert COMMANDS_BY_NAME["reset-all-history"].hidden is True


async def test_gateway_registers_command_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_client(monkeypatch)

    gateway = await _start_fake_gateway(lambda _: asyncio.sleep(0))
    await gateway.stop()

    tree = created[0].tree
    assert tree is not None
    assert tree.commands["new"].description == COMMANDS_BY_NAME["new"].summary


async def test_gateway_syncs_commands_once_across_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_client(monkeypatch)

    gateway = await _start_fake_gateway(lambda _: asyncio.sleep(0))
    client = created[0]
    await client.fire("on_ready")
    await client.fire("on_disconnect")
    await client.fire("on_resumed")
    await client.fire("on_ready")
    await gateway.stop()

    assert client.tree is not None
    assert client.tree.syncs == [None]


async def test_gateway_reports_the_connection_before_syncing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_client(monkeypatch)
    order = []

    async def record_state(_state):
        order.append("connected")

    gateway = await _start_fake_gateway(record_state)
    client = created[0]
    assert client.tree is not None
    original_sync = client.tree.sync

    async def tracking_sync(*, guild=None):
        order.append("synced")
        return await original_sync(guild=guild)

    client.tree.sync = tracking_sync
    await client.fire("on_ready")
    await gateway.stop()

    assert order == ["connected", "synced"]


async def test_gateway_survives_a_command_sync_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    created = _install_fake_client(monkeypatch)
    states, _arrived, record_state = _connection_recorder()

    gateway = await _start_fake_gateway(record_state)
    client = created[0]
    assert client.tree is not None
    client.tree.sync_error = RuntimeError("missing applications.commands scope")
    with caplog.at_level(logging.WARNING, logger="deepagents_talon.channels.discord"):
        await client.fire("on_ready")
    await gateway.stop()

    assert states == [_DiscordConnectionState(connected=True, detail="connected")]
    assert "Could not register Discord application commands" in caplog.text


async def test_gateway_scopes_the_sync_to_a_configured_guild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_client(monkeypatch)

    gateway = await _start_fake_gateway(lambda _: asyncio.sleep(0), command_guild_id="123")
    client = created[0]
    await client.fire("on_ready")
    await gateway.stop()

    assert client.tree is not None
    assert [guild.id for guild in client.tree.syncs] == [123]


async def test_gateway_skips_registration_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_client(monkeypatch)
    states, _arrived, record_state = _connection_recorder()

    gateway = await _start_fake_gateway(record_state, commands_enabled=False)
    client = created[0]
    await client.fire("on_ready")
    await gateway.stop()

    assert client.tree is None
    assert states == [_DiscordConnectionState(connected=True, detail="connected")]


async def test_gateway_command_callback_reports_a_neutral_interaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_client(monkeypatch)
    received, handler = _collector()

    gateway = await _start_fake_gateway(lambda _: asyncio.sleep(0), handle_interaction=handler)
    tree = created[0].tree
    assert tree is not None
    await tree.commands["new"].callback(
        FakeInteraction(channel_id=42, guild_id=None, user_id=7, interaction_id=99),
    )
    await gateway.stop()

    assert len(received) == 1
    assert received[0].command == "new"
    assert received[0].channel_id == "42"
    assert received[0].sender_id == "7"
    assert received[0].interaction_id == "99"
    assert received[0].is_dm is True


async def test_gateway_command_callback_marks_a_guild_interaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_client(monkeypatch)
    received, handler = _collector()

    gateway = await _start_fake_gateway(lambda _: asyncio.sleep(0), handle_interaction=handler)
    tree = created[0].tree
    assert tree is not None
    await tree.commands["stop"].callback(
        FakeInteraction(channel_id=42, guild_id=5, user_id=7, interaction_id=99),
    )
    await gateway.stop()

    assert received[0].is_dm is False


def test_from_talon_config_reads_command_registration_settings(tmp_path):
    config = _talon_config(
        tmp_path,
        {
            "DEEPAGENTS_TALON_DISCORD_BOT_TOKEN": "token",
            "DEEPAGENTS_TALON_DISCORD_OPERATOR_ID": "op-1",
            "DEEPAGENTS_TALON_DISCORD_COMMAND_GUILD_ID": "123",
            "DEEPAGENTS_TALON_DISCORD_SLASH_COMMANDS": "false",
        },
    )

    discord_config = DiscordChannelConfig.from_talon_config(config)

    assert discord_config.command_guild_id == "123"
    assert discord_config.slash_commands_enabled is False


def test_from_talon_config_registers_commands_by_default(tmp_path):
    config = _talon_config(
        tmp_path,
        {
            "DEEPAGENTS_TALON_DISCORD_BOT_TOKEN": "token",
            "DEEPAGENTS_TALON_DISCORD_OPERATOR_ID": "op-1",
        },
    )

    discord_config = DiscordChannelConfig.from_talon_config(config)

    assert discord_config.slash_commands_enabled is True
    assert discord_config.command_guild_id is None


@pytest.mark.parametrize(
    ("env", "field", "match"),
    [
        ({"DEEPAGENTS_TALON_DISCORD_SLASH_COMMANDS": "maybe"}, "slash", "must be a boolean"),
        ({"DEEPAGENTS_TALON_DISCORD_COMMAND_GUILD_ID": "guild"}, "guild", "guild id"),
    ],
)
def test_from_talon_config_rejects_invalid_command_settings(tmp_path, env, field, match):
    del field
    config = _talon_config(
        tmp_path,
        {
            "DEEPAGENTS_TALON_DISCORD_BOT_TOKEN": "token",
            "DEEPAGENTS_TALON_DISCORD_OPERATOR_ID": "op-1",
            **env,
        },
    )

    with pytest.raises(ValueError, match=match):
        DiscordChannelConfig.from_talon_config(config)
