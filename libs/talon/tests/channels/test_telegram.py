from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Literal, Self, cast

import pytest

import deepagents_talon.channels.telegram as telegram_module
from deepagents_talon.channels.base import (
    DEFAULT_MAX_MEDIA_BYTES,
    ChannelExposure,
    ChannelMediaError,
    ExposureMode,
)
from deepagents_talon.channels.telegram import (
    TelegramChannel,
    TelegramChannelConfig,
    _download_file,
    _encode_multipart_form,
    _load_offset,
    _save_offset,
    _TelegramError,
    _TelegramTransport,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import ChannelMedia, ChannelMessage, ChannelReaction


class JsonResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class RecordingTransport:
    """Fake transport that records calls and returns canned responses."""

    def __init__(
        self,
        updates: list[dict[str, object]] | None = None,
    ) -> None:
        self.updates = list(updates) if updates else []
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.uploads: list[tuple[str, str, Path, dict[str, object]]] = []
        self._get_me_called = False

    async def call(self, method: str, **params: object) -> object:  # noqa: C901  # test mock dispatch
        self.calls.append((method, dict(params)))
        if method == "getMe":
            self._get_me_called = True
            return {"id": 123456, "username": "test_bot"}
        if method == "getUpdates":
            updates = self.updates
            self.updates = []
            return updates
        if method == "sendChatAction":
            return True
        if method in ("sendMessage", "sendPhoto", "sendVideo", "sendDocument", "editMessageText"):
            return {"message_id": 42}
        if method == "getFile":
            file_id = params.get("file_id")
            if file_id == "voice123":
                file_path = "voice/file.oga"
            elif file_id in {"video123", "note123"}:
                file_path = "videos/file.mp4"
            elif file_id == "docvideo123":
                file_path = "documents/clip.mp4"
            elif file_id == "doc123":
                file_path = "documents/report.pdf"
            elif file_id == "docaudio123":
                file_path = "music/file.mp3"
            else:
                file_path = "photos/file.jpg"
            return {"file_id": file_id, "file_path": file_path}
        return True

    async def upload(
        self,
        method: str,
        *,
        file_field: str,
        file_path: Path,
        **params: object,
    ) -> object:
        self.uploads.append((method, file_field, file_path, dict(params)))
        return {"message_id": 42}


class ErrorOnFirstSuccessTransport:
    """Transport that raises on the first getUpdates, then returns empty."""

    def __init__(self) -> None:
        self.calls = 0

    async def call(self, method: str, **params: object) -> object:  # noqa: ARG002  # test fake
        if method == "getMe":
            return {"id": 123456, "username": "test_bot"}
        if method == "getUpdates":
            self.calls += 1
            if self.calls == 1:
                msg = "network error"
                raise _TelegramError(msg)
            return []
        return True


def _make_update(  # noqa: PLR0913  # test helper with many optional fields
    *,
    update_id: int = 1,
    chat_id: int = 111,
    sender_id: int = 111,
    text: str = "hello",
    chat_type: str = "private",
    message_id: int = 10,
    photo: list | None = None,
    voice: dict | None = None,
    video: dict | None = None,
    video_note: dict | None = None,
    document: dict | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "message_id": message_id,
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"id": sender_id},
        "text": text,
    }
    if photo is not None:
        message["photo"] = photo
    if voice is not None:
        message["voice"] = voice
    if video is not None:
        message["video"] = video
    if video_note is not None:
        message["video_note"] = video_note
    if document is not None:
        message["document"] = document
    return {"update_id": update_id, "message": message}


def _make_channel_post(
    *,
    update_id: int = 1,
    chat_id: int = -100111,
    text: str = "channel input",
    message_id: int = 10,
) -> dict[str, object]:
    return {
        "update_id": update_id,
        "channel_post": {
            "message_id": message_id,
            "chat": {"id": chat_id, "type": "channel"},
            "text": text,
        },
    }


def _make_reaction_update(  # noqa: PLR0913  # test helper with many optional fields
    *,
    update_id: int = 1,
    chat_id: int = 111,
    sender_id: int | None = 111,
    chat_type: str = "private",
    message_id: int = 10,
    emoji: str = "👍",
    anonymous: bool = False,
    new_reaction: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    reaction: dict[str, object] = {
        "chat": {"id": chat_id, "type": chat_type},
        "message_id": message_id,
        "date": 1_700_000_000,
        "old_reaction": [],
        "new_reaction": new_reaction
        if new_reaction is not None
        else [{"type": "emoji", "emoji": emoji}],
    }
    if sender_id is not None:
        reaction["user"] = {"id": sender_id, "is_bot": False, "first_name": "Operator"}
    if anonymous:
        reaction["actor_chat"] = {"id": chat_id, "type": chat_type, "title": "Anonymous"}
    return {"update_id": update_id, "message_reaction": reaction}


def _make_config(
    tmp_path: Path,
    *,
    exposure: ChannelExposure | None = None,
    operator_id: str | None = None,
    allowed_user_ids: frozenset[str] | None = None,
) -> TelegramChannelConfig:
    return TelegramChannelConfig(
        bot_token="test-token",  # noqa: S106  # inert test token
        session_dir=tmp_path / "telegram",
        inbound_media_dir=tmp_path / "telegram" / "media",
        outbound_media_dir=tmp_path,
        exposure=exposure
        or ChannelExposure(
            operator_ids=frozenset({operator_id}) if operator_id else frozenset(),
        ),
        poll_interval_seconds=60,
        poll_timeout_seconds=1,
        allowed_user_ids=allowed_user_ids or frozenset(),
    )


def _stub_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def download(
        _url: str,
        destination: Path,
        _timeout: float,
        _max_bytes: int,
    ) -> None:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_bytes(b"media-bytes")
        destination.chmod(0o600)

    monkeypatch.setattr(telegram_module, "_download_file", download)


async def _run_channel_once(
    channel: TelegramChannel,
    *,
    expected_messages: int = 1,
) -> list[ChannelMessage]:
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)
    await channel.start()
    try:
        if expected_messages:
            await _wait_for_received(received, expected_messages)
        else:
            await asyncio.sleep(0)
    finally:
        await channel.stop()
    return received


async def _run_channel_reactions_once(
    channel: TelegramChannel,
    *,
    expected_reactions: int = 1,
) -> list[ChannelReaction]:
    received: list[ChannelReaction] = []

    async def record(reaction: ChannelReaction) -> None:
        received.append(reaction)

    channel.set_reaction_handler(record)
    await channel.start()
    try:
        if expected_reactions:
            await _wait_for_reactions(received, expected_reactions)
        else:
            await asyncio.sleep(0)
    finally:
        await channel.stop()
    return received


# --- Config tests ---


def test_config_from_talon_env_maps_telegram_values(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "secret-token",
            "DEEPAGENTS_TALON_TELEGRAM_EXPOSURE": "allowlist",
            "DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS": "123, 456",
            "DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS": "777, 888",
            "DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID": "999",
            "DEEPAGENTS_TALON_TELEGRAM_POLL_TIMEOUT_SECONDS": "45",
            "DEEPAGENTS_TALON_TELEGRAM_POLL_INTERVAL_SECONDS": "2",
        },
        base_home=tmp_path,
    )

    telegram = TelegramChannelConfig.from_talon_config(config)

    assert telegram.bot_token == "secret-token"  # noqa: S105  # inert test token
    assert telegram.session_dir == tmp_path / "assistant" / "channels" / "telegram"
    assert telegram.inbound_media_dir == tmp_path / "assistant" / "media" / "inbound" / "telegram"
    assert telegram.exposure == ChannelExposure(
        mode=ExposureMode.ALLOWLIST,
        operator_ids=frozenset({"999"}),
        conversations=frozenset({"123", "456"}),
    )
    assert telegram.allowed_user_ids == frozenset({"777", "888"})
    assert telegram.poll_timeout_seconds == 45.0
    assert telegram.poll_interval_seconds == 2.0
    assert telegram.max_media_bytes == DEFAULT_MAX_MEDIA_BYTES


def test_config_from_talon_env_maps_multiple_operator_ids(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "secret-token",
            "DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID": "999, 1000",
        },
        base_home=tmp_path,
    )

    telegram = TelegramChannelConfig.from_talon_config(config)

    assert telegram.exposure == ChannelExposure(
        operator_ids=frozenset({"999", "1000"}),
    )
    assert telegram.exposure.allows(
        ChannelMessage(conversation_id="chat", text="hi", sender_id="999")
    )
    assert telegram.exposure.allows(
        ChannelMessage(conversation_id="chat", text="hi", sender_id="1000")
    )
    assert not telegram.exposure.allows(
        ChannelMessage(conversation_id="chat", text="hi", sender_id="1001")
    )


def test_config_from_talon_env_maps_max_media_bytes(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "secret-token",
            "DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID": "999",
            "DEEPAGENTS_TALON_MAX_MEDIA_BYTES": "12345",
        },
        base_home=tmp_path,
    )

    telegram = TelegramChannelConfig.from_talon_config(config)

    assert telegram.max_media_bytes == 12345


def test_config_accepts_telegram_bot_token_alias(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "TELEGRAM_BOT_TOKEN": "alias-token",
            "DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID": "999",
        },
        base_home=tmp_path,
    )

    telegram = TelegramChannelConfig.from_talon_config(config)

    assert telegram.bot_token == "alias-token"  # noqa: S105  # inert test token


def test_config_defaults_outbound_media_dir_to_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "token",
            "DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID": "999",
        },
        base_home=tmp_path / "home",
    )

    telegram = TelegramChannelConfig.from_talon_config(config)

    assert telegram.outbound_media_dir == tmp_path


def test_config_requires_operator_for_self_exposure(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "token",
        },
        base_home=tmp_path,
    )

    with pytest.raises(ValueError, match="TELEGRAM_OPERATOR_ID"):
        TelegramChannelConfig.from_talon_config(config)


def test_config_raises_without_bot_token(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {"AGENT_ASSISTANT_ID": "assistant"},
        base_home=tmp_path,
    )

    with pytest.raises(ValueError, match="bot token is required"):
        TelegramChannelConfig.from_talon_config(config)


def test_config_rejects_open_exposure_without_acknowledgement(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "token",
            "DEEPAGENTS_TALON_TELEGRAM_EXPOSURE": "open",
        },
        base_home=tmp_path,
    )

    with pytest.raises(ValueError, match="allow-arbitrary-senders"):
        TelegramChannelConfig.from_talon_config(config)


def test_config_accepts_open_exposure_with_acknowledgement(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "token",
            "DEEPAGENTS_TALON_TELEGRAM_EXPOSURE": "open",
            "DEEPAGENTS_TALON_TELEGRAM_OPEN_ACK": "allow-arbitrary-senders",
        },
        base_home=tmp_path,
    )

    telegram = TelegramChannelConfig.from_talon_config(config)

    assert telegram.exposure.mode == ExposureMode.OPEN


# --- Polling and exposure tests ---


async def test_channel_polls_and_dispatches_allowed_messages(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[
            _make_update(update_id=10, chat_id=111, sender_id=111, text="allowed"),
            _make_update(update_id=11, chat_id=333, sender_id=222, text="blocked"),
        ],
    )
    channel = TelegramChannel(
        _make_config(tmp_path, operator_id="111"),
        transport=cast("_TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert [msg.text for msg in received] == ["allowed"]
    assert received[0].metadata["provider"] == "telegram"
    assert received[0].conversation_id == "111"
    assert received[0].message_id == "10"
    get_updates_calls = [call for call in transport.calls if call[0] == "getUpdates"]
    assert get_updates_calls[0][1]["allowed_updates"] == [
        "message",
        "channel_post",
        "message_reaction",
    ]


async def test_channel_polls_and_dispatches_allowed_reactions(tmp_path: Path) -> None:
    cases: tuple[tuple[str, int], ...] = (
        ("private", 111),
        ("group", -111),
        ("supergroup", -100111),
        ("channel", -100222),
    )

    for chat_type, chat_id in cases:
        case_dir = tmp_path / chat_type
        transport = RecordingTransport(
            updates=[
                _make_reaction_update(
                    update_id=10,
                    chat_id=chat_id,
                    sender_id=222,
                    chat_type=chat_type,
                    message_id=42,
                    emoji="👎",
                ),
            ],
        )
        channel = TelegramChannel(
            _make_config(
                case_dir,
                exposure=ChannelExposure(
                    mode=ExposureMode.OPEN,
                    operator_ids=frozenset({"222"}),
                ),
            ),
            transport=cast("_TelegramTransport", transport),
        )

        received = await _run_channel_reactions_once(channel)

        assert received == [
            ChannelReaction(
                conversation_id=str(chat_id),
                message_id="42",
                emoji="👎",
                sender_id="222",
                metadata={"provider": "telegram", "chat_type": chat_type},
            )
        ]


async def test_channel_drops_anonymous_or_senderless_reactions(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[
            _make_reaction_update(
                update_id=10,
                chat_id=-100111,
                sender_id=None,
                chat_type="channel",
                anonymous=True,
            ),
            _make_reaction_update(
                update_id=11,
                chat_id=111,
                sender_id=None,
                chat_type="private",
            ),
        ],
    )
    config = _make_config(tmp_path, exposure=ChannelExposure(mode=ExposureMode.OPEN))
    channel = TelegramChannel(config, transport=cast("_TelegramTransport", transport))

    received = await _run_channel_reactions_once(channel, expected_reactions=0)

    assert received == []
    assert _load_offset(config.offset_file) == 12


async def test_channel_drops_reactions_denied_by_exposure(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[
            _make_reaction_update(
                update_id=10,
                chat_id=111,
                sender_id=222,
                chat_type="private",
                message_id=42,
            ),
        ],
    )
    config = _make_config(tmp_path, operator_id="111")
    channel = TelegramChannel(config, transport=cast("_TelegramTransport", transport))

    received = await _run_channel_reactions_once(channel, expected_reactions=0)

    assert received == []
    assert _load_offset(config.offset_file) == 11


async def test_channel_drops_untrusted_reactions_in_allowlisted_conversation(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport(
        updates=[
            _make_reaction_update(
                update_id=10,
                chat_id=-100111,
                sender_id=222,
                chat_type="channel",
                message_id=42,
            ),
            _make_reaction_update(
                update_id=11,
                chat_id=-100111,
                sender_id=111,
                chat_type="channel",
                message_id=42,
            ),
        ],
    )
    config = _make_config(
        tmp_path,
        exposure=ChannelExposure(
            mode=ExposureMode.ALLOWLIST,
            conversations=frozenset({"-100111"}),
        ),
        allowed_user_ids=frozenset({"111"}),
    )
    channel = TelegramChannel(config, transport=cast("_TelegramTransport", transport))

    received = await _run_channel_reactions_once(channel)

    assert [reaction.sender_id for reaction in received] == ["111"]
    assert _load_offset(config.offset_file) == 12


async def test_channel_persists_offset_after_reaction_update(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[
            _make_reaction_update(
                update_id=20,
                chat_id=111,
                sender_id=111,
                message_id=42,
            ),
        ],
    )
    config = _make_config(tmp_path, operator_id="111")
    channel = TelegramChannel(config, transport=cast("_TelegramTransport", transport))

    received = await _run_channel_reactions_once(channel)

    assert received[0].message_id == "42"
    assert _load_offset(config.offset_file) == 21


async def test_channel_polls_and_dispatches_allowed_channel_posts(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[_make_channel_post(update_id=10, chat_id=-100111, text="from channel")],
    )
    channel = TelegramChannel(
        _make_config(
            tmp_path,
            exposure=ChannelExposure(
                mode=ExposureMode.ALLOWLIST,
                conversations=frozenset({"-100111"}),
            ),
        ),
        transport=cast("_TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await _wait_for_received(received, 1)
    await channel.stop()

    assert received[0].text == "from channel"
    assert received[0].conversation_id == "-100111"
    assert received[0].sender_id is None
    assert received[0].message_id == "10"
    assert received[0].metadata["chat_type"] == "channel"


async def test_channel_polls_and_dispatches_allowed_private_users(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[
            _make_update(update_id=10, chat_id=111, sender_id=111, text="allowed user"),
            _make_update(update_id=11, chat_id=222, sender_id=222, text="blocked user"),
        ],
    )
    channel = TelegramChannel(
        _make_config(
            tmp_path,
            exposure=ChannelExposure(
                mode=ExposureMode.ALLOWLIST,
                conversations=frozenset({"-100111"}),
            ),
            allowed_user_ids=frozenset({"111"}),
        ),
        transport=cast("_TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert [msg.text for msg in received] == ["allowed user"]
    assert received[0].conversation_id == "111"
    assert received[0].sender_id == "111"
    assert received[0].metadata["chat_type"] == "private"


async def test_channel_posts_do_not_pass_self_exposure(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[_make_channel_post(update_id=10, chat_id=-100111, text="from channel")],
    )
    channel = TelegramChannel(
        _make_config(tmp_path, operator_id="111"),
        transport=cast("_TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert received == []


async def test_channel_drops_group_messages(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[
            _make_update(update_id=1, chat_id=111, sender_id=111, chat_type="group"),
            _make_update(update_id=2, chat_id=111, sender_id=111, chat_type="supergroup"),
            _make_update(update_id=3, chat_id=111, sender_id=111, chat_type="channel"),
        ],
    )
    channel = TelegramChannel(
        _make_config(tmp_path, exposure=ChannelExposure(mode=ExposureMode.OPEN)),
        transport=cast("_TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert received == []


async def test_channel_does_not_trust_private_chat_sender_as_self(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[
            _make_update(update_id=1, chat_id=111, sender_id=111, text="attacker"),
            _make_update(update_id=2, chat_id=999, sender_id=999, text="operator"),
        ],
    )
    channel = TelegramChannel(
        _make_config(tmp_path, operator_id="999"),
        transport=cast("_TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await _wait_for_received(received, 1)
    await channel.stop()

    assert [msg.text for msg in received] == ["operator"]
    assert received[0].metadata["from_self"] is False


async def test_channel_survives_transient_polling_error(tmp_path: Path) -> None:
    transport = ErrorOnFirstSuccessTransport()
    config = _make_config(tmp_path, exposure=ChannelExposure(mode=ExposureMode.OPEN))
    config = TelegramChannelConfig(
        bot_token="test-token",  # noqa: S106  # inert test token
        session_dir=tmp_path / "telegram",
        inbound_media_dir=tmp_path / "telegram" / "media",
        outbound_media_dir=tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.OPEN),
        poll_interval_seconds=0.01,
        poll_timeout_seconds=1,
    )
    channel = TelegramChannel(
        config,
        transport=cast("_TelegramTransport", transport),
    )

    await channel.start()
    # Allow enough time for error + retry + success.
    await asyncio.sleep(0.1)
    await channel.stop()

    assert transport.calls >= 2  # first errored, second succeeded
    assert (await channel.status()).provider == "telegram"


async def test_channel_identifies_bot_on_start(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(tmp_path, exposure=ChannelExposure(mode=ExposureMode.OPEN)),
        transport=cast("_TelegramTransport", transport),
    )

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert channel._bot_username == "test_bot"


# --- Outbound text tests ---


async def test_send_message_uses_plain_text(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("_TelegramTransport", transport),
    )

    await channel.send_message("123", "hello *world*")

    assert transport.calls[0][0] == "sendMessage"
    params = transport.calls[0][1]
    assert "parse_mode" not in params
    assert params["text"] == "hello *world*"
    assert params["chat_id"] == "123"


async def test_send_message_chunks_long_text(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("_TelegramTransport", transport),
    )

    long_text = "x" * 5000
    await channel.send_message("123", long_text)

    send_calls = [c for c in transport.calls if c[0] == "sendMessage"]
    assert len(send_calls) == 2
    assert len(cast("str", send_calls[0][1]["text"])) <= 4096
    assert len(cast("str", send_calls[1][1]["text"])) <= 4096


async def test_transport_rejects_bot_api_error_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, *, timeout: float) -> JsonResponse:  # noqa: ARG001
        return JsonResponse({"ok": False, "description": "bad request"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transport = _TelegramTransport(
        api_base="https://api.telegram.org",
        token="test-token",  # noqa: S106  # inert test token
        timeout=1,
    )

    with pytest.raises(_TelegramError, match="bad request"):
        await transport.call("sendMessage", chat_id="123", text="hello")


async def test_transport_rejects_upload_error_envelopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, *, timeout: float) -> JsonResponse:  # noqa: ARG001
        return JsonResponse({"ok": False, "description": "upload rejected"})

    image = tmp_path / "image.png"
    image.write_bytes(b"image-data")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transport = _TelegramTransport(
        api_base="https://api.telegram.org",
        token="test-token",  # noqa: S106  # inert test token
        timeout=1,
    )

    with pytest.raises(_TelegramError, match="upload rejected"):
        await transport.upload("sendPhoto", file_field="photo", file_path=image, chat_id="123")


# --- Outbound media tests ---


async def test_send_media_uses_expected_upload_method(tmp_path: Path) -> None:
    cases: tuple[
        tuple[str, Literal["image", "video"], str | None, str, str, dict[str, object]],
        ...,
    ] = (
        (
            "image.png",
            "image",
            "cap",
            "sendPhoto",
            "photo",
            {"chat_id": "123", "caption": "cap"},
        ),
        ("clip.mp4", "video", None, "sendVideo", "video", {"chat_id": "123"}),
    )

    for filename, media_type, caption, method, file_field, params in cases:
        transport = RecordingTransport()
        path = tmp_path / filename
        path.write_bytes(b"media-data")
        channel = TelegramChannel(
            _make_config(tmp_path),
            transport=cast("_TelegramTransport", transport),
        )

        await channel.send_media(
            "123",
            ChannelMedia(path=path, media_type=media_type, caption=caption),
        )

        assert transport.uploads == [(method, file_field, path, params)]


async def test_send_media_sends_long_caption_as_text_before_upload(tmp_path: Path) -> None:
    transport = RecordingTransport()
    image = tmp_path / "image.png"
    image.write_bytes(b"image-data")
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("_TelegramTransport", transport),
    )

    caption = "x" * 1025
    await channel.send_media("123", ChannelMedia(path=image, media_type="image", caption=caption))

    assert transport.calls[0] == ("sendMessage", {"chat_id": "123", "text": caption})
    assert transport.uploads == [("sendPhoto", "photo", image, {"chat_id": "123"})]


def test_multipart_form_encodes_local_file_upload(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image-data")

    body = _encode_multipart_form(
        {"chat_id": "123", "caption": "hello"},
        file_field="photo",
        file_path=image,
        boundary="test-boundary",
    )

    assert b'name="chat_id"\r\n\r\n123' in body
    assert b'name="caption"\r\n\r\nhello' in body
    assert b'name="photo"; filename="image.png"' in body
    assert b"Content-Type: image/png" in body
    assert b"image-data" in body


def _patch_file_size(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    fake_size: int,
) -> None:
    """Patch ``os.stat`` so that *path* reports ``fake_size`` bytes.

    All other paths fall through to the real ``os.stat``.
    """

    real_stat = os.stat

    def patched_stat(
        p: str | bytes | Path,
        *,
        follow_symlinks: bool = True,  # signature must match os.stat
    ) -> os.stat_result:
        if Path(os.fsdecode(p)) == path:
            result = real_stat(p, follow_symlinks=follow_symlinks)
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    result.st_uid,
                    result.st_gid,
                    fake_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                ),
            )
        return real_stat(p, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", patched_stat)


async def test_send_media_rejects_oversized_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases: tuple[tuple[str, Literal["image", "video"], int], ...] = (
        ("big.png", "image", 11),
        ("big.mp4", "video", 11),
    )
    config = _make_config(tmp_path)
    config = replace(config, max_media_bytes=10)

    for filename, media_type, fake_size in cases:
        path = tmp_path / filename
        path.write_bytes(b"x")
        channel = TelegramChannel(config)

        with monkeypatch.context() as patch:
            _patch_file_size(patch, path, fake_size)
            with pytest.raises(ChannelMediaError, match="too large"):
                await channel.send_media(
                    "123",
                    ChannelMedia(path=path, media_type=media_type),
                )


# --- Edit message tests ---


async def test_edit_message_uses_plain_text(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("_TelegramTransport", transport),
    )

    await channel.edit_message("123", "42", "updated *text*")

    assert transport.calls[0][0] == "editMessageText"
    assert "parse_mode" not in transport.calls[0][1]
    assert transport.calls[0][1]["text"] == "updated *text*"
    assert transport.calls[0][1]["message_id"] == 42


# --- Inbound media parsing tests ---


async def test_channel_parses_inbound_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_download(monkeypatch)
    cases: tuple[tuple[str, dict[str, object], dict[str, object]], ...] = (
        (
            "photo",
            _make_update(
                chat_id=111,
                sender_id=111,
                text="",
                photo=[
                    {"file_id": "small", "file_size": 100},
                    {"file_id": "large", "file_size": 500},
                ],
            ),
            {"media_type": "image", "file_id": "large", "suffix": ".jpg"},
        ),
        (
            "voice",
            _make_update(
                chat_id=111,
                sender_id=111,
                text="",
                voice={"file_id": "voice123", "duration": 5},
            ),
            {
                "media_type": "voice",
                "file_id": "voice123",
                "suffix": ".ogg",
                "voice_path": True,
            },
        ),
        (
            "document",
            _make_update(
                chat_id=111,
                sender_id=111,
                text="",
                document={
                    "file_id": "doc123",
                    "file_name": "report.pdf",
                    "mime_type": "application/pdf",
                },
            ),
            {
                "media_type": "document",
                "file_id": "doc123",
                "file_name": "report.pdf",
                "media_mime_types": ["application/pdf"],
                "suffix": ".pdf",
            },
        ),
        (
            "video",
            _make_update(
                chat_id=111,
                sender_id=111,
                text="",
                video={
                    "file_id": "video123",
                    "file_name": "clip.mp4",
                    "mime_type": "video/mp4",
                },
            ),
            {
                "media_type": "video",
                "file_id": "video123",
                "media_mime_types": ["video/mp4"],
                "suffix": ".mp4",
            },
        ),
        (
            "video_note",
            _make_update(
                chat_id=111,
                sender_id=111,
                text="",
                video_note={"file_id": "note123", "duration": 8, "length": 240},
            ),
            {"media_type": "video", "file_id": "note123", "suffix": ".mp4"},
        ),
        (
            "document_video",
            _make_update(
                chat_id=111,
                sender_id=111,
                text="",
                document={
                    "file_id": "docvideo123",
                    "file_name": "clip.mp4",
                    "mime_type": "video/mp4",
                },
            ),
            {
                "media_type": "video",
                "file_id": "docvideo123",
                "file_name": "clip.mp4",
                "media_mime_types": ["video/mp4"],
                "suffix": ".mp4",
            },
        ),
        (
            "document_audio",
            _make_update(
                chat_id=111,
                sender_id=111,
                text="",
                document={
                    "file_id": "docaudio123",
                    "file_name": "song.mp3",
                    "mime_type": "audio/mpeg",
                },
            ),
            {
                "media_type": "voice",
                "file_id": "docaudio123",
                "file_name": "song.mp3",
                "media_mime_types": ["audio/mpeg"],
                "suffix": ".mp3",
                "voice_path": True,
            },
        ),
    )

    for name, update, expected in cases:
        case_dir = tmp_path / name
        transport = RecordingTransport(updates=[update])
        channel = TelegramChannel(
            _make_config(case_dir, operator_id="111"),
            transport=cast("_TelegramTransport", transport),
        )

        received = await _run_channel_once(channel)
        metadata = received[0].metadata
        file_id = cast("str", expected["file_id"])
        expected_path = str(case_dir / "telegram" / "media" / f"10_{file_id}{expected['suffix']}")

        assert metadata["media_type"] == expected["media_type"]
        assert metadata["file_id"] == file_id
        assert metadata["media_paths"] == [expected_path]
        assert metadata["media_path"] == expected_path
        if "file_name" in expected:
            assert metadata["file_name"] == expected["file_name"]
        if "media_mime_types" in expected:
            assert metadata["media_mime_types"] == expected["media_mime_types"]
        if expected.get("voice_path") is True:
            assert metadata["voice_path"] == expected_path
        else:
            assert "voice_path" not in metadata
        media_path = Path(cast("str", metadata["media_path"]))
        assert await asyncio.to_thread(media_path.read_bytes) == b"media-bytes"


async def test_channel_skips_oversized_inbound_media_from_get_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedGetFileTransport(RecordingTransport):
        async def call(self, method: str, **params: object) -> object:
            if method == "getFile":
                self.calls.append((method, dict(params)))
                return {
                    "file_id": params.get("file_id"),
                    "file_path": "documents/report.pdf",
                    "file_size": 11,
                }
            return await super().call(method, **params)

    _stub_download(monkeypatch)
    transport = OversizedGetFileTransport(
        updates=[
            _make_update(
                update_id=10,
                chat_id=111,
                sender_id=111,
                text="oversized",
                document={"file_id": "doc123", "file_name": "report.pdf"},
            ),
        ],
    )
    config = TelegramChannelConfig(
        bot_token="test-token",  # noqa: S106  # inert test token
        session_dir=tmp_path / "telegram",
        inbound_media_dir=tmp_path / "telegram" / "media",
        outbound_media_dir=tmp_path,
        exposure=ChannelExposure(operator_ids=frozenset({"111"})),
        poll_interval_seconds=60,
        poll_timeout_seconds=1,
        max_media_bytes=10,
    )
    channel = TelegramChannel(config, transport=cast("_TelegramTransport", transport))
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await _wait_for_received(received, 1)
    await channel.stop()

    assert received[0].text == "oversized"
    assert received[0].metadata["has_media"] is False
    assert "media_error" in received[0].metadata
    assert "media_paths" not in received[0].metadata
    assert _load_offset(config.offset_file) == 11


def test_download_file_aborts_when_stream_exceeds_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ChunkedResponse:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.chunks = [b"abc", b"def"]

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:  # noqa: ARG002  # urllib-compatible fake
            if not self.chunks:
                return b""
            return self.chunks.pop(0)

    def fake_urlopen(request: object, *, timeout: float) -> ChunkedResponse:  # noqa: ARG001
        return ChunkedResponse()

    destination = tmp_path / "media.bin"
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ChannelMediaError, match="exceeds 5"):
        _download_file("https://example.com/file", destination, 1, 5)

    assert not destination.exists()


# --- Offset persistence tests (ticket 23) ---


def test_offset_round_trip_uses_atomic_file_replace(tmp_path: Path) -> None:
    offset_file = tmp_path / "telegram_offset.json"

    _save_offset(offset_file, 100)
    loaded = _load_offset(offset_file)

    assert loaded == 100
    assert offset_file.is_file()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["telegram_offset.json"]
    assert json.loads(offset_file.read_text(encoding="utf-8")) == {"offset": 100}


def test_offset_invalid_or_missing_file_returns_zero(tmp_path: Path) -> None:
    cases: tuple[tuple[str, str | None], ...] = (
        ("missing.json", None),
        ("corrupt.json", "not valid json {{{"),
        ("invalid.json", json.dumps({"offset": -5})),
    )

    for filename, content in cases:
        offset_file = tmp_path / filename
        if content is not None:
            offset_file.write_text(content, encoding="utf-8")
        assert _load_offset(offset_file) == 0


async def test_channel_persists_offset_after_polling(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[
            _make_update(update_id=10, chat_id=111, sender_id=111, text="msg1"),
            _make_update(update_id=11, chat_id=111, sender_id=111, text="msg2"),
        ],
    )
    config = _make_config(tmp_path, operator_id="111")
    channel = TelegramChannel(config, transport=cast("_TelegramTransport", transport))

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    offset = _load_offset(config.offset_file)
    assert offset == 12  # last update_id (11) + 1


async def test_channel_loads_persisted_offset_on_start(tmp_path: Path) -> None:
    config = _make_config(tmp_path, operator_id="111")
    _save_offset(config.offset_file, 100)

    transport = RecordingTransport(
        updates=[_make_update(update_id=101, chat_id=111, sender_id=111, text="msg1")],
    )
    channel = TelegramChannel(config, transport=cast("_TelegramTransport", transport))

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    # The getUpdates call should have used offset=100.
    get_updates_calls = [c for c in transport.calls if c[0] == "getUpdates"]
    assert get_updates_calls[0][1]["offset"] == 100


async def test_channel_advances_offset_past_failed_media_update(tmp_path: Path) -> None:
    class FailingGetFileTransport(RecordingTransport):
        async def call(self, method: str, **params: object) -> object:
            if method == "getFile":
                self.calls.append((method, dict(params)))
                msg = "temporary getFile failure"
                raise _TelegramError(msg)
            return await super().call(method, **params)

    transport = FailingGetFileTransport(
        updates=[
            _make_update(update_id=10, chat_id=111, sender_id=111, text="before media"),
            _make_update(
                update_id=11,
                chat_id=111,
                sender_id=111,
                text="media",
                photo=[{"file_id": "photo123", "file_size": 100}],
            ),
        ],
    )
    config = _make_config(tmp_path, operator_id="111")
    channel = TelegramChannel(config, transport=cast("_TelegramTransport", transport))
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await _wait_for_received(received, 2)
    await channel.stop()

    assert [msg.text for msg in received] == ["before media", "media"]
    assert _load_offset(config.offset_file) == 12


async def _wait_for_received(messages: list[ChannelMessage], count: int) -> None:
    deadline = asyncio.get_running_loop().time() + 2
    while asyncio.get_running_loop().time() < deadline:
        if len(messages) >= count:
            return
        await asyncio.sleep(0.01)
    msg = f"received {len(messages)} message(s), expected {count}"
    raise AssertionError(msg)


async def _wait_for_reactions(reactions: list[ChannelReaction], count: int) -> None:
    deadline = asyncio.get_running_loop().time() + 2
    while asyncio.get_running_loop().time() < deadline:
        if len(reactions) >= count:
            return
        await asyncio.sleep(0.01)
    msg = f"received {len(reactions)} reaction(s), expected {count}"
    raise AssertionError(msg)
