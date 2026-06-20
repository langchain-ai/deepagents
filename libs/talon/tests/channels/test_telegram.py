from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import cast

import pytest

import deepagents_talon.channels.telegram as telegram_module
from deepagents_talon.channels.base import (
    ChannelExposure,
    ChannelMediaError,
    ExposureMode,
)
from deepagents_talon.channels.telegram import (
    MAX_DOCUMENT_BYTES,
    MAX_PHOTO_BYTES,
    TelegramChannel,
    TelegramChannelConfig,
    TelegramError,
    TelegramTransport,
    _encode_multipart_form,
    _escape_markdown_v2,
    _load_offset,
    _save_offset,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import ChannelMedia, ChannelMessage


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

    async def call(self, method: str, **params: object) -> object:
        self.calls.append((method, dict(params)))
        if method == "getMe":
            self._get_me_called = True
            return {"ok": True, "result": {"id": 123456, "username": "test_bot"}}
        if method == "getUpdates":
            updates = self.updates
            self.updates = []
            return {"ok": True, "result": updates}
        if method == "sendChatAction":
            return {"ok": True, "result": True}
        if method in ("sendMessage", "sendPhoto", "sendDocument", "editMessageText"):
            return {"ok": True, "result": {"message_id": 42}}
        if method == "getFile":
            file_id = params.get("file_id")
            if file_id == "voice123":
                file_path = "voice/file.ogg"
            elif file_id == "doc123":
                file_path = "documents/report.pdf"
            else:
                file_path = "photos/file.jpg"
            return {"ok": True, "result": {"file_id": file_id, "file_path": file_path}}
        return {"ok": True, "result": True}

    async def upload(
        self,
        method: str,
        *,
        file_field: str,
        file_path: Path,
        **params: object,
    ) -> object:
        self.uploads.append((method, file_field, file_path, dict(params)))
        return {"ok": True, "result": {"message_id": 42}}


class ErrorOnFirstSuccessTransport:
    """Transport that raises on the first getUpdates, then returns empty."""

    def __init__(self) -> None:
        self.calls = 0

    async def call(self, method: str, **params: object) -> object:  # noqa: ARG002  # test fake
        if method == "getMe":
            return {"ok": True, "result": {"id": 123456, "username": "test_bot"}}
        if method == "getUpdates":
            self.calls += 1
            if self.calls == 1:
                msg = "network error"
                raise TelegramError(msg)
            return {"ok": True, "result": []}
        return {"ok": True, "result": True}


class MarkdownV2ErrorTransport:
    """Transport that fails MarkdownV2 messages, forcing plain-text fallback."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(self, method: str, **params: object) -> object:
        self.calls.append((method, dict(params)))
        if method == "getMe":
            return {"ok": True, "result": {"id": 123456, "username": "test_bot"}}
        if method == "getUpdates":
            return {"ok": True, "result": []}
        if params.get("parse_mode") == "MarkdownV2":
            msg = "Bad Request: can't parse entities"
            raise TelegramError(msg)
        return {"ok": True, "result": {"message_id": 42}}


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
    if document is not None:
        message["document"] = document
    return {"update_id": update_id, "message": message}


def _make_config(
    tmp_path: Path,
    *,
    exposure: ChannelExposure | None = None,
    operator_id: str | None = None,
) -> TelegramChannelConfig:
    return TelegramChannelConfig(
        bot_token="test-token",  # noqa: S106  # inert test token
        session_dir=tmp_path / "telegram",
        inbound_media_dir=tmp_path / "telegram" / "media",
        outbound_media_dir=tmp_path,
        exposure=exposure or ChannelExposure(operator_id=operator_id),
        poll_interval_seconds=60,
        poll_timeout_seconds=1,
        operator_id=operator_id,
    )


def _stub_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def download(url: str, destination: Path, timeout: float) -> None:  # noqa: ARG001
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_bytes(b"media-bytes")
        destination.chmod(0o600)

    monkeypatch.setattr(telegram_module, "_download_file", download)


# --- Config tests ---


def test_config_from_talon_env_maps_telegram_values(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN": "secret-token",
            "DEEPAGENTS_TALON_TELEGRAM_EXPOSURE": "allowlist",
            "DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS": "123, 456",
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
        operator_id="999",
        conversations=frozenset({"123", "456"}),
    )
    assert telegram.poll_timeout_seconds == 45.0
    assert telegram.poll_interval_seconds == 2.0


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


# --- MarkdownV2 escaping tests ---


def test_escape_markdown_v2_escapes_special_chars() -> None:
    assert _escape_markdown_v2("hello *world*") == r"hello \*world\*"
    assert _escape_markdown_v2("a_b_c") == r"a\_b\_c"
    assert _escape_markdown_v2("list [1]") == r"list \[1\]"
    assert _escape_markdown_v2("not! really.") == r"not\! really\."
    assert _escape_markdown_v2("plain text") == "plain text"


def test_escape_markdown_v2_escapes_backslash() -> None:
    assert _escape_markdown_v2(r"path\to") == r"path\\to"


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
        transport=cast("TelegramTransport", transport),
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
        transport=cast("TelegramTransport", transport),
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
        transport=cast("TelegramTransport", transport),
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
        transport=cast("TelegramTransport", transport),
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
        transport=cast("TelegramTransport", transport),
    )

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    assert channel._bot_username == "test_bot"


# --- Outbound text tests ---


async def test_send_message_uses_markdown_v2(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("TelegramTransport", transport),
    )

    await channel.send_message("123", "hello *world*")

    assert transport.calls[0][0] == "sendMessage"
    params = transport.calls[0][1]
    assert params["parse_mode"] == "MarkdownV2"
    assert params["text"] == r"hello \*world\*"
    assert params["chat_id"] == "123"


async def test_send_message_falls_back_to_plain_text(tmp_path: Path) -> None:
    transport = MarkdownV2ErrorTransport()
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("TelegramTransport", transport),
    )

    await channel.send_message("123", "hello *world*")

    # First call uses MarkdownV2, second is plain text fallback.
    assert transport.calls[0][1]["parse_mode"] == "MarkdownV2"
    assert "parse_mode" not in transport.calls[1][1]
    assert transport.calls[1][1]["text"] == "hello *world*"


async def test_send_message_chunks_long_text(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("TelegramTransport", transport),
    )

    long_text = "x" * 5000
    await channel.send_message("123", long_text)

    send_calls = [c for c in transport.calls if c[0] == "sendMessage"]
    assert len(send_calls) == 2
    assert len(cast("str", send_calls[0][1]["text"])) <= 4096
    assert len(cast("str", send_calls[1][1]["text"])) <= 4096


# --- Outbound media tests ---


async def test_send_media_sends_photo_for_images(tmp_path: Path) -> None:
    transport = RecordingTransport()
    image = tmp_path / "image.png"
    image.write_bytes(b"image-data")
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("TelegramTransport", transport),
    )

    await channel.send_media("123", ChannelMedia(path=image, media_type="image", caption="cap"))

    assert transport.uploads == [
        (
            "sendPhoto",
            "photo",
            image,
            {"chat_id": "123", "caption": "cap", "parse_mode": "MarkdownV2"},
        ),
    ]


async def test_send_media_sends_document_for_videos(tmp_path: Path) -> None:
    transport = RecordingTransport()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-data")
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("TelegramTransport", transport),
    )

    await channel.send_media("123", ChannelMedia(path=video, media_type="video"))

    assert transport.uploads == [("sendDocument", "document", video, {"chat_id": "123"})]


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


async def test_send_media_rejects_oversized_photo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "big.png"
    image.write_bytes(b"x")
    channel = TelegramChannel(_make_config(tmp_path))

    _patch_file_size(monkeypatch, image, MAX_PHOTO_BYTES + 1)

    with pytest.raises(ChannelMediaError, match="too large for Telegram"):
        await channel.send_media("123", ChannelMedia(path=image, media_type="image"))


async def test_send_media_rejects_oversized_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "big.mp4"
    video.write_bytes(b"x")
    channel = TelegramChannel(_make_config(tmp_path))

    _patch_file_size(monkeypatch, video, MAX_DOCUMENT_BYTES + 1)

    with pytest.raises(ChannelMediaError, match="too large for Telegram"):
        await channel.send_media("123", ChannelMedia(path=video, media_type="video"))


# --- Edit message tests ---


async def test_edit_message_uses_markdown_v2(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("TelegramTransport", transport),
    )

    await channel.edit_message("123", "42", "updated *text*")

    assert transport.calls[0][0] == "editMessageText"
    assert transport.calls[0][1]["parse_mode"] == "MarkdownV2"
    assert transport.calls[0][1]["text"] == r"updated \*text\*"
    assert transport.calls[0][1]["message_id"] == 42


async def test_edit_message_falls_back_to_plain_text(tmp_path: Path) -> None:
    transport = MarkdownV2ErrorTransport()
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("TelegramTransport", transport),
    )

    await channel.edit_message("123", "42", "updated *text*")

    edit_calls = [c for c in transport.calls if c[0] == "editMessageText"]
    assert len(edit_calls) == 2
    assert edit_calls[0][1]["parse_mode"] == "MarkdownV2"
    assert "parse_mode" not in edit_calls[1][1]
    assert edit_calls[1][1]["text"] == "updated *text*"


# --- Typing indicator tests ---


async def test_send_typing_calls_send_chat_action(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("TelegramTransport", transport),
    )

    await channel.send_typing("123")

    assert transport.calls[0] == ("sendChatAction", {"chat_id": "123", "action": "typing"})


async def test_send_typing_swallows_errors(tmp_path: Path) -> None:
    class FailingTransport:
        async def call(self, method: str, **params: object) -> object:  # noqa: ARG002  # test fake
            msg = "network error"
            raise TelegramError(msg)

    channel = TelegramChannel(
        _make_config(tmp_path),
        transport=cast("TelegramTransport", FailingTransport()),
    )

    # Should not raise.
    await channel.send_typing("123")


# --- Inbound media parsing tests ---


async def test_channel_parses_inbound_photo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_download(monkeypatch)
    transport = RecordingTransport(
        updates=[
            _make_update(
                update_id=1,
                chat_id=111,
                sender_id=111,
                text="",
                photo=[
                    {"file_id": "small", "file_size": 100},
                    {"file_id": "large", "file_size": 500},
                ],
            ),
        ],
    )
    channel = TelegramChannel(
        _make_config(tmp_path, operator_id="111"),
        transport=cast("TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await _wait_for_received(received, 1)
    await channel.stop()

    assert received[0].metadata["media_type"] == "image"
    assert received[0].metadata["file_id"] == "large"
    assert received[0].metadata["media_paths"] == [
        str(tmp_path / "telegram" / "media" / "10_large.jpg"),
    ]
    assert received[0].metadata["media_path"] == str(
        tmp_path / "telegram" / "media" / "10_large.jpg",
    )
    media_path = Path(cast("str", received[0].metadata["media_path"]))
    assert await asyncio.to_thread(media_path.read_bytes) == b"media-bytes"


async def test_channel_parses_inbound_voice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_download(monkeypatch)
    transport = RecordingTransport(
        updates=[
            _make_update(
                update_id=1,
                chat_id=111,
                sender_id=111,
                text="",
                voice={"file_id": "voice123", "duration": 5},
            ),
        ],
    )
    channel = TelegramChannel(
        _make_config(tmp_path, operator_id="111"),
        transport=cast("TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await _wait_for_received(received, 1)
    await channel.stop()

    assert received[0].metadata["media_type"] == "voice"
    assert received[0].metadata["file_id"] == "voice123"
    assert received[0].metadata["voice_path"] == str(
        tmp_path / "telegram" / "media" / "10_voice123.ogg",
    )


async def test_channel_parses_inbound_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_download(monkeypatch)
    transport = RecordingTransport(
        updates=[
            _make_update(
                update_id=1,
                chat_id=111,
                sender_id=111,
                text="",
                document={
                    "file_id": "doc123",
                    "file_name": "report.pdf",
                    "mime_type": "application/pdf",
                },
            ),
        ],
    )
    channel = TelegramChannel(
        _make_config(tmp_path, operator_id="111"),
        transport=cast("TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def record(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(record)

    await channel.start()
    await _wait_for_received(received, 1)
    await channel.stop()

    assert received[0].metadata["media_type"] == "document"
    assert received[0].metadata["file_id"] == "doc123"
    assert received[0].metadata["file_name"] == "report.pdf"
    assert received[0].metadata["media_mime_types"] == ["application/pdf"]
    assert received[0].metadata["media_paths"] == [
        str(tmp_path / "telegram" / "media" / "10_doc123.pdf"),
    ]


# --- Status tests ---


async def test_status_reports_disconnected_before_start(tmp_path: Path) -> None:
    channel = TelegramChannel(_make_config(tmp_path))

    status = await channel.status()

    assert status.provider == "telegram"
    assert status.connected is False


async def test_status_reports_connected_after_start(tmp_path: Path) -> None:
    transport = RecordingTransport()
    channel = TelegramChannel(
        _make_config(tmp_path, exposure=ChannelExposure(mode=ExposureMode.OPEN)),
        transport=cast("TelegramTransport", transport),
    )

    await channel.start()
    await asyncio.sleep(0)
    status = await channel.status()
    await channel.stop()

    assert status.connected is True


# --- Offset persistence tests (ticket 23) ---


def test_offset_round_trip(tmp_path: Path) -> None:
    offset_file = tmp_path / "telegram_offset.json"

    _save_offset(offset_file, 42)
    loaded = _load_offset(offset_file)

    assert loaded == 42


def test_offset_missing_file_returns_zero(tmp_path: Path) -> None:
    assert _load_offset(tmp_path / "nonexistent.json") == 0


def test_offset_corrupt_file_returns_zero(tmp_path: Path) -> None:
    offset_file = tmp_path / "telegram_offset.json"
    offset_file.write_text("not valid json {{{", encoding="utf-8")

    assert _load_offset(offset_file) == 0


def test_offset_invalid_value_returns_zero(tmp_path: Path) -> None:
    offset_file = tmp_path / "telegram_offset.json"
    offset_file.write_text(json.dumps({"offset": -5}), encoding="utf-8")

    assert _load_offset(offset_file) == 0


def test_offset_atomic_write(tmp_path: Path) -> None:
    offset_file = tmp_path / "telegram_offset.json"

    _save_offset(offset_file, 100)

    # The temp file should have been replaced by the real file.
    assert offset_file.is_file()
    assert not (tmp_path / "telegram_offset.json.tmp").exists()
    data = json.loads(offset_file.read_text(encoding="utf-8"))
    assert data == {"offset": 100}


async def test_channel_persists_offset_after_polling(tmp_path: Path) -> None:
    transport = RecordingTransport(
        updates=[
            _make_update(update_id=10, chat_id=111, sender_id=111, text="msg1"),
            _make_update(update_id=11, chat_id=111, sender_id=111, text="msg2"),
        ],
    )
    config = _make_config(tmp_path, operator_id="111")
    channel = TelegramChannel(config, transport=cast("TelegramTransport", transport))

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
    channel = TelegramChannel(config, transport=cast("TelegramTransport", transport))

    await channel.start()
    await asyncio.sleep(0)
    await channel.stop()

    # The getUpdates call should have used offset=100.
    get_updates_calls = [c for c in transport.calls if c[0] == "getUpdates"]
    assert get_updates_calls[0][1]["offset"] == 100


async def _wait_for_received(messages: list[ChannelMessage], count: int) -> None:
    for _ in range(100):
        if len(messages) >= count:
            return
        await asyncio.sleep(0)
    msg = f"received {len(messages)} message(s), expected {count}"
    raise AssertionError(msg)
