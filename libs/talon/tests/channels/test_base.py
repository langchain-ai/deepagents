from __future__ import annotations

from pathlib import Path

import pytest

from deepagents_talon.channels.base import (
    DEFAULT_MAX_MEDIA_BYTES,
    MAX_INLINE_PHOTO_BYTES,
    ChannelExposure,
    ChannelMediaError,
    ExposureMode,
    chunk_text,
    format_markdown_for_channel,
    max_media_bytes_from_env,
    media_type_for_path,
    message_with_media_paths,
    send_with_retry,
    validate_media,
)
from deepagents_talon.interfaces import ChannelMedia, ChannelMessage, SendResult


def test_default_exposure_allows_only_self_messages() -> None:
    exposure = ChannelExposure(operator_ids=frozenset({"operator"}))

    assert exposure.operator_ids == frozenset({"operator"})
    assert exposure.allows(ChannelMessage(conversation_id="chat", text="hi", sender_id="operator"))
    assert exposure.allows(
        ChannelMessage(
            conversation_id="chat",
            text="hi",
            sender_id="other",
            metadata={"from_self": True},
        ),
    )
    assert not exposure.allows(ChannelMessage(conversation_id="chat", text="hi", sender_id="other"))


def test_default_exposure_allows_multiple_operator_ids() -> None:
    exposure = ChannelExposure(
        operator_ids=frozenset({"operator", "backup-operator"}),
    )

    assert exposure.operator_ids == frozenset({"operator", "backup-operator"})
    assert exposure.allows(ChannelMessage(conversation_id="chat", text="hi", sender_id="operator"))
    assert exposure.allows(
        ChannelMessage(conversation_id="chat", text="hi", sender_id="backup-operator")
    )
    assert not exposure.allows(ChannelMessage(conversation_id="chat", text="hi", sender_id="other"))


def test_allowlist_exposure_allows_chats_and_mention_patterns() -> None:
    exposure = ChannelExposure(
        mode=ExposureMode.ALLOWLIST,
        conversations=frozenset({"allowed"}),
        mention_patterns=("@agent *",),
    )

    assert exposure.allows(ChannelMessage(conversation_id="allowed", text="anything"))
    assert exposure.allows(ChannelMessage(conversation_id="other", text="@agent help"))
    assert not exposure.allows(ChannelMessage(conversation_id="other", text="ignore"))


def test_open_exposure_allows_any_message() -> None:
    exposure = ChannelExposure(mode=ExposureMode.OPEN)

    assert exposure.allows(ChannelMessage(conversation_id="chat", text="hi", sender_id="other"))


def test_format_markdown_for_channel() -> None:
    text = "# Title\nUse **bold**, _italics_, and [docs](https://example.com)."

    assert (
        format_markdown_for_channel(text)
        == "Title\nUse *bold*, _italics_, and docs (https://example.com)."
    )


def test_chunk_text_prefers_word_boundaries() -> None:
    assert chunk_text("alpha beta gamma", limit=10) == ["alpha", "beta gamma"]
    assert chunk_text("abcdefghijk", limit=4) == ["abcd", "efgh", "ijk"]


def test_validate_media_accepts_matching_image(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not-really-a-png")

    media = validate_media(ChannelMedia(path=path, media_type="image", caption="caption"))

    assert media == ChannelMedia(path=path, media_type="image", caption="caption")


def test_validate_media_accepts_relative_path_under_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "image.png"
    path.write_bytes(b"not-really-a-png")

    media = validate_media(ChannelMedia(path=Path("image.png"), media_type="image"), root=root)

    assert media == ChannelMedia(path=path.resolve(), media_type="image")


@pytest.mark.parametrize("name", ["report.pdf", "data.csv", "archive.tar.gz", "README"])
def test_validate_media_accepts_any_file_as_a_document(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"contents")

    media = validate_media(ChannelMedia(path=path, media_type="document"), root=tmp_path)

    assert media == ChannelMedia(path=path.resolve(), media_type="document")


@pytest.mark.parametrize(
    ("name", "size", "expected"),
    [
        ("photo.jpg", 10, "image"),
        ("photo.png", MAX_INLINE_PHOTO_BYTES, "image"),
        ("photo.png", MAX_INLINE_PHOTO_BYTES + 1, "document"),
        ("chart.svg", 10, "document"),
        ("clip.mp4", 10, "video"),
        ("clip.mov", 10, "document"),
    ],
)
def test_only_inline_safe_media_gets_a_native_category(
    tmp_path: Path, name: str, size: int, expected: str
) -> None:
    path = tmp_path / name
    with path.open("wb") as stream:
        stream.truncate(size)

    assert media_type_for_path(path) == expected


def test_validate_media_rejects_configured_global_cap(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"abcd")

    with pytest.raises(ChannelMediaError, match="exceeds 3"):
        validate_media(ChannelMedia(path=path, media_type="image"), max_bytes=3)


def test_validate_media_rejects_path_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not-really-a-png")

    with pytest.raises(ChannelMediaError, match="escapes outbound root"):
        validate_media(ChannelMedia(path=outside, media_type="image"), root=root)


def test_validate_media_rejects_type_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not-really-a-png")

    with pytest.raises(ChannelMediaError, match="does not match"):
        validate_media(ChannelMedia(path=path, media_type="video"))


def test_message_with_media_paths_preserves_provider_media_presence() -> None:
    message = ChannelMessage(
        conversation_id="chat",
        text="",
        metadata={"media_type": "voice"},
    )

    with_media = message_with_media_paths(
        message,
        media_paths=[],
        mime_types=[],
        has_media=True,
    )

    assert "media_paths" not in with_media.metadata
    assert "media_path" not in with_media.metadata
    assert "media_mime_types" not in with_media.metadata
    assert "voice_path" not in with_media.metadata
    assert with_media.metadata["has_media"] is True


def test_message_with_media_paths_adds_voice_path_only_for_voice() -> None:
    voice = message_with_media_paths(
        ChannelMessage(conversation_id="chat", text="", metadata={"media_type": "voice"}),
        media_paths=["voice.ogg"],
    )
    video = message_with_media_paths(
        ChannelMessage(
            conversation_id="chat",
            text="",
            metadata={"media_type": "video", "voice_path": None},
        ),
        media_paths=["clip.mp4"],
    )

    assert voice.metadata["voice_path"] == "voice.ogg"
    assert "voice_path" not in video.metadata


def test_max_media_bytes_from_env_defaults_to_one_gb() -> None:
    assert max_media_bytes_from_env({}) == DEFAULT_MAX_MEDIA_BYTES


def test_max_media_bytes_from_env_accepts_positive_integer() -> None:
    assert max_media_bytes_from_env({"DEEPAGENTS_TALON_MAX_MEDIA_BYTES": "123"}) == 123


def test_max_media_bytes_from_env_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        max_media_bytes_from_env({"DEEPAGENTS_TALON_MAX_MEDIA_BYTES": "0"})


async def test_send_with_retry_converts_exception_to_failed_result() -> None:
    async def raising_send() -> SendResult:
        msg = "transport crashed"
        raise RuntimeError(msg)

    result = await send_with_retry(raising_send, max_retries=0)

    assert result.success is False
    assert "transport crashed" in (result.error or "")
    assert result.retryable is True


async def test_send_with_retry_retries_after_exception() -> None:
    calls = 0

    async def flaky_send() -> SendResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            msg = "connection reset"
            raise RuntimeError(msg)
        return SendResult(success=True)

    result = await send_with_retry(flaky_send, max_retries=2, base_delay=0.01)

    assert result.success is True
    assert calls == 2
