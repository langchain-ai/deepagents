from __future__ import annotations

from pathlib import Path

import pytest

from deepagents_talon.channels.base import (
    ChannelExposure,
    ChannelMediaError,
    ExposureMode,
    chunk_text,
    format_markdown_for_channel,
    validate_media,
)
from deepagents_talon.interfaces import ChannelMedia, ChannelMessage


def test_default_exposure_allows_only_self_messages() -> None:
    exposure = ChannelExposure(operator_id="operator")

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
