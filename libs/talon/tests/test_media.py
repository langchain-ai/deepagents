from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from deepagents_talon.interfaces import ChannelMedia
from deepagents_talon.media import (
    MAX_TEXT_DOCUMENT_BYTES,
    MarkdownMediaRef,
    build_inbound_text,
    build_model_content,
    extract_markdown_media,
    outbound_channel_media,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_inbound_text_injects_readable_document(tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("hello from a document", encoding="utf-8")

    text = build_inbound_text(
        "see attached",
        {"media_type": "document", "media_paths": [str(document)]},
    )

    assert "see attached" in text
    assert "[Content of notes.txt]:" in text
    assert "hello from a document" in text


def test_build_inbound_text_bounds_document_extraction(tmp_path: Path) -> None:
    document = tmp_path / "large.txt"
    document.write_bytes(b"x" * (MAX_TEXT_DOCUMENT_BYTES + 1))

    text = build_inbound_text("", {"media_type": "document", "media_paths": [str(document)]})

    assert "too large to read inline" in text
    assert "large.txt" in text


def test_build_model_content_uses_inbound_image_data_url(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not-a-real-png")

    content = build_model_content(
        "look",
        {
            "media_type": "image",
            "media_paths": [str(image)],
            "media_mime_types": ["image/png"],
        },
    )

    assert isinstance(content, list)
    image_url = cast("dict[str, str]", content[1]["image_url"])
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image_url"
    assert image_url["url"].startswith("data:image/png;base64,")


def test_extract_markdown_media_ignores_code_spans(tmp_path: Path) -> None:
    image = tmp_path / "chart.png"
    ignored = tmp_path / "no.png"
    text = f"send ![chart]({image}) but keep `![code]({ignored})`"

    cleaned, refs = extract_markdown_media(text)

    assert refs == [MarkdownMediaRef(alt="chart", path=image)]
    assert "![chart]" not in cleaned
    assert f"`![code]({ignored})`" in cleaned


def test_outbound_channel_media_rejects_unsupported_file(tmp_path: Path) -> None:
    document = tmp_path / "readme.txt"

    with pytest.raises(ValueError, match="unsupported outbound media"):
        outbound_channel_media(MarkdownMediaRef(alt="doc", path=document))


def test_outbound_channel_media_builds_video_payload(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    ref = MarkdownMediaRef(alt="clip", path=video)

    media = outbound_channel_media(ref, caption="caption")

    assert media == ChannelMedia(path=video, media_type="video", caption="caption")
