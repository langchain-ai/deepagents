"""Unit tests for whatsapp_adapter helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from whatsapp_adapter import extract_markdown_images


class TestExtractMarkdownImages:
    def test_no_image_refs_returns_unchanged(self) -> None:
        text = "Hello world"
        cleaned, refs = extract_markdown_images(text)
        assert cleaned == "Hello world"
        assert refs == []

    def test_single_image_extracted(self) -> None:
        text = "Here: ![chart](/tmp/c.png)"
        cleaned, refs = extract_markdown_images(text)
        assert refs == [("chart", "/tmp/c.png")]
        assert "![chart]" not in cleaned
        assert "/tmp/c.png" not in cleaned
        assert cleaned.startswith("Here:")

    def test_multiple_images_preserve_order(self) -> None:
        text = "a ![one](/a.png) b ![two](/b.png) c"
        cleaned, refs = extract_markdown_images(text)
        assert refs == [("one", "/a.png"), ("two", "/b.png")]
        assert "a " in cleaned and " b " in cleaned and " c" in cleaned

    def test_empty_alt_preserved(self) -> None:
        text = "before ![](/tmp/x.png) after"
        cleaned, refs = extract_markdown_images(text)
        assert refs == [("", "/tmp/x.png")]

    def test_image_in_fenced_block_ignored(self) -> None:
        text = "outside ![x](/outside.png)\n```\n![inside](/inside.png)\n```"
        cleaned, refs = extract_markdown_images(text)
        assert refs == [("x", "/outside.png")]
        assert "![inside](/inside.png)" in cleaned

    def test_image_in_inline_code_ignored(self) -> None:
        text = "Use `![alt](/tmp/x.png)` syntax"
        cleaned, refs = extract_markdown_images(text)
        assert refs == []
        assert "`![alt](/tmp/x.png)`" in cleaned

    def test_excessive_blank_lines_collapsed(self) -> None:
        text = "Here:\n\n![c](/c.png)\n\nDone"
        cleaned, refs = extract_markdown_images(text)
        assert refs == [("c", "/c.png")]
        assert "\n\n\n" not in cleaned
        assert "Here:" in cleaned
        assert "Done" in cleaned


from whatsapp_adapter import MessageEvent, MessageType, _build_inbound_content


def _photo_event(text: str, urls: list[Path], mimes: list[str]) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.PHOTO,
        chat_id="c",
        chat_name="chat",
        chat_type="dm",
        user_id="u",
        user_name="user",
        media_urls=[str(p) for p in urls],
        media_types=mimes,
    )


class TestBuildInboundContent:
    def test_text_event_returns_plain_string(self) -> None:
        event = MessageEvent(
            text="hi",
            message_type=MessageType.TEXT,
            chat_id="c",
            chat_name="chat",
            chat_type="dm",
            user_id="u",
            user_name="user",
        )
        assert _build_inbound_content(event) == "hi"

    def test_photo_event_with_one_image(self, tiny_png: Path) -> None:
        event = _photo_event("look", [tiny_png], ["image/png"])
        content = _build_inbound_content(event)
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "look"}
        assert len(content) == 2
        block = content[1]
        assert block["type"] == "image_url"
        url = block["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert len(url) > len("data:image/png;base64,")

    def test_photo_event_with_empty_body_uses_placeholder(
        self, tiny_png: Path
    ) -> None:
        event = _photo_event("", [tiny_png], ["image/png"])
        content = _build_inbound_content(event)
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "(image)"}

    def test_photo_event_with_two_images(
        self, tiny_png: Path, tiny_jpeg: Path
    ) -> None:
        event = _photo_event(
            "both", [tiny_png, tiny_jpeg], ["image/png", "image/jpeg"]
        )
        content = _build_inbound_content(event)
        assert isinstance(content, list)
        assert len(content) == 3
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[2]["type"] == "image_url"
        assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_oversize_image_dropped(
        self, tiny_png: Path, oversize_image: Path
    ) -> None:
        event = _photo_event(
            "hi",
            [oversize_image, tiny_png],
            ["image/png", "image/png"],
        )
        content = _build_inbound_content(event)
        assert isinstance(content, list)
        assert len(content) == 2  # text + 1 surviving image

    def test_unreadable_image_dropped(
        self, tiny_png: Path, tmp_path: Path
    ) -> None:
        missing = tmp_path / "missing.png"  # never created
        event = _photo_event(
            "hi",
            [missing, tiny_png],
            ["image/png", "image/png"],
        )
        content = _build_inbound_content(event)
        assert isinstance(content, list)
        assert len(content) == 2  # text + 1 surviving image
