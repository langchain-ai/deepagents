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
