"""Tests for the summarization-on-overflow tail-clipping path (`_overflow_clip`).

Focus: `read_file` tail ToolMessages must keep their media (image/audio/video)
blocks and must not get a truncation notice unless the text actually overflows.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from deepagents.middleware._overflow_clip import (
    _READ_FILE_CLIP_CHARS,
    _clip_overflow_tail,
    _slice_read_file_tm,
)


class _DummyBackend:
    """Backend stub; unused on the read_file clip path but required by the signature."""

    def write(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover
        pytest.fail("read_file clipping must not write to the backend")


def _read_file_ai(tool_call_id: str, file_path: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "read_file", "args": {"file_path": file_path}}],
    )


_IMAGE_BLOCK = {"type": "image", "base64": "aGVsbG8=", "mime_type": "image/png"}


def _clip(messages: list) -> list:
    new_messages, _ = _clip_overflow_tail(
        messages,
        _DummyBackend(),
        keep=("tokens", 1),  # threshold of 1 token so clipping always engages
        max_input_tokens=1000,
        token_counter=lambda _msgs: 10_000,  # force the tail batch over threshold
        large_tool_results_prefix="/large_tool_results",
    )
    return new_messages


def _has_image_block(msg: ToolMessage) -> bool:
    return isinstance(msg.content, list) and any(
        isinstance(b, dict) and b.get("type") == "image" for b in msg.content
    )


def test_image_only_read_file_is_left_untouched() -> None:
    """A read_file result that is a bare image block must not be dropped or noticed."""
    tm = ToolMessage(tool_call_id="c1", name="read_file", content=[_IMAGE_BLOCK])
    assert _slice_read_file_tm(tm, "/pic.png") is None

    out = _clip([_read_file_ai("c1", "/pic.png"), tm])
    assert _has_image_block(out[-1])
    assert "Output was truncated" not in str(out[-1].content)


def test_small_text_read_file_gets_no_truncation_notice() -> None:
    """A short read_file result must be returned as-is, without a false notice."""
    tm = ToolMessage(tool_call_id="c1", name="read_file", content="short result")
    assert _slice_read_file_tm(tm, "/f.txt") is None

    out = _clip([_read_file_ai("c1", "/f.txt"), tm])
    assert out[-1].content == "short result"
    assert "Output was truncated" not in str(out[-1].content)


def test_large_text_read_file_is_truncated_with_notice() -> None:
    """A large read_file text result is head-sliced and gets the recovery notice."""
    big = "x" * (_READ_FILE_CLIP_CHARS + 500)
    tm = ToolMessage(tool_call_id="c1", name="read_file", content=big)

    out = _clip([_read_file_ai("c1", "/big.txt"), tm])
    content = out[-1].content
    assert isinstance(content, str)
    assert "Output was truncated" in content
    assert "/big.txt" in content
    # Exactly _READ_FILE_CLIP_CHARS of source text, then the recovery notice.
    assert content[:_READ_FILE_CLIP_CHARS] == "x" * _READ_FILE_CLIP_CHARS
    assert content[_READ_FILE_CLIP_CHARS:].startswith("\n\n[Output was truncated")


def test_large_text_plus_image_keeps_image_and_truncates_text() -> None:
    """When a read_file result mixes large text and an image, keep the image, clip the text."""
    big = "x" * (_READ_FILE_CLIP_CHARS + 500)
    tm = ToolMessage(
        tool_call_id="c1",
        name="read_file",
        content=[{"type": "text", "text": big}, _IMAGE_BLOCK],
    )

    out = _clip([_read_file_ai("c1", "/mixed.png"), tm])
    content = out[-1].content
    assert isinstance(content, list)
    assert any(isinstance(b, dict) and b.get("type") == "image" for b in content)
    text_parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
    assert any("Output was truncated" in t for t in text_parts)
