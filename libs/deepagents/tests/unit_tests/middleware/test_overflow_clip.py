"""Tests for the summarization-on-overflow tail clipping (`_overflow_clip`)."""

from __future__ import annotations

import tempfile

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware._overflow_clip import _aclip_overflow_tail, _clip_overflow_tail


def _backend() -> FilesystemBackend:
    return FilesystemBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)


def _read_file_turn(tool_call_id: str, path: str, content: str | list[dict]) -> list[AnyMessage]:
    ai = AIMessage(content="", tool_calls=[{"id": tool_call_id, "name": "read_file", "args": {"file_path": path}}])
    tm = ToolMessage(tool_call_id=tool_call_id, name="read_file", content=content)
    return [ai, tm]


def _chars(msgs: list[AnyMessage]) -> int:
    return sum(len(str(m.content)) for m in msgs)


def _mixed_batch() -> list[AnyMessage]:
    """One oversized `read_file` result plus a small result from another tool."""
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "big", "name": "read_file", "args": {"file_path": "/big.txt"}},
                {"id": "small", "name": "grep", "args": {"pattern": "x"}},
            ],
        ),
        ToolMessage(tool_call_id="big", name="read_file", content="x" * 10_000),
        ToolMessage(tool_call_id="small", name="grep", content="tiny"),
    ]


def _clip(messages: list[AnyMessage], counter, keep_tokens: int = 1) -> list[AnyMessage]:
    new_messages, _ = _clip_overflow_tail(
        messages,
        _backend(),
        keep=("tokens", keep_tokens),
        max_input_tokens=1000,
        token_counter=counter,
        large_tool_results_prefix="/large_tool_results",
    )
    return new_messages


def test_image_read_file_result_is_replaced_by_a_path_pointer() -> None:
    """A media-only `read_file` result is clipped to a pointer at its original path (#4954)."""
    messages = _read_file_turn("call_1", "/pic.png", [{"type": "image", "base64": "aGVsbG8=", "mime_type": "image/png"}])

    clipped = _clip(messages, lambda _msgs: 10_000)[-1]

    assert isinstance(clipped.content, str)
    assert "Media content was removed" in clipped.content
    assert "/pic.png" in clipped.content
    # The inline payload must not survive -- carrying it over would defeat the clip.
    assert "aGVsbG8=" not in clipped.content


def test_small_results_are_left_untouched() -> None:
    """Clipping stops once the batch fits, so small siblings keep their content (#4954).

    The sibling is a non-`read_file` result, which is always offloaded when
    selected -- so surviving intact proves it was never selected.
    """
    messages = _mixed_batch()

    # Batch is 10_004 "tokens"; clipping the big result alone drops it to 4.
    new_messages = _clip(messages, _chars, keep_tokens=5_000)

    assert "Output was truncated" in new_messages[-2].content
    assert new_messages[-1].content == "tiny"


def test_nothing_to_clip_leaves_messages_alone() -> None:
    """A short text result gains no false truncation notice (#4954)."""
    messages = _read_file_turn("call_1", "/f.txt", "short")

    new_messages = _clip(messages, lambda _msgs: 10_000)

    assert new_messages is messages
    assert new_messages[-1].content == "short"


@pytest.mark.asyncio
async def test_async_clip_matches_sync_selection() -> None:
    """The async path clips the same subset as the sync path (#4954)."""
    new_messages, _ = await _aclip_overflow_tail(
        _mixed_batch(),
        _backend(),
        keep=("tokens", 5_000),
        max_input_tokens=1000,
        token_counter=_chars,
        large_tool_results_prefix="/large_tool_results",
    )

    assert "Output was truncated" in new_messages[-2].content
    assert new_messages[-1].content == "tiny"
