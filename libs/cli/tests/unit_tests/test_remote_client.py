"""Tests for _convert_stream_chunk, _StreamConverter, and _to_uuid."""

import uuid
from typing import Any, NamedTuple

import pytest

from deepagents_cli.remote_client import (
    _convert_stream_chunk,
    _StreamConverter,
    _to_uuid,
)


class StreamPart(NamedTuple):
    event: str
    data: Any


def test_error_event_raises() -> None:
    """Error event with a dict containing 'message' raises RuntimeError."""
    chunk = StreamPart(event="error", data={"message": "something broke"})
    with pytest.raises(RuntimeError, match="something broke"):
        _convert_stream_chunk(chunk, modes=["updates"])


def test_error_event_raises_non_dict() -> None:
    """Error event with a plain string raises RuntimeError."""
    chunk = StreamPart(event="error", data="plain error text")
    with pytest.raises(RuntimeError, match="plain error text"):
        _convert_stream_chunk(chunk, modes=["updates"])


def test_metadata_event_ignored() -> None:
    """Metadata events should be silently ignored (empty list)."""
    chunk = StreamPart(event="metadata", data={"run_id": "abc"})
    result = _convert_stream_chunk(chunk, modes=["updates"])
    assert result == []


def test_end_event_ignored() -> None:
    """End events should be silently ignored (empty list)."""
    chunk = StreamPart(event="end", data=None)
    result = _convert_stream_chunk(chunk, modes=["updates"])
    assert result == []


def test_updates_event() -> None:
    """Updates event with dict data returns a single 3-tuple."""
    data = {"agent": {"messages": [{"role": "ai", "content": "hi"}]}}
    chunk = StreamPart(event="updates", data=data)
    result = _convert_stream_chunk(chunk, modes=["updates"])
    assert result == [((), "updates", data)]


class TestToUuid:
    def test_short_id_becomes_valid_uuid(self) -> None:
        result = _to_uuid("461bc7c2")
        uuid.UUID(result)
        assert result == "461bc7c2-0000-0000-0000-000000000000"

    def test_full_uuid_passthrough(self) -> None:
        full = str(uuid.uuid4())
        assert _to_uuid(full) == full

    def test_deterministic(self) -> None:
        assert _to_uuid("abcd1234") == _to_uuid("abcd1234")


class TestStreamConverterDelta:
    """Verify _StreamConverter emits incremental deltas, not accumulated text."""

    def _make_partial(self, msg_id: str, content: str) -> StreamPart:
        return StreamPart(
            event="messages/partial",
            data={"id": msg_id, "type": "AIMessageChunk", "content": content},
        )

    @staticmethod
    def _get_text(msg: Any) -> str:  # noqa: ANN401
        """Extract plain text from a message's content_blocks."""
        return "".join(
            b.get("text", "") for b in msg.content_blocks if b.get("type") == "text"
        )

    def test_first_chunk_emits_full_text(self) -> None:
        converter = _StreamConverter()
        results = converter.convert(self._make_partial("m1", "Hi"), modes=[])
        assert len(results) == 1
        _, mode, (msg, _meta) = results[0]
        assert mode == "messages"
        assert self._get_text(msg) == "Hi"

    def test_subsequent_chunks_emit_delta_only(self) -> None:
        converter = _StreamConverter()
        converter.convert(self._make_partial("m1", "Hi"), modes=[])
        results = converter.convert(self._make_partial("m1", "Hi! How"), modes=[])
        assert len(results) == 1
        _, _, (msg, _) = results[0]
        assert self._get_text(msg) == "! How"

    def test_full_accumulation_gives_correct_deltas(self) -> None:
        converter = _StreamConverter()
        texts = ["Hi", "Hi! How", "Hi! How can I help you today?"]
        deltas = []
        for text in texts:
            results = converter.convert(self._make_partial("m1", text), modes=[])
            if results:
                _, _, (msg, _) = results[0]
                deltas.append(self._get_text(msg))
        assert "".join(deltas) == "Hi! How can I help you today?"

    def test_empty_delta_skipped(self) -> None:
        converter = _StreamConverter()
        converter.convert(self._make_partial("m1", "Hi"), modes=[])
        results = converter.convert(self._make_partial("m1", "Hi"), modes=[])
        assert results == []

    def test_different_message_ids_tracked_independently(self) -> None:
        converter = _StreamConverter()
        converter.convert(self._make_partial("m1", "Hello"), modes=[])
        results = converter.convert(self._make_partial("m2", "World"), modes=[])
        assert len(results) == 1
        _, _, (msg, _) = results[0]
        assert self._get_text(msg) == "World"

    def test_tool_calls_emitted_even_without_text_delta(self) -> None:
        converter = _StreamConverter()
        chunk = StreamPart(
            event="messages/partial",
            data={
                "id": "m1",
                "type": "AIMessageChunk",
                "content": "",
                "tool_calls": [{"id": "tc1", "name": "search", "args": {"q": "test"}}],
            },
        )
        results = converter.convert(chunk, modes=[])
        assert len(results) == 1

    def test_list_content_delta(self) -> None:
        converter = _StreamConverter()
        chunk1 = StreamPart(
            event="messages/partial",
            data={
                "id": "m1",
                "type": "AIMessageChunk",
                "content": [{"type": "text", "text": "Hi"}],
            },
        )
        chunk2 = StreamPart(
            event="messages/partial",
            data={
                "id": "m1",
                "type": "AIMessageChunk",
                "content": [{"type": "text", "text": "Hi! How"}],
            },
        )
        converter.convert(chunk1, modes=[])
        results = converter.convert(chunk2, modes=[])
        assert len(results) == 1
        _, _, (msg, _) = results[0]
        text_blocks = [
            b for b in msg.content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert text_blocks[0]["text"] == "! How"
