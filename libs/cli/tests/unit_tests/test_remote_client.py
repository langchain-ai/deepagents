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

    def test_complete_event_passes_through_tool_message(self) -> None:
        converter = _StreamConverter()
        chunk = StreamPart(
            event="messages/complete",
            data=[
                {
                    "id": "m2",
                    "type": "tool",
                    "content": "Sunny, 75F",
                    "tool_call_id": "tc1",
                    "name": "search",
                }
            ],
        )
        results = converter.convert(chunk, modes=[])
        assert len(results) == 1
        _, mode, (msg, _) = results[0]
        assert mode == "messages"
        from langchain_core.messages import ToolMessage

        assert isinstance(msg, ToolMessage)
        assert msg.content == "Sunny, 75F"
        assert msg.tool_call_id == "tc1"

    def test_complete_event_empty_content_not_dropped(self) -> None:
        converter = _StreamConverter()
        chunk = StreamPart(
            event="messages/complete",
            data=[
                {
                    "id": "m2",
                    "type": "tool",
                    "content": "",
                    "tool_call_id": "tc1",
                    "name": "search",
                }
            ],
        )
        results = converter.convert(chunk, modes=[])
        assert len(results) == 1

    def test_repeated_tool_call_id_not_re_emitted(self) -> None:
        converter = _StreamConverter()
        chunk1 = StreamPart(
            event="messages/partial",
            data=[
                {
                    "id": "m1",
                    "type": "AIMessageChunk",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc1", "name": "search", "args": {"q": "test"}}
                    ],
                }
            ],
        )
        results1 = converter.convert(chunk1, modes=[])
        assert len(results1) == 1

        chunk2 = StreamPart(
            event="messages/partial",
            data=[
                {
                    "id": "m1",
                    "type": "AIMessageChunk",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc1", "name": "search", "args": {"q": "test query"}}
                    ],
                }
            ],
        )
        results2 = converter.convert(chunk2, modes=[])
        assert results2 == []

    def test_updates_event_extracts_tool_call_messages(self) -> None:
        converter = _StreamConverter()
        chunk = StreamPart(
            event="updates",
            data={
                "agent": {
                    "messages": [
                        {
                            "id": "m1",
                            "type": "ai",
                            "content": "",
                            "tool_calls": [
                                {"id": "tc1", "name": "search", "args": {"q": "test"}}
                            ],
                            "response_metadata": {},
                        }
                    ]
                }
            },
        )
        results = converter.convert(chunk, modes=[])
        updates = [r for r in results if r[1] == "updates"]
        messages = [r for r in results if r[1] == "messages"]
        assert len(updates) == 1
        assert len(messages) == 1
        msg = messages[0][2][0]
        assert msg.tool_calls[0]["name"] == "search"

    def test_updates_event_extracts_tool_result(self) -> None:
        converter = _StreamConverter()
        chunk = StreamPart(
            event="updates",
            data={
                "tools": {
                    "messages": [
                        {
                            "id": "m2",
                            "type": "tool",
                            "content": "Sunny",
                            "tool_call_id": "tc1",
                            "name": "search",
                        }
                    ]
                }
            },
        )
        results = converter.convert(chunk, modes=[])
        messages = [r for r in results if r[1] == "messages"]
        assert len(messages) == 1
        from langchain_core.messages import ToolMessage

        assert isinstance(messages[0][2][0], ToolMessage)

    def test_updates_no_duplicate_with_partial(self) -> None:
        converter = _StreamConverter()
        partial = StreamPart(
            event="messages/partial",
            data=[
                {
                    "id": "m1",
                    "type": "AIMessageChunk",
                    "content": "Hi",
                    "tool_calls": [],
                }
            ],
        )
        converter.convert(partial, modes=[])
        update = StreamPart(
            event="updates",
            data={
                "agent": {
                    "messages": [
                        {
                            "id": "m1",
                            "type": "ai",
                            "content": "Hi",
                            "tool_calls": [],
                            "response_metadata": {},
                        }
                    ]
                }
            },
        )
        results = converter.convert(update, modes=[])
        messages = [r for r in results if r[1] == "messages"]
        assert len(messages) == 0

    def test_full_tool_call_sequence(self) -> None:
        """Simulate the full SSE event sequence for an agent with tool calls."""
        from langchain_core.messages import ToolMessage as LCToolMessage

        converter = _StreamConverter()
        ai_msg = {
            "content": "Let me search.",
            "type": "ai",
            "id": "msg-1",
            "tool_calls": [
                {
                    "name": "search",
                    "args": {"q": "weather"},
                    "id": "tc1",
                    "type": "tool_call",
                }
            ],
            "response_metadata": {},
            "invalid_tool_calls": [],
        }
        tool_msg = {
            "content": "Sunny",
            "type": "tool",
            "tool_call_id": "tc1",
            "id": "msg-2",
            "name": "search",
        }
        events = [
            StreamPart("messages/partial", [{**ai_msg, "content": "Let me "}]),
            StreamPart("messages/partial", [ai_msg]),
            StreamPart("messages/complete", [ai_msg]),
            StreamPart("updates", {"agent": {"messages": [ai_msg]}}),
            StreamPart("messages/complete", [tool_msg]),
            StreamPart("updates", {"tools": {"messages": [tool_msg]}}),
        ]

        all_msgs = []
        for ev in events:
            for _, mode, data in converter.convert(ev, modes=[]):
                if mode == "messages":
                    all_msgs.append(data[0])

        ai_chunks = [m for m in all_msgs if not isinstance(m, LCToolMessage)]
        tool_msgs = [m for m in all_msgs if isinstance(m, LCToolMessage)]

        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "Sunny"

        tc_blocks = [
            b
            for m in ai_chunks
            for b in m.content_blocks
            if b.get("type") == "tool_call"
        ]
        assert len(tc_blocks) == 1
        assert tc_blocks[0]["name"] == "search"

        text = "".join(
            b.get("text", "")
            for m in ai_chunks
            for b in m.content_blocks
            if b.get("type") == "text"
        )
        assert text == "Let me search."
