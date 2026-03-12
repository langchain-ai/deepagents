"""Tests for _convert_stream_chunk, _StreamConverter, _to_uuid, and RemoteAgent."""

import uuid
from datetime import UTC, datetime
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.remote_client import (
    RemoteAgent,
    _convert_stream_chunk,
    _server_thread_to_info,
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
        assert "0000-0000-0000-000000000000" not in result

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
        assert self._get_text(msg) == "! How"

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

    def test_repeated_tool_call_same_args_not_re_emitted(self) -> None:
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

        results2 = converter.convert(chunk1, modes=[])
        assert results2 == []

    def test_tool_call_args_change_emits_delta(self) -> None:
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
        converter.convert(chunk1, modes=[])

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
        assert len(results2) == 1

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
        tool_results = [m for m in all_msgs if isinstance(m, LCToolMessage)]

        assert len(tool_results) == 1
        assert tool_results[0].content == "Sunny"

        tc_blocks = [
            b
            for m in ai_chunks
            for b in m.content_blocks
            if b.get("type") in ("tool_call", "tool_call_chunk")
        ]
        assert len(tc_blocks) >= 1
        assert tc_blocks[0]["name"] == "search"

        text = "".join(
            b.get("text", "")
            for m in ai_chunks
            for b in m.content_blocks
            if b.get("type") == "text"
        )
        assert text == "Let me search."

    def test_content_blocks_format_for_textual_adapter(self) -> None:
        """Verify content_blocks produce tool_call_chunk with string args."""
        converter = _StreamConverter()
        chunk = StreamPart(
            event="messages/partial",
            data=[
                {
                    "id": "m1",
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "web_search",
                            "args": {"query": "weather NYC"},
                            "id": "call_abc123",
                            "type": "tool_call",
                        }
                    ],
                    "response_metadata": {},
                    "invalid_tool_calls": [],
                    "usage_metadata": None,
                }
            ],
        )
        results = converter.convert(chunk, modes=[])
        assert len(results) == 1
        _, mode, (msg, _meta) = results[0]
        assert mode == "messages"

        blocks = msg.content_blocks
        tc_blocks = [
            b for b in blocks if b.get("type") in ("tool_call", "tool_call_chunk")
        ]
        assert len(tc_blocks) == 1
        tc = tc_blocks[0]
        assert tc["type"] == "tool_call_chunk"
        assert tc["name"] == "web_search"
        assert tc["id"] == "call_abc123"
        assert isinstance(tc["args"], str)
        assert "weather NYC" in tc["args"]

    def test_text_then_tool_call_produces_separate_messages(self) -> None:
        """Text and tool call split into separate messages."""
        converter = _StreamConverter()
        converter.convert(
            StreamPart(
                "messages/partial",
                [
                    {
                        "id": "m1",
                        "type": "ai",
                        "content": "Let me ",
                        "tool_calls": [],
                    }
                ],
            ),
            modes=[],
        )
        results = converter.convert(
            StreamPart(
                "messages/partial",
                [
                    {
                        "id": "m1",
                        "type": "ai",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "name": "search",
                                "args": {"q": "test"},
                                "id": "tc1",
                                "type": "tool_call",
                            }
                        ],
                    }
                ],
            ),
            modes=[],
        )
        assert len(results) == 2
        text_msg = results[0][2][0]
        tc_msg = results[1][2][0]
        assert self._get_text(text_msg) == "check."
        tc_blocks = [
            b
            for b in tc_msg.content_blocks
            if b.get("type") in ("tool_call", "tool_call_chunk")
        ]
        assert len(tc_blocks) == 1
        assert tc_blocks[0]["name"] == "search"

    def test_messages_partial_data_as_single_dict(self) -> None:
        """Server may send messages/partial data as a single dict (not list)."""
        converter = _StreamConverter()
        chunk = StreamPart(
            event="messages/partial",
            data={
                "id": "m1",
                "type": "ai",
                "content": "Hello",
                "tool_calls": [],
                "response_metadata": {},
            },
        )
        results = converter.convert(chunk, modes=[])
        assert len(results) == 1
        msg = results[0][2][0]
        text = "".join(
            b.get("text", "") for b in msg.content_blocks if b.get("type") == "text"
        )
        assert text == "Hello"

    def test_complete_ai_message_skipped_if_seen_in_partial(self) -> None:
        """AI messages/complete are skipped if already seen via partial."""
        converter = _StreamConverter()
        converter.convert(
            StreamPart(
                "messages/partial",
                [{"id": "m1", "type": "ai", "content": "Hi", "tool_calls": []}],
            ),
            modes=[],
        )
        results = converter.convert(
            StreamPart(
                "messages/complete",
                [{"id": "m1", "type": "ai", "content": "Hi", "tool_calls": []}],
            ),
            modes=[],
        )
        assert len(results) == 0

    def test_complete_emits_usage_when_skipped_as_duplicate(self) -> None:
        """usage_metadata on a duplicate complete event is still emitted."""
        converter = _StreamConverter()
        converter.convert(
            StreamPart(
                "messages/partial",
                [{"id": "m1", "type": "ai", "content": "Hi", "tool_calls": []}],
            ),
            modes=[],
        )
        results = converter.convert(
            StreamPart(
                "messages/complete",
                [
                    {
                        "id": "m1",
                        "type": "ai",
                        "content": "Hi",
                        "tool_calls": [],
                        "usage_metadata": {
                            "input_tokens": 10,
                            "output_tokens": 20,
                            "total_tokens": 30,
                        },
                    }
                ],
            ),
            modes=[],
        )
        assert len(results) == 1
        msg = results[0][2][0]
        assert msg.usage_metadata == {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }

    def test_complete_no_usage_still_skipped(self) -> None:
        """Duplicate complete without usage_metadata is still fully skipped."""
        converter = _StreamConverter()
        converter.convert(
            StreamPart(
                "messages/partial",
                [{"id": "m1", "type": "ai", "content": "Hi", "tool_calls": []}],
            ),
            modes=[],
        )
        results = converter.convert(
            StreamPart(
                "messages/complete",
                [{"id": "m1", "type": "ai", "content": "Hi", "tool_calls": []}],
            ),
            modes=[],
        )
        assert len(results) == 0

    def test_partial_final_usage_metadata_emitted(self) -> None:
        """Final partial with usage_metadata but no new text emits a stub chunk."""
        converter = _StreamConverter()
        # First partial: text arrives
        converter.convert(
            StreamPart(
                "messages/partial",
                [
                    {
                        "id": "m1",
                        "type": "ai",
                        "content": "Hello",
                        "tool_calls": [],
                        "usage_metadata": None,
                    }
                ],
            ),
            modes=[],
        )
        # Final partial: same text, but now has usage_metadata
        results = converter.convert(
            StreamPart(
                "messages/partial",
                [
                    {
                        "id": "m1",
                        "type": "ai",
                        "content": "Hello",
                        "tool_calls": [],
                        "usage_metadata": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                        },
                    }
                ],
            ),
            modes=[],
        )
        assert len(results) == 1
        msg = results[0][2][0]
        assert msg.usage_metadata == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        # Stub should have empty content
        assert msg.content == ""

    def test_anthropic_tool_call_streaming(self) -> None:
        """Anthropic tool_use blocks with partial_json produce incremental chunks."""
        import json

        converter = _StreamConverter()
        msg_id = "msg-1"
        tc_id = "toolu_01WUnPp2tqBtg5kqA1vGNGPu"
        meta = {"model_name": "claude-opus-4-6", "model_provider": "anthropic"}

        events = [
            StreamPart(
                "messages/partial",
                [
                    {
                        "content": [],
                        "response_metadata": meta,
                        "type": "ai",
                        "id": msg_id,
                        "tool_calls": [],
                    }
                ],
            ),
            StreamPart(
                "messages/partial",
                [
                    {
                        "content": [
                            {
                                "id": tc_id,
                                "input": {},
                                "name": "ls",
                                "type": "tool_use",
                                "index": 0,
                            }
                        ],
                        "response_metadata": meta,
                        "type": "ai",
                        "id": msg_id,
                        "tool_calls": [
                            {
                                "name": "ls",
                                "args": {},
                                "id": tc_id,
                                "type": "tool_call",
                            }
                        ],
                    }
                ],
            ),
            StreamPart(
                "messages/partial",
                [
                    {
                        "content": [
                            {
                                "id": tc_id,
                                "input": {},
                                "name": "ls",
                                "type": "tool_use",
                                "index": 0,
                                "partial_json": "",
                            }
                        ],
                        "response_metadata": meta,
                        "type": "ai",
                        "id": msg_id,
                        "tool_calls": [
                            {
                                "name": "ls",
                                "args": {},
                                "id": tc_id,
                                "type": "tool_call",
                            }
                        ],
                    }
                ],
            ),
            StreamPart(
                "messages/partial",
                [
                    {
                        "content": [
                            {
                                "id": tc_id,
                                "input": {},
                                "name": "ls",
                                "type": "tool_use",
                                "index": 0,
                                "partial_json": '{"path": "/priv',
                            }
                        ],
                        "response_metadata": meta,
                        "type": "ai",
                        "id": msg_id,
                        "tool_calls": [
                            {
                                "name": "ls",
                                "args": {"path": "/priv"},
                                "id": tc_id,
                                "type": "tool_call",
                            }
                        ],
                    }
                ],
            ),
            StreamPart(
                "messages/partial",
                [
                    {
                        "content": [
                            {
                                "id": tc_id,
                                "input": {"path": "/private"},
                                "name": "ls",
                                "type": "tool_use",
                                "index": 0,
                                "partial_json": '{"path": "/private"}',
                            }
                        ],
                        "response_metadata": meta,
                        "type": "ai",
                        "id": msg_id,
                        "tool_calls": [
                            {
                                "name": "ls",
                                "args": {"path": "/private"},
                                "id": tc_id,
                                "type": "tool_call",
                            }
                        ],
                    }
                ],
            ),
        ]

        all_args_parts: list[str] = []
        for ev in events:
            for _, mode, (msg, _) in converter.convert(ev, modes=[]):
                if mode != "messages":
                    continue
                for block in msg.content_blocks:
                    if block.get("type") == "tool_call_chunk":
                        args = block.get("args", "")
                        if args:
                            all_args_parts.append(args)

        joined = "".join(all_args_parts)
        parsed = json.loads(joined)
        assert parsed == {"path": "/private"}


class TestInterruptConversion:
    """Verify interrupt dicts from the server are converted to Interrupt objects."""

    def test_values_event_converts_interrupt_dicts(self) -> None:
        from langgraph.types import Interrupt

        converter = _StreamConverter()
        chunk = StreamPart(
            event="values",
            data={
                "__interrupt__": [
                    {
                        "value": {"type": "ask_user", "question": "Approve?"},
                        "id": "int-1",
                    }
                ]
            },
        )
        results = converter.convert(chunk, modes=[])
        assert len(results) == 1
        _, mode, data = results[0]
        assert mode == "updates"
        interrupts = data["__interrupt__"]
        assert len(interrupts) == 1
        assert isinstance(interrupts[0], Interrupt)
        assert interrupts[0].value == {"type": "ask_user", "question": "Approve?"}
        assert interrupts[0].id == "int-1"

    def test_updates_event_converts_interrupt_dicts(self) -> None:
        from langgraph.types import Interrupt

        converter = _StreamConverter()
        chunk = StreamPart(
            event="updates",
            data={
                "__interrupt__": [
                    {
                        "value": {"type": "ask_user", "question": "OK?"},
                        "id": "int-2",
                    }
                ]
            },
        )
        results = converter.convert(chunk, modes=[])
        updates = [r for r in results if r[1] == "updates"]
        assert len(updates) == 1
        interrupts = updates[0][2]["__interrupt__"]
        assert len(interrupts) == 1
        assert isinstance(interrupts[0], Interrupt)
        assert interrupts[0].value == {"type": "ask_user", "question": "OK?"}

    def test_interrupt_objects_passed_through(self) -> None:
        from langgraph.types import Interrupt

        interrupt = Interrupt(value={"type": "ask_user"}, id="int-3")
        converter = _StreamConverter()
        chunk = StreamPart(
            event="values",
            data={"__interrupt__": [interrupt]},
        )
        results = converter.convert(chunk, modes=[])
        assert len(results) == 1
        interrupts = results[0][2]["__interrupt__"]
        assert interrupts[0] is interrupt


class TestServerThreadToInfo:
    """Tests for _server_thread_to_info helper."""

    def test_basic_conversion(self) -> None:
        server_thread = {
            "thread_id": "abc-123",
            "metadata": {
                "agent_name": "my-agent",
                "git_branch": "main",
                "cwd": "/home/user",
            },
            "updated_at": datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
            "created_at": datetime(2025, 1, 15, 9, 0, tzinfo=UTC),
        }
        info = _server_thread_to_info(server_thread)
        assert info["thread_id"] == "abc-123"
        assert info["agent_name"] == "my-agent"
        assert info["git_branch"] == "main"
        assert info["cwd"] == "/home/user"
        assert "2025-01-15" in info["updated_at"]
        assert "2025-01-15" in info["created_at"]

    def test_missing_metadata(self) -> None:
        server_thread = {
            "thread_id": "abc-123",
            "metadata": None,
            "updated_at": None,
            "created_at": None,
        }
        info = _server_thread_to_info(server_thread)
        assert info["thread_id"] == "abc-123"
        assert info["agent_name"] is None
        assert info["updated_at"] is None

    def test_string_timestamps(self) -> None:
        server_thread = {
            "thread_id": "t1",
            "metadata": {},
            "updated_at": "2025-01-15T10:30:00+00:00",
            "created_at": "2025-01-15T09:00:00+00:00",
        }
        info = _server_thread_to_info(server_thread)
        assert info["updated_at"] == "2025-01-15T10:30:00+00:00"


class TestRemoteAgentListThreads:
    """Tests for RemoteAgent.list_threads."""

    async def test_list_threads_returns_thread_info(self) -> None:
        agent = RemoteAgent("http://localhost:8123")
        mock_client = MagicMock()
        mock_client.threads = MagicMock()
        mock_client.threads.search = AsyncMock(
            return_value=[
                {
                    "thread_id": "t1",
                    "metadata": {"agent_name": "agent-a"},
                    "updated_at": datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
                    "created_at": datetime(2025, 1, 15, 9, 0, tzinfo=UTC),
                },
                {
                    "thread_id": "t2",
                    "metadata": {"agent_name": "agent-b"},
                    "updated_at": datetime(2025, 1, 14, 8, 0, tzinfo=UTC),
                    "created_at": datetime(2025, 1, 14, 7, 0, tzinfo=UTC),
                },
            ]
        )
        agent._client = mock_client

        threads = await agent.list_threads(limit=10, sort_by="updated")
        assert len(threads) == 2
        assert threads[0]["thread_id"] == "t1"
        assert threads[1]["agent_name"] == "agent-b"
        mock_client.threads.search.assert_awaited_once_with(
            limit=10, sort_by="updated", sort_order="desc"
        )

    async def test_list_threads_empty_on_error(self) -> None:
        agent = RemoteAgent("http://localhost:8123")
        mock_client = MagicMock()
        mock_client.threads = MagicMock()
        mock_client.threads.search = AsyncMock(side_effect=RuntimeError("boom"))
        agent._client = mock_client

        threads = await agent.list_threads()
        assert threads == []

    async def test_list_threads_sort_by_created(self) -> None:
        agent = RemoteAgent("http://localhost:8123")
        mock_client = MagicMock()
        mock_client.threads = MagicMock()
        mock_client.threads.search = AsyncMock(return_value=[])
        agent._client = mock_client

        await agent.list_threads(sort_by="created")
        mock_client.threads.search.assert_awaited_once_with(
            limit=20, sort_by="created", sort_order="desc"
        )


class TestRemoteAgentDeleteThread:
    """Tests for RemoteAgent.delete_thread."""

    async def test_delete_success(self) -> None:
        agent = RemoteAgent("http://localhost:8123")
        mock_client = MagicMock()
        mock_client.threads = MagicMock()
        mock_client.threads.delete = AsyncMock(return_value=None)
        agent._client = mock_client

        result = await agent.delete_thread("t1")
        assert result is True

    async def test_delete_not_found(self) -> None:
        import httpx

        agent = RemoteAgent("http://localhost:8123")
        mock_client = MagicMock()
        mock_client.threads = MagicMock()
        response = httpx.Response(404, request=httpx.Request("DELETE", "http://x"))
        exc = httpx.HTTPStatusError(
            "not found", request=response.request, response=response
        )
        mock_client.threads.delete = AsyncMock(side_effect=exc)
        agent._client = mock_client

        result = await agent.delete_thread("t1")
        assert result is False

    async def test_delete_other_error(self) -> None:
        agent = RemoteAgent("http://localhost:8123")
        mock_client = MagicMock()
        mock_client.threads = MagicMock()
        mock_client.threads.delete = AsyncMock(side_effect=RuntimeError("boom"))
        agent._client = mock_client

        result = await agent.delete_thread("t1")
        assert result is False
