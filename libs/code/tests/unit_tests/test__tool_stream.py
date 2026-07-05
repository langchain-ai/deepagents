"""Tests for the shared streaming tool-call buffer and hook-payload builders.

`_tool_stream` is the single source of truth for reassembling streamed tool-call
arguments and building `tool.use` / `tool.result` / `tool.error` payloads across
both execution surfaces, so its contract is exercised directly here (the two
surfaces additionally exercise it end-to-end in their own suites).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code._tool_stream import (
    ToolCallBuffer,
    build_tool_error_payload,
    build_tool_result_payload,
    build_tool_use_payload,
    tool_call_buffer_key,
)
from deepagents_code.hooks import HOOK_TOOL_OUTPUT_LIMIT

if TYPE_CHECKING:
    import pytest


class TestToolCallBufferKey:
    """Precedence of the buffer key: index, then id, then placeholder."""

    def test_index_preferred_over_id(self) -> None:
        """A present index wins even when an id is also available."""
        assert tool_call_buffer_key(0, "toolu_1", 3) == 0

    def test_falls_back_to_id_when_no_index(self) -> None:
        """The tool id keys the buffer when no streaming index is present."""
        assert tool_call_buffer_key(None, "toolu_1", 3) == "toolu_1"

    def test_placeholder_when_no_index_or_id(self) -> None:
        """A count-based placeholder keeps unrelated id-less calls distinct."""
        assert tool_call_buffer_key(None, None, 2) == "unknown-2"
        assert tool_call_buffer_key(None, None, 5) == "unknown-5"

    def test_zero_index_is_not_treated_as_missing(self) -> None:
        """Index 0 is a valid key, not a falsy stand-in for absent."""
        assert tool_call_buffer_key(0, None, 0) == 0


class TestToolCallBufferIngest:
    """Folding streamed chunk fields into the buffer."""

    def test_name_and_id_captured(self) -> None:
        """Name and id from a chunk populate the buffer."""
        buffer = ToolCallBuffer()
        buffer.ingest(name="write_file", tool_id="toolu_1", args=None)
        assert buffer.name == "write_file"
        assert buffer.tool_id == "toolu_1"

    def test_dict_args_replace_accumulated_fragments(self) -> None:
        """A whole-value dict delivery discards any partial fragments."""
        buffer = ToolCallBuffer(args_parts=['{"partial": '])
        buffer.ingest(name=None, tool_id=None, args={"path": "foo.py"})
        assert buffer.args == {"path": "foo.py"}
        assert buffer.args_parts == []

    def test_string_fragments_accumulate(self) -> None:
        """String fragments append in order."""
        buffer = ToolCallBuffer()
        buffer.ingest(name=None, tool_id=None, args='{"a":')
        buffer.ingest(name=None, tool_id=None, args=" 1}")
        assert buffer.args_parts == ['{"a":', " 1}"]

    def test_empty_string_fragment_skipped(self) -> None:
        """An empty fragment carries no payload and is not appended."""
        buffer = ToolCallBuffer()
        buffer.ingest(name=None, tool_id=None, args="")
        assert buffer.args_parts == []

    def test_adjacent_identical_fragments_preserved(self) -> None:
        """Repeated identical fragments are real content, not de-duplicated."""
        buffer = ToolCallBuffer()
        for fragment in ('{"content": "', "hi", "hi", '"}'):
            buffer.ingest(name=None, tool_id=None, args=fragment)
        assert buffer.parse_args() == {"content": "hihi"}

    def test_non_string_non_dict_scalar_stored(self) -> None:
        """A non-None scalar that is neither dict nor str is stored as-is."""
        buffer = ToolCallBuffer()
        buffer.ingest(name=None, tool_id=None, args=7)
        assert buffer.args == 7


class TestToolCallBufferParseArgs:
    """Argument reassembly and completeness gating."""

    def test_dict_returned_as_is(self) -> None:
        buffer = ToolCallBuffer(args={"path": "foo.py"})
        assert buffer.parse_args() == {"path": "foo.py"}

    def test_scalar_wrapped(self) -> None:
        buffer = ToolCallBuffer(args=42)
        assert buffer.parse_args() == {"value": 42}

    def test_fragments_joined_and_parsed(self) -> None:
        buffer = ToolCallBuffer(args_parts=['{"command": "uv run', ' pytest"}'])
        assert buffer.parse_args() == {"command": "uv run pytest"}

    def test_incomplete_object_returns_none(self) -> None:
        buffer = ToolCallBuffer(args_parts=['{"command": "uv run'])
        assert buffer.parse_args() is None

    def test_non_object_json_wrapped(self) -> None:
        buffer = ToolCallBuffer(args_parts=["[1, 2, 3]"])
        assert buffer.parse_args() == {"value": [1, 2, 3]}

    def test_empty_and_whitespace_return_none(self) -> None:
        assert ToolCallBuffer().parse_args() is None
        assert ToolCallBuffer(args_parts=["   "]).parse_args() is None

    def test_malformed_complete_json_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A complete-but-invalid payload logs exactly once via the latch.

        The buffer is retained across chunks on failure, so `parse_args` runs
        again on every later chunk; the `warned` latch keeps that from spamming
        an identical warning per fragment.
        """
        buffer = ToolCallBuffer(args_parts=["{bad json}"])
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert buffer.parse_args() is None
            assert buffer.parse_args() is None
            assert buffer.parse_args() is None
        warnings = [
            r
            for r in caplog.records
            if "look complete but failed to parse" in r.message
        ]
        assert len(warnings) == 1
        assert buffer.warned is True


class TestPayloadBuilders:
    """Fixed-shape hook payloads and the output truncation invariant."""

    def test_tool_use_payload_shape(self) -> None:
        assert build_tool_use_payload("write_file", "toolu_1", {"path": "f"}) == {
            "tool_name": "write_file",
            "tool_id": "toolu_1",
            "tool_args": {"path": "f"},
        }

    def test_tool_error_payload_shape(self) -> None:
        assert build_tool_error_payload("execute") == {"tool_names": ["execute"]}

    def test_tool_result_payload_shape(self) -> None:
        assert build_tool_result_payload(
            "write_file", "toolu_1", {"path": "f"}, "success", "ok"
        ) == {
            "tool_name": "write_file",
            "tool_id": "toolu_1",
            "tool_args": {"path": "f"},
            "tool_status": "success",
            "tool_output": "ok",
        }

    def test_tool_output_truncated_to_limit(self) -> None:
        """`tool_output` is capped at `HOOK_TOOL_OUTPUT_LIMIT`."""
        payload = build_tool_result_payload(
            "read_file", "toolu_1", {}, "success", "x" * (HOOK_TOOL_OUTPUT_LIMIT + 500)
        )
        assert len(payload["tool_output"]) == HOOK_TOOL_OUTPUT_LIMIT

    def test_tool_args_not_truncated(self) -> None:
        """`tool_args` is passed through in full; only output is capped."""
        big_content = "y" * (HOOK_TOOL_OUTPUT_LIMIT + 500)
        payload = build_tool_result_payload(
            "write_file", "toolu_1", {"content": big_content}, "success", ""
        )
        assert payload["tool_args"]["content"] == big_content
