"""Tests for the shared streaming tool-call buffer and hook-payload builders.

`_tool_stream` is the single source of truth for reassembling streamed tool-call
arguments and building `tool.use` / `tool.result` / `tool.error` payloads across
both execution surfaces, so its contract is exercised directly here (the two
surfaces additionally exercise it end-to-end in their own suites).
"""

from __future__ import annotations

import pytest

from deepagents_code._tool_stream import (
    ToolCallBuffer,
    build_tool_error_payload,
    build_tool_result_payload,
    build_tool_use_payload,
    normalize_tool_status,
    tool_call_buffer_key,
)
from deepagents_code.hooks import HOOK_TOOL_OUTPUT_LIMIT


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


class TestToolCallBufferConstruction:
    """The `args` XOR `args_parts` invariant is enforced at construction."""

    def test_both_args_and_args_parts_rejected(self) -> None:
        """Constructing with both populated raises, making the state unrepresentable.

        `ingest` maintains the XOR on every chunk; `__post_init__` guarantees no
        buffer can be built in the illegal both-set state that `parse_args` would
        otherwise mask via read order.
        """
        with pytest.raises(ValueError, match="cannot hold both"):
            ToolCallBuffer(args={"a": 1}, args_parts=['{"a":'])

    def test_only_args_allowed(self) -> None:
        """A whole value alone is a legal buffer."""
        assert ToolCallBuffer(args={"a": 1}).args == {"a": 1}

    def test_only_args_parts_allowed(self) -> None:
        """Fragments alone are a legal buffer."""
        assert ToolCallBuffer(args_parts=['{"a":']).args_parts == ['{"a":']


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

    def test_same_tool_id_keeps_accumulating(self) -> None:
        """Repeated chunks of one call (same id, then id-less) keep appending."""
        buffer = ToolCallBuffer()
        buffer.ingest(name="write_file", tool_id="toolu_1", args='{"a":')
        buffer.ingest(name=None, tool_id="toolu_1", args=" 1}")
        buffer.ingest(name=None, tool_id=None, args="")
        assert buffer.parse_args() == {"a": 1}

    def test_new_tool_id_resets_stale_call_state(self) -> None:
        """A differing id (reused streaming index) discards old call state.

        Indices restart per message, so a buffer retained from an earlier call
        (e.g. one whose args never parsed) can be handed to a new call via the
        same key. The new id must reset the old call's arguments and metadata so
        they cannot leak into chunks for the new call.
        """
        buffer = ToolCallBuffer(
            name="read_file",
            tool_id="toolu_a",
            args_parts=["{bad"],
            displayed=True,
        )
        buffer.ingest(name=None, tool_id="toolu_b", args='{"x": 1}')
        assert buffer.tool_id == "toolu_b"
        assert buffer.name is None
        assert buffer.displayed is False
        assert buffer.parse_args() == {"x": 1}

    def test_new_tool_id_resets_warned_latch(self) -> None:
        """A reused index resets the `warned` latch.

        The new call's own malformed payload is still surfaced once, rather than
        being suppressed by the previous call's latch.
        """
        buffer = ToolCallBuffer(tool_id="toolu_a", args_parts=["{bad json}"])
        assert buffer.parse_args() is None  # sets warned for call a
        assert buffer.warned is True
        buffer.ingest(name="write_file", tool_id="toolu_b", args="{also bad}")
        assert buffer.warned is False

    def test_string_fragment_clears_prior_whole_value(self) -> None:
        """A fragment supersedes a prior whole value.

        Enforces the `args` XOR `args_parts` invariant at write time rather than
        leaving both populated for `parse_args` read-order to disambiguate.
        """
        buffer = ToolCallBuffer(args={"stale": True})
        buffer.ingest(name=None, tool_id=None, args='{"real":')
        assert buffer.args is None
        assert buffer.args_parts == ['{"real":']

    def test_scalar_clears_prior_fragments(self) -> None:
        """A whole scalar value discards accumulated fragments (XOR invariant)."""
        buffer = ToolCallBuffer(args_parts=['{"partial":'])
        buffer.ingest(name=None, tool_id=None, args=7)
        assert buffer.args == 7
        assert buffer.args_parts == []


class TestNormalizeToolStatus:
    """Fail-closed mapping of a raw `ToolMessage.status` to the hook domain."""

    def test_success_passthrough(self) -> None:
        assert normalize_tool_status("success", "read_file") == "success"

    def test_error_passthrough(self) -> None:
        assert normalize_tool_status("error", "read_file") == "error"

    def test_unexpected_status_treated_as_error_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unknown present status fails closed to error and is logged."""
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert normalize_tool_status("cancelled", "execute") == "error"
        assert any("Unexpected ToolMessage.status" in r.message for r in caplog.records)

    def test_none_status_treated_as_error(self) -> None:
        """An explicit `None` status is unexpected, not a silent success."""
        assert normalize_tool_status(None, "execute") == "error"

    def test_success_default_not_warned(self, caplog: pytest.LogCaptureFixture) -> None:
        """The missing-status caller default (`"success"`) must not warn."""
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            normalize_tool_status("success", "read_file")
        assert not any(
            "Unexpected ToolMessage.status" in r.message for r in caplog.records
        )


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
