"""Tests for the shared streaming tool-call buffer and hook-payload builders.

`_tool_stream` is the single source of truth for reassembling streamed tool-call
arguments and building `tool.use` / `tool.result` / `tool.error` payloads across
both execution surfaces, so its contract is exercised directly here (the two
surfaces additionally exercise it end-to-end in their own suites).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from deepagents_code._tool_stream import (
    INVALID_ARGS_PREVIEW_LIMIT,
    MAX_JSON_CONTAINER_DEPTH,
    TOOL_OUTPUT_TRUNCATION_MARKER,
    ToolCallBuffer,
    build_tool_error_payload,
    build_tool_result_payload,
    build_tool_use_payload,
    count_unemitted_tool_calls,
    normalize_tool_status,
    tool_call_buffer_key,
)
from deepagents_code.hooks import HOOK_TOOL_OUTPUT_LIMIT

if TYPE_CHECKING:
    import pytest


class TestToolCallBufferKey:
    """Precedence of the buffer key: index, then id, then placeholder."""


class TestToolCallBufferConstruction:
    """The `args` XOR `args_parts` invariant is enforced at construction."""


class TestToolCallBufferIngest:
    """Folding streamed chunk fields into the buffer."""

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


class TestNormalizeToolStatus:
    """Fail-closed mapping of a raw `ToolMessage.status` to the hook domain."""

    def test_unexpected_status_treated_as_error_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unknown present status fails closed to error and is logged."""
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert normalize_tool_status("cancelled", "execute") == "error"
        assert any("Unexpected ToolMessage.status" in r.message for r in caplog.records)


class TestToolCallBufferParseArgs:
    """Argument reassembly and completeness gating."""

    def test_warned_latch_does_not_strand_a_later_payload(self) -> None:
        """A malformed payload does not poison the next one in the same buffer.

        `parse_args` short-circuits on `warned`, so the latch has to be cleared
        with the rest of the per-payload state. A whole-value chunk resets the
        fragment state mid-buffer; if it left `warned` set, the following
        fragment stream would return `None` forever and silently drop a valid
        `tool.use`.
        """
        buffer = ToolCallBuffer()
        buffer.ingest(name="write_file", tool_id="t1", args="{bad json}")
        assert buffer.parse_args() is None
        assert buffer.warned is True

        buffer.ingest(name=None, tool_id=None, args={"whole": 1})
        assert buffer.parse_args() == {"whole": 1}
        assert buffer.warned is False

        buffer.ingest(name=None, tool_id=None, args='{"good": 1}')
        assert buffer.parse_args() == {"good": 1}

    def test_parse_cache_is_invalidated_by_later_fragments(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A cached parse is dropped as soon as another fragment arrives.

        Once a container parses, the result is memoized so repeated reads are
        free. A later fragment invalidates that memo: appending to an already
        closed value makes it malformed, and returning the stale dict would
        dispatch `tool.use` with args the model did not send.
        """
        over_closed = ToolCallBuffer()
        over_closed.ingest(name=None, tool_id=None, args='{"a": 1}')
        assert over_closed.parse_args() == {"a": 1}
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            over_closed.ingest(name=None, tool_id=None, args="}")
            assert over_closed.parse_args() is None

        continued = ToolCallBuffer()
        continued.ingest(name=None, tool_id=None, args='{"a": 1}')
        assert continued.parse_args() == {"a": 1}
        continued.ingest(name=None, tool_id=None, args=', "b": 2}')
        assert continued.parse_args() is None

    def test_parse_args_returns_the_cached_dict_by_identity(self) -> None:
        """Repeated reads share one dict, so callers must treat it as read-only.

        Pinned deliberately: both surfaces forward this object into hook
        payloads and retain it on the in-flight record, and the end-of-stream
        diagnostic re-reads it. A caller that mutated it would corrupt every
        other holder.
        """
        buffer = ToolCallBuffer(args_parts=['{"a": 1}'])
        first = buffer.parse_args()
        assert first is buffer.parse_args()

        wrapped = ToolCallBuffer(args_parts=["[1, 2]"])
        assert wrapped.parse_args() is wrapped.parse_args()

    def test_escape_state_carries_across_fragment_boundaries(self) -> None:
        """A backslash ending a fragment still escapes the next fragment's char.

        The escape flag is scanned once per fragment and must survive the
        boundary. Losing it makes an escaped quote look like a closing quote (or
        vice versa), which flips the computed string state and leaves a complete
        payload permanently unparsed.
        """
        # Fragment boundary splits `\"`: the quote is escaped, so the string
        # stays open and the `}` inside it is not a real close.
        escaped_quote = ToolCallBuffer(args_parts=[r'{"a": "x' + "\\"])
        escaped_quote.ingest(name=None, tool_id=None, args=r'"y}"}')
        assert escaped_quote.parse_args() == {"a": 'x"y}'}

        # Fragment boundary splits `\\`: the backslash is literal, so the next
        # quote really does close the string.
        escaped_backslash = ToolCallBuffer(args_parts=[r'{"a": "x\\'])
        escaped_backslash.ingest(name=None, tool_id=None, args='"}')
        assert escaped_backslash.parse_args() == {"a": "x\\"}

    def test_open_string_after_balanced_container_not_warned(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An open string with balanced brackets is incomplete, not malformed.

        Complements `test_trailing_brace_inside_open_string_not_warned`, where
        the depth term alone catches the payload. Here the depth is back to zero
        and only the open-string term can tell that the value is still
        streaming, so dropping that term would warn on a healthy fragment.
        """
        buffer = ToolCallBuffer(args_parts=['{"a": 1} "x}'])
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert buffer.parse_args() is None
        assert buffer.warned is False
        assert not any(
            "are unparseable and cannot be completed" in r.message
            for r in caplog.records
        )

    def test_over_depth_is_skipped_without_parsing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The depth guard fires on nesting the C scanner would happily parse.

        Nesting just past `MAX_JSON_CONTAINER_DEPTH` is well within what
        `json.loads` accepts, so a `None` here can only come from the guard —
        which pins it against rotting into a no-op behind the `RecursionError`
        arm. The payload arrives in many small fragments, so it also pins the
        high-water mark surviving fragment boundaries: the final depth is zero,
        and only the running maximum records how deep it went.
        """
        depth = MAX_JSON_CONTAINER_DEPTH + 10
        assert json.loads("[" * depth + "]" * depth) is not None

        buffer = ToolCallBuffer()
        for _ in range(depth):
            buffer.ingest(name=None, tool_id=None, args="[")
        for _ in range(depth):
            buffer.ingest(name=None, tool_id=None, args="]")
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert buffer.parse_args() is None
        assert buffer.warned is True

    def test_over_closed_json_fed_incrementally_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Over-closing is detected across fragments, not just in one chunk.

        The over-close flag latches during the per-fragment scan, so the stray
        closer is still recognised when it arrives in its own chunk long after
        the value closed.
        """
        buffer = ToolCallBuffer()
        buffer.ingest(name=None, tool_id=None, args='{"a": ')
        buffer.ingest(name=None, tool_id=None, args="1}")
        buffer.ingest(name=None, tool_id=None, args="}")
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert buffer.parse_args() is None
        assert buffer.warned is True
        assert any(
            "are unparseable and cannot be completed" in r.message
            for r in caplog.records
        )

    def test_invalid_args_warning_is_length_bounded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning previews a bounded prefix, not the whole payload.

        Streamed args are unbounded, so the log line must not be. The preview
        walks fragments and stops at the cap rather than slicing a joined string
        that may never have been built.
        """
        filler = "z" * 50
        buffer = ToolCallBuffer(args_parts=["{"])
        for _ in range(40):
            buffer.ingest(name=None, tool_id=None, args=filler)
        buffer.ingest(name=None, tool_id=None, args="}")
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert buffer.parse_args() is None

        (record,) = [
            r
            for r in caplog.records
            if "are unparseable and cannot be completed" in r.message
        ]
        assert len("".join(buffer.args_parts)) > 2_000
        assert INVALID_ARGS_PREVIEW_LIMIT < len(record.message) < 300

    def test_whole_value_chunk_resets_lexer_state_mid_string(self) -> None:
        """A dict chunk clears string state left by an abandoned fragment run.

        The fragment stream is discarded mid-literal, so the open-string flag
        has to go with it. Left set, it would report the *next* fragment run as
        forever incomplete.
        """
        buffer = ToolCallBuffer()
        buffer.ingest(name=None, tool_id=None, args='{"a": "unterminated')
        assert buffer.parse_args() is None

        buffer.ingest(name=None, tool_id=None, args={"whole": 1})
        assert buffer.parse_args() == {"whole": 1}

        buffer.ingest(name=None, tool_id=None, args='{"b": 2}')
        assert buffer.parse_args() == {"b": 2}

    def test_midstream_nested_json_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A partial nested payload that happens to end in `}` is not warned.

        A chunk boundary landing right after an inner object closes leaves the
        outer container open (`{"edits": [{"a": 1}`). The old "starts with {/[
        and ends with }/]" heuristic mistook this for a complete-but-malformed
        value and logged a WARNING on a perfectly healthy stream. The
        string-aware balance check treats it as still-incomplete: no warning,
        `warned` stays unset, and the next fragment can still complete it.
        """
        buffer = ToolCallBuffer(args_parts=['{"edits": [{"a": 1}'])
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert buffer.parse_args() is None
        assert buffer.warned is False
        assert not any(
            "are unparseable and cannot be completed" in r.message
            for r in caplog.records
        )
        # The completing fragments still parse once they arrive.
        buffer.ingest(name=None, tool_id=None, args=', {"b": 2}]}')
        assert buffer.parse_args() == {"edits": [{"a": 1}, {"b": 2}]}

    def test_trailing_brace_inside_open_string_not_warned(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A `}` that lives inside an unterminated string is not "complete".

        The payload ends in `}` (so it clears the cheap pre-check and reaches
        `json.loads`), but that brace is inside an open string literal, so the
        outer object is still unbalanced. The string-aware balance check must
        report it incomplete — no warning — rather than treating the trailing
        brace as a real close.
        """
        buffer = ToolCallBuffer(args_parts=['{"content": "a } b}'])
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert buffer.parse_args() is None
        assert buffer.warned is False
        assert not any(
            "are unparseable and cannot be completed" in r.message
            for r in caplog.records
        )

    def test_pathologically_nested_json_is_skipped_not_raised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Deeply nested model output is one skipped call, not an escaped error.

        The depth guard pre-empts `json.loads` entirely here, so this pins the
        guard rather than the `RecursionError` arm behind it — with CPython's C
        scanner, `json.loads` does not consume a Python frame per level and only
        raises far past this depth. `test_over_depth_is_skipped_without_parsing`
        covers the guard at its boundary; the `RecursionError` arm remains
        defense-in-depth for a pure-Python-scanner build.
        """
        depth = 100_000
        nested = "[" * depth + "]" * depth
        buffer = ToolCallBuffer(args_parts=[nested])
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert buffer.parse_args() is None
        assert buffer.warned is True
        assert any(
            "are unparseable and cannot be completed" in r.message
            for r in caplog.records
        )

    def test_over_closed_json_warns_via_balance_check(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A payload with more closers than openers is complete-but-malformed.

        `{"a": 1}}` ends in `}` (so it clears the cheap pre-check and reaches
        `json.loads`, which rejects the trailing brace). The string-aware balance
        scan hits the `depth < 0` branch and reports the value complete — a stray
        closer can never be finished by more input — so the failed parse is
        warned once rather than mistaken for a still-open mid-stream fragment.
        """
        buffer = ToolCallBuffer(args_parts=['{"a": 1}}'])
        with caplog.at_level("WARNING", logger="deepagents_code._tool_stream"):
            assert buffer.parse_args() is None
        assert buffer.warned is True
        assert any(
            "are unparseable and cannot be completed" in r.message
            for r in caplog.records
        )


class TestPayloadBuilders:
    """Fixed-shape hook payloads and the output truncation invariant."""


class TestCountUnemittedToolCalls:
    """Classification of buffered tool calls that never fired a `tool.use`."""
