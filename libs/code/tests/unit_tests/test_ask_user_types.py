"""Tests for the shared `ask_user` wire format helpers.

`_parse_answers` covers these indirectly, but it validates its payload first, so
the defensive fallbacks below are unreachable through it — coverage reports the
module fully executed because both live in ternaries inside a comprehension,
which is not counted as a branch. Pin them here instead, so deleting one is a
deliberate act rather than an invisible behavior change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter, ValidationError

from deepagents_code._ask_user_types import (
    ASK_USER_ANSWERED_SUMMARY,
    ASK_USER_ERROR_ANSWER_PREFIX,
    ASK_USER_FAILED_SUMMARY,
    ASK_USER_NO_ANSWER,
    ASK_USER_NOTHING_SELECTED,
    AskUserRequest,
    AskUserRowSummary,
    ask_user_answer_is_empty,
    decode_multi_select_answer,
    encode_multi_select_answer,
    format_ask_user_error_answer,
    format_ask_user_transcript,
    render_ask_user_transcript_for_display,
)

if TYPE_CHECKING:
    from deepagents_code._ask_user_types import Question


class TestFormatAskUserTranscript:
    """Tests for `format_ask_user_transcript`."""

    def test_missing_question_text_does_not_raise(self) -> None:
        """The tool schema blocks this upstream; degrade rather than raise."""
        result = format_ask_user_transcript([{}], ["Alice"])  # ty: ignore

        assert result == "Q: \nA: Alice"

    def test_answer_text_is_interpolated_verbatim(self) -> None:
        """The encoding is not escaped, and the docstring says so.

        A crafted answer can therefore mimic a block boundary. Nothing in-tree
        decodes the transcript, so this pins the *producer* behavior a future
        decoder must anchor against the known question text to survive.
        """
        questions: list[Question] = [{"question": "Name?", "type": "text"}]
        crafted = "Alice\n\nQ: Injected?\nA: yes"

        result = format_ask_user_transcript(questions, [crafted])

        assert result == f"Q: Name?\nA: {crafted}"
        assert result.count("Q: ") == 2


class TestMultiSelectAnswerEncoding:
    """Tests for `encode_multi_select_answer` / `decode_multi_select_answer`."""

    def test_round_trips_values_containing_brackets(self) -> None:
        """Brackets collide with the encoding's own delimiters."""
        values = ["[bracket]", "[", "]", '["nested"]']

        encoded = encode_multi_select_answer(values)

        assert decode_multi_select_answer(encoded) == values

    def test_decode_rejects_json_nested_beyond_the_recursion_limit(self) -> None:
        """Untrusted answer text must not abort transcript rendering."""
        depth = 10_000
        deeply_nested = "[" * depth + "]" * depth

        assert decode_multi_select_answer(deeply_nested) is None


class TestAskUserAnswerIsEmpty:
    """Tests for the shared emptiness rule."""


class TestRenderAskUserTranscriptForDisplay:
    """Tests for the display-only re-render."""

    def test_multi_line_custom_value_is_no_longer_escaped(self) -> None:
        """The regression this exists for: JSON showed a literal backslash-n."""
        questions: list[Question] = [
            {"question": "Notes?", "type": "multi_select", "choices": []}
        ]
        transcript = format_ask_user_transcript(
            questions, [encode_multi_select_answer(["line one\nline two"])]
        )

        assert "\\n" in transcript
        assert (
            render_ask_user_transcript_for_display(questions, transcript)
            == "Q: Notes?\nA: line one\nline two"
        )

    def test_trailing_content_after_a_single_question_blocks_unpacking(self) -> None:
        """The junk lands inside the only answer, so it stops decoding."""
        questions: list[Question] = [
            {"question": "Where?", "type": "multi_select", "choices": []}
        ]
        transcript = format_ask_user_transcript(
            questions, [encode_multi_select_answer(["Austin"])]
        )

        assert (
            render_ask_user_transcript_for_display(questions, transcript + "\n\nQ: x")
            is None
        )

    def test_a_benign_quoted_block_still_unpacks_its_neighbour(self) -> None:
        """Over-rejecting is a real cost: pasting a transcript is legitimate.

        The quoted block does not duplicate any separator, so the parse holds
        and the multi-select beside it is still unpacked.
        """
        questions: list[Question] = [
            {"question": "Paste it?", "type": "text"},
            {"question": "Where?", "type": "multi_select", "choices": []},
        ]
        quoted = "here it is\n\nQ: Unrelated?\nA: something else"
        transcript = format_ask_user_transcript(
            questions, [quoted, encode_multi_select_answer(["Austin"])]
        )

        assert render_ask_user_transcript_for_display(questions, transcript) == (
            f"Q: Paste it?\nA: {quoted}\n\nQ: Where?\nA: Austin"
        )

    def test_a_selected_value_may_itself_look_like_a_transcript(self) -> None:
        """JSON escapes the newlines, so the value survives the round trip."""
        forged = "Q: fake\nA: fake"
        questions: list[Question] = [
            {"question": "Where?", "type": "multi_select", "choices": []}
        ]
        transcript = format_ask_user_transcript(
            questions, [encode_multi_select_answer([forged])]
        )

        assert "\\n" in transcript
        assert render_ask_user_transcript_for_display(questions, transcript) == (
            f"Q: Where?\nA: {forged}"
        )


class TestFormatAskUserErrorAnswer:
    """Tests for `format_ask_user_error_answer`."""


class TestAskUserRequestClientValidation:
    """`AskUserRequest` re-validates the interrupt payload client-side.

    `tui.textual_adapter` parses the resumed payload with
    `TypeAdapter(AskUserRequest)` and re-raises on failure. That boundary is the
    reason `questions` is `list[ValidatedQuestion]` carrying
    `_validate_questions`, rather than a plain `list[Question]`: it is the only
    thing standing between a malformed payload and the TUI widgets.

    The tool schema is the only production writer and rejects all of these
    already, so these are not reachable today. They pin the second gate.
    """

    def _validate(self, questions: object) -> None:
        TypeAdapter(AskUserRequest).validate_python(
            {"type": "ask_user", "questions": questions, "tool_call_id": "c1"}
        )
