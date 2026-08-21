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

    def test_pairs_answers_positionally(self) -> None:
        questions: list[Question] = [
            {"question": "Name?", "type": "text"},
            {"question": "Color?", "type": "text"},
        ]

        result = format_ask_user_transcript(questions, ["Alice", "blue"])

        assert result == "Q: Name?\nA: Alice\n\nQ: Color?\nA: blue"

    def test_blank_answer_stays_blank(self) -> None:
        """A deliberately empty answer is not the missing-answer placeholder."""
        questions: list[Question] = [{"question": "Name?", "type": "text"}]

        result = format_ask_user_transcript(questions, [""])

        assert result == "Q: Name?\nA: "
        assert ASK_USER_NO_ANSWER not in result

    def test_short_answer_list_falls_back_to_placeholder(self) -> None:
        """Unreachable via `_parse_answers`, which rejects a count mismatch."""
        questions: list[Question] = [
            {"question": "Name?", "type": "text"},
            {"question": "Color?", "type": "text"},
        ]

        result = format_ask_user_transcript(questions, ["Alice"])

        assert result == f"Q: Name?\nA: Alice\n\nQ: Color?\nA: {ASK_USER_NO_ANSWER}"

    def test_extra_answers_are_dropped(self) -> None:
        questions: list[Question] = [{"question": "Name?", "type": "text"}]

        result = format_ask_user_transcript(questions, ["Alice", "blue"])

        assert result == "Q: Name?\nA: Alice"
        assert "blue" not in result

    def test_missing_question_text_does_not_raise(self) -> None:
        """The tool schema blocks this upstream; degrade rather than raise."""
        result = format_ask_user_transcript([{}], ["Alice"])  # ty: ignore

        assert result == "Q: \nA: Alice"

    def test_no_questions_yields_empty_string(self) -> None:
        assert format_ask_user_transcript([], []) == ""

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

    def test_round_trips_selected_values(self) -> None:
        values = ["Boston, MA", 'quoted "value"', "line one\nline two", "🚀"]

        encoded = encode_multi_select_answer(values)

        assert encoded == (
            '["Boston, MA", "quoted \\"value\\"", "line one\\nline two", "🚀"]'
        )
        assert decode_multi_select_answer(encoded) == values

    def test_round_trips_values_containing_brackets(self) -> None:
        """Brackets collide with the encoding's own delimiters."""
        values = ["[bracket]", "[", "]", '["nested"]']

        encoded = encode_multi_select_answer(values)

        assert decode_multi_select_answer(encoded) == values

    def test_round_trips_an_empty_selection(self) -> None:
        """An untouched optional multi-select encodes as `[]`, not `""`."""
        assert encode_multi_select_answer([]) == "[]"
        assert decode_multi_select_answer("[]") == []

    def test_encode_produces_a_single_line(self) -> None:
        """The transcript is blank-line separated, so the encoding must not be."""
        encoded = encode_multi_select_answer(["a\nb", "c"])

        assert "\n" not in encoded

    def test_decode_rejects_non_json(self) -> None:
        assert decode_multi_select_answer("a, b") is None

    def test_decode_rejects_json_nested_beyond_the_recursion_limit(self) -> None:
        """Untrusted answer text must not abort transcript rendering."""
        depth = 10_000
        deeply_nested = "[" * depth + "]" * depth

        assert decode_multi_select_answer(deeply_nested) is None

    def test_decode_rejects_a_json_object(self) -> None:
        assert decode_multi_select_answer('{"a": 1}') is None

    def test_decode_rejects_non_string_elements(self) -> None:
        assert decode_multi_select_answer('["a", 1]') is None

    def test_decode_rejects_a_bare_json_string(self) -> None:
        """A JSON string is decodable but is not a multi-select answer."""
        assert decode_multi_select_answer('"a"') is None

    def test_decode_rejects_the_empty_string(self) -> None:
        """The old encoding of an untouched question must not decode to `[]`.

        `textual_adapter` still synthesizes `""` answers on its error and cancel
        paths, so a lenient decode here would quietly resurrect the pre-JSON
        semantics.
        """
        assert decode_multi_select_answer("") is None

    def test_decode_rejects_json_scalars(self) -> None:
        assert decode_multi_select_answer("null") is None
        assert decode_multi_select_answer("1") is None
        assert decode_multi_select_answer("true") is None

    def test_decode_rejects_nested_arrays(self) -> None:
        assert decode_multi_select_answer('[["a"]]') is None
        assert decode_multi_select_answer('["a", ["b"]]') is None

    def test_decode_rejects_a_null_element(self) -> None:
        assert decode_multi_select_answer("[null]") is None


class TestAskUserAnswerIsEmpty:
    """Tests for the shared emptiness rule."""

    def test_unselected_multi_select_is_empty_despite_being_truthy(self) -> None:
        assert ask_user_answer_is_empty("[]", "multi_select")

    def test_selected_multi_select_is_not_empty(self) -> None:
        assert not ask_user_answer_is_empty('["a"]', "multi_select")

    def test_multi_select_holding_a_blank_value_is_not_empty(self) -> None:
        """The user selected something, even if the value renders as nothing."""
        assert not ask_user_answer_is_empty('[" "]', "multi_select")

    def test_malformed_multi_select_is_empty(self) -> None:
        """Fail closed: the TUI re-prompts, Auto withholds the consent evidence."""
        assert ask_user_answer_is_empty("not json", "multi_select")

    def test_blank_text_answer_is_empty(self) -> None:
        assert ask_user_answer_is_empty("   ", "text")

    def test_literal_brackets_in_a_text_answer_are_not_empty(self) -> None:
        """The multi-select rule must not leak onto other types."""
        assert not ask_user_answer_is_empty("[]", "text")

    def test_missing_type_takes_the_blank_test(self) -> None:
        assert not ask_user_answer_is_empty("[]", None)


class TestRenderAskUserTranscriptForDisplay:
    """Tests for the display-only re-render."""

    def test_unpacks_multi_select_values_one_per_line(self) -> None:
        questions: list[Question] = [
            {"question": "Where?", "type": "multi_select", "choices": []}
        ]
        transcript = format_ask_user_transcript(
            questions, [encode_multi_select_answer(["Boston, MA", "Austin"])]
        )

        assert (
            render_ask_user_transcript_for_display(questions, transcript)
            == "Q: Where?\nA: Boston, MA\nAustin"
        )

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

    def test_empty_selection_reads_as_nothing_selected(self) -> None:
        """An answered-but-unselected question is not a missing answer.

        `ASK_USER_NO_ANSWER` keeps its single meaning — no positionally matching
        answer — so the display path uses its own placeholder.
        """
        questions: list[Question] = [
            {"question": "Where?", "type": "multi_select", "choices": []}
        ]
        transcript = format_ask_user_transcript(
            questions, [encode_multi_select_answer([])]
        )

        assert render_ask_user_transcript_for_display(questions, transcript) == (
            f"Q: Where?\nA: {ASK_USER_NOTHING_SELECTED}"
        )
        assert ASK_USER_NOTHING_SELECTED != ASK_USER_NO_ANSWER

    def test_other_answers_survive_a_multi_select_neighbour(self) -> None:
        questions: list[Question] = [
            {"question": "Where?", "type": "multi_select", "choices": []},
            {"question": "Why?", "type": "text"},
        ]
        transcript = format_ask_user_transcript(
            questions, [encode_multi_select_answer(["Austin"]), "because\n\nreasons"]
        )

        assert render_ask_user_transcript_for_display(questions, transcript) == (
            "Q: Where?\nA: Austin\n\nQ: Why?\nA: because\n\nreasons"
        )

    def test_returns_none_when_nothing_needed_unpacking(self) -> None:
        """The caller keeps its literal rendering rather than a rebuilt copy."""
        questions: list[Question] = [{"question": "Why?", "type": "text"}]
        transcript = format_ask_user_transcript(questions, ["because"])

        assert render_ask_user_transcript_for_display(questions, transcript) is None

    def test_returns_none_for_the_cancelled_placeholder(self) -> None:
        """Placeholders are not JSON, so the row keeps showing them verbatim."""
        questions: list[Question] = [
            {"question": "Where?", "type": "multi_select", "choices": []}
        ]
        transcript = format_ask_user_transcript(questions, ["(cancelled)"])

        assert render_ask_user_transcript_for_display(questions, transcript) is None

    def test_returns_none_when_the_questions_do_not_match(self) -> None:
        """Give up rather than guess at a transcript built from something else."""
        questions: list[Question] = [
            {"question": "Where?", "type": "multi_select", "choices": []}
        ]
        transcript = format_ask_user_transcript(
            [{"question": "Somewhere else?", "type": "multi_select", "choices": []}],
            [encode_multi_select_answer(["Austin"])],
        )

        assert render_ask_user_transcript_for_display(questions, transcript) is None

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

    def test_trailing_content_is_re_emitted_rather_than_rejected(self) -> None:
        """The final answer runs to the end, so junk cannot be told apart.

        There is no guard against this and there cannot be one: appended text is
        indistinguishable from text the user typed at the end of the last
        answer. It is display-only, so it is carried through verbatim.
        """
        questions: list[Question] = [
            {"question": "Where?", "type": "multi_select", "choices": []},
            {"question": "Why?", "type": "text"},
        ]
        transcript = format_ask_user_transcript(
            questions, [encode_multi_select_answer(["Austin"]), "because"]
        )

        assert render_ask_user_transcript_for_display(
            questions, transcript + "\n\nQ: junk\nA: x"
        ) == ("Q: Where?\nA: Austin\n\nQ: Why?\nA: because\n\nQ: junk\nA: x")

    def test_returns_none_for_three_identically_worded_questions(self) -> None:
        """Their shared anchor makes every separator ambiguous.

        A legitimate transcript, refused: the uniqueness rule cannot tell this
        apart from a crafted answer, and guessing is what it exists to prevent.
        """
        questions: list[Question] = [
            {"question": "Which?", "type": "multi_select", "choices": []},
            {"question": "Which?", "type": "multi_select", "choices": []},
            {"question": "Which?", "type": "multi_select", "choices": []},
        ]
        transcript = format_ask_user_transcript(
            questions, [encode_multi_select_answer(["a"])] * 3
        )

        assert render_ask_user_transcript_for_display(questions, transcript) is None

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

    def test_returns_none_when_an_answer_quotes_a_later_block(self) -> None:
        """A text answer can legitimately contain transcript-like text.

        Without the uniqueness check the parser anchors on the quoted copy, so
        the real multi-select block folds into the first answer and the display
        shows `["fake"]` as the second answer.
        """
        questions: list[Question] = [
            {"question": "Paste it?", "type": "text"},
            {"question": "Where?", "type": "multi_select", "choices": []},
            {"question": "Why?", "type": "text"},
        ]
        quoted = 'here it is\n\nQ: Where?\nA: ["fake"]\n\nQ: Why?\nA: copied verbatim'
        transcript = format_ask_user_transcript(
            questions, [quoted, encode_multi_select_answer(["Austin"]), "because"]
        )

        assert render_ask_user_transcript_for_display(questions, transcript) is None

    def test_returns_none_for_no_questions(self) -> None:
        assert render_ask_user_transcript_for_display([], "") is None


class TestFormatAskUserErrorAnswer:
    """Tests for `format_ask_user_error_answer`."""

    def test_wraps_detail_in_the_sentinel(self) -> None:
        result = format_ask_user_error_answer("boom")

        assert result == "(error: boom)"
        assert result.startswith(ASK_USER_ERROR_ANSWER_PREFIX)
        assert result.endswith(")")

    def test_empty_detail_still_closes_the_sentinel(self) -> None:
        assert format_ask_user_error_answer("") == "(error: )"


def test_row_summary_alias_matches_its_constants() -> None:
    """`AskUserRowSummary` restates the constants, so keep the two in step.

    `Literal[...]` cannot reference a name, so the alias duplicates these values.
    Rewording a constant without updating the alias would make `defer_success`
    reject the very summary it is meant to accept.
    """
    assert set(AskUserRowSummary.__args__) == {  # ty: ignore
        ASK_USER_ANSWERED_SUMMARY,
        ASK_USER_FAILED_SUMMARY,
    }


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

    def test_well_formed_payload_is_accepted(self) -> None:
        """The negative control for the rejections below."""
        self._validate([{"question": "How?", "type": "text"}])

    def test_choice_question_without_choices_is_rejected(self) -> None:
        """Otherwise `_QuestionWidget` degrades it to a text box.

        The widget keeps that degrade as a defence, but this is what makes the
        degrade unreachable rather than a live gap.
        """
        with pytest.raises(ValidationError, match="requires a non-empty 'choices'"):
            self._validate([{"question": "Pick", "type": "multiple_choice"}])

    def test_empty_questions_is_rejected(self) -> None:
        """Otherwise `AskUserMenu([])` renders a prompt with no questions.

        The result is a titled panel with no question widgets and nothing
        focusable. This rule lives on the tool's `questions` parameter too; it
        is attached here as well so both gates enforce it.
        """
        with pytest.raises(ValidationError, match="at least one question"):
            self._validate([])

    def test_blank_question_text_is_rejected(self) -> None:
        """Field-level rules are unconditional, so they apply here too."""
        with pytest.raises(ValidationError, match="must not be blank"):
            self._validate([{"question": "  ", "type": "text"}])
