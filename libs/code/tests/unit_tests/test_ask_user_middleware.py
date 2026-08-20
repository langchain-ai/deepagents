"""Unit tests for ask_user middleware helpers and prompt injection."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import TypeAdapter, ValidationError

from deepagents_code._ask_user_types import (
    ASK_USER_AUTHORIZATION_METADATA_KEY,
    CHOICE_QUESTION_TYPES,
    MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS,
    QUESTION_TYPES,
    Question,
    ValidatedQuestion,
    _requires_choices,
    decode_multi_select_answer,
    encode_multi_select_answer,
)
from deepagents_code.ask_user import (
    AskUserMiddleware,
    _format_validation_error,
    _parse_answers,
)

if TYPE_CHECKING:
    from langgraph.types import Command


def _extract_tool_message(command: Command[object]) -> ToolMessage:
    update = command.update
    assert isinstance(update, dict)
    messages = update.get("messages")
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, ToolMessage)
    return message


def _extract_tool_message_content(command: Command[object]) -> str:
    """Extract `ToolMessage.content` from a command update payload."""
    return str(_extract_tool_message(command).content)


_VALIDATION_ADAPTER = TypeAdapter(list[ValidatedQuestion])
"""Parses raw tool-args payloads the way the tool schema's `questions` field does.

This adapter is the unit under test for the validation rules: it applies the
same `Literal`/strict-bool/`AfterValidator` checks the tool's pydantic model
applies, without needing a live `ToolNode`. The empty-list rule lives one
level up, on the `questions` parameter itself, and is covered by
`test_ask_user_tool_rejects_empty_questions` instead."""


def _validate(questions: object) -> None:
    """Parse `questions` against the validated schema, raising on any violation."""
    _VALIDATION_ADAPTER.validate_python(questions)


class TestValidateQuestions:
    """Tests for the pydantic validation rules on `ValidatedQuestion`.

    These rules replace the old imperative `_validate_questions` body: raising
    `ValueError` from a validator surfaces as a pydantic `ValidationError`,
    which `ToolNode` converts to an error `ToolMessage` the model can correct.
    """

    def test_rejects_empty_question_text(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 character"):
            _validate([{"question": "", "type": "text"}])

    def test_rejects_multiple_choice_without_choices(self) -> None:
        with pytest.raises(ValidationError, match="requires a non-empty 'choices'"):
            _validate(
                [{"question": "Pick one", "type": "multiple_choice", "choices": []}]
            )

    def test_rejects_text_question_with_choices(self) -> None:
        with pytest.raises(ValidationError, match="must not define 'choices'"):
            _validate(
                [
                    {
                        "question": "Name?",
                        "type": "text",
                        "choices": [{"value": "Alice"}],
                    }
                ]
            )

    def test_rejects_multi_select_without_choices(self) -> None:
        with pytest.raises(
            ValidationError, match=r"multi_select question .* non-empty"
        ):
            _validate(
                [{"question": "Pick some", "type": "multi_select", "choices": []}]
            )

    def test_rejects_blank_choice_value(self) -> None:
        """A blank label would render as a selectable option with no answer."""
        with pytest.raises(ValidationError, match="missing or blank 'value'"):
            _validate(
                [
                    {
                        "question": "Pick some",
                        "type": "multi_select",
                        "choices": [{"value": "logs"}, {"value": "  "}],
                    }
                ]
            )

    def test_rejects_non_string_choice_value(self) -> None:
        """The `Choice.value` field type rejects a non-string before the validator."""
        with pytest.raises(ValidationError):
            _validate(
                [
                    {
                        "question": "Color?",
                        "type": "multiple_choice",
                        "choices": [{"value": 1}],
                    }
                ]
            )

    def test_allows_comma_in_multi_select_choice_value(self) -> None:
        """The JSON-array answer encoding keeps a comma inside a value exact."""
        _validate(
            [
                {
                    "question": "Where?",
                    "type": "multi_select",
                    "choices": [{"value": "Boston, MA"}, {"value": "Austin"}],
                }
            ]
        )

    def test_allows_comma_in_multiple_choice_value(self) -> None:
        """Choice values are returned as-is, so a comma needs no special handling."""
        _validate(
            [
                {
                    "question": "Where?",
                    "type": "multiple_choice",
                    "choices": [{"value": "Boston, MA"}],
                }
            ]
        )

    def test_rejects_unknown_question_type(self) -> None:
        """Nothing outside `QuestionType` may reach the interrupt."""
        with pytest.raises(ValidationError, match="Input should be"):
            _validate([{"question": "Q?", "type": "multiselect"}])

    def test_tool_schema_rejects_non_boolean_required(self) -> None:
        """Pydantic must reject `required: "false"` rather than coercing it.

        This is the check that actually runs in production, and it has to be
        strict. `_ask_user_question_count` reads the *raw* tool args and requires
        a real bool, so a coerced `"false"` would render the prompt, let the user
        answer, and then return `None` — dropping every answer in the call as
        same-turn authorization with no error.
        """
        adapter = TypeAdapter(list[Question])

        # A real bool is still accepted, in both Python and JSON form.
        assert adapter.validate_python(
            [{"question": "Q?", "type": "text", "required": False}]
        ) == [{"question": "Q?", "type": "text", "required": False}]
        assert adapter.validate_json(
            '[{"question": "Q?", "type": "text", "required": true}]'
        ) == [{"question": "Q?", "type": "text", "required": True}]

        for coercible in ("false", "true", 0, 1):
            with pytest.raises(ValidationError):
                adapter.validate_python(
                    [{"question": "Q?", "type": "text", "required": coercible}]
                )

    def test_accepts_every_declared_question_type(self) -> None:
        """Guards against a `QuestionType` member the validator rejects.

        Note this cannot catch a member *added* to `QuestionType`, since the
        fixture derives its shape from `CHOICE_QUESTION_TYPES`. That direction is
        covered by `test_choice_question_types_covers_every_question_type` and by
        the widget-side `assert_never` in `_QuestionWidget.compose`.
        """
        for question_type in sorted(QUESTION_TYPES):
            question: dict[str, Any] = {
                "question": "Q?",
                "type": question_type,
            }
            if question_type in CHOICE_QUESTION_TYPES:
                question["choices"] = [{"value": "a"}, {"value": "b"}]
            _validate([question])

    def test_choice_question_types_covers_every_question_type(self) -> None:
        """`CHOICE_QUESTION_TYPES` must partition `QUESTION_TYPES`, not lag it.

        Non-tautological in the direction that matters: a `QuestionType` member
        missing from `_requires_choices` would pass `_validate_question` with no
        choices validation *and* make `_ask_user_question_count` return `None`
        for any payload that does carry choices.
        """
        assert CHOICE_QUESTION_TYPES <= QUESTION_TYPES
        assert {
            question_type
            for question_type in QUESTION_TYPES
            if _requires_choices(cast("Any", question_type))
        } == CHOICE_QUESTION_TYPES

    def test_non_choice_question_types_reject_choices(self) -> None:
        """Every non-choice type must refuse a `choices` list."""
        for question_type in sorted(QUESTION_TYPES - CHOICE_QUESTION_TYPES):
            questions = [
                {
                    "question": "Q?",
                    "type": question_type,
                    "choices": [{"value": "a"}],
                }
            ]
            with pytest.raises(ValidationError, match="must not define 'choices'"):
                _validate(questions)

    def test_accepts_valid_question_set(self) -> None:
        _validate(
            [
                {"question": "Name?", "type": "text"},
                {
                    "question": "Color?",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                },
                {
                    "question": "Toppings?",
                    "type": "multi_select",
                    "choices": [{"value": "cheese"}, {"value": "olives"}],
                },
            ]
        )


class TestParseAnswers:
    """Tests for `_parse_answers`."""

    def test_parses_answered_payload(self) -> None:
        cmd = _parse_answers(
            {"answers": ["Alice"]},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert "Q: Name?" in _extract_tool_message_content(cmd)
        assert "A: Alice" in _extract_tool_message_content(cmd)

    def test_records_trusted_same_turn_authorization_receipt(self) -> None:
        cmd = _parse_answers(
            {"answers": ["Rebase my commit onto the remote branch"]},
            [
                {
                    "question": "How should I integrate the remote branch?",
                    "type": "multiple_choice",
                    "choices": [
                        {"value": "Rebase my commit onto the remote branch"},
                        {"value": "Merge the remote branch"},
                    ],
                }
            ],
            "ask-1",
            thread_id="thread-1",
            turn_id="turn-1",
        )

        message = _extract_tool_message(cmd)
        assert message.name == "ask_user"
        assert message.additional_kwargs[ASK_USER_AUTHORIZATION_METADATA_KEY] == {
            "version": 1,
            "thread_id": "thread-1",
            "turn_id": "turn-1",
            "tool_call_id": "ask-1",
            "answers": ["Rebase my commit onto the remote branch"],
        }

    @pytest.mark.parametrize(
        ("response", "questions", "thread_id", "turn_id"),
        [
            (
                {"status": "cancelled", "answers": ["ignored"]},
                [{"question": "Proceed?", "type": "text"}],
                "thread-1",
                "turn-1",
            ),
            (
                {"status": "error", "error": "prompt failed"},
                [{"question": "Proceed?", "type": "text"}],
                "thread-1",
                "turn-1",
            ),
            (
                "malformed",
                [{"question": "Proceed?", "type": "text"}],
                "thread-1",
                "turn-1",
            ),
            (
                {},
                [{"question": "Proceed?", "type": "text"}],
                "thread-1",
                "turn-1",
            ),
            (
                {"answers": ["yes"]},
                [
                    {"question": "Proceed?", "type": "text"},
                    {"question": "Target?", "type": "text"},
                ],
                "thread-1",
                "turn-1",
            ),
            (
                {"answers": [True]},
                [{"question": "Proceed?", "type": "text"}],
                "thread-1",
                "turn-1",
            ),
            (
                {"answers": ["x" * (MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS + 1)]},
                [{"question": "Proceed?", "type": "text"}],
                "thread-1",
                "turn-1",
            ),
            (
                {"answers": ["yes"]},
                [{"question": "Proceed?", "type": "text"}],
                None,
                "turn-1",
            ),
            (
                {"answers": ["yes"]},
                [{"question": "Proceed?", "type": "text"}],
                "thread-1",
                None,
            ),
        ],
    )
    def test_invalid_answer_has_no_authorization_receipt(
        self,
        response: object,
        questions: list[Question],
        thread_id: str | None,
        turn_id: str | None,
    ) -> None:
        cmd = _parse_answers(
            response,
            questions,
            "ask-1",
            thread_id=thread_id,
            turn_id=turn_id,
        )

        assert (
            ASK_USER_AUTHORIZATION_METADATA_KEY
            not in _extract_tool_message(cmd).additional_kwargs
        )

    def test_json_escaping_can_push_an_answer_over_the_receipt_cap(self) -> None:
        """The per-answer budget is measured on the encoded string.

        Escaping inflates the wire form, so a selection whose decoded content
        fits can still lose its receipt. Fail-closed, but worth pinning: the
        units changed when the encoding did.
        """
        values = ["\n" * ((MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS // 2) - 1)]
        answer = encode_multi_select_answer(values)
        assert len(values[0]) < MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS
        assert len(answer) > MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS

        cmd = _parse_answers(
            {"answers": [answer]},
            [{"question": "Which?", "type": "multi_select", "choices": []}],
            "ask-1",
            thread_id="thread-1",
            turn_id="turn-1",
        )

        assert (
            ASK_USER_AUTHORIZATION_METADATA_KEY
            not in _extract_tool_message(cmd).additional_kwargs
        )

    def test_cancelled_status_uses_cancelled_placeholder(self) -> None:
        cmd = _parse_answers(
            {"status": "cancelled", "answers": ["ignored"]},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert "A: (cancelled)" in _extract_tool_message_content(cmd)

    def test_error_status_uses_error_placeholder(self) -> None:
        cmd = _parse_answers(
            {"status": "error", "error": "failed to display ask_user prompt"},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert (
            "A: (error: failed to display ask_user prompt)"
            in _extract_tool_message_content(cmd)
        )

    def test_error_status_marks_the_tool_message_as_errored(self) -> None:
        """A failed prompt must not be recorded as a successful tool call.

        `status` defaults to `"success"`, which told the model the tool had
        succeeded and made a reloaded thread render the `(error: ...)` transcript
        as an ordinary answered row.
        """
        cmd = _parse_answers(
            {"status": "error", "error": "failed to display ask_user prompt"},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert _extract_tool_message(cmd).status == "error"

    def test_answered_status_marks_the_tool_message_as_successful(self) -> None:
        cmd = _parse_answers(
            {"status": "answered", "answers": ["Alice"]},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert _extract_tool_message(cmd).status == "success"

    def test_cancelled_status_marks_the_tool_message_as_successful(self) -> None:
        """Cancelling is a user choice, not a tool failure."""
        cmd = _parse_answers(
            {"status": "cancelled", "answers": []},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert _extract_tool_message(cmd).status == "success"

    @pytest.mark.parametrize(
        ("response", "expected_detail"),
        [
            ("not-a-dict", "invalid ask_user response payload"),
            ({}, "missing ask_user answers payload"),
            ({"answers": "Alice"}, "invalid ask_user answers payload"),
            (
                {"status": "unexpected", "answers": ["Alice"]},
                "invalid ask_user response status",
            ),
        ],
        ids=["not-a-dict", "missing-answers", "non-list-answers", "unknown-status"],
    )
    def test_malformed_payloads_are_explicit_errors(
        self, response: object, expected_detail: str
    ) -> None:
        """Every malformed payload errors the `ToolMessage`, not just its text.

        The status is asserted alongside the transcript because the two are set in
        different places: a regression that narrowed the `status=` expression to,
        say, a caller-supplied status rather than the locally reassigned one would
        keep every transcript assertion green while re-marking these payloads as
        successful — the exact bug this branch exists to prevent, and the value
        that now also drives the row badge on reload.
        """
        message = _extract_tool_message(
            _parse_answers(response, [{"question": "Name?", "type": "text"}], "tc-1")
        )

        assert message.status == "error"
        assert f"A: (error: {expected_detail})" in str(message.content)

    def test_caller_declared_error_detail_wins_over_a_local_one(self) -> None:
        """An explicit `error` from the caller is the root cause; keep it.

        A caller that declares `status="error"` knows why. A payload that instead
        claims `"answered"` and fails validation here may still carry a stale
        `error` field, and that must not describe a defect this function found —
        so the two details are tracked separately rather than overwriting.
        """
        declared = _extract_tool_message(
            _parse_answers(
                {"status": "error", "error": "widget crashed", "answers": "bad"},
                [{"question": "Name?", "type": "text"}],
                "tc-1",
            )
        )
        assert "A: (error: widget crashed)" in str(declared.content)

        stale = _extract_tool_message(
            _parse_answers(
                {"status": "answered", "error": "stale", "answers": "bad"},
                [{"question": "Name?", "type": "text"}],
                "tc-1",
            )
        )
        assert "A: (error: invalid ask_user answers payload)" in str(stale.content)

    def test_error_status_without_a_detail_uses_the_default(self) -> None:
        """The third arm of the detail chain: neither a caller nor a local detail.

        `status="error"` with no `error` field and a well-formed answer list reaches
        neither `client_error_text` nor `local_error_text`, so the generic fallback
        is what the model sees.
        """
        message = _extract_tool_message(
            _parse_answers(
                {"status": "error", "answers": [""]},
                [{"question": "Name?", "type": "text"}],
                "tc-1",
            )
        )

        assert message.status == "error"
        assert "A: (error: ask_user interaction failed)" in str(message.content)

    def test_non_string_answers_are_coerced_loudly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A coerced answer is presented to the model as the user's own words.

        The coercion is deliberate — the model still gets something per question —
        but it also silently withholds the authorization receipt, so an operator
        relying on that receipt needs a reason for its absence.
        """
        with caplog.at_level(logging.WARNING):
            message = _extract_tool_message(
                _parse_answers(
                    {"answers": [42]},
                    [{"question": "How many?", "type": "text"}],
                    "tc-1",
                    thread_id="thread-1",
                    turn_id="turn-1",
                )
            )

        assert message.status == "success"
        assert "A: 42" in str(message.content)
        assert ASK_USER_AUTHORIZATION_METADATA_KEY not in message.additional_kwargs
        assert any(
            "non-string answer element" in record.message for record in caplog.records
        )

    def test_answer_count_mismatch_is_an_error(self) -> None:
        """A short answer list is a failed prompt, not a partial one.

        Padding with `(no answer)` would keep `status="success"` while silently
        re-attributing every answer after the gap to the wrong question — here
        `"Alice"` would stay on `Name?` only by luck of it being first. The model
        must be told the payload was unusable rather than handed a confident
        wrong pairing.
        """
        cmd = _parse_answers(
            {"answers": ["Alice"]},
            [
                {"question": "Name?", "type": "text"},
                {"question": "Color?", "type": "text"},
            ],
            "tc-1",
        )
        message = _extract_tool_message(cmd)
        assert message.status == "error"
        content = str(message.content)
        assert "Q: Name?\nA: (error: ask_user answer count mismatch" in content
        assert "Q: Color?\nA: (error: ask_user answer count mismatch" in content
        assert "expected 2, got 1" in content
        assert "Alice" not in content

    def test_extra_answers_are_also_an_error(self) -> None:
        """A long list is equally untrustworthy; extras would be dropped."""
        cmd = _parse_answers(
            {"answers": ["Alice", "blue"]},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        message = _extract_tool_message(cmd)
        assert message.status == "error"
        assert "expected 1, got 2" in str(message.content)


class TestMultiSelectEncoding:
    """Tests for the JSON-array encoding of `multi_select` answers on the wire."""

    def test_answer_with_commas_quotes_and_newlines_round_trips(self) -> None:
        """Punctuation that broke the joined encoding must survive verbatim."""
        questions: list[Question] = [
            {
                "question": "Which constraints apply?",
                "type": "multi_select",
                "choices": [
                    {"value": 'push-to-main — no PR label, always "strict"'},
                    {"value": "line one\nline two"},
                ],
            }
        ]
        answer = encode_multi_select_answer(
            ['push-to-main — no PR label, always "strict"', "line one\nline two"]
        )

        cmd = _parse_answers(
            {"answers": [answer]},
            questions,
            "tc-1",
            thread_id="thread-1",
            turn_id="turn-1",
        )

        message = _extract_tool_message(cmd)
        assert message.status == "success"
        assert f"A: {answer}" in str(message.content)
        receipt = message.additional_kwargs[ASK_USER_AUTHORIZATION_METADATA_KEY]
        assert decode_multi_select_answer(receipt["answers"][0]) == [
            'push-to-main — no PR label, always "strict"',
            "line one\nline two",
        ]

    def test_transcript_renders_the_json_array_verbatim(self) -> None:
        """The model sees the self-delimiting form, not a re-joined string."""
        questions: list[Question] = [
            {
                "question": "Where?",
                "type": "multi_select",
                "choices": [{"value": "Boston, MA"}, {"value": "Austin"}],
            }
        ]
        answer = encode_multi_select_answer(["Boston, MA", "Austin"])

        content = _extract_tool_message_content(
            _parse_answers({"answers": [answer]}, questions, "tc-1")
        )

        assert content == 'Q: Where?\nA: ["Boston, MA", "Austin"]'

    def test_empty_multi_select_answer_keeps_its_receipt(self) -> None:
        """`[]` is a real answer — it must not read as a blank `A:` line."""
        questions: list[Question] = [
            {
                "question": "Extras?",
                "type": "multi_select",
                "choices": [{"value": "docs"}],
                "required": False,
            }
        ]

        cmd = _parse_answers(
            {"answers": [encode_multi_select_answer([])]},
            questions,
            "tc-1",
            thread_id="thread-1",
            turn_id="turn-1",
        )

        message = _extract_tool_message(cmd)
        assert "A: []" in str(message.content)
        receipt = message.additional_kwargs[ASK_USER_AUTHORIZATION_METADATA_KEY]
        assert decode_multi_select_answer(receipt["answers"][0]) == []


def _turn_state(turn_id: str) -> dict[str, object]:
    from deepagents_code.auto_mode import USER_PROMPT_METADATA_KEY

    return {
        "messages": [
            HumanMessage(
                content="request",
                additional_kwargs={
                    USER_PROMPT_METADATA_KEY: {
                        "literal_user_text": "request",
                        "referenced_paths": [],
                        "turn_id": turn_id,
                    }
                },
            )
        ]
    }


class TestAskUserTool:
    def test_runtime_identity_is_bound_to_resumed_answer(self) -> None:
        ask_tool = cast("Any", AskUserMiddleware().tools[0])
        questions = [{"question": "How should I integrate?", "type": "text"}]
        runtime = SimpleNamespace(
            context={"thread_id": "thread-1", "turn_id": "turn-1"},
            execution_info=SimpleNamespace(thread_id="thread-1"),
            tool_call_id="ask-1",
            state=_turn_state("turn-1"),
        )

        with patch(
            "deepagents_code.ask_user.interrupt",
            return_value={"answers": ["Rebase my commit"]},
        ):
            command = ask_tool.func(
                questions=questions,
                tool_call_id="ask-1",
                runtime=runtime,
            )

        receipt = _extract_tool_message(command).additional_kwargs[
            ASK_USER_AUTHORIZATION_METADATA_KEY
        ]
        assert receipt["thread_id"] == "thread-1"
        assert receipt["turn_id"] == "turn-1"
        assert receipt["tool_call_id"] == "ask-1"
        assert set(ask_tool.args) == {"questions"}

    @pytest.mark.parametrize(
        "runtime",
        [
            SimpleNamespace(
                context={"thread_id": "other-thread", "turn_id": "turn-1"},
                execution_info=SimpleNamespace(thread_id="thread-1"),
                tool_call_id="ask-1",
                state=_turn_state("turn-1"),
            ),
            SimpleNamespace(
                context={"thread_id": "thread-1", "turn_id": "turn-1"},
                execution_info=None,
                tool_call_id="ask-1",
                state=_turn_state("turn-1"),
            ),
            SimpleNamespace(
                context={"thread_id": "thread-1"},
                execution_info=SimpleNamespace(thread_id="thread-1"),
                tool_call_id="ask-1",
                state=_turn_state("turn-1"),
            ),
            SimpleNamespace(
                context={"thread_id": "thread-1", "turn_id": "turn-1"},
                execution_info=SimpleNamespace(thread_id="thread-1"),
                tool_call_id="different-call",
                state=_turn_state("turn-1"),
            ),
            SimpleNamespace(
                context={"thread_id": "thread-1", "turn_id": "turn-1"},
                execution_info=SimpleNamespace(thread_id="thread-1"),
                tool_call_id="ask-1",
                state=_turn_state("older-turn"),
            ),
        ],
    )
    def test_invalid_runtime_identity_does_not_mint_receipt(
        self, runtime: object
    ) -> None:
        ask_tool = cast("Any", AskUserMiddleware().tools[0])
        with patch(
            "deepagents_code.ask_user.interrupt",
            return_value={"answers": ["yes"]},
        ):
            command = ask_tool.func(
                questions=[{"question": "Proceed?", "type": "text"}],
                tool_call_id="ask-1",
                runtime=runtime,
            )

        assert (
            ASK_USER_AUTHORIZATION_METADATA_KEY
            not in _extract_tool_message(command).additional_kwargs
        )


class TestWrapModelCall:
    """Tests for ask_user prompt injection wrappers."""

    def test_wrap_model_call_appends_system_prompt(self) -> None:
        middleware = AskUserMiddleware(system_prompt="ASK_USER_PROMPT")
        request = Mock()
        request.system_message = SystemMessage(
            content=[{"type": "text", "text": "Base prompt"}]
        )
        overridden_request = Mock()
        request.override.return_value = overridden_request
        handler = Mock(return_value="ok")

        result = middleware.wrap_model_call(request, handler)

        request.override.assert_called_once()
        override_kwargs = request.override.call_args.kwargs
        system_message = override_kwargs["system_message"]
        assert isinstance(system_message, SystemMessage)
        assert system_message.content_blocks[-1]["text"] == "\n\nASK_USER_PROMPT"
        handler.assert_called_once_with(overridden_request)
        assert result == "ok"

    def test_wrap_model_call_creates_system_prompt_when_missing(self) -> None:
        middleware = AskUserMiddleware(system_prompt="ASK_USER_PROMPT")
        request = Mock()
        request.system_message = None
        overridden_request = Mock()
        request.override.return_value = overridden_request
        handler = Mock(return_value="ok")

        middleware.wrap_model_call(request, handler)

        override_kwargs = request.override.call_args.kwargs
        system_message = override_kwargs["system_message"]
        assert isinstance(system_message, SystemMessage)
        assert system_message.content_blocks == [
            {"type": "text", "text": "ASK_USER_PROMPT"}
        ]

    async def test_awrap_model_call_appends_system_prompt(self) -> None:
        middleware = AskUserMiddleware(system_prompt="ASK_USER_PROMPT")
        request = Mock()
        request.system_message = SystemMessage(
            content=[{"type": "text", "text": "Base prompt"}]
        )
        overridden_request = Mock()
        request.override.return_value = overridden_request
        handler = AsyncMock(return_value="ok")

        result = await middleware.awrap_model_call(request, handler)

        request.override.assert_called_once()
        override_kwargs = request.override.call_args.kwargs
        system_message = override_kwargs["system_message"]
        assert isinstance(system_message, SystemMessage)
        assert system_message.content_blocks[-1]["text"] == "\n\nASK_USER_PROMPT"
        handler.assert_awaited_once_with(overridden_request)
        assert result == "ok"


def _invoke_ask_user(args: dict[str, Any]) -> ToolMessage:
    """Invoke the middleware's `ask_user` tool on raw model-authored args.

    Exercises the same parse-and-validate path `ToolNode` drives: a rejection
    surfaces as an error `ToolMessage` (via `handle_validation_error`), not a
    raised exception.
    """
    tool = AskUserMiddleware().tools[0]
    result = tool.invoke(
        {"args": args, "name": "ask_user", "id": "c1", "type": "tool_call"}
    )
    assert isinstance(result, ToolMessage)
    return result


class TestToolArgumentValidation:
    """Bad `ask_user` arguments come back as error `ToolMessage`s, not crashes.

    Validation lives on the tool schema, so `ToolNode`'s built-in
    `ValidationError` handling — not a `ToolErrorMiddleware` — is what makes a
    malformed call recoverable.
    """

    def test_empty_questions_becomes_error_tool_message(self) -> None:
        result = _invoke_ask_user({"questions": []})

        assert result.status == "error"
        assert result.tool_call_id == "c1"
        assert "`ask_user` failed" in str(result.content)
        assert "at least one question" in str(result.content)

    def test_blank_choice_value_names_the_field(self) -> None:
        result = _invoke_ask_user(
            {
                "questions": [
                    {
                        "question": "Pick some",
                        "type": "multi_select",
                        "choices": [{"value": "logs"}, {"value": "  "}],
                    }
                ]
            }
        )

        assert result.status == "error"
        assert "missing or blank 'value'" in str(result.content)

    def test_unknown_question_type_is_rejected(self) -> None:
        result = _invoke_ask_user(
            {"questions": [{"question": "Q?", "type": "multiselect"}]}
        )

        assert result.status == "error"
        assert "Input should be" in str(result.content)

    def test_handle_validation_error_only_covers_validation(self) -> None:
        """`handle_validation_error` fires only for `ValidationError`.

        The tool body still raises for real faults (`interrupt()` signals, a
        bad resume payload); `BaseTool.run` routes those to `handle_tool_error`
        (unset here) rather than the validation handler, so they stay fatal.
        This pins that the registered handler is the validation one only.
        """
        tool = AskUserMiddleware().tools[0]

        assert tool.handle_validation_error is _format_validation_error
        assert not tool.handle_tool_error


class TestFormatValidationError:
    """`_format_validation_error` locates the failing field for the model."""

    def test_names_loc_and_detail(self) -> None:
        # Errors arrive from the tool's pydantic model with the parameter name
        # leading `loc`; `from_exception_data` reproduces that shape.
        exc = ValidationError.from_exception_data(
            "ask_user",
            [
                {
                    "type": "literal_error",
                    "loc": ("questions", 0, "type"),
                    "input": "bogus",
                    "ctx": {"expected": "'text', 'multiple_choice' or 'multi_select'"},
                }
            ],
        )

        message = _format_validation_error(exc)

        assert message.startswith("`ask_user` failed: questions.0.type: ")
        assert "Input should be 'text', 'multiple_choice' or 'multi_select'" in message
        assert message.endswith("Fix the input and retry.")

    def test_empty_questions_list(self) -> None:
        # The empty-list rule lives on the `questions` parameter, one level
        # above `list[ValidatedQuestion]`, so its error is shaped by hand here.
        exc = ValidationError.from_exception_data(
            "ask_user",
            [
                {
                    "type": "value_error",
                    "loc": ("questions",),
                    "input": [],
                    "ctx": {
                        "error": ValueError("ask_user requires at least one question")
                    },
                }
            ],
        )

        message = _format_validation_error(exc)

        assert message == (
            "`ask_user` failed: questions: Value error, "
            "ask_user requires at least one question. Fix the input and retry."
        )
