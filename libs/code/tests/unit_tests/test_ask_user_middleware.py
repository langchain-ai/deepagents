"""Unit tests for ask_user middleware helpers and prompt injection."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import (
    ToolNode,
    _filter_validation_errors,
)
from langgraph.types import Command
from pydantic import BaseModel, TypeAdapter, ValidationError

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
    _parse_answers,
)


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
applies, without needing a live tool invocation.

It is a *parallel* schema, not the tool's own, so it cannot catch the tool
losing an annotation. `TestToolArgumentValidation` covers each rule through a
real invocation for that reason. The empty-list rule is not visible here at
all: it is attached to the `questions` parameter and to
`AskUserRequest.questions`, not to the item type."""


def _validate(questions: object) -> None:
    """Parse `questions` against the validated schema, raising on any violation."""
    _VALIDATION_ADAPTER.validate_python(questions)


class TestValidateQuestions:
    """Tests for the pydantic validation rules on `ValidatedQuestion`.

    These rules replace the old imperative validation in `ask_user.py`: raising
    `ValueError` from a validator surfaces as a pydantic `ValidationError`,
    which `ToolNode` converts to an error `ToolMessage` the model can
    correct.
    """

    def test_rejects_blank_question_text(self) -> None:
        """Empty or whitespace-only text would render as a blank prompt.

        The `AfterValidator` runs before `min_length=1` is consulted, so the
        empty string lands on the same "blank" rejection as whitespace-only
        text.
        """
        for text in ("", "   ", "\t\n ", "\u00a0"):
            with pytest.raises(ValidationError, match="must not be blank"):
                _validate([{"question": text, "type": "text"}])

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
        with pytest.raises(ValidationError, match="blank 'value'"):
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


def _harness_runtime() -> ToolRuntime[Any, Any]:
    """Build the `runtime` object `ToolNode` would inject.

    A real `ToolRuntime` rather than a stand-in: it is a dataclass field on the
    tool's `args_schema`, so pydantic rejects anything else — which is the very
    harness fault `TestHarnessFaultIsNotBlamedOnTheModel` covers.
    """
    return ToolRuntime[Any, Any](
        state=_turn_state("turn-1"),
        context={"thread_id": "t1", "turn_id": "turn-1"},
        config={},
        stream_writer=lambda _: None,
        tool_call_id="c1",
        store=None,
        tools=[],
        execution_info=None,
        server_info=None,
    )


def _invoke_ask_user(questions: object) -> object:
    """Invoke the middleware's `ask_user` tool on raw model-authored args.

    `tool_call_id` and `runtime` are injected the way `ToolNode` injects them,
    so the only validation error in play is the one the caller is testing.
    Malformed `questions` raise `ValidationError` — nothing on the tool converts
    that to a `ToolMessage`.
    """
    return _invoke_ask_user_raw(
        {"questions": questions, "tool_call_id": "c1", "runtime": _harness_runtime()}
    )


def _invoke_ask_user_raw(args: dict[str, Any]) -> object:
    """Invoke the tool on a complete args dict, injected arguments included.

    A malformed call raises `ValidationError` out of argument parsing; a call
    that clears the schema runs the tool body and returns a `Command`.
    """
    tool = AskUserMiddleware().tools[0]
    return tool.invoke(
        {"args": args, "name": "ask_user", "id": "c1", "type": "tool_call"}
    )


class TestToolArgumentValidation:
    """Bad `ask_user` arguments become a `ValidationError` during parsing.

    No handling is wired on the tool. `ToolNode` converts the error into a
    recoverable `ToolMessage` and strips the injected arguments from it first,
    which is what `test_end_to_end` covers. These tests pin that the tool
    rejects at parse time and leaves the conversion to the framework.
    """

    def test_valid_questions_clear_the_schema(self) -> None:
        """The negative control: well-formed args must not be rejected.

        Without this, a schema change that rejected *every* input would leave
        the rest of this class green.
        """
        with patch(
            "deepagents_code.ask_user.interrupt",
            return_value={"answers": ["Rebase"]},
        ):
            result = _invoke_ask_user_raw(
                {
                    "questions": [{"question": "How?", "type": "text"}],
                    "tool_call_id": "c1",
                    "runtime": _harness_runtime(),
                }
            )

        assert isinstance(result, Command)
        message = _extract_tool_message(cast("Command[object]", result))
        assert message.status != "error"

    def test_empty_questions_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one question"):
            _invoke_ask_user([])

    def test_blank_choice_value_names_the_field(self) -> None:
        with pytest.raises(ValidationError, match="blank 'value'"):
            _invoke_ask_user(
                [
                    {
                        "question": "Pick some",
                        "type": "multi_select",
                        "choices": [{"value": "logs"}, {"value": "  "}],
                    }
                ]
            )

    def test_unknown_question_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Input should be"):
            _invoke_ask_user([{"question": "Q?", "type": "multiselect"}])

    def test_choice_question_without_choices_is_rejected(self) -> None:
        """Pins `ValidatedQuestion` on the tool's own `questions` annotation.

        Both cross-field rules live on that alias and nowhere else on the tool,
        so this test and its sibling below are what stop the annotation from
        silently degrading to `list[Question]`. Every other case in this class
        survives that change, because each is enforced by the parameter-level
        validator, by a `Choice` validator, or by `Literal`.
        """
        with pytest.raises(ValidationError, match="requires a non-empty 'choices'"):
            _invoke_ask_user([{"question": "Pick", "type": "multiple_choice"}])

    def test_non_choice_question_with_choices_is_rejected(self) -> None:
        """The other half of the cross-field rule. See the sibling above."""
        with pytest.raises(ValidationError, match="must not define 'choices'"):
            _invoke_ask_user(
                [
                    {
                        "question": "Why?",
                        "type": "text",
                        "choices": [{"value": "a"}],
                    }
                ]
            )

    def test_blank_question_text_is_rejected(self) -> None:
        """Covered against the parallel adapter too, but pinned here as well.

        A blank prompt is the most visible of the failures these rules prevent,
        so it is worth holding at the boundary the model actually reaches.
        """
        with pytest.raises(ValidationError, match="must not be blank"):
            _invoke_ask_user([{"question": "   ", "type": "text"}])

    def test_stringly_typed_required_is_rejected(self) -> None:
        """`strict=True` must survive on the tool's own schema.

        This is the case with the quietest failure mode if it regresses: a
        coerced `"false"` renders the prompt, and then
        `_ask_user_question_count` — which reads the raw tool args and requires
        a real bool — drops every answer in the call as same-turn
        authorization, with no error anywhere.
        """
        with pytest.raises(ValidationError, match="valid boolean"):
            _invoke_ask_user([{"question": "Q?", "type": "text", "required": "false"}])

    def test_min_length_reaches_the_model_facing_schema(self) -> None:
        """`min_length=1` exists only to emit `minLength` for the model.

        `_validate_question_text` runs first and rejects everything the
        constraint would, so nothing else in the suite would notice its
        removal — but the model would stop being told the field has a minimum.
        """
        tool = AskUserMiddleware().tools[0]
        # `tool_call_schema`, not `args_schema`: the latter still carries the
        # injected `runtime`, which has no JSON schema representation.
        schema = TypeAdapter(tool.tool_call_schema).json_schema()
        question = schema["$defs"]["Question"]["properties"]["question"]
        assert question["minLength"] == 1

    def test_no_error_handling_is_wired_on_the_tool(self) -> None:
        """The tool must leave both error hooks unset.

        `handle_validation_error` is undocumented in LangChain v1, and the
        migration guide says schema mismatches are already handled by the
        framework. Setting either hook here would intercept inside
        `BaseTool.run`, which (a) bypasses `_filter_validation_errors`, so a
        harness fault would be reported to the model as its own bad input, and
        (b) makes `BaseTool.run` call `on_tool_end`, so tracing would record a
        rejected call as a success. Setting `handle_tool_error` would also
        swallow the `interrupt()` signal.
        """
        tool = AskUserMiddleware().tools[0]

        assert not tool.handle_validation_error
        assert not tool.handle_tool_error


class TestBodyFaultsStayFatal:
    """A fault raised after parsing must not become model-facing input.

    `_parse_answers` raises plain `ValueError` for a malformed resume payload.
    These are not model-authored arguments, and no `handle_tool_error` is set,
    so they propagate and halt the run.

    A `ValidationError` is the exception, and the reason the tool body carries
    an explicit guard: `ToolNode` wraps the body in the same `try` as argument
    parsing, so one escaping from here would be reported to the model as its
    own bad input. `test_body_raised_validation_error_is_fatal` pins the guard.
    """

    def test_body_raised_value_error_is_fatal(self) -> None:
        with (
            patch(
                "deepagents_code.ask_user.interrupt",
                side_effect=ValueError("bad resume payload"),
            ),
            pytest.raises(ValueError, match="bad resume payload"),
        ):
            _invoke_ask_user([{"question": "How?", "type": "text"}])

    def test_non_value_error_is_fatal(self) -> None:
        with (
            patch(
                "deepagents_code.ask_user.interrupt",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            _invoke_ask_user([{"question": "How?", "type": "text"}])

    def test_body_raised_validation_error_is_fatal(self) -> None:
        """A `ValidationError` from the body must not blame the model.

        Without the guard in `_ask_user`, this surfaces to the model as an
        error `ToolMessage` naming a field that is not on the tool schema,
        against arguments the model wrote correctly, while the user's answer is
        discarded and the run continues. The re-raise keeps it fatal by making
        it a type `_default_handle_tool_errors` refuses to convert.
        """

        class _Inner(BaseModel):
            count: int

        def _raise_validation_error(_request: object) -> None:
            _Inner(count="not-an-int")  # type: ignore[arg-type]

        with (
            patch(
                "deepagents_code.ask_user.interrupt",
                side_effect=_raise_validation_error,
            ),
            pytest.raises(RuntimeError, match="not a model-authored error") as excinfo,
        ):
            _invoke_ask_user([{"question": "How?", "type": "text"}])

        assert isinstance(excinfo.value.__cause__, ValidationError)


class TestHarnessFaultIsNotBlamedOnTheModel:
    """A malformed *injected* argument is a harness fault, not model input.

    `tool_call_id` and `runtime` sit on the same `args_schema` as `questions`,
    so pydantic reports them the same way. The model cannot rewrite either, so
    reporting one back would loop it to the recursion limit. `ToolNode` filters
    `runtime` out of the message; `tool_call_id` stays out because
    `ToolInvocationError` is built from the pre-injection arguments.
    """

    def test_missing_runtime_is_a_validation_error_at_the_boundary(self) -> None:
        """The fault is raised, not silently defaulted.

        This is the raw tool boundary, below `ToolNode`, so the error is still a
        `ValidationError` here and names the injected field.
        """
        with pytest.raises(ValidationError, match="runtime") as excinfo:
            _invoke_ask_user_raw(
                {
                    "questions": [{"question": "How?", "type": "text"}],
                    "tool_call_id": "c1",
                }
            )

        assert "runtime" in {str(e["loc"][0]) for e in excinfo.value.errors()}

    def test_tool_node_filters_the_injected_argument_out(self) -> None:
        """`runtime` must not survive into the model-facing message.

        The end-to-end test cannot pin this: `ToolNode` injects `runtime`
        correctly on every real call, so it is never the field that failed. The
        error has to be forced here instead.

        This reaches into `langgraph` internals on purpose. The tool wires no
        `handle_validation_error` *because* this filtering exists, so if the
        private helper moves or changes shape, that decision needs revisiting
        and this test is the alarm.
        """
        tool = AskUserMiddleware().tools[0]
        with pytest.raises(ValidationError) as excinfo:
            _invoke_ask_user_raw(
                {
                    "questions": [{"question": "How?", "type": "text"}],
                    "tool_call_id": "c1",
                }
            )

        node = ToolNode([tool])
        filtered = _filter_validation_errors(
            excinfo.value,
            node._injected_args.get("ask_user"),
        )

        assert "runtime" not in {str(e["loc"][0]) for e in filtered}
