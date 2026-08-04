"""Unit tests for ask_user middleware helpers and prompt injection."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from deepagents_code._ask_user_types import (
    ASK_USER_AUTHORIZATION_METADATA_KEY,
    MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS,
    Question,
)
from deepagents_code.ask_user import (
    AskUserMiddleware,
    _parse_answers,
    _validate_questions,
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


class TestValidateQuestions:
    """Tests for `_validate_questions`."""

    def test_rejects_empty_questions(self) -> None:
        with pytest.raises(ValueError, match="at least one question"):
            _validate_questions([])

    def test_rejects_empty_question_text(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'question'"):
            _validate_questions([{"question": "   ", "type": "text"}])

    def test_rejects_multiple_choice_without_choices(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty 'choices'"):
            _validate_questions(
                [{"question": "Pick one", "type": "multiple_choice", "choices": []}]
            )

    def test_rejects_text_question_with_choices(self) -> None:
        with pytest.raises(ValueError, match="must not define 'choices'"):
            _validate_questions(
                [
                    {
                        "question": "Name?",
                        "type": "text",
                        "choices": [{"value": "Alice"}],
                    }
                ]
            )

    def test_accepts_valid_question_set(self) -> None:
        _validate_questions(
            [
                {"question": "Name?", "type": "text"},
                {
                    "question": "Color?",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
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
