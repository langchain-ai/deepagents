"""Unit tests for ask_user middleware helpers and prompt injection."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import SystemMessage, ToolMessage

from deepagents_cli.ask_user import (
    AskUserMiddleware,
    _parse_answers,
    _validate_questions,
)

if TYPE_CHECKING:
    from langgraph.types import Command


def _extract_tool_message_content(command: Command[object]) -> str:
    """Extract `ToolMessage.content` from a command update payload."""
    update = command.update
    assert isinstance(update, dict)
    messages = update.get("messages")
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, ToolMessage)
    return str(message.content)


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

    def test_malformed_payload_is_explicit_error(self) -> None:
        cmd = _parse_answers(
            "not-a-dict",
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert (
            "A: (error: invalid ask_user response payload)"
            in _extract_tool_message_content(cmd)
        )

    def test_missing_answers_on_answered_status_is_explicit_error(self) -> None:
        cmd = _parse_answers(
            {},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert (
            "A: (error: missing ask_user answers payload)"
            in _extract_tool_message_content(cmd)
        )

    def test_non_list_answers_payload_is_explicit_error(self) -> None:
        cmd = _parse_answers(
            {"answers": "Alice"},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert (
            "A: (error: invalid ask_user answers payload)"
            in _extract_tool_message_content(cmd)
        )

    def test_unknown_status_is_explicit_error(self) -> None:
        cmd = _parse_answers(
            {"status": "unexpected", "answers": ["Alice"]},
            [{"question": "Name?", "type": "text"}],
            "tc-1",
        )
        assert (
            "A: (error: invalid ask_user response status)"
            in _extract_tool_message_content(cmd)
        )

    def test_answer_count_mismatch_falls_back_to_no_answer(self) -> None:
        cmd = _parse_answers(
            {"answers": ["Alice"]},
            [
                {"question": "Name?", "type": "text"},
                {"question": "Color?", "type": "text"},
            ],
            "tc-1",
        )
        content = _extract_tool_message_content(cmd)
        assert "Q: Name?\nA: Alice" in content
        assert "Q: Color?\nA: (no answer)" in content


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
