"""Tests for transient rubric grader transport retries."""

from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from deepagents.middleware.rubric import GraderResponse, RubricState
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.reliable_rubric import (
    ReliableRubricMiddleware,
    _is_transient_grader_transport_error,
)


def _state() -> RubricState:
    return cast(
        "RubricState",
        {
            "rubric": "tests pass",
            "messages": [
                HumanMessage(content="implement it"),
                AIMessage(content="implementation complete"),
            ],
        },
    )


def _satisfied_result() -> dict[str, Any]:
    return {
        "structured_response": GraderResponse(
            result="satisfied",
            explanation="all checks pass",
            criteria=[],
        )
    }


class TestTransientGraderTransportClassification:
    def test_read_error_is_transient(self) -> None:
        error = httpx.ReadError(
            "connection closed while reading",
            request=httpx.Request("POST", "https://grader.test"),
        )

        assert _is_transient_grader_transport_error(error) is True

    def test_transfer_encoding_error_in_cause_chain_is_transient(self) -> None:
        error_type = type(
            "TransferEncodingError",
            (Exception,),
            {"__module__": "aiohttp.http_exceptions"},
        )
        cause = error_type("Not enough data to satisfy transfer length header")
        wrapper = RuntimeError("grader request failed")
        wrapper.__cause__ = cause

        assert _is_transient_grader_transport_error(wrapper) is True

    def test_unrelated_exception_is_not_transient(self) -> None:
        assert _is_transient_grader_transport_error(RuntimeError("bug")) is False


class TestReliableRubricMiddleware:
    async def test_retries_only_grading_without_mutating_agent_transcript(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        error = httpx.ReadError(
            "connection closed while reading",
            request=httpx.Request("POST", "https://grader.test"),
        )
        grader = AsyncMock()
        grader.ainvoke.side_effect = [error, _satisfied_result()]
        middleware._grader = grader
        state = _state()
        messages_before = list(state["messages"])

        result = await middleware._agrade(state, 0)

        assert result.result == "satisfied"
        assert grader.ainvoke.await_count == 2
        assert state["messages"] == messages_before

    async def test_does_not_retry_unrelated_exception(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = AsyncMock()
        grader.ainvoke.side_effect = RuntimeError("programming error")
        middleware._grader = grader

        with pytest.raises(RuntimeError, match="programming error"):
            await middleware._agrade(_state(), 0)

        grader.ainvoke.assert_awaited_once()
