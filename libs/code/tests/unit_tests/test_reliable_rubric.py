"""Tests for transient rubric grader transport retries."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from deepagents.middleware.rubric import GraderResponse, RubricState
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code._constants import SDK_DEFAULT_RUBRIC_MAX_ITERATIONS
from deepagents_code.reliable_rubric import (
    ReliableRubricMiddleware,
    _is_transient_grader_transport_error,
)


def _read_error() -> httpx.ReadError:
    return httpx.ReadError(
        "connection closed while reading",
        request=httpx.Request("POST", "https://grader.test"),
    )


def _typed_error(module: str, name: str, message: str = "boom") -> Exception:
    """Build an exception whose type mimics an external library's error class."""
    error_type = type(name, (Exception,), {"__module__": module})
    return error_type(message)


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
        assert _is_transient_grader_transport_error(_read_error()) is True

    def test_remote_protocol_error_is_transient(self) -> None:
        assert (
            _is_transient_grader_transport_error(httpx.RemoteProtocolError("boom"))
            is True
        )

    def test_httpcore_read_error_is_transient(self) -> None:
        # httpcore errors cannot be caught via a stable isinstance, so the
        # classifier matches them by module/name; exercise that path directly.
        error = _typed_error("httpcore", "ReadError")

        assert _is_transient_grader_transport_error(error) is True

    def test_httpcore_remote_protocol_error_is_transient(self) -> None:
        error = _typed_error("httpcore._exceptions", "RemoteProtocolError")

        assert _is_transient_grader_transport_error(error) is True

    def test_read_error_in_exception_group_is_transient(self) -> None:
        group = ExceptionGroup(
            "grading failed", [ValueError("unrelated"), _read_error()]
        )

        assert _is_transient_grader_transport_error(group) is True

    def test_read_error_in_context_chain_is_transient(self) -> None:
        wrapper = RuntimeError("grader request failed")
        wrapper.__context__ = _read_error()

        assert _is_transient_grader_transport_error(wrapper) is True

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
    def test_displayed_max_iterations_default_matches_sdk(self) -> None:
        """Drift guard for the TUI-display duplicate of the SDK default.

        The constant must equal the `RubricMiddleware` default that the app
        actually instantiates.
        """
        middleware = ReliableRubricMiddleware(model="fake-model")

        assert middleware.max_iterations == SDK_DEFAULT_RUBRIC_MAX_ITERATIONS

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

    def test_sync_grade_retries_transient_transport_failure(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.invoke.side_effect = [_read_error(), _satisfied_result()]
        middleware._grader = grader

        result = middleware._grade(_state(), 0)

        assert result.result == "satisfied"
        assert grader.invoke.call_count == 2

    async def test_second_transient_failure_propagates_async(self) -> None:
        # The retry is bounded to one attempt: a second transient failure must
        # surface so the base middleware can report it as a grader_error.
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = AsyncMock()
        grader.ainvoke.side_effect = [_read_error(), _read_error()]
        middleware._grader = grader

        with pytest.raises(httpx.ReadError):
            await middleware._agrade(_state(), 0)

        assert grader.ainvoke.await_count == 2

    def test_second_transient_failure_propagates_sync(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.invoke.side_effect = [_read_error(), _read_error()]
        middleware._grader = grader

        with pytest.raises(httpx.ReadError):
            middleware._grade(_state(), 0)

        assert grader.invoke.call_count == 2
