"""Tests for BlockingCallGuardMiddleware."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.blocking_guard import (
    _RECOVERABLE_MESSAGE,
    BlockingCallGuardMiddleware,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _BlockingError(Exception):
    """Stand-in matching BlockBuster's `BlockingError` by class name.

    The guard matches on the class name so the package has no runtime
    dependency on `blockbuster`, which is only present under `langgraph dev`.
    """


_BlockingError.__name__ = "BlockingError"

_SENTINEL = "not-a-blocking-error"


def _make_request() -> ModelRequest:
    """Build a minimal model request for the guard under test."""
    runtime = SimpleNamespace(context=None)
    return ModelRequest(
        model=cast("Any", SimpleNamespace()),
        messages=[HumanMessage(content="hi")],
        tools=[],
        runtime=cast("Any", runtime),
        model_settings=None,
    )


def _ok_response() -> ModelResponse[Any]:
    """Build a normal downstream response."""
    return ModelResponse(result=[AIMessage(content="response")])


async def test_awrap_recovers_from_blocking_error() -> None:
    """A BlockingError from an inner handler becomes a recoverable message."""

    async def handler(_request: ModelRequest) -> ModelResponse[Any]:
        await asyncio.sleep(0)
        raise _BlockingError

    guard = BlockingCallGuardMiddleware()
    result = await guard.awrap_model_call(_make_request(), handler)

    assert isinstance(result, ModelResponse)
    assert len(result.result) == 1
    message = result.result[0]
    assert isinstance(message, AIMessage)
    assert message.content == _RECOVERABLE_MESSAGE
    assert not getattr(message, "tool_calls", None)


async def test_awrap_passes_through_normal_response() -> None:
    """A successful handler response is returned unchanged."""

    async def handler(_request: ModelRequest) -> ModelResponse[Any]:
        await asyncio.sleep(0)
        return _ok_response()

    guard = BlockingCallGuardMiddleware()
    result = await guard.awrap_model_call(_make_request(), handler)

    assert result.result[0].content == "response"


async def test_awrap_reraises_other_exceptions() -> None:
    """Non-blocking exceptions still propagate so real bugs are not masked."""

    async def handler(_request: ModelRequest) -> ModelResponse[Any]:
        await asyncio.sleep(0)
        raise ValueError(_SENTINEL)

    guard = BlockingCallGuardMiddleware()
    with pytest.raises(ValueError, match=_SENTINEL):
        await guard.awrap_model_call(_make_request(), handler)


def test_wrap_recovers_from_blocking_error() -> None:
    """The sync path recovers a BlockingError the same way as the async path."""

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        raise _BlockingError

    guard = BlockingCallGuardMiddleware()
    result = guard.wrap_model_call(_make_request(), cast("Callable[..., Any]", handler))

    assert isinstance(result, ModelResponse)
    assert result.result[0].content == _RECOVERABLE_MESSAGE


def test_wrap_reraises_other_exceptions() -> None:
    """Non-blocking exceptions still propagate on the sync path."""

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        raise RuntimeError(_SENTINEL)

    guard = BlockingCallGuardMiddleware()
    with pytest.raises(RuntimeError, match=_SENTINEL):
        guard.wrap_model_call(_make_request(), cast("Callable[..., Any]", handler))
