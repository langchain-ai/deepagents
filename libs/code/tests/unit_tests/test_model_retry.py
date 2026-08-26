"""Tests for dcode model-node retry middleware and retry-count resolution."""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.errors import GraphBubbleUp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )

from deepagents_code import model_retry
from deepagents_code.config import (
    ASCII_GLYPHS,
    DEFAULT_MODEL_RETRIES,
    MODEL_RETRIES_ATTR,
    UNICODE_GLYPHS,
)
from deepagents_code.model_retry import (
    _JITTER_FRACTION,
    CodeModelRetryMiddleware,
    _is_retryable_model_error,
    _retry_after_seconds,
    build_retry_event,
    format_retry_status,
    retry_model_call,
    retry_status_from_event,
)

_UNSET = object()

_READ_ERROR = httpx.ReadError("connection dropped")
_CONNECT_ERROR = httpx.ConnectError("connection refused")
_VALUE_ERROR = ValueError("bad request")


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _ResponseStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("resp")
        self.response = SimpleNamespace(status_code=status_code)


class APIConnectionError(Exception):
    pass


class AuthenticationError(Exception):
    def __init__(self) -> None:
        super().__init__("auth")
        self.status_code = 401


class _GoogleAPICoreError(Exception):
    code: int

    def __init__(self, code: int) -> None:
        super().__init__(f"google status {code}")
        self.code = code


_GoogleAPICoreError.__module__ = "google.api_core.exceptions"


class ResourceExhausted(Exception):  # noqa: N818  # mirrors the Google SDK name
    pass


class _RetryingStreamingModel(BaseChatModel):
    attempts: int = 0

    @property
    def _llm_type(self) -> str:
        return "retrying-stream"

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="final"))]
        )

    def _stream(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Iterator[ChatGenerationChunk]:
        self.attempts += 1
        if self.attempts == 1:
            yield ChatGenerationChunk(message=AIMessageChunk(content="orphaned"))
            raise _READ_ERROR
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="final", chunk_position="last")
        )


class _LiveStreamingModel(BaseChatModel):
    gate: asyncio.Event

    @property
    def _llm_type(self) -> str:
        return "live-stream"

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="firstsecond"))]
        )

    async def _astream(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="first"))
        await self.gate.wait()
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="second", chunk_position="last")
        )


def _req(
    events: list[dict[str, object]] | None = None,
    *,
    model_retries: object = _UNSET,
) -> ModelRequest:
    writer = (lambda event: events.append(event)) if events is not None else None
    model = SimpleNamespace()
    if model_retries is not _UNSET:
        setattr(model, MODEL_RETRIES_ATTR, model_retries)
    return cast(
        "ModelRequest",
        SimpleNamespace(runtime=SimpleNamespace(stream_writer=writer), model=model),
    )


def _handler(
    function: Callable[[object], object],
) -> Callable[[ModelRequest], ModelResponse]:
    return cast("Callable[[ModelRequest], ModelResponse]", function)


def _async_handler(
    function: Callable[[object], Awaitable[object]],
) -> Callable[[ModelRequest], Awaitable[ModelResponse]]:
    return cast("Callable[[ModelRequest], Awaitable[ModelResponse]]", function)


async def _no_sleep(*_args: object, **_kwargs: object) -> None:
    pass


@pytest.mark.parametrize(
    "exc",
    [
        ModelRateLimitError("x"),
        ModelAPIError("x"),
        ModelConnectionError("x"),
        ModelTimeoutError("x"),
    ],
)
def test_predicate_uses_retryable_model_taxonomy(exc: Exception) -> None:
    assert _is_retryable_model_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ModelAuthenticationError("x"),
        ModelPermissionDeniedError("x"),
        ModelInvalidRequestError("x"),
        ContextOverflowError("x"),
    ],
)
def test_predicate_uses_non_retryable_model_taxonomy(exc: Exception) -> None:
    assert _is_retryable_model_error(exc) is False


class _RetryableTransportModelError(httpx.ReadError, ModelInvalidRequestError):
    pass


def test_model_taxonomy_takes_precedence_over_legacy_fallback() -> None:
    assert _is_retryable_model_error(_RetryableTransportModelError("x")) is False


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadError("x"),
        httpx.ConnectError("x"),
        httpx.WriteError("x"),
        httpx.RemoteProtocolError("x"),
        httpx.ConnectTimeout("x"),
        httpx.ReadTimeout("x"),
        httpx.PoolTimeout("x"),
        _StatusError(408),
        _StatusError(409),
        _StatusError(429),
        _StatusError(500),
        _StatusError(503),
        _ResponseStatusError(502),
        APIConnectionError("x"),
        TimeoutError("x"),
        ConnectionError("x"),
    ],
)
def test_predicate_retryable(exc: Exception) -> None:
    assert _is_retryable_model_error(exc) is True


@pytest.mark.parametrize("code", [408, 409, 429, 500, 503, 504])
def test_predicate_retries_google_api_core_status_codes(code: int) -> None:
    assert _is_retryable_model_error(_GoogleAPICoreError(code)) is True


def test_predicate_retries_google_api_core_transient_class_without_code() -> None:
    assert _is_retryable_model_error(ResourceExhausted("quota unavailable")) is True


def test_predicate_rejects_google_api_core_permanent_status_code() -> None:
    assert _is_retryable_model_error(_GoogleAPICoreError(400)) is False


@pytest.mark.parametrize(
    "exc",
    [
        _StatusError(400),
        _StatusError(401),
        _StatusError(403),
        _StatusError(404),
        AuthenticationError(),
        ValueError("bad request"),
        KeyError("schema"),
        RuntimeError("model config error"),
    ],
)
def test_predicate_not_retryable(exc: Exception) -> None:
    assert _is_retryable_model_error(exc) is False


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(
            httpx.UnsupportedProtocol("bad scheme"), id="unsupported-protocol"
        ),
        pytest.param(httpx.LocalProtocolError("bad request"), id="local-protocol"),
        pytest.param(httpx.ProxyError("bad proxy"), id="proxy"),
    ],
)
def test_permanent_transport_errors_are_not_retried(exc: Exception) -> None:
    assert isinstance(exc, httpx.TransportError)
    assert _is_retryable_model_error(exc) is False


def test_middleware_defaults() -> None:
    mw = CodeModelRetryMiddleware()
    assert mw.max_retries == DEFAULT_MODEL_RETRIES


def test_middleware_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        CodeModelRetryMiddleware(max_retries=-1)


def test_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    events: list[dict[str, object]] = []
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _READ_ERROR
        return "OK"

    mw = CodeModelRetryMiddleware(max_retries=5)
    assert mw.wrap_model_call(_req(events), _handler(handler)) == "OK"
    assert calls["n"] == 3
    assert [e["type"] for e in events] == ["model_retry", "model_retry"]
    assert events[0]["message"] == "Retrying model request 1/5"


def test_auxiliary_call_uses_model_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_retry._retry_delay_seconds", lambda *_: 0
    )
    model = SimpleNamespace()
    setattr(model, MODEL_RETRIES_ATTR, 2)
    calls = 0

    def call() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _READ_ERROR
        return "OK"

    assert retry_model_call(model, call) == "OK"
    assert calls == 3


def test_exhaustion_reraises_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=2)
    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(_req(), _handler(handler))
    assert calls["n"] == 3


def test_non_retryable_raises_immediately() -> None:
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _VALUE_ERROR

    mw = CodeModelRetryMiddleware(max_retries=5)
    with pytest.raises(ValueError, match="bad request"):
        mw.wrap_model_call(_req(), _handler(handler))
    assert calls["n"] == 1


def test_graph_bubble_up_is_never_handled() -> None:
    mw = CodeModelRetryMiddleware(max_retries=5)

    def handler(_req_arg: object) -> str:
        raise GraphBubbleUp

    with pytest.raises(GraphBubbleUp):
        mw.wrap_model_call(_req(), _handler(handler))


def test_zero_retries_calls_handler_once() -> None:
    mw = CodeModelRetryMiddleware(max_retries=0)
    assert mw.max_retries == 0
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _READ_ERROR

    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(_req(), _handler(handler))
    assert calls["n"] == 1


def test_retry_reinvokes_only_the_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    seen_requests: list[object] = []

    def handler(req_arg: object) -> str:
        seen_requests.append(req_arg)
        if len(seen_requests) < 2:
            raise _CONNECT_ERROR
        return "OK"

    mw = CodeModelRetryMiddleware(max_retries=3)
    request = _req()
    assert mw.wrap_model_call(request, _handler(handler)) == "OK"
    assert seen_requests == [request, request]


def test_backoff_grows_exponentially_and_is_capped() -> None:
    mw = CodeModelRetryMiddleware(max_retries=12)
    delays = [mw._compute_delay(n) for n in range(12)]  # policy under test

    for n, delay in enumerate(delays):
        nominal = min(0.2 * (2.0**n), 10.0)
        assert delay == pytest.approx(nominal, rel=_JITTER_FRACTION)
    assert delays[-1] <= 10.0 * (1 + _JITTER_FRACTION)
    assert delays[3] > delays[0]


def test_backoff_jitter_varies_between_calls() -> None:
    mw = CodeModelRetryMiddleware()
    samples = {mw._compute_delay(5) for _ in range(20)}  # policy under test
    assert len(samples) > 1


def test_retry_uses_the_computed_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(
        "deepagents_code.model_retry.time.sleep", lambda seconds: slept.append(seconds)
    )
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        if calls["n"] < 4:
            raise _READ_ERROR
        return "OK"

    mw = CodeModelRetryMiddleware(max_retries=5)
    assert mw.wrap_model_call(_req(), _handler(handler)) == "OK"
    assert len(slept) == 3
    assert slept[0] < slept[1] < slept[2]


class _RateLimitError(Exception):
    def __init__(self, retry_after: str = "1") -> None:
        super().__init__("rate limited")
        self.response = SimpleNamespace(
            status_code=429, headers={"retry-after": retry_after}
        )


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        pytest.param("30", 30.0, id="delta-seconds"),
        pytest.param("  45  ", 45.0, id="padded"),
        pytest.param("0", 0.0, id="zero"),
        pytest.param("-5", 0.0, id="negative-clamped"),
        pytest.param("600", 60.0, id="capped"),
    ],
)
def test_retry_after_seconds_parses_delta(header: str, expected: float) -> None:
    assert _retry_after_seconds(_RateLimitError(header)) == pytest.approx(expected)


def test_retry_after_seconds_parses_http_date() -> None:
    when = datetime.now(UTC) + timedelta(seconds=20)
    header = format_datetime(when, usegmt=True)
    result = _retry_after_seconds(_RateLimitError(header))
    assert result is not None
    assert result == pytest.approx(20.0, abs=2.0)


def test_retry_after_seconds_past_date_clamps_to_zero() -> None:
    when = datetime.now(UTC) - timedelta(seconds=30)
    header = format_datetime(when, usegmt=True)
    assert _retry_after_seconds(_RateLimitError(header)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "header",
    [
        pytest.param("soon", id="unparseable"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("nan", id="nan"),
        pytest.param("inf", id="infinity"),
    ],
)
def test_retry_after_seconds_ignores_unusable_headers(header: str) -> None:
    assert _retry_after_seconds(_RateLimitError(header)) is None


def test_retry_after_seconds_absent_header() -> None:
    assert _retry_after_seconds(_READ_ERROR) is None
    assert _retry_after_seconds(_StatusError(429)) is None


_RATE_LIMIT_25S = _RateLimitError("25")
_RATE_LIMIT_18S = _RateLimitError("18")


def test_retry_after_overrides_the_backoff_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(
        "deepagents_code.model_retry.time.sleep", lambda seconds: slept.append(seconds)
    )
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _RATE_LIMIT_25S
        return "OK"

    mw = CodeModelRetryMiddleware(max_retries=3)
    assert mw.wrap_model_call(_req(), _handler(handler)) == "OK"
    assert slept == [pytest.approx(25.0)]


async def test_async_retry_after_overrides_the_backoff_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def _record(seconds: float, *_a: object, **_k: object) -> None:  # noqa: RUF029
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _record)
    calls = {"n": 0}

    async def handler(_req_arg: object) -> str:  # noqa: RUF029  # awaited by middleware; no internal await needed
        calls["n"] += 1
        if calls["n"] < 2:
            raise _RATE_LIMIT_18S
        return "OK"

    mw = CodeModelRetryMiddleware(max_retries=3)
    assert await mw.awrap_model_call(_req(), _async_handler(handler)) == "OK"
    assert slept == [pytest.approx(18.0)]


def test_exhaustion_logs_the_attempt_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)

    def handler(_req_arg: object) -> str:
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=2)
    with (
        caplog.at_level(logging.ERROR, logger="deepagents_code.model_retry"),
        pytest.raises(httpx.ReadError),
    ):
        mw.wrap_model_call(_req(), _handler(handler))

    assert "after 3 attempts" in caplog.text
    assert "retry budget 2 exhausted" in caplog.text


def test_non_transient_failure_is_not_logged_as_exhaustion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(_req_arg: object) -> str:
        raise _VALUE_ERROR

    mw = CodeModelRetryMiddleware(max_retries=5)
    with (
        caplog.at_level(logging.ERROR, logger="deepagents_code.model_retry"),
        pytest.raises(ValueError, match="bad request"),
    ):
        mw.wrap_model_call(_req(), _handler(handler))

    assert "exhausted" not in caplog.text


async def test_async_exhaustion_logs_the_attempt_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    async def handler(_req_arg: object) -> str:  # noqa: RUF029  # awaited by middleware; no internal await needed
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=1)
    with (
        caplog.at_level(logging.ERROR, logger="deepagents_code.model_retry"),
        pytest.raises(httpx.ReadError),
    ):
        await mw.awrap_model_call(_req(), _async_handler(handler))

    assert "after 2 attempts" in caplog.text


def test_model_budget_overrides_startup_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=5)
    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(_req(model_retries=1), _handler(handler))
    assert calls["n"] == 2


def test_zero_startup_budget_still_retries_runtime_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    events: list[dict[str, object]] = []
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _READ_ERROR
        return "OK"

    mw = CodeModelRetryMiddleware(max_retries=0)
    assert mw.wrap_model_call(_req(events, model_retries=3), _handler(handler)) == "OK"
    assert calls["n"] == 3
    assert events[0]["message"] == "Retrying model request 1/3"


def test_model_budget_zero_disables_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=5)
    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(_req(model_retries=0), _handler(handler))
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "stamped",
    [
        pytest.param(True, id="bool-true"),
        pytest.param(-1, id="negative"),
        pytest.param("3", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_invalid_model_budget_falls_back(
    monkeypatch: pytest.MonkeyPatch, stamped: object
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=2)
    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(_req(model_retries=stamped), _handler(handler))
    assert calls["n"] == 3


def test_request_without_model_uses_startup_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=1)
    request = cast(
        "ModelRequest",
        SimpleNamespace(runtime=SimpleNamespace(stream_writer=None)),
    )
    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(request, _handler(handler))
    assert calls["n"] == 2


async def test_async_model_budget_overrides_startup_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    async def handler(_req_arg: object) -> str:  # noqa: RUF029  # awaited by middleware; no internal await needed
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=5)
    with pytest.raises(httpx.ReadError):
        await mw.awrap_model_call(_req(model_retries=1), _async_handler(handler))
    assert calls["n"] == 2


async def test_async_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    async def handler(_req_arg: object) -> str:  # noqa: RUF029  # awaited by middleware; no internal await needed
        calls["n"] += 1
        if calls["n"] < 2:
            raise _READ_ERROR
        return "OK"

    mw = CodeModelRetryMiddleware(max_retries=3)
    assert await mw.awrap_model_call(_req(), _async_handler(handler)) == "OK"
    assert calls["n"] == 2


async def test_successful_stream_chunks_are_delivered_live() -> None:
    gate = asyncio.Event()
    agent = create_agent(
        _LiveStreamingModel(gate=gate),
        middleware=[CodeModelRetryMiddleware(max_retries=1)],
    )
    stream = agent.astream(
        {"messages": [HumanMessage("hi")]},
        stream_mode=["messages"],
        subgraphs=True,
    )

    first = await asyncio.wait_for(anext(stream), timeout=1)
    gate.set()
    chunks = [first]
    chunks.extend([chunk async for chunk in stream])

    message_text = "".join(
        message.text
        for _namespace, mode, data in chunks
        if mode == "messages"
        for message in [data[0]]
        if isinstance(message, AIMessageChunk)
    )
    assert message_text == "firstsecond"


async def test_failed_attempt_is_not_retried_after_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    model = _RetryingStreamingModel()
    agent = create_agent(
        model,
        middleware=[CodeModelRetryMiddleware(max_retries=1)],
    )

    stream = agent.astream(
        {"messages": [HumanMessage("hi")]},
        stream_mode=["messages"],
        subgraphs=True,
    )
    first = await anext(stream)
    with pytest.raises(httpx.ReadError):
        await anext(stream)
    chunks = [first]
    message_text = "".join(
        message.text
        for _namespace, mode, data in chunks
        if mode == "messages"
        for message in [data[0]]
        if isinstance(message, AIMessageChunk)
    )

    assert model.attempts == 1
    assert message_text == "orphaned"


def test_status_helpers() -> None:
    status = format_retry_status(1, 5)
    assert status == "Retrying model request 1/5"
    assert not status.endswith((UNICODE_GLYPHS.ellipsis, ASCII_GLYPHS.ellipsis))
    event = build_retry_event(2, 5)
    assert event["type"] == "model_retry"
    assert event["attempt"] == 2
    assert event["max_retries"] == 5
    assert event["message"] == "Retrying model request 2/5"


def test_stream_tracker_setup_failure_preserves_the_model_error() -> None:
    class _TrackerSetupError(Exception):
        pass

    @contextmanager
    def _broken_tracker(_tracker: object) -> Iterator[None]:
        raise _TrackerSetupError
        yield  # pragma: no cover  # makes this a generator

    middleware = CodeModelRetryMiddleware(max_retries=2)

    def _handler(_request: ModelRequest) -> ModelResponse:
        msg = "model is down"
        raise ModelConnectionError(msg)

    with (
        patch.object(model_retry, "_track_message_streams", _broken_tracker),
        pytest.raises(_TrackerSetupError),
    ):
        middleware.wrap_model_call(
            cast("ModelRequest", SimpleNamespace(model=None)), _handler
        )


def test_transient_failure_with_retries_disabled_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(_req_arg: object) -> str:
        msg = "connection reset"
        raise ModelConnectionError(msg)

    mw = CodeModelRetryMiddleware(max_retries=0)
    with (
        caplog.at_level(logging.WARNING, logger="deepagents_code.model_retry"),
        pytest.raises(ModelConnectionError),
    ):
        mw.wrap_model_call(_req(), _handler(handler))

    assert "retries are disabled" in caplog.text
    assert "non-transient" not in caplog.text


@pytest.mark.parametrize(
    "event",
    [
        pytest.param({"attempt": 7, "max_retries": 5}, id="attempt-past-budget"),
        pytest.param({"attempt": 0, "max_retries": 5}, id="attempt-not-1-indexed"),
        pytest.param({"attempt": True, "max_retries": 5}, id="bool-attempt"),
        pytest.param({"attempt": "1", "max_retries": 5}, id="string-attempt"),
        pytest.param({"max_retries": 5}, id="missing-attempt"),
        pytest.param({}, id="empty"),
    ],
)
def test_malformed_retry_event_is_rejected_and_logged(
    event: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="deepagents_code.model_retry"):
        assert retry_status_from_event(event) == "Retrying model request"
    assert "malformed model_retry payload" in caplog.text


def test_valid_retry_event_renders_the_shared_status() -> None:
    event = build_retry_event(2, 5)
    assert retry_status_from_event(event) == format_retry_status(2, 5)


@pytest.mark.parametrize(
    "handler_name", ["StreamMessagesHandler", "StreamMessagesHandlerV2"]
)
def test_tracked_dedup_ids_reach_the_original_handler(handler_name: str) -> None:
    from langchain_core.callbacks import BaseCallbackManager as _Manager
    from langgraph.pregel import _messages as _lg_messages

    handler_cls = getattr(_lg_messages, handler_name)
    source = handler_cls(lambda _chunk: None, subgraphs=False)
    manager = _Manager(handlers=[source], inheritable_handlers=[source])

    tracker = model_retry._MessageStreamTracker()
    tracked_callbacks = tracker.callbacks_with_tracked_messages(manager)
    assert tracked_callbacks is not None
    tracked = tracked_callbacks.handlers[0]
    assert isinstance(tracked, handler_cls)
    assert tracked is not source

    tracked.seen.add("msg-1")
    assert "msg-1" not in source.seen

    tracker.merge_seen()
    assert "msg-1" in source.seen
