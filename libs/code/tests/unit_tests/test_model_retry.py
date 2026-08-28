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
    build_attempt_event,
    build_retry_event,
    format_retry_status,
    model_attempt_from_event,
    model_retry_from_event,
    retry_model_call,
    retry_status_from_event,
)

_UNSET = object()

_READ_ERROR = httpx.ReadError("connection dropped")
_CONNECT_ERROR = httpx.ConnectError("connection refused")
_VALUE_ERROR = ValueError("bad request")
_DROPPED = "connection dropped"
_RETRY_AFTER_30 = "30"
_RETRY_AFTER_1 = "1"


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


# Classification is keyed by `(root package, class name)`, so a fixture standing
# in for a provider SDK error has to claim that package.
APIConnectionError.__module__ = "openai"


class _UnrelatedAPIConnectionError(Exception):
    """Same class name, different package: must not be classified transient."""


_UnrelatedAPIConnectionError.__name__ = "APIConnectionError"
_UnrelatedAPIConnectionError.__module__ = "some_unrelated_lib.errors"


class AuthenticationError(Exception):
    def __init__(self) -> None:
        super().__init__("auth")
        self.status_code = 401


def _typed_error(module: str, name: str, message: str = "boom") -> Exception:
    """Build an exception whose type mimics an optional transport library."""
    error_type = type(name, (Exception,), {"__module__": module})
    return error_type(message)


class _GoogleAPICoreError(Exception):
    code: int

    def __init__(self, code: int) -> None:
        super().__init__(f"google status {code}")
        self.code = code


_GoogleAPICoreError.__module__ = "google.api_core.exceptions"


class ResourceExhausted(Exception):  # noqa: N818  # mirrors the Google SDK name
    pass


ResourceExhausted.__module__ = "google.api_core.exceptions"


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


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(
            _typed_error("botocore.exceptions", "ConnectionClosedError"),
            id="botocore-connection-closed",
        ),
        pytest.param(_typed_error("httpcore", "ReadError"), id="httpcore-read"),
        pytest.param(
            _typed_error("httpcore._exceptions", "RemoteProtocolError"),
            id="httpcore-remote-protocol",
        ),
        pytest.param(
            _typed_error(
                "aiohttp.http_exceptions",
                "TransferEncodingError",
                "Not enough data to satisfy transfer length header",
            ),
            id="aiohttp-incomplete-transfer",
        ),
    ],
)
def test_predicate_retries_lower_level_transport_errors(exc: Exception) -> None:
    assert _is_retryable_model_error(exc) is True


def test_predicate_retries_transport_error_in_exception_group() -> None:
    exc = ExceptionGroup("request failed", [ValueError("bad"), _READ_ERROR])

    assert _is_retryable_model_error(exc) is True


def test_predicate_retries_transport_error_in_context_chain() -> None:
    exc = RuntimeError("request failed")
    exc.__context__ = _typed_error("httpcore", "RemoteProtocolError")

    assert _is_retryable_model_error(exc) is True


def test_predicate_retries_transport_error_in_cause_chain() -> None:
    exc = RuntimeError("request failed")
    exc.__cause__ = _typed_error(
        "aiohttp.http_exceptions",
        "TransferEncodingError",
        "Not enough data to satisfy transfer length header",
    )

    assert _is_retryable_model_error(exc) is True


@pytest.mark.parametrize("code", [408, 409, 429, 500, 503, 504])
def test_predicate_retries_google_api_core_status_codes(code: int) -> None:
    assert _is_retryable_model_error(_GoogleAPICoreError(code)) is True


def test_predicate_retries_google_api_core_transient_class_without_code() -> None:
    assert _is_retryable_model_error(ResourceExhausted("quota unavailable")) is True


def test_predicate_ignores_a_transient_class_name_from_another_package() -> None:
    """`Aborted`, `APIConnectionError` and friends are generic words.

    Matching a bare class name would classify any dependency's identically named
    error as a transient provider failure, burning the whole retry budget on a
    permanent fault.
    """
    assert _is_retryable_model_error(_UnrelatedAPIConnectionError("x")) is False


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
    assert mw.stream_output_is_visible is True


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
    assert [(e["type"], e.get("phase"), e["attempt"]) for e in events] == [
        ("model_attempt", "start", 0),
        ("model_retry", None, 1),
        ("model_attempt", "start", 1),
        ("model_retry", None, 2),
        ("model_attempt", "start", 2),
        ("model_attempt", "complete", 2),
    ]
    retry_events = [e for e in events if e["type"] == "model_retry"]
    assert retry_events[0]["message"] == "Retrying model request 1/5"


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


def test_graph_bubble_up_is_never_handled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """See the async twin: the log assertion is what pins the clause."""
    mw = CodeModelRetryMiddleware(max_retries=5)

    def handler(_req_arg: object) -> str:
        raise GraphBubbleUp

    with (
        caplog.at_level(logging.DEBUG, logger="deepagents_code.model_retry"),
        pytest.raises(GraphBubbleUp),
    ):
        mw.wrap_model_call(_req(), _handler(handler))

    assert "non-transient" not in caplog.text


async def test_async_graph_bubble_up_is_never_handled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The async loop needs its own guard, and the server runs the async loop.

    `_aretry_call` carries a separate `except GraphBubbleUp: raise` from its
    sync twin, and only `wrap_model_call` was covered. Asserting propagation
    alone does not pin the clause: without it `GraphBubbleUp` falls into
    `except Exception`, is judged non-retryable, and is re-raised anyway. The
    clause earns its place by keeping the give-up log out of it -- otherwise a
    graph interrupt is reported as a "non-transient" model failure, burying
    real control flow in a misleading line.
    """
    mw = CodeModelRetryMiddleware(max_retries=5)

    async def handler(_req_arg: object) -> str:  # noqa: RUF029  # awaited by middleware; no internal await needed
        raise GraphBubbleUp

    with (
        caplog.at_level(logging.DEBUG, logger="deepagents_code.model_retry"),
        pytest.raises(GraphBubbleUp),
    ):
        await mw.awrap_model_call(_req(), _async_handler(handler))

    assert "non-transient" not in caplog.text


def _req_with_writer(writer: Callable[[dict[str, object]], None]) -> ModelRequest:
    return cast(
        "ModelRequest",
        SimpleNamespace(
            runtime=SimpleNamespace(stream_writer=writer), model=SimpleNamespace()
        ),
    )


def test_writer_graph_bubble_up_propagates() -> None:
    """A `GraphBubbleUp` from the stream writer is control flow, not a fault.

    `_emit_retry_status` catches `Exception` to keep a broken writer from
    failing the run, and `GraphBubbleUp` subclasses `Exception`, so the
    narrower clause must come first. Deleting it left the suite green while
    converting a LangGraph interrupt raised through the writer into a logged
    warning that is then discarded -- the agent continues as if nothing
    happened.
    """
    calls = {"n": 0}

    def writer(_event: dict[str, object]) -> None:
        raise GraphBubbleUp

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=5)
    with pytest.raises(GraphBubbleUp):
        mw.wrap_model_call(_req_with_writer(writer), _handler(handler))
    # The interrupt surfaces from the first lifecycle event, before the
    # handler or the retry budget is touched.
    assert calls["n"] == 0


def test_writer_failure_is_logged_and_retry_proceeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken writer costs the status event, never the retry itself."""

    def writer(_event: dict[str, object]) -> None:
        msg = "writer exploded"
        raise RuntimeError(msg)

    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _READ_ERROR
        return "ok"

    mw = CodeModelRetryMiddleware(max_retries=5)
    with (
        caplog.at_level(logging.WARNING, logger="deepagents_code.model_retry"),
        patch.object(model_retry.time, "sleep"),
    ):
        result = mw.wrap_model_call(_req_with_writer(writer), _handler(handler))

    assert result == "ok"
    assert calls["n"] == 2
    assert "Failed to emit model_retry stream event" in caplog.text


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
    # The backoff curve is module policy, not middleware state.
    delays = [model_retry._compute_backoff_delay(n) for n in range(12)]

    for n, delay in enumerate(delays):
        nominal = min(0.2 * (2.0**n), 10.0)
        assert delay == pytest.approx(nominal, rel=_JITTER_FRACTION)
    assert delays[-1] <= 10.0 * (1 + _JITTER_FRACTION)
    assert delays[3] > delays[0]


def test_backoff_jitter_varies_between_calls() -> None:
    samples = {model_retry._compute_backoff_delay(5) for _ in range(20)}
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


def test_retry_after_seconds_past_date_is_unusable() -> None:
    """An elapsed hint must not cancel the backoff.

    Returning 0.0 here is not the same as returning `None`: 0.0 is a real
    delay, so both loops skip the sleep and burn the whole budget in a tight
    loop against a server that just asked us to wait.
    """
    when = datetime.now(UTC) - timedelta(seconds=30)
    header = format_datetime(when, usegmt=True)
    assert _retry_after_seconds(_RateLimitError(header)) is None


@pytest.mark.parametrize(
    "header",
    [
        pytest.param("soon", id="unparseable"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("nan", id="nan"),
        pytest.param("inf", id="infinity"),
        pytest.param("0", id="zero-carries-no-wait"),
        pytest.param("-5", id="negative-carries-no-wait"),
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
    retry_events = [e for e in events if e["type"] == "model_retry"]
    assert retry_events[0]["message"] == "Retrying model request 1/3"


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


async def test_failed_attempt_is_retried_after_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-stream transient drop must recover while retry budget remains."""
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    model = _RetryingStreamingModel()
    agent = create_agent(
        model,
        middleware=[CodeModelRetryMiddleware(max_retries=1)],
    )

    chunks = [
        chunk
        async for chunk in agent.astream(
            {"messages": [HumanMessage("hi")]},
            stream_mode=["messages", "custom"],
            subgraphs=True,
        )
    ]
    message_text = "".join(
        message.text
        for _namespace, mode, data in chunks
        if mode == "messages"
        for message in [data[0]]
        if isinstance(message, AIMessageChunk)
    )
    events = [
        cast("dict[str, Any]", data)
        for _namespace, mode, data in chunks
        if mode == "custom"
    ]

    assert model.attempts == 2
    assert message_text == "orphanedfinal"
    call_ids = {event["call_id"] for event in events}
    assert len(call_ids) == 1
    assert [
        (event["type"], event.get("phase"), event["attempt"]) for event in events
    ] == [
        ("model_attempt", "start", 0),
        ("model_retry", None, 1),
        ("model_attempt", "start", 1),
        ("model_attempt", "complete", 1),
    ]
    assert events[1]["failed_attempt"] == 0
    assert events[1]["output_may_have_started"] is True


def test_status_helpers() -> None:
    status = format_retry_status(1, 5)
    assert status == "Retrying model request 1/5"
    assert not status.endswith((UNICODE_GLYPHS.ellipsis, ASCII_GLYPHS.ellipsis))
    event = build_retry_event(2, 5)
    assert event["type"] == "model_retry"
    assert event["attempt"] == 2
    assert event["max_retries"] == 5
    assert event["message"] == "Retrying model request 2/5"


def test_stream_tracker_setup_failure_propagates() -> None:
    """A tracker that cannot be set up must fail loudly, not be swallowed.

    The handler never runs here, so there is no model error to preserve --
    the earlier name claimed otherwise. What matters is that a broken tracker
    surfaces instead of leaving the loop retrying against half-initialised
    streaming state.
    """

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


def test_retry_event_correlation_fields_round_trip() -> None:
    event = build_retry_event(
        2, 5, call_id="abc123", failed_attempt=1, output_may_have_started=True
    )
    assert event["call_id"] == "abc123"
    assert event["failed_attempt"] == 1
    assert event["output_may_have_started"] is True
    assert retry_status_from_event(event) == format_retry_status(2, 5)


def test_retry_correlation_parser_accepts_valid_event() -> None:
    event = build_retry_event(
        2, 5, call_id="abc123", failed_attempt=1, output_may_have_started=True
    )
    assert model_retry_from_event(event) == {
        "call_id": "abc123",
        "failed_attempt": 1,
        "output_may_have_started": True,
    }


@pytest.mark.parametrize(
    "event",
    [
        pytest.param({}, id="legacy"),
        pytest.param(
            {
                "call_id": "bad id",
                "failed_attempt": 0,
                "output_may_have_started": True,
            },
            id="invalid-call-id",
        ),
        pytest.param(
            {"call_id": "abc", "failed_attempt": True, "output_may_have_started": True},
            id="bool-attempt",
        ),
        pytest.param(
            {"call_id": "abc", "failed_attempt": 0, "output_may_have_started": 1},
            id="non-bool-visible",
        ),
    ],
)
def test_retry_correlation_parser_rejects_untrusted_fields(
    event: dict[str, object],
) -> None:
    assert model_retry_from_event(event) is None


def test_retry_event_without_correlation_keeps_legacy_shape() -> None:
    """Producers without lifecycle support must emit the original payload."""
    event = build_retry_event(1, 3)
    assert "call_id" not in event
    assert "failed_attempt" not in event
    assert "output_may_have_started" not in event


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"call_id": "abc"}, "provided together", id="call-id-without-attempt"
        ),
        pytest.param(
            {"failed_attempt": 0}, "provided together", id="attempt-without-call-id"
        ),
    ],
)
def test_retry_event_rejects_partial_correlation(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_retry_event(1, 3, **cast("Any", kwargs))


def test_build_attempt_event() -> None:
    assert build_attempt_event("call-1", 2, phase="start") == {
        "type": "model_attempt",
        "phase": "start",
        "call_id": "call-1",
        "attempt": 2,
    }
    assert build_attempt_event("call-1", 2, phase="complete")["phase"] == "complete"


def test_build_attempt_event_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="phase must be one of"):
        build_attempt_event("call-1", 0, phase="explode")


def test_attempt_parser_accepts_valid_events() -> None:
    for phase in ("start", "complete"):
        event = {"phase": phase, "call_id": "x" * 64, "attempt": 0, "extra": 1}
        parsed = model_attempt_from_event(event)
        assert parsed == {
            "type": "model_attempt",
            "phase": phase,
            "call_id": "x" * 64,
            "attempt": 0,
        }


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            {"phase": "explode", "call_id": "abc", "attempt": 0},
            id="unknown-phase",
        ),
        pytest.param(
            {"phase": ["start"], "call_id": "abc", "attempt": 0},
            id="list-phase",
        ),
        pytest.param(
            {"phase": {"value": "start"}, "call_id": "abc", "attempt": 0},
            id="object-phase",
        ),
        pytest.param({"phase": "start", "attempt": 0}, id="missing-call-id"),
        pytest.param(
            {"phase": "start", "call_id": 123, "attempt": 0}, id="non-string-call-id"
        ),
        pytest.param(
            {"phase": "start", "call_id": "", "attempt": 0}, id="empty-call-id"
        ),
        pytest.param(
            {"phase": "start", "call_id": "x" * 65, "attempt": 0},
            id="overlong-call-id",
        ),
        pytest.param(
            {"phase": "start", "call_id": "a b\tc", "attempt": 0},
            id="control-chars-in-call-id",
        ),
        pytest.param(
            {"phase": "start", "call_id": "abc", "attempt": True}, id="bool-attempt"
        ),
        pytest.param(
            {"phase": "start", "call_id": "abc", "attempt": "0"},
            id="string-attempt",
        ),
        pytest.param(
            {"phase": "start", "call_id": "abc", "attempt": -1},
            id="negative-attempt",
        ),
        pytest.param({}, id="empty"),
    ],
)
def test_malformed_attempt_event_is_rejected_and_logged(
    event: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="deepagents_code.model_retry"):
        assert model_attempt_from_event(event) is None
    assert "malformed model_attempt lifecycle fields" in caplog.text


def test_middleware_call_id_is_stable_per_invocation_and_unique_across_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    middleware = CodeModelRetryMiddleware(max_retries=1)
    invocation_ids: list[set[object]] = []

    def run_once() -> None:
        events: list[dict[str, object]] = []
        calls = {"n": 0}

        def handler(_req_arg: object) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _READ_ERROR
            return "OK"

        assert middleware.wrap_model_call(_req(events), _handler(handler)) == "OK"
        invocation_ids.append({e["call_id"] for e in events})

    run_once()
    run_once()

    assert all(len(ids) == 1 for ids in invocation_ids)
    assert invocation_ids[0] != invocation_ids[1]


def test_lifecycle_events_stop_at_permanent_error_and_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a decided retry emits `model_retry`; failures end after `start`."""
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    middleware = CodeModelRetryMiddleware(max_retries=1)

    permanent_events: list[dict[str, object]] = []

    def permanent_handler(_req_arg: object) -> str:
        raise _VALUE_ERROR

    with pytest.raises(ValueError, match="bad request"):
        middleware.wrap_model_call(_req(permanent_events), _handler(permanent_handler))
    assert [(e["type"], e.get("phase")) for e in permanent_events] == [
        ("model_attempt", "start")
    ]

    exhausted_events: list[dict[str, object]] = []

    def transient_handler(_req_arg: object) -> str:
        raise _READ_ERROR

    with pytest.raises(httpx.ReadError):
        middleware.wrap_model_call(_req(exhausted_events), _handler(transient_handler))
    assert [(e["type"], e.get("phase")) for e in exhausted_events] == [
        ("model_attempt", "start"),
        ("model_retry", None),
        ("model_attempt", "start"),
    ]


def test_hidden_model_call_marks_retry_output_as_not_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtered nested streams retry without claiming visible supersession."""
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    events: list[dict[str, object]] = []
    calls = 0

    @contextmanager
    def _already_streamed(
        tracker: model_retry._MessageStreamTracker,
    ) -> Iterator[None]:
        tracker.has_streamed = True
        yield

    def _handler(_request: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadError(_DROPPED)
        return cast("ModelResponse", "verdict")

    middleware = CodeModelRetryMiddleware(
        max_retries=1,
        stream_output_is_visible=False,
    )
    with patch.object(model_retry, "_track_message_streams", _already_streamed):
        assert middleware.wrap_model_call(_req(events), _handler) == "verdict"

    retry_events = [e for e in events if e["type"] == "model_retry"]
    assert retry_events[0]["output_may_have_started"] is False


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


class APIConnectionError(Exception):
    """Name-matched transient SDK error that also carries a status code.

    Named and packaged to collide with `_TRANSIENT_SDK_EXC_NAMES` on purpose:
    the status check must win over the name heuristic.
    """

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


APIConnectionError.__module__ = "openai"


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        pytest.param(400, False, id="bad-request-beats-name-match"),
        pytest.param(401, False, id="auth-beats-name-match"),
        pytest.param(403, False, id="permission-beats-name-match"),
        pytest.param(503, True, id="server-error-agrees-with-name-match"),
        pytest.param(429, True, id="rate-limit-agrees-with-name-match"),
    ],
)
def test_status_code_outranks_the_sdk_name_heuristic(
    status: int, retryable: bool
) -> None:
    """A status code must decide before the class-name fallback runs.

    Without the ordering, a provider error class named `APIConnectionError`
    that carries a 401 is retried the full budget, burning the wait and
    delaying the auth failure the user has to act on. The existing fixtures
    never combine the two signals, so reordering the block passes the suite.
    """
    assert _is_retryable_model_error(APIConnectionError(status)) is retryable


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        pytest.param(499, False, id="below-server-range"),
        pytest.param(500, True, id="server-range-floor"),
        pytest.param(599, True, id="server-range-ceiling"),
        pytest.param(600, False, id="above-server-range"),
    ],
)
def test_server_error_range_boundaries(status: int, retryable: bool) -> None:
    """Pin both ends of the 5xx window so the ceiling cannot drift."""
    assert _is_retryable_model_error(_StatusError(status)) is retryable


def test_bool_status_code_is_not_read_as_http_one() -> None:
    """`status_code = True` must not classify as HTTP status 1."""
    exc = _StatusError(True)  # type: ignore[arg-type]
    assert model_retry._extract_status_code(exc) is None


class _AttributeShapedError(Exception):
    """Carry provider-specific status shapes on a real exception."""

    def __init__(self, **shape: object) -> None:
        super().__init__("provider error")
        for name, value in shape.items():
            setattr(self, name, value)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        pytest.param(
            _AttributeShapedError(http_status=503), 503, id="http-status-attribute"
        ),
        pytest.param(
            _AttributeShapedError(
                response={"ResponseMetadata": {"HTTPStatusCode": 500}}
            ),
            500,
            id="botocore-response-mapping",
        ),
    ],
)
def test_status_code_fallback_attributes(exc: Exception, expected: int) -> None:
    """The documented fallbacks are load-bearing for Bedrock-shaped errors."""
    assert model_retry._extract_status_code(exc) == expected


def test_unstamped_model_falls_back_to_the_default_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unstamped model must not silently disable auxiliary retries.

    `_install_summary_model_retries` replaces LangChain's unconditional
    three-attempt `with_retry`, so a zero fallback would drop compaction
    summarization to a single attempt -- a regression disguised as a feature.
    """
    attempts = 0

    def call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise httpx.ReadError(_DROPPED)
        return "ok"

    with caplog.at_level(logging.WARNING, logger="deepagents_code.model_retry"):
        assert retry_model_call(SimpleNamespace(), call) == "ok"

    assert attempts == 3
    assert "carries no dcode retry metadata" in caplog.text


def test_stamped_model_budget_still_wins_over_the_fallback() -> None:
    """Explicit metadata of zero still means zero."""
    model = SimpleNamespace()
    setattr(model, MODEL_RETRIES_ATTR, 0)
    attempts = 0

    def call() -> str:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadError(_DROPPED)

    with pytest.raises(httpx.ReadError):
        retry_model_call(model, call)
    assert attempts == 1


def test_max_delay_gives_up_rather_than_sleeping_past_a_deadline() -> None:
    """A retry that cannot fit the caller's budget must surface the real error.

    The auto-mode classifier runs its retries inside a hard `asyncio.timeout`.
    Honouring a 30s `Retry-After` there gets cancelled mid-sleep and resurfaces
    as a classifier timeout, blaming the wrong subsystem for a rate limit.
    """
    model = SimpleNamespace()
    setattr(model, MODEL_RETRIES_ATTR, 5)
    attempts = 0

    def call() -> str:
        nonlocal attempts
        attempts += 1
        raise _RateLimitError(_RETRY_AFTER_30)

    with pytest.raises(_RateLimitError):
        retry_model_call(model, call, max_total_delay=5.0)
    assert attempts == 1, "a 30s Retry-After must not be waited out under a 5s cap"


def test_max_delay_still_allows_a_retry_that_fits() -> None:
    """The cap must not disable retries that comfortably fit the budget."""
    model = SimpleNamespace()
    setattr(model, MODEL_RETRIES_ATTR, 5)
    attempts = 0

    def call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _RateLimitError(_RETRY_AFTER_1)
        return "ok"

    assert retry_model_call(model, call, max_total_delay=5.0) == "ok"
    assert attempts == 2


def test_streaming_flag_is_set_before_the_chunk_is_forwarded() -> None:
    """A writer that raises mid-chunk has still shown output to the user.

    Setting the flag afterwards leaves it `False` on a broken pipe, and since
    `ConnectionError` classifies retryable the retried call would wrongly
    report `output_may_have_started=False` for a response whose first chunk
    already rendered -- leaving the client to append the replay after text it
    cannot correlate.
    """
    from langchain_core.callbacks import BaseCallbackManager as _Manager
    from langgraph.pregel import _messages as _lg_messages

    observed: list[bool] = []
    tracker = model_retry._MessageStreamTracker()

    def explode(_chunk: object) -> None:
        # The tracker's view of itself at the moment the writer runs is the
        # whole contract: it must already believe output has escaped.
        observed.append(tracker.has_streamed)
        msg = "broken pipe"
        raise ConnectionError(msg)

    source = _lg_messages.StreamMessagesHandler(explode, subgraphs=False)
    manager = _Manager(handlers=[source], inheritable_handlers=[source])

    tracked_callbacks = tracker.callbacks_with_tracked_messages(manager)
    assert tracked_callbacks is not None
    tracked = tracked_callbacks.handlers[0]

    with pytest.raises(ConnectionError):
        cast("Any", tracked).stream(("chunk", {}))

    assert observed == [True], "flag must already be set when the writer runs"
    assert tracker.has_streamed is True


def test_sync_model_call_is_retried_after_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync loop must retry after streamed output, like the async one.

    `test_failed_attempt_is_retried_after_streaming` drives `astream`, so it
    only covers `awrap_model_call`. The sync path is a verbatim duplicate and
    an async-only fix leaves it still ending the turn on a mid-stream drop.
    """
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    events: list[dict[str, object]] = []
    calls = 0

    @contextmanager
    def _already_streamed(
        tracker: model_retry._MessageStreamTracker,
    ) -> Iterator[None]:
        tracker.has_streamed = True
        yield

    def _handler(_request: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadError(_DROPPED)
        return cast("ModelResponse", "recovered")

    middleware = CodeModelRetryMiddleware(max_retries=3)
    with patch.object(model_retry, "_track_message_streams", _already_streamed):
        result = middleware.wrap_model_call(_req(events), _handler)

    assert result == "recovered"
    assert calls == 2
    retry_events = [e for e in events if e["type"] == "model_retry"]
    assert retry_events[0]["output_may_have_started"] is True


async def test_hidden_model_call_is_retried_after_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested stream filtered by the clients may retry its failed model node."""
    monkeypatch.setattr(
        "deepagents_code.model_retry._retry_delay_seconds", lambda *_: 0
    )
    calls = 0

    @contextmanager
    def _already_streamed(
        tracker: model_retry._MessageStreamTracker,
    ) -> Iterator[None]:
        tracker.has_streamed = True
        yield

    async def _handler(_request: ModelRequest) -> ModelResponse:
        nonlocal calls
        await asyncio.sleep(0)
        calls += 1
        if calls == 1:
            raise httpx.ReadError(_DROPPED)
        return cast("ModelResponse", "verdict")

    middleware = CodeModelRetryMiddleware(
        max_retries=1,
        stream_output_is_visible=False,
    )
    with patch.object(model_retry, "_track_message_streams", _already_streamed):
        result = await middleware.awrap_model_call(
            cast("ModelRequest", SimpleNamespace(model=None)), _handler
        )

    assert result == "verdict"
    assert calls == 2


def test_sync_model_call_still_retries_before_streaming() -> None:
    """The guard must not disable retries on an attempt that emitted nothing."""
    calls = 0

    @contextmanager
    def _nothing_streamed(
        _tracker: model_retry._MessageStreamTracker,
    ) -> Iterator[None]:
        yield

    def _handler(_request: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadError(_DROPPED)
        return cast("ModelResponse", "ok")

    middleware = CodeModelRetryMiddleware(max_retries=3)
    with patch.object(model_retry, "_track_message_streams", _nothing_streamed):
        result = middleware.wrap_model_call(
            cast("ModelRequest", SimpleNamespace(model=None)), _handler
        )

    assert result == "ok"
    assert calls == 2


def test_predicate_ignores_stdlib_timeout_reached_only_through_context() -> None:
    """A permanent error is not retryable because a timeout preceded it.

    Python sets `__context__` on anything raised inside an `except` block, so
    treating the bare stdlib fallback as a chain signal would retry a genuine
    configuration fault five times whenever a timeout happened to precede it.
    """
    exc = ValueError("permanent")
    exc.__context__ = TimeoutError("transient")

    assert _is_retryable_model_error(exc) is False


def test_predicate_still_retries_a_stdlib_timeout_raised_directly() -> None:
    """The narrowing must not stop the raised exception from classifying."""
    assert _is_retryable_model_error(TimeoutError("transient")) is True


def test_predicate_lets_a_definite_verdict_decide_its_own_branch() -> None:
    """A non-retryable taxonomy verdict outranks whatever it wraps.

    A provider that raises an authentication failure while handling a dropped
    connection must not be retried: the credentials will not become valid, and
    the wrapped transport fault is incidental. Descending past a definite
    verdict would turn every such failure into a full budget of doomed calls.
    """
    exc = ModelAuthenticationError("bad key")
    exc.__context__ = _READ_ERROR

    assert _is_retryable_model_error(exc) is False


def test_predicate_finds_a_transport_fault_beside_a_permanent_member() -> None:
    """A definite verdict decides its own branch only, not its siblings."""
    exc = ExceptionGroup(
        "request failed", [ModelAuthenticationError("bad key"), _READ_ERROR]
    )

    assert _is_retryable_model_error(exc) is True


def test_deadline_guard_stays_quiet_for_a_non_retryable_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A permanent error must not be reported as a deadline give-up.

    The guard used to run before the eligibility check, so an authentication
    failure raised under a tight cap was logged as a retry that would have
    waited past the caller deadline -- pointing the reader at the classifier
    budget instead of the invalid credentials.
    """
    model = SimpleNamespace()
    setattr(model, MODEL_RETRIES_ATTR, 5)
    permanent = ModelAuthenticationError("bad key")

    def call() -> str:
        raise permanent

    with (
        caplog.at_level(logging.DEBUG, logger=model_retry.__name__),
        pytest.raises(ModelAuthenticationError),
    ):
        retry_model_call(model, call, max_total_delay=0.0)

    assert not [r for r in caplog.records if "past the caller deadline" in r.message]
    assert [r for r in caplog.records if "non-transient" in r.message]


def test_exhausted_budget_is_logged_even_under_a_deadline_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Budget exhaustion must stay visible when a guard is also installed."""
    model = SimpleNamespace()
    setattr(model, MODEL_RETRIES_ATTR, 0)

    def call() -> str:
        raise _RateLimitError(_RETRY_AFTER_30)

    with (
        caplog.at_level(logging.DEBUG, logger=model_retry.__name__),
        pytest.raises(_RateLimitError),
    ):
        retry_model_call(model, call, max_total_delay=0.0)

    assert [r for r in caplog.records if "retries are disabled" in r.message]


def test_retry_sleep_budget_is_cumulative_not_per_delay() -> None:
    """Several waits that each fit the cap must not overrun it together.

    A per-delay ceiling bounds nothing: the auto-mode classifier's five
    retries each cleared a quarter of its 20s deadline and still spent the
    whole budget sleeping, so the rate limit resurfaced as a classifier
    timeout.
    """
    model = SimpleNamespace()
    setattr(model, MODEL_RETRIES_ATTR, 5)
    attempts = 0

    def call() -> str:
        nonlocal attempts
        attempts += 1
        raise _RateLimitError(_RETRY_AFTER_1)

    with patch.object(model_retry.time, "sleep"), pytest.raises(_RateLimitError):
        retry_model_call(model, call, max_total_delay=2.5)

    assert attempts == 3, "a 1s Retry-After must run out a 2.5s total budget"


def test_middleware_rejects_a_bool_budget() -> None:
    """`True` must not pass as a retry budget of one.

    `True >= 0` holds and `range(True + 1)` runs a single retry, so an
    unchecked bool silently becomes a budget nobody asked for.
    """
    with pytest.raises(TypeError, match="max_retries"):
        CodeModelRetryMiddleware(max_retries=cast("int", True))


def test_middleware_rejects_non_bool_stream_visibility() -> None:
    with pytest.raises(TypeError, match="stream_output_is_visible"):
        CodeModelRetryMiddleware(stream_output_is_visible=cast("bool", 1))
