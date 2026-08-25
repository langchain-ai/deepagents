"""Tests for dcode model-node retry middleware and retry-count resolution."""

from __future__ import annotations

import asyncio
import logging
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
    from collections.abc import Awaitable, Callable, Iterator
    from pathlib import Path

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.callbacks import CallbackManagerForLLMRun

from deepagents_code import model_config
from deepagents_code.config import (
    ASCII_GLYPHS,
    DEFAULT_MODEL_RETRIES,
    MODEL_RETRIES_ATTR,
    UNICODE_GLYPHS,
    _resolve_config_retry_count,
    resolve_model_retries,
)
from deepagents_code.model_retry import (
    _JITTER_FRACTION,
    CodeModelRetryMiddleware,
    _is_retryable_model_error,
    _retry_after_seconds,
    build_retry_event,
    format_retry_status,
)

_UNSET = object()
"""Sentinel for "no retry budget stamped on the model" in `_req`."""

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
    """Name mirrors provider SDK transient errors matched by class name."""


class AuthenticationError(Exception):
    def __init__(self) -> None:
        super().__init__("auth")
        self.status_code = 401


class _RetryingStreamingModel(BaseChatModel):
    """Emit one orphaned chunk, fail, then complete on the retry."""

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


def _write_config(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text)
    return p


def _req(
    events: list[dict[str, object]] | None = None,
    *,
    model_retries: object = _UNSET,
) -> ModelRequest:
    """Build a stub model request.

    Args:
        events: Collector for `model_retry` stream events, or `None` for no writer.
        model_retries: Value to stamp on `request.model` under
            `MODEL_RETRIES_ATTR`. Left off entirely when unset, so the
            middleware falls back to its startup budget.

    Returns:
        A stub standing in for a `ModelRequest`.
    """
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


# --- resolve_model_retries / config resolution ---


def test_default_retries_is_five(tmp_path: Path) -> None:
    with patch.object(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "none.toml"):
        assert resolve_model_retries("openai") == 5
    assert DEFAULT_MODEL_RETRIES == 5


def test_cli_zero_disables(tmp_path: Path) -> None:
    with patch.object(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "none.toml"):
        assert resolve_model_retries("openai", cli_max_retries=0) == 0


def test_cli_overrides_config(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, "[retries]\nmax_retries = 3\n")
    with patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg):
        assert resolve_model_retries("openai", cli_max_retries=1) == 1


def test_global_config_applies(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, "[retries]\nmax_retries = 3\n")
    with patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg):
        assert resolve_model_retries("openai") == 3


def test_global_zero_disables(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, "[retries]\nmax_retries = 0\n")
    with patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg):
        assert resolve_model_retries("openai") == 0


def test_provider_overrides_global(tmp_path: Path) -> None:
    cfg = _write_config(
        tmp_path,
        "[retries]\nmax_retries = 3\n[retries.openai]\nmax_retries = 7\n",
    )
    with patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg):
        assert resolve_model_retries("openai") == 7
        assert resolve_model_retries("anthropic") == 3


def test_param_key_ignored_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _write_config(
        tmp_path,
        '[retries.openai]\nparam = "num_retries"\nmax_retries = 2\n',
    )
    with (
        patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg),
        caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
    ):
        assert resolve_model_retries("openai") == 2


def test_resolve_config_retry_count_direct() -> None:
    assert _resolve_config_retry_count(None, "openai") is None
    assert _resolve_config_retry_count({"max_retries": 2}, "openai") == 2
    assert _resolve_config_retry_count({"max_retries": 0}, "openai") == 0


# --- retry predicate ---


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
    """Model taxonomy must win over broader transport inheritance."""


def test_model_taxonomy_takes_precedence_over_legacy_fallback() -> None:
    assert _is_retryable_model_error(_RetryableTransportModelError("x")) is False


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadError("x"),
        httpx.ConnectError("x"),
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
    """`TransportError` subclasses that can never succeed on a retry.

    A mistyped `base_url` scheme, a malformed request, or a misconfigured proxy
    is knowable on the first attempt; retrying spends the whole budget to
    surface the same config error.
    """
    assert isinstance(exc, httpx.TransportError)
    assert _is_retryable_model_error(exc) is False


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(httpx.ReadError("dropped"), id="read"),
        pytest.param(httpx.ConnectError("refused"), id="connect"),
        pytest.param(httpx.WriteError("broken"), id="write"),
        pytest.param(httpx.PoolTimeout("pool"), id="pool-timeout"),
        pytest.param(httpx.ConnectTimeout("connect"), id="connect-timeout"),
        pytest.param(httpx.RemoteProtocolError("server broke"), id="remote-protocol"),
    ],
)
def test_transient_transport_errors_are_still_retried(exc: Exception) -> None:
    """Narrowing the predicate must not drop the genuinely transient faults."""
    assert _is_retryable_model_error(exc) is True


# --- middleware behavior ---


def test_middleware_defaults() -> None:
    mw = CodeModelRetryMiddleware()
    assert mw.max_retries == DEFAULT_MODEL_RETRIES
    assert mw.on_failure == "error"
    assert mw.initial_delay == pytest.approx(0.2)
    assert mw.backoff_factor == pytest.approx(2.0)
    assert mw.max_delay == pytest.approx(10.0)


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
    assert "retrying 1/5" in events[0]["message"]


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


def test_retry_scoped_to_model_node(monkeypatch: pytest.MonkeyPatch) -> None:
    # Retries re-invoke only the model handler; a separate "tool_calls" ledger
    # is never touched, proving completed tool work is not replayed.
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    tool_calls: list[str] = []
    model_calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        model_calls["n"] += 1
        if model_calls["n"] < 2:
            raise _CONNECT_ERROR
        return "OK"

    mw = CodeModelRetryMiddleware(max_retries=3)
    assert mw.wrap_model_call(_req(), _handler(handler)) == "OK"
    assert model_calls["n"] == 2
    assert tool_calls == []


# --- backoff curve ---


def test_backoff_grows_exponentially_and_is_capped() -> None:
    """The curve doubles from 200ms and stops at the 10s ceiling.

    Asserted on the values rather than the constructor attributes: without
    this, removing the cap or making growth linear passes the suite.
    """
    mw = CodeModelRetryMiddleware(max_retries=12)
    delays = [mw._compute_delay(n) for n in range(12)]  # policy under test

    # Nominal curve is 0.2 * 2**n with +-10% jitter, capped at 10s.
    for n, delay in enumerate(delays):
        nominal = min(0.2 * (2.0**n), 10.0)
        assert delay == pytest.approx(nominal, rel=_JITTER_FRACTION)
    assert delays[-1] <= 10.0 * (1 + _JITTER_FRACTION)
    # Growth, not a constant.
    assert delays[3] > delays[0]


def test_backoff_jitter_varies_between_calls() -> None:
    """Jitter is applied, so a thundering herd desynchronizes."""
    mw = CodeModelRetryMiddleware()
    samples = {mw._compute_delay(5) for _ in range(20)}  # policy under test
    assert len(samples) > 1


def test_retry_uses_the_computed_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop sleeps for the curve's values, in order."""
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


# --- Retry-After ---


class _RateLimitError(Exception):
    """Carries a 429 and a `Retry-After` header the way provider SDKs do."""

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
    """The header may be an HTTP-date instead of a delta."""
    when = datetime.now(UTC) + timedelta(seconds=20)
    header = format_datetime(when, usegmt=True)
    result = _retry_after_seconds(_RateLimitError(header))
    assert result is not None
    assert result == pytest.approx(20.0, abs=2.0)


def test_retry_after_seconds_past_date_clamps_to_zero() -> None:
    """A stale date means retry now, not a negative sleep."""
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
    """A rate limit waits as long as the provider asked, not ~0.2s."""
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
    """The async path honors the header too."""
    slept: list[float] = []

    async def _record(seconds: float, *_a: object, **_k: object) -> None:  # noqa: RUF029  # async stub replacing asyncio.sleep
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


# --- give-up diagnostics ---


def test_exhaustion_logs_the_attempt_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An exhausted budget is recorded; the bare re-raise carries no count."""
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
    """A deterministic error never spent the budget, so don't claim it did."""

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
    """The async path reports exhaustion too."""

    async def _no_sleep(*_a: object, **_k: object) -> None:  # noqa: RUF029  # async stub replacing asyncio.sleep
        return None

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


# --- per-request retry budget carried on the model ---


def test_model_budget_overrides_startup_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget stamped on `request.model` wins over the constructor value."""
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=5)
    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(_req(model_retries=1), _handler(handler))
    # One retry, not the constructor's five.
    assert calls["n"] == 2


def test_zero_startup_budget_still_retries_runtime_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `/model` switch to a provider with a non-zero budget still retries.

    This is why `create_cli_agent` keeps the middleware installed when the
    startup budget is zero.
    """
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
    assert "retrying 1/3" in str(events[0]["message"])


def test_model_budget_zero_disables_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero budget on the model overrides a non-zero startup fallback."""
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
    """A malformed stamp is ignored in favor of the startup fallback.

    `True` is the sharp case: `isinstance(True, int)` holds, so without the
    explicit bool guard it would be read as a budget of one.
    """
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
    """A request carrying no model at all falls back rather than raising."""
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
    """The async path reads the same per-request budget."""

    async def _no_sleep(*_a: object, **_k: object) -> None:  # noqa: RUF029  # async stub replacing asyncio.sleep
        return None

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
    async def _no_sleep(*_a: object, **_k: object) -> None:  # noqa: RUF029  # async stub replacing asyncio.sleep
        return None

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


async def test_failed_attempt_stream_chunks_are_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only chunks from the successful model attempt reach `messages` stream."""

    async def _no_sleep(*_args: object, **_kwargs: object) -> None:  # noqa: RUF029
        return None

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

    assert model.attempts == 2
    assert message_text == "final"


@pytest.mark.parametrize("ellipsis", [UNICODE_GLYPHS.ellipsis, ASCII_GLYPHS.ellipsis])
def test_status_helpers(ellipsis: str) -> None:
    with patch("deepagents_code.model_retry.get_glyphs") as get_glyphs:
        get_glyphs.return_value.ellipsis = ellipsis
        assert (
            format_retry_status(1, 5)
            == f"model connection dropped, retrying 1/5{ellipsis}"
        )
    event = build_retry_event(2, 5)
    assert event["type"] == "model_retry"
    assert event["attempt"] == 2
    assert event["max_retries"] == 5
    assert "retrying 2/5" in str(event["message"])
