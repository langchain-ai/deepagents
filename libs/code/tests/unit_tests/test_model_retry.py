"""Tests for dcode model-node retry middleware and retry-count resolution."""

from __future__ import annotations

import asyncio
import logging
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
    CodeModelRetryMiddleware,
    _is_retryable_model_error,
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
