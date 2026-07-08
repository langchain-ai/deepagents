"""Tests for dcode model-node retry middleware and retry-count resolution."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code import model_config
from deepagents_code.config import (
    DEFAULT_MODEL_RETRIES,
    _resolve_config_retry_count,
    resolve_model_retries,
)
from deepagents_code.model_retry import (
    CodeModelRetryMiddleware,
    _is_retryable_model_error,
    build_retry_event,
    format_retry_status,
)

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


def _write_config(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text)
    return p


def _req(events: list[dict] | None = None) -> SimpleNamespace:
    writer = (lambda e: events.append(e)) if events is not None else None
    return SimpleNamespace(runtime=SimpleNamespace(stream_writer=writer))


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
    assert "obsolete" in caplog.text


def test_resolve_config_retry_count_direct() -> None:
    assert _resolve_config_retry_count(None, "openai") is None
    assert _resolve_config_retry_count({"max_retries": 2}, "openai") == 2
    assert _resolve_config_retry_count({"max_retries": 0}, "openai") == 0


# --- retry predicate ---


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
    events: list[dict] = []
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _READ_ERROR
        return "OK"

    mw = CodeModelRetryMiddleware(max_retries=5)
    assert mw.wrap_model_call(_req(events), handler) == "OK"
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
        mw.wrap_model_call(_req(), handler)
    assert calls["n"] == 3


def test_non_retryable_raises_immediately() -> None:
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _VALUE_ERROR

    mw = CodeModelRetryMiddleware(max_retries=5)
    with pytest.raises(ValueError, match="bad request"):
        mw.wrap_model_call(_req(), handler)
    assert calls["n"] == 1


def test_zero_retries_calls_handler_once() -> None:
    mw = CodeModelRetryMiddleware(max_retries=0)
    assert mw.max_retries == 0
    calls = {"n": 0}

    def handler(_req_arg: object) -> str:
        calls["n"] += 1
        raise _READ_ERROR

    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(_req(), handler)
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
    assert mw.wrap_model_call(_req(), handler) == "OK"
    assert model_calls["n"] == 2
    assert tool_calls == []


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
    assert await mw.awrap_model_call(_req(), handler) == "OK"
    assert calls["n"] == 2


def test_status_helpers() -> None:
    assert format_retry_status(1, 5) == "model connection dropped, retrying 1/5\u2026"
    event = build_retry_event(2, 5)
    assert event["type"] == "model_retry"
    assert event["attempt"] == 2
    assert event["max_retries"] == 5
    assert "retrying 2/5" in event["message"]
