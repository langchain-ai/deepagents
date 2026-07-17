"""Tests for dcode model-node retry middleware and retry-count resolution."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code import model_config
from deepagents_code._cli_context import CLIContext
from deepagents_code.config import (
    CLI_MAX_RETRIES_KEY,
    DEFAULT_MODEL_RETRIES,
    MODEL_RETRIES_ATTR,
    _resolve_config_retry_count,
    create_model,
    reset_glyphs_cache,
    resolve_model_retries,
)
from deepagents_code.configurable_model import ConfigurableModelMiddleware
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


class EndpointConnectionError(Exception):
    """Name mirrors botocore's endpoint connection error."""


class ConnectionClosedError(Exception):
    """Name mirrors botocore's dropped-connection error."""


class ReadTimeoutError(Exception):
    """Name mirrors botocore's read-timeout error."""


class _BedrockClientError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"bedrock status {status_code}")
        self.response = {"ResponseMetadata": {"HTTPStatusCode": status_code}}


class AuthenticationError(Exception):
    def __init__(self) -> None:
        super().__init__("auth")
        self.status_code = 401


def _write_config(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text)
    return p


def _req(
    events: list[dict] | None = None,
    *,
    model_retries: int | None = None,
) -> ModelRequest:
    writer = (lambda event: events.append(event)) if events is not None else None
    runtime = SimpleNamespace(stream_writer=writer)
    model = MagicMock(spec=BaseChatModel)
    if model_retries is not None:
        setattr(model, MODEL_RETRIES_ATTR, model_retries)
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="test")],
        tools=[],
        runtime=cast("Any", runtime),
    )


def _response() -> ModelResponse[Any]:
    return ModelResponse(result=[AIMessage(content="OK")])


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


def test_param_key_does_not_change_middleware_count(tmp_path: Path) -> None:
    cfg = _write_config(
        tmp_path,
        '[retries.openai]\nparam = "num_retries"\nmax_retries = 2\n',
    )
    with patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg):
        assert resolve_model_retries("openai") == 2


@pytest.mark.parametrize("provider", ["bedrock", "custom"])
def test_valid_retry_provider_tables_do_not_warn(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    provider: str,
) -> None:
    models = (
        ""
        if provider == "bedrock"
        else ('[models.providers.custom]\nclass_path = "example.models:ChatCustom"\n')
    )
    cfg = _write_config(
        tmp_path,
        f"{models}[retries.{provider}]\nmax_retries = 4\n",
    )
    with (
        patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg),
        caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
    ):
        assert resolve_model_retries(provider) == 4
    assert "not a known provider" not in caplog.text


@pytest.mark.parametrize("cli_retries", [0, 3])
def test_create_model_disables_provider_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_retries: int,
) -> None:
    cfg = _write_config(
        tmp_path,
        ('[retries.anthropic]\nparam = "num_retries"\nmax_retries = 7\n'),
    )
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    monkeypatch.setattr(model_config, "has_provider_credentials", lambda _: True)
    model_config.clear_caches()
    with (
        patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg),
        patch("langchain.chat_models.init_chat_model", return_value=model) as init,
    ):
        result = create_model(
            "anthropic:claude-sonnet-4-5",
            extra_kwargs={
                "max_retries": 99,
                CLI_MAX_RETRIES_KEY: cli_retries,
            },
        )
    model_config.clear_caches()

    assert init.call_args.kwargs["max_retries"] == 0
    assert "num_retries" not in init.call_args.kwargs
    assert result.model_retries == cli_retries
    assert getattr(result.model, MODEL_RETRIES_ATTR) == cli_retries


def test_custom_retry_param_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _write_config(
        tmp_path,
        """
[models.providers.custom]
class_path = "example.models:ChatCustom"

[retries.custom]
param = "num_retries"
max_retries = 6
""",
    )
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    monkeypatch.setattr(model_config, "has_provider_credentials", lambda _: True)
    model_config.clear_caches()
    with (
        patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg),
        patch(
            "deepagents_code.config._create_model_from_class",
            return_value=model,
        ) as create,
    ):
        result = create_model("custom:test")
    model_config.clear_caches()

    assert create.call_args.args[3]["num_retries"] == 0
    assert result.model_retries == 6


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
        _BedrockClientError(503),
        APIConnectionError("x"),
        EndpointConnectionError("x"),
        ConnectionClosedError("x"),
        ReadTimeoutError("x"),
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
        _BedrockClientError(400),
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
    response = _response()

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _READ_ERROR
        return response

    mw = CodeModelRetryMiddleware(max_retries=5)
    assert mw.wrap_model_call(_req(events), handler) is response
    assert calls["n"] == 3
    assert [e["type"] for e in events] == ["model_retry", "model_retry"]
    assert "retrying 1/5" in events[0]["message"]


def test_exhaustion_reraises_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=2)
    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(_req(), handler)
    assert calls["n"] == 3


def test_non_retryable_raises_immediately() -> None:
    calls = {"n": 0}

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
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

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        raise _READ_ERROR

    with pytest.raises(httpx.ReadError):
        mw.wrap_model_call(_req(), handler)
    assert calls["n"] == 1


def test_request_model_overrides_startup_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime-selected model carries its own retry budget per request."""
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}
    events: list[dict] = []

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        raise _READ_ERROR

    middleware = CodeModelRetryMiddleware(max_retries=0)
    with pytest.raises(httpx.ReadError):
        middleware.wrap_model_call(_req(events, model_retries=3), handler)

    assert calls["n"] == 4
    assert [event["max_retries"] for event in events] == [3, 3, 3]


def test_runtime_model_switch_uses_selected_models_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configurable-model layer passes the switched model's budget on."""
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    original = MagicMock(spec=BaseChatModel)
    switched = MagicMock(spec=BaseChatModel)
    setattr(original, MODEL_RETRIES_ATTR, 1)
    setattr(switched, MODEL_RETRIES_ATTR, 3)
    runtime = SimpleNamespace(
        context=CLIContext(
            model="openai:gpt-5.5",
            model_params={CLI_MAX_RETRIES_KEY: 3},
        ),
        stream_writer=lambda _event: None,
    )
    request = ModelRequest(
        model=original,
        messages=[HumanMessage(content="test")],
        tools=[],
        runtime=cast("Any", runtime),
    )
    model_result = SimpleNamespace(
        model=switched,
        model_name="gpt-5.5",
        provider="openai",
        context_limit=None,
        unsupported_modalities=frozenset(),
        model_retries=3,
    )
    calls = {"n": 0}
    response = _response()

    def provider_handler(selected: ModelRequest) -> ModelResponse[Any]:
        assert selected.model is switched
        assert CLI_MAX_RETRIES_KEY not in selected.model_settings
        calls["n"] += 1
        if calls["n"] < 4:
            raise _READ_ERROR
        return response

    retry = CodeModelRetryMiddleware(max_retries=1)
    configurable = ConfigurableModelMiddleware(persist_model_state=False)

    def retry_handler(selected: ModelRequest) -> ModelResponse[Any]:
        result = retry.wrap_model_call(selected, provider_handler)
        assert isinstance(result, ModelResponse)
        return result

    with (
        patch(
            "deepagents_code.configurable_model.model_matches_spec",
            return_value=False,
        ),
        patch(
            "deepagents_code.config.create_model",
            return_value=model_result,
        ) as create,
    ):
        result = configurable.wrap_model_call(request, retry_handler)

    assert result is response
    assert calls["n"] == 4
    create.assert_called_once_with(
        "openai:gpt-5.5",
        extra_kwargs={CLI_MAX_RETRIES_KEY: 3},
    )


def test_retry_scoped_to_model_node(monkeypatch: pytest.MonkeyPatch) -> None:
    # Retries re-invoke only the model handler; a separate "tool_calls" ledger
    # is never touched, proving completed tool work is not replayed.
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    tool_calls: list[str] = []
    model_calls = {"n": 0}
    response = _response()

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        model_calls["n"] += 1
        if model_calls["n"] < 2:
            raise _CONNECT_ERROR
        return response

    mw = CodeModelRetryMiddleware(max_retries=3)
    assert mw.wrap_model_call(_req(), handler) is response
    assert model_calls["n"] == 2
    assert tool_calls == []


async def test_async_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_sleep(*_a: object, **_k: object) -> None:  # noqa: RUF029  # async stub replacing asyncio.sleep
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}
    response = _response()

    async def handler(  # noqa: RUF029  # awaited by middleware; no internal await needed
        _request: ModelRequest,
    ) -> ModelResponse[Any]:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _READ_ERROR
        return response

    mw = CodeModelRetryMiddleware(max_retries=3)
    assert await mw.awrap_model_call(_req(), handler) is response
    assert calls["n"] == 2


@pytest.mark.parametrize(
    ("mode", "suffix"),
    [("unicode", "\u2026"), ("ascii", "...")],
)
def test_status_helpers_respect_charset(
    monkeypatch: pytest.MonkeyPatch, mode: str, suffix: str
) -> None:
    monkeypatch.setenv("UI_CHARSET_MODE", mode)
    monkeypatch.setenv("DEEPAGENTS_CODE_UI_CHARSET_MODE", mode)
    reset_glyphs_cache()
    try:
        assert format_retry_status(1, 5) == (
            f"model connection dropped, retrying 1/5{suffix}"
        )
        event = build_retry_event(2, 5)
        assert event["type"] == "model_retry"
        assert event["attempt"] == 2
        assert event["max_retries"] == 5
        assert event["message"] == f"model connection dropped, retrying 2/5{suffix}"
    finally:
        reset_glyphs_cache()
