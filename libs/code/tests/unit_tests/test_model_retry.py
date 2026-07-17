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
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code import model_config
from deepagents_code._cli_context import CLIContext
from deepagents_code.config import (
    CLI_MAX_RETRIES_KEY,
    DEFAULT_MODEL_RETRIES,
    MODEL_RETRIES_ATTR,
    MODEL_RETRY_OVERRIDE_ATTR,
    ModelResult,
    _provider_retry_disable_kwargs,
    _resolve_config_retry_count,
    create_model,
    reset_glyphs_cache,
    resolve_model_retries,
)
from deepagents_code.configurable_model import ConfigurableModelMiddleware
from deepagents_code.model_retry import (
    CodeModelRetryMiddleware,
    _describe_error,
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


class _CodeStatusError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"code {code}")
        self.code = code


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


class _QuotaError(Exception):
    """OpenAI-style permanent billing error riding a retryable 429 status."""

    def __init__(self) -> None:
        super().__init__("insufficient_quota")
        self.status_code = 429
        self.code = "insufficient_quota"


class _ThrottlingError(Exception):
    """botocore-style rate limit surfaced behind a fatal-looking HTTP 400."""

    def __init__(self) -> None:
        super().__init__("throttled")
        self.response = {
            "Error": {"Code": "ThrottlingException"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        }


class ResourceExhausted(Exception):  # noqa: N818  # mirrors Google's real class name
    """Google api_core transient error over gRPC (`.code` is a non-int enum)."""

    def __init__(self) -> None:
        super().__init__("resource exhausted")
        # `grpc.StatusCode` enum member: truthy, non-int, carries a `name`.
        self.code = SimpleNamespace(name="RESOURCE_EXHAUSTED")


class ServiceUnavailable(Exception):  # noqa: N818  # mirrors Google's real class name
    """Google api_core transient error over gRPC (`.code` is a non-int enum)."""

    def __init__(self) -> None:
        super().__init__("service unavailable")
        self.code = SimpleNamespace(name="UNAVAILABLE")


class _SubclassedConnectionError(APIConnectionError):
    """Own name is absent from the transient set; a base name matches via MRO."""


def _typed_error(module: str, name: str, message: str = "boom") -> Exception:
    error_type = type(name, (Exception,), {"__module__": module})
    return error_type(message)


def _write_config(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text)
    return p


def _req(
    events: list[dict] | None = None,
    *,
    model_retries: int | None = None,
    runtime_retries: int | None = None,
) -> ModelRequest:
    writer = (lambda event: events.append(event)) if events is not None else None
    context = (
        CLIContext(model_params={CLI_MAX_RETRIES_KEY: runtime_retries})
        if runtime_retries is not None
        else None
    )
    runtime = SimpleNamespace(stream_writer=writer, context=context)
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
    assert getattr(result.model, MODEL_RETRY_OVERRIDE_ATTR) == cli_retries


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


@pytest.mark.parametrize("bad", [-1, 1.5, True, False, "3"])
def test_resolve_config_retry_count_drops_invalid_global(
    caplog: pytest.LogCaptureFixture, bad: object
) -> None:
    """A malformed global `max_retries` is dropped with a warning."""
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
        assert _resolve_config_retry_count({"max_retries": bad}, "openai") is None
    assert "expected int >= 0" in caplog.text


def test_resolve_config_retry_count_provider_overrides_global() -> None:
    """A provider `max_retries` overrides the global value."""
    section = {"max_retries": 2, "openai": {"max_retries": 7}}
    assert _resolve_config_retry_count(section, "openai") == 7
    assert _resolve_config_retry_count(section, "anthropic") == 2


def test_resolve_config_retry_count_warns_unknown_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Scalar junk keys are ignored with a warning; `param` is not."""
    section = {"bogus": 9, "openai": {"max_retries": 3, "param": "num_retries"}}
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
        assert _resolve_config_retry_count(section, "openai") == 3
    assert "Ignoring [retries].bogus" in caplog.text
    assert "param" not in caplog.text


def test_read_config_toml_retries_warns_on_mistyped_provider(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A mistyped provider sub-table surfaces a 'not a known provider' warning."""
    from deepagents_code.config import _read_config_toml_retries

    cfg = _write_config(tmp_path, "[retries.fireorks]\nmax_retries = 4\n")
    with (
        patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg),
        caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
    ):
        _read_config_toml_retries()
    assert "not a known provider" in caplog.text


def test_read_config_toml_retries_warns_on_malformed_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Malformed TOML yields None and a 'Could not read' warning."""
    from deepagents_code.config import _read_config_toml_retries

    cfg = _write_config(tmp_path, "[retries]\nthis is not = = valid toml\n")
    with (
        patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg),
        caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
    ):
        assert _read_config_toml_retries() is None
    assert "Could not read retries config" in caplog.text


def test_read_config_toml_retries_ignores_scalar_section(tmp_path: Path) -> None:
    """A scalar `retries = 5` (not a table) is treated as absent."""
    from deepagents_code.config import _read_config_toml_retries

    cfg = _write_config(tmp_path, "retries = 5\n")
    with patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg):
        assert _read_config_toml_retries() is None


def test_disable_kwargs_registered_provider() -> None:
    """A registered provider disables its own SDK retries."""
    assert _provider_retry_disable_kwargs(None, "anthropic", {}) == {"max_retries": 0}


def test_disable_kwargs_configured_param() -> None:
    """A `[retries.<provider>].param` names the kwarg to zero out."""
    section = {"custom": {"param": "num_retries"}}
    assert _provider_retry_disable_kwargs(section, "custom", {}) == {"num_retries": 0}


def test_disable_kwargs_falls_back_to_max_retries_in_kwargs() -> None:
    """An unregistered provider that already passes `max_retries` is disabled."""
    assert _provider_retry_disable_kwargs(None, "custom", {"max_retries": 4}) == {
        "max_retries": 0
    }


def test_disable_kwargs_unidentifiable_provider_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unidentifiable provider yields no disable kwarg and a visible warning.

    This is the branch where the provider's own SDK retry loop stays active and
    can multiply the middleware's budget, so the warning must be surfaced.
    """
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
        assert _provider_retry_disable_kwargs(None, "mystery", {}) == {}
    assert "SDK retries stay active" in caplog.text


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
        _CodeStatusError(429),
        _CodeStatusError(503),
        _ResponseStatusError(502),
        _BedrockClientError(503),
        _ThrottlingError(),
        APIConnectionError("x"),
        EndpointConnectionError("x"),
        ConnectionClosedError("x"),
        ReadTimeoutError("x"),
        TimeoutError("x"),
        ConnectionError("x"),
        _typed_error("httpcore._exceptions", "ReadError"),
        _typed_error(
            "aiohttp.http_exceptions",
            "TransferEncodingError",
            "Not enough data to satisfy transfer length header",
        ),
        # Google gRPC transient errors whose `.code` is a non-int enum, so they
        # can only be classified by name across the MRO.
        ResourceExhausted(),
        ServiceUnavailable(),
        # A subclass whose own name is not in the transient set matches via a
        # base class name in its MRO.
        _SubclassedConnectionError("x"),
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
        _CodeStatusError(400),
        AuthenticationError(),
        _BedrockClientError(400),
        _QuotaError(),
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


def test_wrapped_transient_error_retries_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            wrapped = RuntimeError("model graph failed")
            wrapped.__cause__ = _READ_ERROR
            raise wrapped
        return _response()

    result = CodeModelRetryMiddleware(max_retries=1).wrap_model_call(_req(), handler)

    assert result.result[0].text == "OK"
    assert calls["n"] == 2


def test_exception_group_transient_error_retries_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            group = ExceptionGroup("model graph failed", [ValueError("x"), _READ_ERROR])
            raise group
        return _response()

    CodeModelRetryMiddleware(max_retries=1).wrap_model_call(_req(), handler)

    assert calls["n"] == 2


def test_does_not_retry_after_streamed_output() -> None:
    """A partial streamed attempt must never be replayed into the client."""
    events: list[dict] = []
    calls = {"n": 0}
    model = FakeListChatModel(responses=["first", "second"])
    setattr(model, MODEL_RETRIES_ATTR, 3)
    request = _req(events).override(model=model)

    def handler(selected: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        assert selected.model is model
        stream = selected.model.stream(selected.messages)
        assert next(stream).text == "f"
        raise _READ_ERROR

    middleware = CodeModelRetryMiddleware(max_retries=3)
    with pytest.raises(httpx.ReadError):
        middleware.wrap_model_call(request, handler)

    assert calls["n"] == 1
    assert model.i == 1
    assert events == []


async def test_async_does_not_retry_after_streamed_output() -> None:
    """The async path tracks output without copying the stateful model."""
    events: list[dict] = []
    calls = {"n": 0}
    model = FakeListChatModel(responses=["first", "second"])
    setattr(model, MODEL_RETRIES_ATTR, 3)
    request = _req(events).override(model=model)

    async def handler(selected: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        assert selected.model is model
        stream = selected.model.astream(selected.messages)
        assert (await anext(stream)).text == "f"
        raise _READ_ERROR

    middleware = CodeModelRetryMiddleware(max_retries=3)
    with pytest.raises(httpx.ReadError):
        await middleware.awrap_model_call(request, handler)

    assert calls["n"] == 1
    assert model.i == 1
    assert events == []


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


def test_runtime_context_overrides_same_models_attached_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        raise _READ_ERROR

    middleware = CodeModelRetryMiddleware(max_retries=5)
    with pytest.raises(httpx.ReadError):
        middleware.wrap_model_call(
            _req(model_retries=4, runtime_retries=1),
            handler,
        )

    assert calls["n"] == 2


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
        assert format_retry_status(1, 5) == (f"model call failed, retrying 1/5{suffix}")
        event = build_retry_event(2, 5)
        assert event["type"] == "model_retry"
        assert event["attempt"] == 2
        assert event["max_retries"] == 5
        assert event["message"] == f"model call failed, retrying 2/5{suffix}"
    finally:
        reset_glyphs_cache()


# --- error classification helpers ---


def test_describe_error_includes_status_and_code() -> None:
    desc = _describe_error(_QuotaError())
    assert "_QuotaError" in desc
    assert "status=429" in desc
    assert "code=insufficient_quota" in desc


def test_meta_present_in_retry_param_map() -> None:
    # `meta` is a wired provider; it must stay in the disable-list so its SDK
    # retry loop cannot multiply the middleware budget.
    assert model_config.RETRY_PARAM_BY_PROVIDER.get("meta") == "max_retries"


def test_model_result_rejects_negative_retries() -> None:
    with pytest.raises(ValueError, match="model_retries must be >= 0"):
        ModelResult(
            model=MagicMock(spec=BaseChatModel),
            model_name="m",
            provider="openai",
            model_retries=-1,
        )


# --- backoff delay ---


def test_compute_delay_grows_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin jitter to zero to assert the exponential curve and the cap exactly.
    monkeypatch.setattr(
        "deepagents_code.model_retry.random.uniform", lambda _a, _b: 0.0
    )
    mw = CodeModelRetryMiddleware(max_retries=10)
    assert mw._compute_delay(0) == pytest.approx(0.2)
    assert mw._compute_delay(1) == pytest.approx(0.4)
    assert mw._compute_delay(2) == pytest.approx(0.8)
    # Exponential growth is bounded by the max-delay cap.
    assert mw._compute_delay(20) == pytest.approx(10.0)


def test_compute_delay_jitter_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    mw = CodeModelRetryMiddleware(max_retries=5)
    base = 0.4  # initial_delay * factor**1
    # random.uniform(-amount, +amount); return each extreme deterministically.
    monkeypatch.setattr(
        "deepagents_code.model_retry.random.uniform", lambda low, _high: low
    )
    low = mw._compute_delay(1)
    monkeypatch.setattr(
        "deepagents_code.model_retry.random.uniform", lambda _low, high: high
    )
    high = mw._compute_delay(1)
    assert low == pytest.approx(base * 0.9)
    assert high == pytest.approx(base * 1.1)
    assert low >= 0.0


# --- defensive attribute guards ---


def test_bool_status_code_is_not_treated_as_int() -> None:
    err = _StatusError(500)
    err.status_code = True  # type: ignore[assignment]  # bool is not a status
    assert _is_retryable_model_error(err) is False


def test_bool_model_retries_falls_back_to_startup() -> None:
    mw = CodeModelRetryMiddleware(max_retries=2)
    model = MagicMock(spec=BaseChatModel)
    setattr(model, MODEL_RETRIES_ATTR, True)
    assert mw._model_max_retries(model) == 2


def test_writer_failure_does_not_break_retry_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}
    response = _response()

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _READ_ERROR
        return response

    def bad_writer(_event: dict[str, object]) -> None:
        msg = "stream closed"
        raise RuntimeError(msg)

    runtime = SimpleNamespace(stream_writer=bad_writer)
    request = ModelRequest(
        model=MagicMock(spec=BaseChatModel),
        messages=[HumanMessage(content="test")],
        tools=[],
        runtime=cast("Any", runtime),
    )
    mw = CodeModelRetryMiddleware(max_retries=5)
    assert mw.wrap_model_call(request, handler) is response
    assert calls["n"] == 3


async def test_async_exhaustion_reraises_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _no_sleep(*_a: object, **_k: object) -> None:  # noqa: RUF029  # async stub replacing asyncio.sleep
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    async def handler(  # noqa: RUF029  # awaited by middleware; no internal await needed
        _request: ModelRequest,
    ) -> ModelResponse[Any]:
        calls["n"] += 1
        raise _READ_ERROR

    mw = CodeModelRetryMiddleware(max_retries=2)
    with pytest.raises(httpx.ReadError):
        await mw.awrap_model_call(_req(), handler)
    assert calls["n"] == 3
