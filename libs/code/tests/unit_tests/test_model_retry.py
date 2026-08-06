"""Tests for dcode model-node retry middleware and retry-count resolution."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
    from langchain_core.runnables import RunnableConfig

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
    _runtime_model_retry_override,
    _should_retry_after_failure,
    _StreamOutputTracker,
    build_retry_event,
    format_retry_status,
)


def _read_error() -> httpx.ReadError:
    """Return a fresh transient transport error.

    Deliberately a factory, not a shared instance: the retry driver annotates an
    exception whose budget it exhausted (so an enclosing driver does not restart
    it), and a module-level singleton would carry that mark into later tests.
    """
    return httpx.ReadError("connection dropped")


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


def test_late_resolved_provider_still_gets_its_retry_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider only `init_chat_model` can name must still pick up its config.

    For a bare model name that `detect_provider` cannot place, the pre-construction
    provider is `""`, so the retry lookup finds no `[retries.<provider>]` table and
    falls back to the global count. The provider becomes known once the model
    exists, so the budget is re-resolved rather than silently discarding the
    user's per-provider setting.
    """
    cfg = _write_config(
        tmp_path,
        "[retries]\nmax_retries = 3\n[retries.openai]\nmax_retries = 10\n",
    )
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    # What `init_chat_model` records after inferring the provider itself.
    model._model_provider = "openai"
    monkeypatch.setattr(model_config, "has_provider_credentials", lambda _: True)
    model_config.clear_caches()
    with (
        patch.object(model_config, "DEFAULT_CONFIG_PATH", cfg),
        patch("deepagents_code.config.detect_provider", return_value=None),
        patch("langchain.chat_models.init_chat_model", return_value=model),
    ):
        result = create_model("some-unrecognized-model")
    model_config.clear_caches()

    assert result.provider == "openai"
    # 10 from `[retries.openai]`, not the global 3.
    assert result.model_retries == 10
    assert getattr(model, MODEL_RETRIES_ATTR) == 10


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

    # Built-in registry wins over mistyped config param; user max_retries is
    # mapped onto dcode's budget (CLI carrier wins here) and SDK loop is zeroed.
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


def test_built_in_retry_param_wins_over_config_for_known_provider() -> None:
    """Known providers ignore mistyped `[retries.<provider>].param` values."""
    assert _provider_retry_disable_kwargs(
        {"anthropic": {"param": "num_retries"}},
        "anthropic",
        {"max_retries": 12},
    ) == {"max_retries": 0}


def test_custom_provider_config_param_still_disables_retries() -> None:
    """Custom providers can still declare their retry-disable constructor kwarg."""
    assert _provider_retry_disable_kwargs(
        {"custom": {"param": "num_retries"}},
        "custom",
        {},
    ) == {"num_retries": 0}


def test_string_startup_model_uses_retry_aware_creation(tmp_path: Path) -> None:
    """Public string callers cannot bypass SDK retry disabling or metadata."""
    from deepagents_code.agent import create_cli_agent
    from deepagents_code.model_retry import CodeModelRetryMiddleware
    from deepagents_code.offload_middleware import (
        CLICompactionMiddleware,
        RetryingSummarizationMiddleware,
    )

    model = FakeListChatModel(responses=["ok"])
    model.profile = {"max_input_tokens": 20_000}
    result = ModelResult(
        model=model,
        model_name="model",
        provider="provider",
        model_retries=2,
    )
    graph = MagicMock()
    graph.with_config.return_value = graph
    with (
        patch("deepagents_code.config.create_model", return_value=result) as create,
        patch("deepagents_code.agent.list_subagents", return_value=[]),
        patch("deepagents_code.agent.create_deep_agent", return_value=graph) as build,
    ):
        create_cli_agent(
            model="provider:model",
            assistant_id="retry-test",
            auto_approve=True,
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            system_prompt="test",
            cwd=tmp_path,
        )

    create.assert_called_once_with("provider:model", extra_kwargs=None)
    assert build.call_args.kwargs["model"] is model
    main_middleware = build.call_args.kwargs["middleware"]
    main_retry = next(
        item for item in main_middleware if isinstance(item, CodeModelRetryMiddleware)
    )
    assert main_retry.max_retries == 2
    # `CLICompactionMiddleware` owns the automatic-summarization slot on the main
    # agent, so the retrying summarizer must not also be installed there: both
    # report `name == "SummarizationMiddleware"` and `_apply_custom_middleware`
    # silently drops all but the last entry for a given name.
    assert not any(
        isinstance(item, RetryingSummarizationMiddleware) for item in main_middleware
    )
    assert any(isinstance(item, CLICompactionMiddleware) for item in main_middleware)
    names = [item.name for item in main_middleware]
    assert len(names) == len(set(names)), f"duplicate middleware names: {names}"
    for subagent in build.call_args.kwargs["subagents"]:
        assert any(
            isinstance(item, RetryingSummarizationMiddleware)
            for item in subagent["middleware"]
        )


def test_subagent_missing_credentials_does_not_block_startup(tmp_path: Path) -> None:
    """A declarative subagent naming an unusable provider defers to invocation.

    `create_model` runs dcode's fail-fast credential precheck, so constructing a
    subagent model eagerly would let one unused `.deepagents/agents/*.md` file
    stop the whole CLI from starting.
    """
    from deepagents_code.agent import create_cli_agent
    from deepagents_code.model_config import MissingCredentialsError

    model = FakeListChatModel(responses=["ok"])
    model.profile = {"max_input_tokens": 20_000}
    graph = MagicMock()
    graph.with_config.return_value = graph
    subagent_meta = {
        "name": "researcher",
        "description": "research things",
        "system_prompt": "research",
        "model": "anthropic:claude-sonnet-4-5",
    }

    def fake_create_model(spec: str, **_kwargs: object) -> ModelResult:
        if spec == "anthropic:claude-sonnet-4-5":
            msg = "No credentials found for provider 'anthropic'"
            raise MissingCredentialsError(
                msg, provider="anthropic", env_var="ANTHROPIC_API_KEY"
            )
        return ModelResult(
            model=model, model_name="model", provider="openai", model_retries=2
        )

    with (
        patch("deepagents_code.config.create_model", side_effect=fake_create_model),
        patch("deepagents_code.agent.list_subagents", return_value=[subagent_meta]),
        patch("deepagents_code.agent.create_deep_agent", return_value=graph) as build,
    ):
        create_cli_agent(
            model="openai:gpt-5",
            assistant_id="retry-test",
            auto_approve=True,
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            system_prompt="test",
            cwd=tmp_path,
        )

    subagent = next(
        item
        for item in build.call_args.kwargs["subagents"]
        if item["name"] == "researcher"
    )
    # Unresolvable: kept as the lazy spec so the failure surfaces on invocation.
    assert subagent["model"] == "anthropic:claude-sonnet-4-5"


def test_prebuilt_model_budget_does_not_forge_a_cli_override(tmp_path: Path) -> None:
    """A config-resolved count must not masquerade as `--max-retries`.

    Both CLI entry points build the model with `create_model` and pass the
    resolved count back into `create_cli_agent`. Recording that as a CLI override
    would make a runtime `/model` switch reuse the old provider's number instead
    of consulting `[retries.<new-provider>]`.
    """
    from deepagents_code.agent import create_cli_agent
    from deepagents_code.config import (
        MODEL_RETRY_OVERRIDE_ATTR,
        get_model_retries,
        set_model_retry_metadata,
    )

    model = FakeListChatModel(responses=["ok"])
    model.profile = {"max_input_tokens": 20_000}
    # What `create_model` leaves behind for `[retries.openai].max_retries = 2`
    # with no `--max-retries` on the command line.
    set_model_retry_metadata(model, retries=2, cli_override=None)
    graph = MagicMock()
    graph.with_config.return_value = graph
    with (
        patch("deepagents_code.agent.list_subagents", return_value=[]),
        patch("deepagents_code.agent.create_deep_agent", return_value=graph),
    ):
        create_cli_agent(
            model=model,
            assistant_id="retry-test",
            auto_approve=True,
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            system_prompt="test",
            cwd=tmp_path,
            model_retries=2,
        )

    assert get_model_retries(model, 99) == 2
    assert getattr(model, MODEL_RETRY_OVERRIDE_ATTR) is None


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
            raise _read_error()
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
            wrapped.__cause__ = _read_error()
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
            group = ExceptionGroup(
                "model graph failed", [ValueError("x"), _read_error()]
            )
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
        raise _read_error()

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
        raise _read_error()

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
        raise _read_error()

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
        raise _read_error()

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
        raise _read_error()

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
        raise _read_error()

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
            raise _read_error()
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
            raise _read_error()
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


def test_retry_loop_sleeps_the_computed_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop must actually wait, with the attempt-indexed delay.

    `_compute_delay` is well covered in isolation, but nothing connected it to the
    loop: dropping the sleep, or passing a fixed index instead of `attempt`, turns
    the budget into a tight hammer loop against an already-struggling provider and
    every other test still passes.
    """
    monkeypatch.setattr(
        "deepagents_code.model_retry.random.uniform", lambda _a, _b: 0.0
    )
    slept: list[float] = []
    monkeypatch.setattr(
        "deepagents_code.model_retry.time.sleep", lambda delay: slept.append(delay)
    )
    middleware = CodeModelRetryMiddleware(max_retries=3)
    model = MagicMock(spec=BaseChatModel)

    def handler() -> ModelResponse[Any]:
        raise _read_error()

    with pytest.raises(httpx.ReadError):
        middleware.run_with_retry(model, handler, max_retries=3)

    # One sleep per retry (not per attempt), following the exponential curve.
    assert slept == pytest.approx([0.2, 0.4, 0.8])


async def test_async_retry_loop_sleeps_the_computed_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async twin of `test_retry_loop_sleeps_the_computed_backoff`."""
    monkeypatch.setattr(
        "deepagents_code.model_retry.random.uniform", lambda _a, _b: 0.0
    )
    slept: list[float] = []

    # Must stay a coroutine function to stand in for `asyncio.sleep`, and must not
    # await anything itself -- awaiting the patched `sleep` would recurse.
    async def _fake_sleep(delay: float) -> None:  # noqa: RUF029
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    middleware = CodeModelRetryMiddleware(max_retries=2)
    model = MagicMock(spec=BaseChatModel)

    async def handler() -> ModelResponse[Any]:  # noqa: RUF029  # awaited by the driver
        raise _read_error()

    with pytest.raises(httpx.ReadError):
        await middleware.arun_with_retry(model, handler, max_retries=2)

    assert slept == pytest.approx([0.2, 0.4])


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
            raise _read_error()
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
        raise _read_error()

    mw = CodeModelRetryMiddleware(max_retries=2)
    with pytest.raises(httpx.ReadError):
        await mw.awrap_model_call(_req(), handler)
    assert calls["n"] == 3


async def test_async_request_model_overrides_startup_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async model node resolves the per-request budget like the sync path."""

    async def _no_sleep(*_a: object, **_k: object) -> None:  # noqa: RUF029  # async stub replacing asyncio.sleep
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}
    events: list[dict] = []

    async def handler(  # noqa: RUF029  # awaited by middleware; no internal await needed
        _request: ModelRequest,
    ) -> ModelResponse[Any]:
        calls["n"] += 1
        raise _read_error()

    middleware = CodeModelRetryMiddleware(max_retries=0)
    with pytest.raises(httpx.ReadError):
        await middleware.awrap_model_call(
            _req(events, model_retries=4, runtime_retries=1), handler
        )

    # The runtime override (1) wins over the model's attached budget (4):
    # one retry, so two calls and a single status event.
    assert calls["n"] == 2
    assert [event["max_retries"] for event in events] == [1]


# --- runtime retry-override carrier validation ---


@pytest.mark.parametrize(
    "context",
    [
        None,
        "not-a-context",
        SimpleNamespace(model_params=None),
        SimpleNamespace(model_params=["not", "a", "mapping"]),
        {"model_params": {}},
        {"model_params": {CLI_MAX_RETRIES_KEY: True}},
        {"model_params": {CLI_MAX_RETRIES_KEY: -1}},
        {"model_params": {CLI_MAX_RETRIES_KEY: "3"}},
    ],
)
def test_runtime_model_retry_override_rejects_invalid(context: object) -> None:
    """A malformed runtime carrier yields `None`, never a bad budget."""
    runtime = SimpleNamespace(context=context)
    assert _runtime_model_retry_override(runtime) is None


@pytest.mark.parametrize("value", [0, 3])
def test_runtime_model_retry_override_accepts_valid(value: int) -> None:
    runtime = SimpleNamespace(context={"model_params": {CLI_MAX_RETRIES_KEY: value}})
    assert _runtime_model_retry_override(runtime) == value


# --- negative-budget clamp (fail-loud guard stays unreachable) ---


def test_negative_override_runs_once_and_reraises() -> None:
    """A negative override clamps to a single attempt, not a skipped call."""
    mw = CodeModelRetryMiddleware(max_retries=5)
    model = MagicMock(spec=BaseChatModel)
    calls = {"n": 0}

    def handler() -> ModelResponse[Any]:
        calls["n"] += 1
        raise _read_error()

    with pytest.raises(httpx.ReadError):
        mw.run_with_retry(model, handler, max_retries=-1)
    assert calls["n"] == 1


async def test_async_negative_override_runs_once_and_reraises() -> None:
    mw = CodeModelRetryMiddleware(max_retries=5)
    model = MagicMock(spec=BaseChatModel)
    calls = {"n": 0}

    async def handler() -> ModelResponse[Any]:  # noqa: RUF029  # awaited by middleware
        calls["n"] += 1
        raise _read_error()

    with pytest.raises(httpx.ReadError):
        await mw.arun_with_retry(model, handler, max_retries=-1)
    assert calls["n"] == 1


# --- top-level deterministic error wins over a chained transient one ---


def test_should_retry_top_level_deterministic_status_wins() -> None:
    """A 401 carrying a transient __cause__ is not retryable."""
    err = AuthenticationError()  # status_code = 401
    err.__cause__ = _read_error()
    assert _should_retry_after_failure(err) is False


def test_should_retry_opaque_wrapper_with_transient_cause() -> None:
    """A statusless opaque wrapper still retries via the chain scan."""
    wrapped = RuntimeError("opaque model graph failure")
    wrapped.__cause__ = _read_error()
    assert _should_retry_after_failure(wrapped) is True


def test_should_retry_throttle_400_survives_deterministic_guard() -> None:
    """AWS throttling behind HTTP 400 is not mistaken for a fatal 4xx."""
    assert _should_retry_after_failure(_ThrottlingError()) is True


def test_deterministic_error_with_transient_cause_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: an auth failure with a transient cause surfaces at once."""
    monkeypatch.setattr("deepagents_code.model_retry.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        err = AuthenticationError()
        err.__cause__ = _read_error()
        raise err

    mw = CodeModelRetryMiddleware(max_retries=5)
    with pytest.raises(AuthenticationError):
        mw.wrap_model_call(_req(), handler)
    assert calls["n"] == 1


# --- create_model must not mutate the caller's extra_kwargs ---


def test_create_model_does_not_mutate_caller_extra_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`create_model` must keep the CLI carrier in the caller's dict.

    The app retains a single `extra_kwargs` dict and reuses it across runtime
    `/model` switches; stripping `CLI_MAX_RETRIES_KEY` here would silently
    disable `--max-retries` on every switch after the first.
    """
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    monkeypatch.setattr(model_config, "has_provider_credentials", lambda _: True)
    extra_kwargs = {CLI_MAX_RETRIES_KEY: 2, "temperature": 0.1}
    model_config.clear_caches()
    with (
        patch.object(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "none.toml"),
        patch("langchain.chat_models.init_chat_model", return_value=model),
    ):
        result = create_model("anthropic:claude-sonnet-4-5", extra_kwargs=extra_kwargs)
    model_config.clear_caches()

    assert extra_kwargs == {CLI_MAX_RETRIES_KEY: 2, "temperature": 0.1}
    assert result.model_retries == 2


class Aborted(Exception):  # noqa: N818  # mirrors Google's real class name
    """Google api_core Aborted with real HTTP 409 on `.code`."""

    def __init__(self) -> None:
        super().__init__("aborted")
        self.code = 409


def test_google_aborted_http_409_is_retryable() -> None:
    """Aborted→409 must still retry via the known-transient name allowlist."""
    assert _is_retryable_model_error(Aborted()) is True
    assert _should_retry_after_failure(Aborted()) is True


def test_exception_chain_walks_context_even_when_cause_set() -> None:
    from deepagents_code.model_retry import exception_chain

    outer = RuntimeError("outer")
    cause = ValueError("cause")
    context = _read_error()
    outer.__cause__ = cause
    outer.__context__ = context
    names = [type(item).__name__ for item in exception_chain(outer)]
    assert "RuntimeError" in names
    assert "ValueError" in names
    assert "ReadError" in names


def test_should_retry_exception_group_top_level_auth_wins() -> None:
    """A top-level group member that is definitive non-retryable blocks retry."""
    group = ExceptionGroup(
        "model graph failed",
        [AuthenticationError(), _read_error()],
    )
    assert _should_retry_after_failure(group) is False


def test_should_retry_nested_exception_group_auth_wins() -> None:
    """A nested group's definitive member blocks retry despite a transient sibling."""
    group = ExceptionGroup(
        "model graph failed",
        [
            ExceptionGroup("inner", [AuthenticationError(), _read_error()]),
        ],
    )
    assert _should_retry_after_failure(group) is False


def test_should_retry_nested_exception_group_all_transient_still_retries() -> None:
    group = ExceptionGroup(
        "model graph failed",
        [
            ExceptionGroup("inner", [_read_error(), _StatusError(503)]),
        ],
    )
    assert _should_retry_after_failure(group) is True


def test_should_retry_exception_group_all_transient_still_retries() -> None:
    group = ExceptionGroup(
        "model graph failed",
        [_read_error(), _StatusError(503)],
    )
    assert _should_retry_after_failure(group) is True


def test_transient_wrapper_does_not_launder_a_permanent_cause() -> None:
    """A permanent failure under a transient wrapper must not be retried.

    The definitive verdict can arrive *below* the transient one, so checking only
    the top level (or only group members) would burn the whole budget re-issuing a
    request that can never succeed.
    """
    # The chain `raise ReadError() from AuthenticationError()` leaves behind.
    exc = _read_error()
    exc.__cause__ = AuthenticationError()
    assert _should_retry_after_failure(exc) is False


@pytest.mark.parametrize(
    "deterministic",
    [
        ContextOverflowError("too large"),
        GraphInterrupt(()),
    ],
    ids=["context_overflow", "graph_interrupt"],
)
def test_never_retry_types_beat_a_transient_sibling(
    deterministic: Exception,
) -> None:
    """Control-flow and deterministic failures carry no status to classify by.

    Without an explicit never-retry entry, the transient sibling would win the
    chain scan -- suppressing an approval prompt, or delaying compaction, for the
    full retry budget.
    """
    group = ExceptionGroup("model graph failed", [_read_error(), deterministic])
    assert _should_retry_after_failure(group) is False


def test_raise_from_none_severs_an_incidental_transient_context() -> None:
    """`raise X from None` states the in-flight error is unrelated.

    A deterministic error raised while a transport fault happens to be in flight
    must not inherit that fault's retryability.
    """
    # Exactly the state `raise ValueError(...) from None` leaves behind inside an
    # `except httpx.ReadError` block: context present, no cause, context suppressed.
    exc = ValueError("deterministic")
    exc.__context__ = _read_error()
    exc.__suppress_context__ = True
    assert _should_retry_after_failure(exc) is False

    # Without the `from None`, the transport fault is still part of the story.
    incidental = ValueError("deterministic")
    incidental.__context__ = _read_error()
    assert _should_retry_after_failure(incidental) is True


def test_classification_failure_surfaces_the_model_error() -> None:
    """A raising error attribute must not replace the failure being classified."""

    class _LazyParseError(Exception):
        @property
        def code(self) -> str:
            msg = "lazy parse failed"
            raise RuntimeError(msg)

    middleware = CodeModelRetryMiddleware(max_retries=2)
    model = MagicMock(spec=BaseChatModel)
    calls = 0

    def handler() -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        raise _LazyParseError

    with pytest.raises(_LazyParseError):
        middleware.run_with_retry(model, handler, max_retries=2)
    # Unclassifiable, so treated as non-retryable rather than retried blindly.
    assert calls == 1


def test_nested_drivers_do_not_multiply_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outer driver must not restart a budget an inner driver already spent.

    Retry drivers nest in production (the model node wraps middleware that runs
    its own summarizer/grader/classifier retries), so without this an exhausted
    inner budget becomes `(retries + 1) ** 2` model calls behind a status line
    still reporting the outer budget.
    """
    monkeypatch.setattr(
        CodeModelRetryMiddleware, "_compute_delay", lambda _self, _attempt: 0.0
    )
    inner = CodeModelRetryMiddleware(max_retries=2)
    outer = CodeModelRetryMiddleware(max_retries=2)
    model = MagicMock(spec=BaseChatModel)
    calls = 0

    def leaf() -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        raise _read_error()

    def middle() -> ModelResponse[Any]:
        return inner.run_with_retry(model, leaf, max_retries=2)

    with pytest.raises(httpx.ReadError):
        outer.run_with_retry(model, middle, max_retries=2)

    assert calls == 3


def test_explicit_non_retryable_marker_opts_out_of_timeout_fallback() -> None:
    class _LocalDeadlineError(TimeoutError):
        dcode_model_retryable = False

    assert _is_retryable_model_error(_LocalDeadlineError()) is False
    assert _should_retry_after_failure(_LocalDeadlineError()) is False


def _stream_from(model: BaseChatModel, *, lc_source: str | None = None) -> None:
    """Consume one streamed token, optionally tagged as internal output."""
    config: RunnableConfig | None = (
        {"metadata": {"lc_source": lc_source}} if lc_source else None
    )
    next(iter(model.stream("prompt", config=config)))


def test_hidden_summarizer_output_does_not_veto_retry() -> None:
    """An inner summarizer's tokens must not suppress the guarded call's retry.

    The tracker rides the ambient callback config, so it sees every model run
    inside the handler. Summarizer output is filtered out of the transcript, so
    retrying after it cannot duplicate anything the user saw.
    """
    summarizer = FakeListChatModel(responses=["a summary"])
    model = FakeListChatModel(responses=["real"])
    setattr(model, MODEL_RETRIES_ATTR, 3)
    request = _req().override(model=model)
    calls = {"n": 0}

    def handler(selected: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        assert selected.model is model
        _stream_from(summarizer, lc_source="summarization")
        raise _read_error()

    middleware = CodeModelRetryMiddleware(max_retries=3)
    with (
        patch.object(middleware, "_compute_delay", return_value=0.0),
        pytest.raises(httpx.ReadError),
    ):
        middleware.wrap_model_call(request, handler)

    assert calls["n"] == 4


async def test_async_hidden_summarizer_output_does_not_veto_retry() -> None:
    """Async twin: hidden inner output must not suppress retries."""
    summarizer = FakeListChatModel(responses=["a summary"])
    model = FakeListChatModel(responses=["real"])
    setattr(model, MODEL_RETRIES_ATTR, 3)
    request = _req().override(model=model)
    calls = {"n": 0}

    async def handler(selected: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        assert selected.model is model
        async for _ in summarizer.astream(
            "prompt", config={"metadata": {"lc_source": "summarization"}}
        ):
            break
        raise _read_error()

    middleware = CodeModelRetryMiddleware(max_retries=3)
    with (
        patch.object(middleware, "_compute_delay", return_value=0.0),
        pytest.raises(httpx.ReadError),
    ):
        await middleware.awrap_model_call(request, handler)

    assert calls["n"] == 4


def test_hidden_classifier_output_does_not_veto_retry() -> None:
    """Auto mode's classifier output is hidden too, so it must not veto."""
    classifier = FakeListChatModel(responses=["allow"])
    model = FakeListChatModel(responses=["real"])
    request = _req().override(model=model)
    calls = {"n": 0}

    def handler(selected: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        assert selected.model is model
        _stream_from(classifier, lc_source="auto_mode_classifier")
        raise _read_error()

    middleware = CodeModelRetryMiddleware(max_retries=2)
    with (
        patch.object(middleware, "_compute_delay", return_value=0.0),
        pytest.raises(httpx.ReadError),
    ):
        middleware.wrap_model_call(request, handler)

    assert calls["n"] == 3


def test_untagged_inner_output_still_vetoes_retry() -> None:
    """Unrecognized runs stay fail-safe: they still count as visible output."""
    other = FakeListChatModel(responses=["something"])
    model = FakeListChatModel(responses=["real"])
    request = _req().override(model=model)
    calls = {"n": 0}

    def handler(selected: ModelRequest) -> ModelResponse[Any]:
        calls["n"] += 1
        assert selected.model is model
        _stream_from(other)
        raise _read_error()

    middleware = CodeModelRetryMiddleware(max_retries=3)
    with pytest.raises(httpx.ReadError):
        middleware.wrap_model_call(request, handler)

    assert calls["n"] == 1


def test_protocol_stream_event_vetoes_retry_through_wrap_model_call() -> None:
    """The tracker must be reachable via the ambient config, not just in isolation.

    Real providers signal output with `on_stream_event`, while `FakeListChatModel`
    (used by the other veto tests) only emits `on_llm_new_token`. Dispatching
    through the ambient callback manager the way a provider does is what proves
    `_stream_tracking_config` actually installed the tracker where the model can
    reach it -- otherwise every streaming provider would replay partial output.
    """
    from langchain_core.callbacks.manager import handle_event
    from langchain_core.runnables import ensure_config

    events: list[dict] = []
    calls = 0
    request = _req(events, model_retries=3)

    def handler(_selected: ModelRequest) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        manager = cast("CallbackManager", ensure_config()["callbacks"])
        # Exactly how a streaming provider announces a chunk.
        handle_event(
            manager.handlers,
            "on_stream_event",
            "ignore_llm",
            object(),
            run_id=uuid4(),
        )
        raise _read_error()

    middleware = CodeModelRetryMiddleware(max_retries=3)
    with pytest.raises(httpx.ReadError):
        middleware.wrap_model_call(request, handler)

    assert calls == 1
    assert events == []


async def test_async_protocol_stream_event_vetoes_retry() -> None:
    """Async twin: the async callback manager must carry the tracker too."""
    from langchain_core.callbacks.manager import ahandle_event
    from langchain_core.runnables import ensure_config

    events: list[dict] = []
    calls = 0
    request = _req(events, model_retries=3)

    async def handler(_selected: ModelRequest) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        manager = cast("AsyncCallbackManager", ensure_config()["callbacks"])
        await ahandle_event(
            manager.handlers,
            "on_stream_event",
            "ignore_llm",
            object(),
            run_id=uuid4(),
        )
        raise _read_error()

    middleware = CodeModelRetryMiddleware(max_retries=3)
    with pytest.raises(httpx.ReadError):
        await middleware.awrap_model_call(request, handler)

    assert calls == 1
    assert events == []


def test_on_stream_event_marks_output_visible() -> None:
    """The protocol-stream callback must veto retries like token streaming.

    Real providers dispatch `on_stream_event` rather than `on_llm_new_token`,
    so this path is what actually guards duplicate output in production.
    """
    tracker = _StreamOutputTracker()
    assert tracker.emitted is False
    tracker.on_stream_event(object(), run_id=uuid4())
    assert tracker.emitted is True


def test_on_stream_event_ignores_hidden_run() -> None:
    """Hidden runs must not latch `emitted` via the protocol-stream path."""
    tracker = _StreamOutputTracker()
    run_id = uuid4()
    tracker.on_chat_model_start(
        {}, [], run_id=run_id, metadata={"lc_source": "summarization"}
    )
    tracker.on_stream_event(object(), run_id=run_id)
    assert tracker.emitted is False
