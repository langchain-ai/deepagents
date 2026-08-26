"""Tests for ConfigurableModelMiddleware."""

import asyncio
import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code._cli_context import CLIContext, CLIContextSchema
from deepagents_code.agent import build_model_identity_section
from deepagents_code.configurable_model import (
    ConfigurableModelMiddleware,
    _cache_endpoint_identity,
    _checkpoint_command,
    _effective_cache_params,
    _get_context,
    _is_anthropic_model,
    _is_fireworks_model,
    _is_openai_model,
    _model_spec_from_model,
    _ResolvedModelRequest,
)


def _make_model(name: str) -> MagicMock:
    """Create a mock BaseChatModel with model_name set."""
    model = MagicMock(spec=BaseChatModel)
    model.model_name = name
    model.model_dump.return_value = {"model_name": name}
    model._get_ls_params.return_value = {"ls_provider": "openai"}
    model.root_client = SimpleNamespace(base_url="https://api.openai.com/v1")
    return model


def _make_request(
    model: BaseChatModel,
    context: object = None,
    model_settings: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> ModelRequest:
    """Create a ModelRequest with a runtime that carries CLIContext."""
    runtime = SimpleNamespace(context=context)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [HumanMessage(content="hi")],
        "tools": [],
        "runtime": cast("Any", runtime),
        "model_settings": model_settings,
    }
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return ModelRequest(**kwargs)


def _make_response() -> ModelResponse[Any]:
    """Create a minimal model response for handler mocks."""
    return ModelResponse(result=[AIMessage(content="response")])


def _checkpoint_update(
    result: ModelResponse[Any] | ExtendedModelResponse[Any],
) -> dict[str, Any]:
    """Return the checkpoint update emitted by the middleware."""
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert isinstance(result.command.update, dict)
    update = dict(result.command.update)
    timestamp = update.pop("_last_model_request_at")
    assert isinstance(timestamp, str)
    cache_model_spec = update.pop("_last_cache_model_spec")
    assert isinstance(cache_model_spec, str)
    cache_endpoint = update.pop("_last_cache_endpoint")
    assert isinstance(cache_endpoint, str)
    return update


def _make_model_result(
    model: MagicMock,
    *,
    model_name: str = "",
    provider: str = "",
    context_limit: int | None = None,
    unsupported_modalities: frozenset[str] = frozenset(),
) -> SimpleNamespace:
    """Create a mock ModelResult with model metadata."""
    return SimpleNamespace(
        model=model,
        model_name=model_name or model.model_name,
        provider=provider,
        context_limit=context_limit,
        unsupported_modalities=unsupported_modalities,
    )


_PATCH_CREATE = "deepagents_code.config.create_model"

# The shared instance pins the OpenAI cache-key flag explicitly so it does not
# read config at import time — that keeps it hermetic regardless of a
# developer's env/config.toml. Tests that exercise flag *resolution* construct
# their own instances after patching the config lookup.
_mw = ConfigurableModelMiddleware(openai_prompt_cache_key=True)


class TestCheckpointPersistence:
    """Tests for private resume-state checkpoint updates."""

    def test_startup_custom_provider_uses_configured_spec(self) -> None:
        """Custom classes must checkpoint their configured provider alias."""
        from deepagents_code.config import settings

        model = _make_model("fake")
        model._get_ls_params.return_value = {
            "ls_provider": "deterministicintegrationchatmodel"
        }
        with (
            patch.object(settings, "model_provider", "itest"),
            patch.object(settings, "model_name", "fake"),
        ):
            assert _model_spec_from_model(model) == "itest:fake"

    def test_records_request_start_only_after_success(self) -> None:
        middleware = ConfigurableModelMiddleware(openai_prompt_cache_key=True)
        request = _make_request(_make_model("gpt-5.6"))

        with patch(
            "deepagents_code.configurable_model._utc_now_iso",
            return_value="2026-08-11T12:30:00+00:00",
        ):
            result = middleware.wrap_model_call(
                request,
                lambda _request: _make_response(),
            )

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        update = result.command.update
        assert isinstance(update, dict)
        assert update["_last_model_request_at"] == "2026-08-11T12:30:00+00:00"
        assert update["_last_cache_model_spec"] == "openai:gpt-5.6"
        assert update["_last_cache_endpoint"] == "default"

    def test_timestamp_is_captured_before_the_model_call(self) -> None:
        """Cache age must be measured from when the prefix was written.

        Stamping after `handler()` returns would make a twenty-minute turn
        look twenty minutes fresher than it is, under-warning on exactly the
        long, expensive turns this feature targets.
        """
        middleware = ConfigurableModelMiddleware(openai_prompt_cache_key=True)
        request = _make_request(_make_model("gpt-5.6"))
        clock = iter(
            ["2026-08-11T12:30:00+00:00", "2026-08-11T12:50:00+00:00"],
        )

        def slow_handler(_request: ModelRequest) -> ModelResponse[Any]:
            # Consumes the second reading, as a real elapsed call would.
            next(clock)
            return _make_response()

        with patch(
            "deepagents_code.configurable_model._utc_now_iso",
            side_effect=lambda: next(clock),
        ):
            result = middleware.wrap_model_call(request, slow_handler)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        update = result.command.update
        assert isinstance(update, dict)
        assert update["_last_model_request_at"] == "2026-08-11T12:30:00+00:00"

    def test_timestamp_is_omitted_when_the_model_spec_is_unknown(self) -> None:
        """Timing and identity are one fact and must be written together.

        A timestamp without a spec reads back as a permanent "model changed",
        warning on every send with copy naming a change that never happened.
        """
        resolved = _ResolvedModelRequest(
            _make_request(_make_model("gpt-5.6")),
            None,
            model_params_known=True,
        )

        command = _checkpoint_command(resolved, "2026-08-11T12:30:00+00:00", "default")

        update = command.update
        assert isinstance(update, dict)
        assert "_last_model_request_at" not in update
        assert "_last_cache_model_spec" not in update
        assert "_last_cache_endpoint" not in update

    def test_failed_call_does_not_return_checkpoint_update(self) -> None:
        middleware = ConfigurableModelMiddleware(openai_prompt_cache_key=True)
        request = _make_request(_make_model("gpt-5.6"))

        def fail(_request: ModelRequest) -> ModelResponse[Any]:
            msg = "provider failed"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="provider failed"):
            middleware.wrap_model_call(request, fail)

    def test_can_disable_model_state_persistence(self) -> None:
        middleware = ConfigurableModelMiddleware(persist_model_state=False)
        request = _make_request(_make_model("gpt-5.5"))

        result = middleware.wrap_model_call(request, lambda _request: _make_response())

        assert isinstance(result, ModelResponse)


class TestNoOverride:
    """Cases where the middleware should pass the request through unchanged."""

    def test_no_context(self) -> None:
        request = _make_request(_make_model("claude-sonnet-4-6"), context=None)
        captured: list[ModelRequest] = []
        result = _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0].model is request.model
        assert _checkpoint_update(result) == {"_model_spec": "openai:claude-sonnet-4-6"}

    def test_empty_context(self) -> None:
        request = _make_request(_make_model("claude-sonnet-4-6"), context=CLIContext())
        captured: list[ModelRequest] = []
        result = _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request
        assert _checkpoint_update(result) == {
            "_model_spec": "openai:claude-sonnet-4-6",
            "_model_params": None,
            "_last_cache_params": None,
        }

    def test_dict_context_reconstructs_approval_fields(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context={
                "auto_approve": True,
                "approval_mode_key": "approval-key",
                "thread_id": "thread-123",
            },
        )

        ctx = _get_context(request)

        assert ctx is not None
        assert ctx.auto_approve is True
        assert ctx.approval_mode_key == "approval-key"
        assert ctx.thread_id == "thread-123"

    @pytest.mark.parametrize("key", [None, 1, object()])
    def test_dict_context_coerces_non_string_approval_key(self, key: object) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context={
                "auto_approve": True,
                "approval_mode_key": key,
            },
        )

        ctx = _get_context(request)

        assert ctx is not None
        assert ctx.auto_approve is True
        assert ctx.approval_mode_key is None

    def test_dict_context_carries_classifier_model(self) -> None:
        """A serialized context must keep the Auto classifier override."""
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context={"classifier_model": "openai:gpt-5.5-mini"},
        )

        ctx = _get_context(request)

        assert ctx is not None
        assert ctx.classifier_model == "openai:gpt-5.5-mini"

    @pytest.mark.parametrize("classifier_model", [None, 1, object()])
    def test_dict_context_coerces_non_string_classifier_model(
        self, classifier_model: object
    ) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context={"classifier_model": classifier_model},
        )

        ctx = _get_context(request)

        assert ctx is not None
        assert ctx.classifier_model is None

    @pytest.mark.parametrize("thread_id", [None, 1, object()])
    def test_dict_context_coerces_non_string_thread_id(self, thread_id: object) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context={"thread_id": thread_id},
        )

        ctx = _get_context(request)

        assert ctx is not None
        assert ctx.thread_id is None

    def test_same_model_spec(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="claude-sonnet-4-6"),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request

    def test_provider_prefixed_spec_matches(self) -> None:
        request = _make_request(
            _make_model("gpt-5.5"),
            context=CLIContext(model="openai:gpt-5.5"),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request

    def test_provider_prefixed_spec_mismatch_overrides_same_model_name(self) -> None:
        request = _make_request(
            _make_model("gpt-5.5"),
            context=CLIContextSchema(model="openai_codex:gpt-5.5"),
        )
        replacement = _make_model("gpt-5.5")
        replacement._get_ls_params.return_value = {"ls_provider": "openai-codex"}
        captured: list[ModelRequest] = []

        with patch(
            _PATCH_CREATE, return_value=_make_model_result(replacement)
        ) as create:
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        create.assert_called_once_with("openai_codex:gpt-5.5")
        assert captured[0].model is replacement

    def test_none_runtime(self) -> None:
        request = ModelRequest(
            model=_make_model("claude-sonnet-4-6"),
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=None,
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0].model is request.model

    def test_non_dict_context_ignored(self) -> None:
        runtime = SimpleNamespace(context="not-a-dict")
        request = ModelRequest(
            model=_make_model("claude-sonnet-4-6"),
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=cast("Any", runtime),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0].model is request.model

    def test_empty_model_params(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model_params={}),
        )
        captured: list[ModelRequest] = []
        result = _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request
        assert _checkpoint_update(result) == {
            "_model_spec": "openai:claude-sonnet-4-6",
            "_model_params": None,
            "_last_cache_params": None,
        }


def test_cache_endpoint_identity_uses_configured_provider_params() -> None:
    """A `params.base_url` gateway must not be recorded as the default API."""
    from deepagents_code.model_config import ModelConfig

    config = ModelConfig(
        providers={"openai": {"params": {"base_url": "https://proxy.example.com/v1"}}}
    )
    with patch("deepagents_code.model_config.ModelConfig.load", return_value=config):
        assert _cache_endpoint_identity("openai:gpt-5.5") == (
            "https://proxy.example.com/v1"
        )


def test_checkpoint_records_effective_cache_params() -> None:
    """Configured cache params reach the checkpoint, not just runtime overrides.

    Regression: with `prompt_cache_retention` in config and no session
    override, storing only the runtime overrides (`None`) makes the next turn's
    identity check compare `{"prompt_cache_retention": ...}` against `None` and
    report a false `identity_changed` every turn. The projection lands in the
    dedicated `_last_cache_params` channel so resume never re-reads it as
    per-session overrides.
    """
    from deepagents_code.model_config import ModelConfig

    config = ModelConfig(
        providers={"openai": {"params": {"prompt_cache_retention": "24h"}}}
    )
    request = _make_request(
        _make_model("gpt-5.5"),
        context=CLIContext(),
    )
    with patch("deepagents_code.model_config.ModelConfig.load", return_value=config):
        result = ConfigurableModelMiddleware().wrap_model_call(
            request, lambda _r: _make_response()
        )

    update = _checkpoint_update(result)
    assert update["_last_cache_params"] == {"prompt_cache_retention": "24h"}
    assert update["_model_params"] is None


def test_checkpoint_cache_params_exclude_unrelated_config() -> None:
    """Only cache-identity keys may be persisted for the cold-cache check.

    Regression: `_model_params` is read back on resume as runtime overrides,
    so persisting the full effective kwargs there (or anywhere resume reads)
    pins configured defaults like `temperature` into old threads and silently
    overrides newer config.
    """
    from deepagents_code.model_config import ModelConfig

    config = ModelConfig(
        providers={
            "openai": {
                "params": {
                    "prompt_cache_retention": "24h",
                    "temperature": 0.7,
                    "max_retries": 5,
                    "default_headers": {"x-trace": "abc"},
                    "base_url": "https://proxy.example.com/v1",
                }
            }
        }
    )
    request = _make_request(
        _make_model("gpt-5.5"),
        context=CLIContext(model_params={"reasoning_effort": "high"}),
    )
    with patch("deepagents_code.model_config.ModelConfig.load", return_value=config):
        result = ConfigurableModelMiddleware().wrap_model_call(
            request, lambda _r: _make_response()
        )

    update = _checkpoint_update(result)
    # Only the identity key survives; `base_url` is tracked separately as the
    # endpoint identity, and the runtime override is not a cache-identity key.
    assert update["_last_cache_params"] == {"prompt_cache_retention": "24h"}
    # Resume semantics are untouched: exactly the runtime overrides.
    assert update["_model_params"] == {"reasoning_effort": "high"}


def test_cache_endpoint_identity_normalizes_the_provider() -> None:
    """The reader lowercases before looking up, so the writer must too.

    `get_kwargs`/`get_base_url` are exact-key lookups, so without this the two
    sides resolve different endpoints for the same request and disagree on every
    turn -- a disagreement that never self-heals.
    """
    from deepagents_code.model_config import ModelConfig

    config = ModelConfig(
        providers={"openai": {"params": {"base_url": "https://proxy.example.com/v1"}}}
    )
    with patch("deepagents_code.model_config.ModelConfig.load", return_value=config):
        for spec in ("openai:gpt-5.5", "OpenAI:gpt-5.5", " openai:gpt-5.5"):
            assert _cache_endpoint_identity(spec) == "https://proxy.example.com/v1"


def test_cache_endpoint_identity_degrades_instead_of_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A completed, billed model call must not be lost to a config surprise.

    Both call sites run after `handler()` returns, so propagating here would
    discard a paid response over a value used only for change detection.
    """
    config = MagicMock()
    config.get_effective_kwargs.side_effect = RuntimeError("config exploded")
    config.get_base_url.side_effect = RuntimeError("config exploded")

    with (
        patch("deepagents_code.model_config.ModelConfig.load", return_value=config),
        caplog.at_level(logging.WARNING, logger="deepagents_code.configurable_model"),
    ):
        assert _cache_endpoint_identity("openai:gpt-5.5") == "default"

    assert "Could not resolve the cache endpoint" in caplog.text


class TestModelSwap:
    """Cases where the middleware should swap the model."""

    def test_different_model_swapped(self) -> None:
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-5.5")
        request = _make_request(original, context=CLIContext(model="openai:gpt-5.5"))

        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=_make_model_result(override)):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is override
        assert request.model is original  # original unchanged

    def test_profile_overrides_forwarded_to_swapped_model(self) -> None:
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-5.5")
        profile_overrides = {"max_input_tokens": 180_000}
        request = _make_request(
            original,
            context=CLIContext(
                model="openai:gpt-5.5",
                profile_overrides=profile_overrides,
            ),
        )

        with patch(_PATCH_CREATE, return_value=_make_model_result(override)) as create:
            _mw.wrap_model_call(request, lambda _: _make_response())

        create.assert_called_once_with(
            "openai:gpt-5.5",
            profile_overrides=profile_overrides,
        )

    async def test_async_model_swapped(self) -> None:
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-5.5")
        request = _make_request(original, context=CLIContext(model="openai:gpt-5.5"))

        captured: list[ModelRequest] = []
        offloaded: list[
            tuple[Callable[..., object], tuple[object, ...], dict[str, object]]
        ] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        async def fake_to_thread(
            func: Callable[..., object], /, *args: object, **kwargs: object
        ) -> object:
            await asyncio.sleep(0)
            offloaded.append((func, args, kwargs))
            return func(*args, **kwargs)

        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)) as create,
            patch(
                "deepagents_code.configurable_model.asyncio.to_thread", fake_to_thread
            ),
        ):
            result = await _mw.awrap_model_call(request, handler)

        assert captured[0].model is override
        # Blocking calls must be offloaded: model construction, the
        # endpoint-identity lookup, and effective cache-param resolution (both
        # read the config and credential store). Doing any inline would trip
        # `blockbuster` on the server event loop.
        assert offloaded == [
            (create, ("openai:gpt-5.5",), {}),
            (_cache_endpoint_identity, ("openai:gpt-5.5",), {}),
            (_effective_cache_params, ("openai:gpt-5.5", None), {}),
        ]
        assert "_last_cache_params" in _checkpoint_update(result)

    async def test_async_profile_overrides_forwarded_to_swapped_model(self) -> None:
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-5.5")
        profile_overrides = {"max_input_tokens": 180_000}
        request = _make_request(
            original,
            context=CLIContext(
                model="openai:gpt-5.5",
                profile_overrides=profile_overrides,
            ),
        )

        async def handler(_: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            return _make_response()

        with patch(_PATCH_CREATE, return_value=_make_model_result(override)) as create:
            await _mw.awrap_model_call(request, handler)

        create.assert_called_once_with(
            "openai:gpt-5.5",
            profile_overrides=profile_overrides,
        )

    def test_class_path_provider_swapped(self) -> None:
        """Config-defined class_path provider resolves through create_model."""
        original = _make_model("claude-sonnet-4-6")
        custom = _make_model("my-model")
        request = _make_request(original, context=CLIContext(model="custom:my-model"))

        captured: list[ModelRequest] = []
        with patch(
            _PATCH_CREATE, return_value=_make_model_result(custom)
        ) as mock_create:
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is custom
        mock_create.assert_called_once_with("custom:my-model")

    def test_create_model_error_falls_back_to_original(self) -> None:
        """ModelConfigError falls back to original model instead of crashing."""
        from deepagents_code.model_config import ModelConfigError

        original = _make_model("claude-sonnet-4-6")
        original._get_ls_params.return_value = {"ls_provider": "anthropic"}
        request = _make_request(
            original,
            context=CLIContext(
                model="unknown:bad-model",
                model_params={"temperature": 0.7},
            ),
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, side_effect=ModelConfigError("no such provider")):
            result = _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is original
        assert captured[0].model_settings == {}
        # `_model_params` is deliberately absent rather than `None`: the
        # override never reached `_build_overrides`, so the params in effect
        # are unknown and the checkpoint's previous value must stand. Writing
        # `None` here would clear it while the app still holds its override,
        # pinning the cold-cache identity check to a permanent false
        # "model changed".
        assert _checkpoint_update(result) == {
            "_model_spec": "anthropic:claude-sonnet-4-6",
        }

    def test_model_policy_error_does_not_fall_back_to_original(self) -> None:
        """A blocked runtime switch propagates instead of using the old model."""
        from deepagents_code.model_config import ModelNotAllowedError

        original = _make_model("claude-sonnet-4-6")
        request = _make_request(
            original,
            context=CLIContext(model="openai:blocked"),
        )
        denial = ModelNotAllowedError(
            model_spec="openai:blocked",
            source="managed config",
            allowed_models=("anthropic:allowed",),
        )

        with (
            patch(_PATCH_CREATE, side_effect=denial),
            pytest.raises(ModelNotAllowedError, match="administrator-managed"),
        ):
            _mw.wrap_model_call(request, lambda _request: _make_response())

    async def test_async_model_policy_error_does_not_fall_back(self) -> None:
        """The asynchronous runtime-switch path propagates policy denials."""
        from deepagents_code.model_config import ModelNotAllowedError

        original = _make_model("claude-sonnet-4-6")
        request = _make_request(
            original,
            context=CLIContext(model="openai:blocked"),
        )
        denial = ModelNotAllowedError(
            model_spec="openai:blocked",
            source="config.toml",
            allowed_models=("anthropic:allowed",),
        )

        async def handler(_request: ModelRequest) -> ModelResponse[Any]:
            await asyncio.sleep(0)
            return _make_response()

        with (
            patch(_PATCH_CREATE, side_effect=denial),
            pytest.raises(ModelNotAllowedError, match=r"config\.toml"),
        ):
            await _mw.awrap_model_call(request, handler)

    def test_failed_override_records_original_as_cache_identity(self) -> None:
        """The cache model spec tracks the model that served the call."""
        from deepagents_code.model_config import ModelConfigError

        original = _make_model("claude-sonnet-4-6")
        original._get_ls_params.return_value = {"ls_provider": "anthropic"}
        request = _make_request(original, context=CLIContext(model="unknown:bad-model"))

        with patch(_PATCH_CREATE, side_effect=ModelConfigError("no such provider")):
            result = _mw.wrap_model_call(request, lambda _request: _make_response())

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        update = result.command.update
        assert isinstance(update, dict)
        assert update["_last_cache_model_spec"] == "anthropic:claude-sonnet-4-6"

    def test_successful_swap_records_resolved_model_spec(self) -> None:
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-5.5")
        request = _make_request(original, context=CLIContext(model="openai:gpt-5.5"))

        with patch(
            _PATCH_CREATE,
            return_value=_make_model_result(
                override,
                model_name="gpt-5.5",
                provider="openai",
            ),
        ):
            result = _mw.wrap_model_call(request, lambda _request: _make_response())

        assert _checkpoint_update(result) == {
            "_model_spec": "openai:gpt-5.5",
            "_model_params": None,
            "_last_cache_params": None,
        }


class TestAnthropicSettingsStripped:
    """Anthropic-specific model_settings stripped on cross-provider swap.

    When swapping from Anthropic to a non-Anthropic model, provider-specific
    settings like `cache_control` must be stripped to avoid TypeError on the
    target provider's API (e.g. OpenAI/Groq).
    """

    def test_cache_control_stripped_on_swap(self) -> None:
        override = _make_model("gpt-5.5")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-5.5"),
            model_settings={"cache_control": {"type": "ephemeral", "ttl": "5m"}},
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_code.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert "cache_control" not in captured[0].model_settings

    def test_cache_control_preserved_for_anthropic_swap(self) -> None:
        override = _make_model("claude-opus-4-6")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="anthropic:claude-opus-4-6"),
            model_settings={"cache_control": {"type": "ephemeral", "ttl": "5m"}},
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_code.configurable_model._is_anthropic_model",
                return_value=True,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings["cache_control"] == {
            "type": "ephemeral",
            "ttl": "5m",
        }

    def test_other_settings_preserved_on_swap(self) -> None:
        override = _make_model("gpt-5.5")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-5.5"),
            model_settings={
                "cache_control": {"type": "ephemeral"},
                "max_tokens": 2048,
            },
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_code.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings == {"max_tokens": 2048}

    async def test_async_cache_control_stripped(self) -> None:
        override = _make_model("gpt-5.5")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-5.5"),
            model_settings={"cache_control": {"type": "ephemeral"}},
        )
        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_code.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            await _mw.awrap_model_call(request, handler)

        assert "cache_control" not in captured[0].model_settings

    def test_swap_with_model_params_and_cache_control(self) -> None:
        """Stripping operates on the merged settings, not the original."""
        override = _make_model("gpt-5.5")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(
                model="openai:gpt-5.5",
                model_params={"temperature": 0.7},
            ),
            model_settings={
                "cache_control": {"type": "ephemeral"},
                "max_tokens": 2048,
            },
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_code.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings == {
            "max_tokens": 2048,
            "temperature": 0.7,
        }

    def test_only_cache_control_results_in_empty_settings(self) -> None:
        override = _make_model("gpt-5.5")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-5.5"),
            model_settings={"cache_control": {"type": "ephemeral"}},
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=_make_model_result(override)),
            patch(
                "deepagents_code.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings == {}


class TestFireworksSessionSettings:
    """Fireworks model calls receive session settings from the thread ID."""

    def _fireworks_model(self) -> MagicMock:
        model = _make_model("accounts/fireworks/models/kimi-k2p7-code")
        model._get_ls_params.return_value = {"ls_provider": "fireworks"}
        return model

    def test_fireworks_model_gets_session_settings(self) -> None:
        request = _make_request(
            self._fireworks_model(),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model is request.model
        assert captured[0].model_settings == {
            "prompt_cache_key": "thread-123",
            "extra_headers": {"x-session-affinity": "thread-123"},
        }

    def test_existing_headers_preserved_and_session_affinity_not_overwritten(
        self,
    ) -> None:
        request = _make_request(
            self._fireworks_model(),
            context=CLIContext(thread_id="thread-123"),
            model_settings={
                "extra_headers": {
                    "Authorization": "Bearer custom",
                    "X-Session-Affinity": "custom-session",
                }
            },
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {
            "extra_headers": {
                "Authorization": "Bearer custom",
                "X-Session-Affinity": "custom-session",
            }
        }

    def test_non_fireworks_non_openai_model_unchanged_with_thread_id(self) -> None:
        model = _make_model("gemini-3.6-flash")
        model._get_ls_params.return_value = {"ls_provider": "google_genai"}
        request = _make_request(
            model,
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0] is request

    def test_fireworks_swap_gets_session_settings(self) -> None:
        override = self._fireworks_model()
        request = _make_request(
            _make_model("gpt-5.5"),
            context=CLIContext(
                model="fireworks:accounts/fireworks/models/kimi-k2p7-code",
                thread_id="thread-123",
            ),
        )
        captured: list[ModelRequest] = []

        with patch(_PATCH_CREATE, return_value=_make_model_result(override)):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is override
        assert captured[0].model_settings == {
            "prompt_cache_key": "thread-123",
            "extra_headers": {"x-session-affinity": "thread-123"},
        }

    async def test_async_fireworks_model_gets_session_settings(self) -> None:
        request = _make_request(
            self._fireworks_model(),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        await _mw.awrap_model_call(request, handler)

        assert captured[0].model_settings == {
            "prompt_cache_key": "thread-123",
            "extra_headers": {"x-session-affinity": "thread-123"},
        }

    def test_empty_thread_id_skips_session_settings(self) -> None:
        """A blank thread ID must not inject empty session settings."""
        request = _make_request(
            self._fireworks_model(),
            context=CLIContext(thread_id=""),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0] is request

    def test_non_mapping_extra_headers_skips_injection(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed `extra_headers` leaves the request untouched and warns."""
        request = _make_request(
            self._fireworks_model(),
            context=CLIContext(thread_id="thread-123"),
            model_settings={"extra_headers": ["not", "a", "mapping"]},
        )
        captured: list[ModelRequest] = []

        with caplog.at_level(
            logging.WARNING, logger="deepagents_code.configurable_model"
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0] is request
        assert captured[0].model_settings == {"extra_headers": ["not", "a", "mapping"]}
        assert "extra_headers" in caplog.text

    def test_existing_prompt_cache_key_not_overwritten(self) -> None:
        request = _make_request(
            self._fireworks_model(),
            context=CLIContext(thread_id="thread-123"),
            model_settings={"prompt_cache_key": "custom-cache"},
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {
            "prompt_cache_key": "custom-cache",
            "extra_headers": {"x-session-affinity": "thread-123"},
        }

    def test_existing_session_affinity_header_case_insensitive(self) -> None:
        """A differently-cased session-affinity header is not duplicated."""
        request = _make_request(
            self._fireworks_model(),
            context=CLIContext(thread_id="thread-123"),
            model_settings={"extra_headers": {"X-Session-Affinity": "custom-session"}},
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {
            "extra_headers": {"X-Session-Affinity": "custom-session"},
        }

    def test_caller_model_settings_not_mutated(self) -> None:
        """Injection copies the caller's dicts instead of mutating in place."""
        original_headers = {"Authorization": "Bearer token"}
        model_settings = {"extra_headers": original_headers}
        request = _make_request(
            self._fireworks_model(),
            context=CLIContext(thread_id="thread-123"),
            model_settings=model_settings,
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert original_headers == {"Authorization": "Bearer token"}
        assert model_settings == {"extra_headers": {"Authorization": "Bearer token"}}
        assert captured[0].model_settings["extra_headers"] is not original_headers

    def test_openai_opt_out_does_not_affect_fireworks(self) -> None:
        """The OpenAI opt-out gates only the OpenAI branch, not Fireworks."""
        middleware = ConfigurableModelMiddleware(openai_prompt_cache_key=False)
        request = _make_request(
            self._fireworks_model(),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        middleware.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {
            "prompt_cache_key": "thread-123",
            "extra_headers": {"x-session-affinity": "thread-123"},
        }


class TestOpenAIPromptCacheKey:
    """OpenAI model calls receive a `prompt_cache_key` from the thread ID."""

    def test_openai_model_gets_prompt_cache_key(self) -> None:
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model is request.model
        assert captured[0].model_settings == {"prompt_cache_key": "thread-123"}

    def test_prompt_cache_key_merged_with_existing_settings(self) -> None:
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
            model_settings={"temperature": 0.5},
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {
            "temperature": 0.5,
            "prompt_cache_key": "thread-123",
        }

    def test_existing_prompt_cache_key_not_overwritten(self) -> None:
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
            model_settings={"prompt_cache_key": "custom-cache"},
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0] is request
        assert captured[0].model_settings == {"prompt_cache_key": "custom-cache"}

    def test_model_prompt_cache_key_not_overwritten(self) -> None:
        model = _make_model("gpt-5.6")
        model.model_kwargs = {"prompt_cache_key": "model-cache"}
        request = _make_request(
            model,
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0] is request
        assert captured[0].model_settings == {}

    def test_non_mapping_model_kwargs_still_injects(self) -> None:
        """A non-mapping `model_kwargs` is treated as no key present."""
        model = _make_model("gpt-5.6")
        model.model_kwargs = ["not", "a", "mapping"]
        request = _make_request(
            model,
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {"prompt_cache_key": "thread-123"}

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.openai.com/v1",
            "https://eu.api.openai.com/v1",
            "https://gateway.smith.langchain.com/openai/v1",
            "https://proxy.example/v1",
        ],
    )
    def test_any_openai_endpoint_gets_prompt_cache_key(self, base_url: str) -> None:
        """The key is attempted for every OpenAI-provider endpoint.

        Official, regional, the LangSmith gateway, and arbitrary OpenAI-compatible
        base URLs all report `ls_provider == "openai"`, so the additive
        `prompt_cache_key` is injected regardless of host. Endpoints that reject
        it opt out via `models.openai_prompt_cache_key`.
        """
        model = _make_model("gpt-5.6")
        model.root_client = SimpleNamespace(base_url=base_url)
        request = _make_request(
            model,
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {"prompt_cache_key": "thread-123"}

    def test_default_config_injects_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no override, construction resolves the opt-out to on (default).

        Exercises the real `models.openai_prompt_cache_key` resolution through
        `__init__` (env cleared by conftest, `config.toml` stubbed empty),
        pinning that the default is on end-to-end rather than only asserting the
        config helper in isolation.
        """
        from deepagents_code import config_manifest

        monkeypatch.setattr(config_manifest, "load_config_toml", dict)
        middleware = ConfigurableModelMiddleware()
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        middleware.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {"prompt_cache_key": "thread-123"}

    def test_opt_out_skips_prompt_cache_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The `models.openai_prompt_cache_key` opt-out suppresses injection.

        The opt-out is resolved once at construction, so patch the config lookup
        before building the middleware.
        """
        monkeypatch.setattr(
            "deepagents_code.config.is_openai_prompt_cache_key_enabled",
            lambda: False,
        )
        middleware = ConfigurableModelMiddleware()
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        middleware.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0] is request

    def test_explicit_opt_out_param_skips(self) -> None:
        """An explicit `openai_prompt_cache_key=False` bypasses config and skips."""
        middleware = ConfigurableModelMiddleware(openai_prompt_cache_key=False)
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        middleware.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0] is request

    async def test_async_explicit_opt_out_param_skips(self) -> None:
        """The async path honors the opt-out flag like the sync path.

        `awrap_model_call` threads `self._openai_prompt_cache_key` through
        `_apply_overrides_async` symmetrically with the sync path; this pins that
        wiring so a future edit dropping the kwarg on only one path is caught.
        """
        middleware = ConfigurableModelMiddleware(openai_prompt_cache_key=False)
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        await middleware.awrap_model_call(request, handler)

        assert captured[0] is request

    def test_opt_out_preserves_user_supplied_key(self) -> None:
        """Disabling injection still forwards a user-supplied key untouched."""
        middleware = ConfigurableModelMiddleware(openai_prompt_cache_key=False)
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
            model_settings={"prompt_cache_key": "custom-cache"},
        )
        captured: list[ModelRequest] = []

        middleware.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {"prompt_cache_key": "custom-cache"}

    def test_config_read_failure_defaults_to_injecting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed opt-out lookup falls back to injecting the key (fail-open).

        The resolver's fail-open runs at construction, so the raising config
        lookup must be patched before the middleware is built.
        """

        def _boom() -> bool:
            msg = "config exploded"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "deepagents_code.config.is_openai_prompt_cache_key_enabled",
            _boom,
        )
        middleware = ConfigurableModelMiddleware()
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        middleware.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {"prompt_cache_key": "thread-123"}

    def test_blocking_error_propagates_not_fail_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `BlockingError` from the config read is re-raised, never masked.

        Fail-open must not swallow a blocking-I/O-on-the-event-loop violation:
        that would hide the regression and silently defeat the opt-out. The
        resolver matches by class name (blockbuster is not a runtime dep), so a
        stand-in exception named `BlockingError` reproduces the path.
        """

        class BlockingError(Exception):
            """Stand-in matching the resolver's by-name check."""

        def _boom() -> bool:
            raise BlockingError

        monkeypatch.setattr(
            "deepagents_code.config.is_openai_prompt_cache_key_enabled",
            _boom,
        )
        with pytest.raises(BlockingError):
            ConfigurableModelMiddleware()

    def test_empty_thread_id_skips_prompt_cache_key(self) -> None:
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id=""),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0] is request

    def test_no_prompt_cache_key_without_thread_id(self) -> None:
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0] is request

    def test_openai_swap_gets_prompt_cache_key(self) -> None:
        base = _make_model("claude-sonnet-4-6")
        base._get_ls_params.return_value = {"ls_provider": "anthropic"}
        override = _make_model("gpt-5.6")
        request = _make_request(
            base,
            context=CLIContext(model="openai:gpt-5.6", thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        with patch(_PATCH_CREATE, return_value=_make_model_result(override)):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is override
        assert captured[0].model_settings == {"prompt_cache_key": "thread-123"}

    def test_swap_to_openai_injects_key_and_strips_cache_control(self) -> None:
        """Anthropic→OpenAI swap injects the key and strips `cache_control`.

        The real `/model` mid-thread scenario: a session running
        `AnthropicPromptCachingMiddleware` (which sets `cache_control`) switches
        to an OpenAI model. Injection and the Anthropic-only strip must both run
        in the same pass, leaving only the cache key — otherwise `cache_control`
        would reach the OpenAI SDK and raise `TypeError`.
        """
        base = _make_model("claude-sonnet-4-6")
        base._get_ls_params.return_value = {"ls_provider": "anthropic"}
        override = _make_model("gpt-5.6")
        request = _make_request(
            base,
            context=CLIContext(model="openai:gpt-5.6", thread_id="thread-123"),
            model_settings={"cache_control": {"type": "ephemeral"}},
        )
        captured: list[ModelRequest] = []

        with patch(_PATCH_CREATE, return_value=_make_model_result(override)):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is override
        assert captured[0].model_settings == {"prompt_cache_key": "thread-123"}

    def test_prompt_cache_key_layered_over_model_params(self) -> None:
        """The key is added on top of a `model_params` merge, not instead of it."""
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(
                model_params={"temperature": 0.7}, thread_id="thread-123"
            ),
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {
            "temperature": 0.7,
            "prompt_cache_key": "thread-123",
        }

    async def test_async_openai_model_gets_prompt_cache_key(self) -> None:
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
        )
        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        await _mw.awrap_model_call(request, handler)

        assert captured[0].model_settings == {"prompt_cache_key": "thread-123"}

    def test_caller_model_settings_not_mutated(self) -> None:
        """Injection copies the caller's dict instead of mutating in place."""
        model_settings = {"temperature": 0.5}
        request = _make_request(
            _make_model("gpt-5.6"),
            context=CLIContext(thread_id="thread-123"),
            model_settings=model_settings,
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert model_settings == {"temperature": 0.5}
        assert captured[0].model_settings is not model_settings


class TestIsFireworksModel:
    """Direct tests for the `_is_fireworks_model` helper."""

    def test_returns_true_for_fireworks(self) -> None:
        model = _make_model("accounts/fireworks/models/kimi-k2p7-code")
        model._get_ls_params.return_value = {"ls_provider": "fireworks"}
        assert _is_fireworks_model(model) is True

    def test_returns_false_for_non_fireworks(self) -> None:
        assert _is_fireworks_model(_make_model("gpt-5.5")) is False

    def test_returns_false_for_plain_object(self) -> None:
        assert _is_fireworks_model(object()) is False

    def test_returns_false_when_ls_params_returns_none(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        model._get_ls_params.return_value = None
        assert _is_fireworks_model(model) is False

    def test_returns_false_when_ls_provider_not_str(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        model._get_ls_params.return_value = {"ls_provider": 123}
        assert _is_fireworks_model(model) is False


class TestIsOpenAIModel:
    """Direct tests for the `_is_openai_model` helper."""

    def test_returns_true_for_openai(self) -> None:
        assert _is_openai_model(_make_model("gpt-5.6")) is True

    def test_returns_true_for_official_openai_endpoint(self) -> None:
        model = _make_model("gpt-5.6")
        model.root_client = SimpleNamespace(base_url="https://api.openai.com/v1")
        assert _is_openai_model(model) is True

    def test_returns_true_for_custom_openai_endpoint(self) -> None:
        """A custom base URL still resolves the OpenAI provider, so it is eligible."""
        model = _make_model("gpt-5.6")
        model.root_client = SimpleNamespace(base_url="https://proxy.example/v1")
        assert _is_openai_model(model) is True

    def test_returns_true_for_gateway_endpoint(self) -> None:
        """The LangSmith gateway is an OpenAI-provider endpoint and is eligible."""
        model = _make_model("gpt-5.6")
        model.root_client = SimpleNamespace(
            base_url="https://gateway.smith.langchain.com/openai/v1"
        )
        assert _is_openai_model(model) is True

    def test_returns_true_without_endpoint_metadata(self) -> None:
        """Eligibility depends only on the provider, not a discoverable base URL."""
        model = MagicMock(spec=BaseChatModel)
        model._get_ls_params.return_value = {"ls_provider": "openai"}
        assert _is_openai_model(model) is True

    def test_returns_false_for_non_openai(self) -> None:
        model = _make_model("accounts/fireworks/models/kimi-k2p7-code")
        model._get_ls_params.return_value = {"ls_provider": "fireworks"}
        assert _is_openai_model(model) is False

    def test_returns_false_for_plain_object(self) -> None:
        assert _is_openai_model(object()) is False

    def test_returns_false_when_ls_params_returns_none(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        model._get_ls_params.return_value = None
        assert _is_openai_model(model) is False

    def test_returns_false_when_ls_provider_not_str(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        model._get_ls_params.return_value = {"ls_provider": object()}
        assert _is_openai_model(model) is False


class TestIsAnthropicModel:
    """Direct tests for the `_is_anthropic_model` helper."""

    def test_returns_true_for_anthropic(self) -> None:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model_name="claude-sonnet-4-6")
        assert _is_anthropic_model(model) is True

    def test_returns_false_for_non_anthropic(self) -> None:
        assert _is_anthropic_model(_make_model("gpt-5.5")) is False

    def test_returns_false_for_plain_object(self) -> None:
        assert _is_anthropic_model(object()) is False

    def test_returns_false_when_ls_params_returns_none(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        model._get_ls_params.return_value = None
        assert _is_anthropic_model(model) is False

    def test_returns_false_when_ls_params_raises(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        model._get_ls_params.side_effect = RuntimeError("not initialized")
        assert _is_anthropic_model(model) is False


class TestModelParams:
    """Cases where model_params are merged into model_settings."""

    def test_params_merged(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model_params={"temperature": 0.7}),
        )
        captured: list[ModelRequest] = []
        result = _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model is request.model
        assert captured[0].model_settings == {"temperature": 0.7}
        assert _checkpoint_update(result) == {
            "_model_spec": "openai:claude-sonnet-4-6",
            "_model_params": {"temperature": 0.7},
            "_last_cache_params": None,
        }

    def test_reasoning_effort_reaches_model_settings(self) -> None:
        """`reasoning_effort` from `/effort` must survive intact to the model.

        Hermetic regression anchor for the effort-delivery path: a bug in the
        override plumbing could silently drop or duplicate `reasoning_effort`
        before it reaches the model constructor. Provider-specific translation
        of the value is LangChain's responsibility; this pins the deepagents
        contract that the resolved effort is carried into `model_settings`
        (and checkpointed for resume) without mutation.
        """
        request = _make_request(
            _make_model("claude-opus-4-5"),
            context=CLIContext(model_params={"reasoning_effort": "high"}),
        )
        captured: list[ModelRequest] = []
        result = _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {"reasoning_effort": "high"}
        assert _checkpoint_update(result) == {
            "_model_spec": "openai:claude-opus-4-5",
            "_model_params": {"reasoning_effort": "high"},
            "_last_cache_params": None,
        }

    def test_params_merge_preserves_existing(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model_params={"temperature": 0.5}),
            model_settings={"max_tokens": 2048},
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model_settings == {"max_tokens": 2048, "temperature": 0.5}

    def test_params_with_model_swap(self) -> None:
        override = _make_model("gpt-5.5")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(
                model="openai:gpt-5.5", model_params={"max_tokens": 1024}
            ),
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=_make_model_result(override)):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is override
        assert captured[0].model_settings == {"max_tokens": 1024}

    async def test_async_params(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model_params={"temperature": 0.3}),
        )
        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        await _mw.awrap_model_call(request, handler)
        assert captured[0].model_settings == {"temperature": 0.3}


class TestModelIdentityPatch:
    """System prompt Model Identity section is updated on model swap."""

    _OLD_PROMPT = (
        "Some preamble.\n\n---\n\n"
        "### Model Identity\n\n"
        "You are running as model `claude-opus-4-6` (provider: anthropic).\n"
        "Your context window is 200,000 tokens.\n\n"
        "### Skills Directory\n\nYour skills are stored at: `/tmp/skills`\n"
    )

    def test_identity_replaced_on_swap(self) -> None:
        override = _make_model("gpt-5.5")
        result = _make_model_result(
            override, model_name="gpt-5.5", provider="openai", context_limit=128_000
        )
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="openai:gpt-5.5"),
            system_prompt=self._OLD_PROMPT,
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=result):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        prompt = captured[0].system_prompt
        assert prompt is not None
        assert "`gpt-5.5`" in prompt
        assert "(provider: openai)" in prompt
        assert "128,000 tokens" in prompt
        assert "`claude-opus-4-6`" not in prompt
        # Surrounding content must survive the replacement
        assert "Some preamble." in prompt
        assert "### Skills Directory" in prompt
        assert "`/tmp/skills`" in prompt

    def test_no_identity_section_left_unchanged(self) -> None:
        """Prompt without identity section is not modified."""
        bare_prompt = "You are a helpful assistant.\n\n### Skills Directory\n"
        override = _make_model("gpt-5.5")
        result = _make_model_result(override, model_name="gpt-5.5", provider="openai")
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="openai:gpt-5.5"),
            system_prompt=bare_prompt,
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=result):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].system_prompt == bare_prompt

    def test_no_system_prompt_skips_patch(self) -> None:
        """When system_prompt is None, no patching is attempted."""
        override = _make_model("gpt-5.5")
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="openai:gpt-5.5"),
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=_make_model_result(override)):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is override

    def test_identity_at_end_of_prompt(self) -> None:
        """Identity section at the very end (no trailing ###) is still replaced."""
        prompt = (
            "Preamble.\n\n### Model Identity\n\nYou are running as model `old`.\n\n"
        )
        override = _make_model("gpt-5.5")
        result = _make_model_result(override, model_name="gpt-5.5", provider="openai")
        request = _make_request(
            _make_model("old"),
            context=CLIContext(model="openai:gpt-5.5"),
            system_prompt=prompt,
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=result):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        patched = captured[0].system_prompt
        assert patched is not None
        assert "`gpt-5.5`" in patched
        assert "`old`" not in patched
        assert "Preamble." in patched

    def test_identity_without_context_limit(self) -> None:
        result = build_model_identity_section("gpt-5.5", provider="openai")
        assert "`gpt-5.5`" in result
        assert "(provider: openai)" in result
        assert "context window" not in result

    def test_identity_without_provider(self) -> None:
        result = build_model_identity_section("local-llama", context_limit=4096)
        assert "`local-llama`" in result
        assert "provider" not in result
        assert "4,096 tokens" in result

    def test_identity_no_model_name(self) -> None:
        assert build_model_identity_section(None) == ""

    def test_modality_line_replaced_on_swap(self) -> None:
        """Swapping replaces old modality warning with the new model's."""
        prompt_with_modality = (
            "Preamble.\n\n### Model Identity\n\n"
            "You are running as model `deepseek-r1` (provider: deepseek).\n"
            "Your context window is 64,000 tokens.\n"
            "Audio, image, pdf, video input may not be available for this model.\n\n"
            "### Skills Directory\n\nSkills here.\n"
        )
        override = _make_model("claude-sonnet-4-6")
        result = _make_model_result(
            override,
            model_name="claude-sonnet-4-6",
            provider="anthropic",
            context_limit=200_000,
            unsupported_modalities=frozenset(),
        )
        request = _make_request(
            _make_model("deepseek-r1"),
            context=CLIContext(model="anthropic:claude-sonnet-4-6"),
            system_prompt=prompt_with_modality,
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=result):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        patched = captured[0].system_prompt
        assert patched is not None
        assert "`claude-sonnet-4-6`" in patched
        assert "200,000 tokens" in patched
        assert "may not be available" not in patched
        assert "`deepseek-r1`" not in patched
        assert "### Skills Directory" in patched


class TestRuntimeModelRetryBudget:
    """A `/model` switch must carry the explicit CLI retry budget."""

    @staticmethod
    def _switch(
        model_params: dict[str, Any], *, cli_max_retries: int | None = None
    ) -> tuple[MagicMock, ModelRequest]:
        """Drive a runtime model switch and return the `create_model` mock.

        Returns:
            The patched `create_model` mock and the request the handler saw.
        """
        request = _make_request(
            _make_model("gpt-5.5"),
            context=CLIContextSchema(
                model="anthropic:claude-sonnet-4-6", model_params=model_params
            ),
        )
        replacement = _make_model("claude-sonnet-4-6")
        replacement._get_ls_params.return_value = {"ls_provider": "anthropic"}
        captured: list[ModelRequest] = []
        middleware = ConfigurableModelMiddleware(
            openai_prompt_cache_key=True,
            cli_max_retries=cli_max_retries,
        )
        with patch(
            _PATCH_CREATE, return_value=_make_model_result(replacement)
        ) as create:
            middleware.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )
        return create, captured[0]

    def test_budget_survives_the_switch(self) -> None:
        """Without this the switched-to model reverts to the default budget.

        `--max-retries 0` or a raised budget would silently stop applying the
        moment the user ran `/model`.
        """
        create, _request = self._switch({}, cli_max_retries=2)
        _args, kwargs = create.call_args
        assert kwargs["cli_max_retries"] == 2

    def test_cli_budget_stays_separate_from_model_settings(self) -> None:
        """The explicit retry field is never a provider request parameter."""
        _create, request = self._switch({"top_p": 1}, cli_max_retries=2)
        assert request.model_settings == {"top_p": 1}

    def test_cli_budget_alone_leaves_settings_untouched(self) -> None:
        """A retry override alone must not create model settings."""
        _create, request = self._switch({}, cli_max_retries=2)
        assert not request.model_settings

    def test_real_params_still_merge(self) -> None:
        """The explicit retry field must not drop actual model overrides."""
        _create, request = self._switch({"top_p": 1}, cli_max_retries=2)
        assert (request.model_settings or {})["top_p"] == 1

    def test_absent_sentinel_sends_no_extra_kwargs(self) -> None:
        """A model with no flag set must resolve its own configured budget."""
        create, _request = self._switch({"top_p": 1})
        _args, kwargs = create.call_args
        assert "cli_max_retries" not in kwargs
