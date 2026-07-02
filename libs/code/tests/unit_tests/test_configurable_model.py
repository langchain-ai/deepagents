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
from deepagents_code.configurable_model import (
    ConfigurableModelMiddleware,
    _get_context,
    _is_anthropic_model,
    _is_fireworks_model,
)


def _make_model(name: str) -> MagicMock:
    """Create a mock BaseChatModel with model_name set."""
    model = MagicMock(spec=BaseChatModel)
    model.model_name = name
    model.model_dump.return_value = {"model_name": name}
    model._get_ls_params.return_value = {"ls_provider": "openai"}
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
    return result.command.update


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

_mw = ConfigurableModelMiddleware()


class TestCheckpointPersistence:
    """Tests for private resume-state checkpoint updates."""

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
        }


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
            await _mw.awrap_model_call(request, handler)

        assert captured[0].model is override
        assert offloaded == [(create, ("openai:gpt-5.5",), {})]

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
        assert _checkpoint_update(result) == {
            "_model_spec": "anthropic:claude-sonnet-4-6",
            "_model_params": None,
        }

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

    def test_non_fireworks_model_unchanged_with_thread_id(self) -> None:
        request = _make_request(
            _make_model("gpt-5.5"),
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


class TestModelIdentityReRender:
    """On model swap, the whole system prompt is re-rendered via `prompt_for`."""

    def test_prompt_rerendered_on_swap(self) -> None:
        """A model swap replaces the system prompt with `prompt_for`'s output."""
        override = _make_model("gpt-5.5")
        result = _make_model_result(
            override, model_name="gpt-5.5", provider="openai", context_limit=128_000
        )
        mw = ConfigurableModelMiddleware(
            prompt_for=lambda mr: (
                f"BASE\n\nrunning as `{mr.model_name}` ({mr.provider})"
            )
        )
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="openai:gpt-5.5"),
            system_prompt="stale prompt naming claude-opus-4-6",
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=result):
            mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].system_prompt == "BASE\n\nrunning as `gpt-5.5` (openai)"
        assert captured[0].model is override

    def test_no_prompt_for_leaves_prompt_unchanged(self) -> None:
        """Without `prompt_for` (subagents), the model swaps but the prompt does not."""
        override = _make_model("gpt-5.5")
        result = _make_model_result(override, model_name="gpt-5.5", provider="openai")
        mw = ConfigurableModelMiddleware()  # prompt_for=None
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="openai:gpt-5.5"),
            system_prompt="unchanged prompt",
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=result):
            mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].system_prompt == "unchanged prompt"
        assert captured[0].model is override

    def test_prompt_for_not_invoked_without_swap(self) -> None:
        """When the model doesn't change, `prompt_for` is not called."""
        prompt_for = MagicMock()
        mw = ConfigurableModelMiddleware(prompt_for=prompt_for)
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(),
            system_prompt="original",
        )
        captured: list[ModelRequest] = []
        mw.wrap_model_call(request, lambda r: (captured.append(r), _make_response())[1])

        prompt_for.assert_not_called()
        assert captured[0].system_prompt == "original"
