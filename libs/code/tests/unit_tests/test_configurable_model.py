"""Tests for ConfigurableModelMiddleware."""

import asyncio
import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from deepagents import HarnessProfile
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code._cli_context import CLIContext, CLIContextSchema
from deepagents_code._glm_5p2_profile import _SYSTEM_PROMPT_SUFFIX
from deepagents_code.agent import build_model_identity_section
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
_PATCH_HARNESS_PROFILE = (
    "deepagents.profiles.harness.harness_profiles._harness_profile_for_model"
)

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

    def test_baseten_glm_5p2_suffix_added_on_swap(self) -> None:
        override = _make_model("zai-org/GLM-5.2")
        result = _make_model_result(
            override,
            model_name="zai-org/GLM-5.2",
            provider="baseten",
        )
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="baseten:zai-org/GLM-5.2"),
            system_prompt=self._OLD_PROMPT,
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=result),
            patch(
                _PATCH_HARNESS_PROFILE,
                side_effect=[
                    HarnessProfile(),
                    HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
                ],
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        prompt = captured[0].system_prompt
        assert prompt is not None
        assert prompt.endswith(_SYSTEM_PROMPT_SUFFIX)
        assert prompt.count(_SYSTEM_PROMPT_SUFFIX) == 1

    def test_glm_5p2_suffix_removed_on_swap_away(self) -> None:
        override = _make_model("gpt-5.5")
        result = _make_model_result(
            override,
            model_name="gpt-5.5",
            provider="openai",
        )
        request = _make_request(
            _make_model("zai-org/GLM-5.2"),
            context=CLIContext(model="openai:gpt-5.5"),
            system_prompt=f"{self._OLD_PROMPT}\n\n{_SYSTEM_PROMPT_SUFFIX}",
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=result),
            patch(
                _PATCH_HARNESS_PROFILE,
                side_effect=[
                    HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
                    HarnessProfile(),
                ],
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        prompt = captured[0].system_prompt
        assert prompt is not None
        assert _SYSTEM_PROMPT_SUFFIX not in prompt

    def test_glm_5p2_suffix_removed_using_initial_spec(self) -> None:
        """Compiled spec removes suffix when current-model introspection is lossy."""
        override = _make_model("gpt-5.5")
        result = _make_model_result(
            override,
            model_name="gpt-5.5",
            provider="openai",
        )
        current = _make_model("z-ai/glm-5.2")
        current._get_ls_params.return_value = {"ls_provider": "openai"}
        request = _make_request(
            current,
            context=CLIContext(model="openai:gpt-5.5"),
            system_prompt=f"{self._OLD_PROMPT}\n\n{_SYSTEM_PROMPT_SUFFIX}",
        )
        middleware = ConfigurableModelMiddleware(
            initial_model_spec="openrouter:z-ai/glm-5.2"
        )

        def _profile_for_model(
            _model: BaseChatModel, spec: str | None
        ) -> HarnessProfile:
            if spec == "openrouter:z-ai/glm-5.2":
                return HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX)
            return HarnessProfile()

        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=result),
            patch(_PATCH_HARNESS_PROFILE, side_effect=_profile_for_model),
        ):
            middleware.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        prompt = captured[0].system_prompt
        assert prompt is not None
        assert _SYSTEM_PROMPT_SUFFIX not in prompt

    def test_glm_5p2_suffix_not_duplicated_on_provider_swap(self) -> None:
        override = _make_model("zai-org/GLM-5.2")
        result = _make_model_result(
            override,
            model_name="zai-org/GLM-5.2",
            provider="baseten",
        )
        request = _make_request(
            _make_model("accounts/fireworks/models/glm-5p2"),
            context=CLIContext(model="baseten:zai-org/GLM-5.2"),
            system_prompt=f"{self._OLD_PROMPT}\n\n{_SYSTEM_PROMPT_SUFFIX}",
        )
        captured: list[ModelRequest] = []
        profile = HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX)
        with (
            patch(_PATCH_CREATE, return_value=result),
            patch(_PATCH_HARNESS_PROFILE, side_effect=[profile, profile]),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        prompt = captured[0].system_prompt
        assert prompt is not None
        assert prompt.count(_SYSTEM_PROMPT_SUFFIX) == 1

    def test_swap_to_glm_applies_real_registered_suffix(self) -> None:
        """End-to-end: real registration + unmocked lookup + spec computation.

        Every other transition test patches `_harness_profile_for_model`, so
        the glue that computes `target_spec` and resolves it against the real
        registry is otherwise untested. Here the profile is genuinely
        registered and the lookup runs unmocked, so a wrong `target_spec` (or a
        registration that never matches) would drop the suffix and fail.
        """
        from deepagents.profiles.harness import harness_profiles

        from deepagents_code import _glm_5p2_profile

        saved = dict(harness_profiles._HARNESS_PROFILES)
        saved_flag = _glm_5p2_profile._registered
        _glm_5p2_profile._registered = False
        try:
            _glm_5p2_profile._ensure_glm_5p2_profile_registered()
            override = _make_model("zai-org/GLM-5.2")
            result = _make_model_result(
                override,
                model_name="zai-org/GLM-5.2",
                provider="baseten",
            )
            request = _make_request(
                _make_model("claude-opus-4-6"),
                context=CLIContext(model="baseten:zai-org/GLM-5.2"),
                system_prompt=self._OLD_PROMPT,
            )
            captured: list[ModelRequest] = []
            # NOTE: _PATCH_HARNESS_PROFILE is deliberately NOT patched here.
            with patch(_PATCH_CREATE, return_value=result):
                _mw.wrap_model_call(
                    request, lambda r: (captured.append(r), _make_response())[1]
                )
        finally:
            harness_profiles._HARNESS_PROFILES.clear()
            harness_profiles._HARNESS_PROFILES.update(saved)
            _glm_5p2_profile._registered = saved_flag

        prompt = captured[0].system_prompt
        assert prompt is not None
        assert prompt.endswith(_SYSTEM_PROMPT_SUFFIX)
        assert prompt.count(_SYSTEM_PROMPT_SUFFIX) == 1

    def test_structural_profile_change_warns_restart_required(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        override = _make_model("target-model")
        result = _make_model_result(
            override,
            model_name="target-model",
            provider="test-provider",
        )
        request = _make_request(
            _make_model("source-model"),
            context=CLIContext(model="test-provider:target-model"),
            system_prompt=self._OLD_PROMPT,
        )
        with (
            patch(_PATCH_CREATE, return_value=result),
            patch(
                _PATCH_HARNESS_PROFILE,
                side_effect=[
                    HarnessProfile(),
                    HarnessProfile(excluded_tools=frozenset({"execute"})),
                ],
            ),
            caplog.at_level(
                logging.WARNING, logger="deepagents_code.configurable_model"
            ),
        ):
            _mw.wrap_model_call(request, lambda _request: _make_response())

        assert "restart with this model" in caplog.text
        # The warning names the specific field that cannot be applied so the
        # user knows what steering is missing until they restart.
        assert "excluded_tools" in caplog.text

    def test_suffix_only_swap_does_not_warn_restart(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A runtime-safe suffix swap must not emit the restart warning."""
        override = _make_model("zai-org/GLM-5.2")
        result = _make_model_result(
            override,
            model_name="zai-org/GLM-5.2",
            provider="baseten",
        )
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="baseten:zai-org/GLM-5.2"),
            system_prompt=self._OLD_PROMPT,
        )
        with (
            patch(_PATCH_CREATE, return_value=result),
            patch(
                _PATCH_HARNESS_PROFILE,
                side_effect=[
                    HarnessProfile(),
                    HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
                ],
            ),
            caplog.at_level(
                logging.WARNING, logger="deepagents_code.configurable_model"
            ),
        ):
            _mw.wrap_model_call(request, lambda _request: _make_response())

        assert "restart with this model" not in caplog.text

    def test_suffix_swap_sets_override_when_identity_unchanged(self) -> None:
        """The override gate keys off the suffix swap, not just identity.

        Both the model name/provider are unchanged here, so the identity
        substitution is a no-op; only the profile suffix changes. The override
        must still be applied, proving the final gate accounts for the suffix
        transition rather than relying on the identity patch.
        """
        override = _make_model("zai-org/GLM-5.2")
        result = _make_model_result(
            override,
            model_name="zai-org/GLM-5.2",
            provider="baseten",
        )
        # `identity` already ends with a blank-line separator, so append the
        # next heading directly; this keeps the identity substitution a true
        # no-op (matched region == replacement) and isolates the suffix swap.
        identity = build_model_identity_section("zai-org/GLM-5.2", provider="baseten")
        stable_prompt = f"{identity}### Skills Directory\n`/tmp/skills`"
        request = _make_request(
            override,
            context=CLIContext(model="baseten:zai-org/GLM-5.2"),
            system_prompt=stable_prompt,
        )
        captured: list[ModelRequest] = []
        with (
            patch(_PATCH_CREATE, return_value=result),
            patch(
                _PATCH_HARNESS_PROFILE,
                side_effect=[
                    HarnessProfile(),
                    HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
                ],
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        prompt = captured[0].system_prompt
        assert prompt is not None
        assert prompt.endswith(_SYSTEM_PROMPT_SUFFIX)

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
