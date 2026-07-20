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
    _get_context,
    _is_anthropic_model,
    _is_fireworks_model,
    _is_google_genai_model,
    _is_openai_model,
)
from deepagents_code.model_config import ModelConfigError


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
            await _mw.awrap_model_call(request, handler)

        assert captured[0].model is override
        assert offloaded == [(create, ("openai:gpt-5.5",), {})]

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

    def test_non_fireworks_non_openai_model_unchanged_with_thread_id(self) -> None:
        model = _make_model("gemini-3.5-flash")
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

    def test_custom_openai_endpoint_skips_prompt_cache_key(self) -> None:
        model = _make_model("gpt-5.6")
        model.root_client = SimpleNamespace(base_url="https://proxy.example/v1")
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

    def test_regional_openai_endpoint_gets_prompt_cache_key(self) -> None:
        model = _make_model("gpt-5.6")
        model.root_client = SimpleNamespace(base_url="https://eu.api.openai.com/v1")
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
            "https://api.openai.com.example/v1",
            "https://eu.api.openai.com.example/v1",
            "https://fake-api.openai.com/v1",
        ],
    )
    def test_lookalike_openai_endpoint_skips_prompt_cache_key(
        self, base_url: str
    ) -> None:
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

        assert captured[0] is request
        assert captured[0].model_settings == {}

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

    def test_returns_false_for_custom_openai_endpoint(self) -> None:
        model = _make_model("gpt-5.6")
        model.root_client = SimpleNamespace(base_url="https://proxy.example/v1")
        assert _is_openai_model(model) is False

    def test_returns_false_without_endpoint_metadata(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        model._get_ls_params.return_value = {"ls_provider": "openai"}
        assert _is_openai_model(model) is False

    def test_falls_back_to_openai_api_base_for_official(self) -> None:
        model = _make_model("gpt-5.6")
        model.root_client = None
        model.openai_api_base = "https://api.openai.com/v1"
        assert _is_openai_model(model) is True

    def test_falls_back_to_openai_api_base_for_custom(self) -> None:
        model = _make_model("gpt-5.6")
        model.root_client = None
        model.openai_api_base = "https://proxy.example/v1"
        assert _is_openai_model(model) is False

    def test_returns_false_for_malformed_base_url(self) -> None:
        model = _make_model("gpt-5.6")
        model.root_client = SimpleNamespace(base_url="http://[::1")
        assert _is_openai_model(model) is False

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


def _make_gemini_model(name: str = "gemini-3.1-pro-preview") -> MagicMock:
    """Create a mock BaseChatModel reporting the google_genai provider."""
    model = _make_model(name)
    model._get_ls_params.return_value = {"ls_provider": "google_genai"}
    return model


class TestGoogleGenAINormalization:
    """Gemini requests are normalized so trivial turns don't 400."""

    def test_single_word_message_produces_valid_request(self) -> None:
        """A bare 'hi' turn reaches the provider with a non-empty content part."""
        request = _make_request(_make_gemini_model())
        request.messages[0].content = "hi"
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].messages[0].content == "hi"

    def test_empty_content_message_gets_placeholder(self) -> None:
        """An empty-content message is padded so Gemini does not reject it."""
        request = _make_request(_make_gemini_model())
        request.messages[0].content = ""
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].messages[0].content == " "

    def test_invalid_generation_config_dropped(self) -> None:
        """Out-of-range generation-config values are stripped before dispatch."""
        request = _make_request(
            _make_gemini_model(),
            model_settings={"temperature": 5, "max_tokens": -1, "top_p": 0.9},
        )
        captured: list[ModelRequest] = []

        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        settings = captured[0].model_settings
        assert settings is not None
        assert "temperature" not in settings
        assert "max_tokens" not in settings
        assert settings["top_p"] == pytest.approx(0.9)

    def test_http_400_surfaced_as_model_config_error(self) -> None:
        """A Gemini 400 becomes an actionable ModelConfigError naming the model."""
        request = _make_request(_make_gemini_model("gemini-3.1-pro-preview"))

        def _raise(_request: ModelRequest) -> ModelResponse[Any]:
            msg = "ChatGoogleGenerativeAIError: 400 Bad Request"
            raise RuntimeError(msg)

        with pytest.raises(ModelConfigError) as excinfo:
            _mw.wrap_model_call(request, _raise)

        assert "gemini-3.1-pro-preview" in str(excinfo.value)

    def test_non_gemini_400_propagates_unchanged(self) -> None:
        """A 400 from a non-Gemini model is not rewrapped."""
        request = _make_request(_make_model("gpt-5.5"))

        def _raise(_request: ModelRequest) -> ModelResponse[Any]:
            msg = "400 Bad Request"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError):
            _mw.wrap_model_call(request, _raise)

    def test_is_google_genai_model_detection(self) -> None:
        assert _is_google_genai_model(_make_gemini_model()) is True
        assert _is_google_genai_model(_make_model("gpt-5.5")) is False
