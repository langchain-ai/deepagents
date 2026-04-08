"""Tests for ConfigurableModelMiddleware."""

import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_cli.configurable_model import (
    CLIContext,
    ConfigurableModelMiddleware,
    _is_anthropic_model,
)


def _make_model(name: str) -> MagicMock:
    """Create a mock BaseChatModel with model_name set."""
    model = MagicMock(spec=BaseChatModel)
    model.model_name = name
    model.model_dump.return_value = {"model_name": name}
    return model


def _make_request(
    model: BaseChatModel,
    context: CLIContext | None = None,
    model_settings: dict[str, Any] | None = None,
) -> ModelRequest:
    """Create a ModelRequest with a runtime that carries CLIContext."""
    runtime = SimpleNamespace(context=context)
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hi")],
        tools=[],
        runtime=cast("Any", runtime),
        model_settings=model_settings,
    )


def _make_response() -> ModelResponse[Any]:
    """Create a minimal model response for handler mocks."""
    return ModelResponse(result=[AIMessage(content="response")])


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


_PATCH_CREATE = "deepagents_cli.config.create_model"

_mw = ConfigurableModelMiddleware()


class TestNoOverride:
    """Cases where the middleware should pass the request through unchanged."""

    def test_no_context(self) -> None:
        request = _make_request(_make_model("claude-sonnet-4-6"), context=None)
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0].model is request.model

    def test_empty_context(self) -> None:
        request = _make_request(_make_model("claude-sonnet-4-6"), context=CLIContext())
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request

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
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="anthropic:claude-sonnet-4-6"),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request

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
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0] is request


class TestModelSwap:
    """Cases where the middleware should swap the model."""

    def test_different_model_swapped(self) -> None:
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(original, context=CLIContext(model="openai:gpt-4o"))

        captured: list[ModelRequest] = []
        with patch(
            "deepagents_cli.configurable_model.resolve_model", return_value=override
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model is override
        assert request.model is original  # original unchanged

    async def test_async_model_swapped(self) -> None:
        original = _make_model("claude-sonnet-4-6")
        override = _make_model("gpt-4o")
        request = _make_request(original, context=CLIContext(model="openai:gpt-4o"))

        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        with patch(
            "deepagents_cli.configurable_model.resolve_model", return_value=override
        ):
            await _mw.awrap_model_call(request, handler)

        assert captured[0].model is override


class TestAnthropicSettingsStripped:
    """Anthropic-specific model_settings stripped on cross-provider swap.

    When swapping from Anthropic to a non-Anthropic model, provider-specific
    settings like `cache_control` must be stripped to avoid TypeError on the
    target provider's API (e.g. OpenAI/Groq).
    """

    def test_cache_control_stripped_on_swap(self) -> None:
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
            model_settings={"cache_control": {"type": "ephemeral", "ttl": "5m"}},
        )
        captured: list[ModelRequest] = []
        with (
            patch(
                "deepagents_cli.configurable_model.resolve_model", return_value=override
            ),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
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
            patch(
                "deepagents_cli.configurable_model.resolve_model", return_value=override
            ),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
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
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
            model_settings={
                "cache_control": {"type": "ephemeral"},
                "max_tokens": 2048,
            },
        )
        captured: list[ModelRequest] = []
        with (
            patch(
                "deepagents_cli.configurable_model.resolve_model", return_value=override
            ),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings == {"max_tokens": 2048}

    async def test_async_cache_control_stripped(self) -> None:
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
            model_settings={"cache_control": {"type": "ephemeral"}},
        )
        captured: list[ModelRequest] = []

        async def handler(r: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            captured.append(r)
            return _make_response()

        with (
            patch(
                "deepagents_cli.configurable_model.resolve_model", return_value=override
            ),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            await _mw.awrap_model_call(request, handler)

        assert "cache_control" not in captured[0].model_settings

    def test_swap_with_model_params_and_cache_control(self) -> None:
        """Stripping operates on the merged settings, not the original."""
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(
                model="openai:gpt-4o",
                model_params={"temperature": 0.7},
            ),
            model_settings={
                "cache_control": {"type": "ephemeral"},
                "max_tokens": 2048,
            },
        )
        captured: list[ModelRequest] = []
        with (
            patch(
                "deepagents_cli.configurable_model.resolve_model", return_value=override
            ),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
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
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
            model_settings={"cache_control": {"type": "ephemeral"}},
        )
        captured: list[ModelRequest] = []
        with (
            patch(
                "deepagents_cli.configurable_model.resolve_model", return_value=override
            ),
            patch(
                "deepagents_cli.configurable_model._is_anthropic_model",
                return_value=False,
            ),
        ):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        assert captured[0].model_settings == {}


class TestIsAnthropicModel:
    """Direct tests for the `_is_anthropic_model` helper."""

    def test_returns_true_for_anthropic(self) -> None:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model_name="claude-sonnet-4-6")
        assert _is_anthropic_model(model) is True

    def test_returns_false_for_non_anthropic(self) -> None:
        assert _is_anthropic_model(_make_model("gpt-4o")) is False

    def test_returns_false_when_import_missing(self) -> None:
        with patch.dict("sys.modules", {"langchain_anthropic": None}):
            assert _is_anthropic_model(_make_model("anything")) is False


class TestModelParams:
    """Cases where model_params are merged into model_settings."""

    def test_params_merged(self) -> None:
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(model_params={"temperature": 0.7}),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )

        assert captured[0].model is request.model
        assert captured[0].model_settings == {"temperature": 0.7}

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
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-sonnet-4-6"),
            context=CLIContext(
                model="openai:gpt-4o", model_params={"max_tokens": 1024}
            ),
        )
        captured: list[ModelRequest] = []
        with patch(
            "deepagents_cli.configurable_model.resolve_model", return_value=override
        ):
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
        override = _make_model("gpt-4o")
        result = _make_model_result(
            override, model_name="gpt-4o", provider="openai", context_limit=128_000
        )
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
            system_prompt=self._OLD_PROMPT,
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=result):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        prompt = captured[0].system_prompt
        assert prompt is not None
        assert "`gpt-4o`" in prompt
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
        override = _make_model("gpt-4o")
        result = _make_model_result(override, model_name="gpt-4o", provider="openai")
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
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
        override = _make_model("gpt-4o")
        request = _make_request(
            _make_model("claude-opus-4-6"),
            context=CLIContext(model="openai:gpt-4o"),
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
        override = _make_model("gpt-4o")
        result = _make_model_result(override, model_name="gpt-4o", provider="openai")
        request = _make_request(
            _make_model("old"),
            context=CLIContext(model="openai:gpt-4o"),
            system_prompt=prompt,
        )
        captured: list[ModelRequest] = []
        with patch(_PATCH_CREATE, return_value=result):
            _mw.wrap_model_call(
                request, lambda r: (captured.append(r), _make_response())[1]
            )

        patched = captured[0].system_prompt
        assert patched is not None
        assert "`gpt-4o`" in patched
        assert "`old`" not in patched
        assert "Preamble." in patched

    def test_identity_without_context_limit(self) -> None:
        result = build_model_identity_section("gpt-4o", provider="openai")
        assert "`gpt-4o`" in result
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
