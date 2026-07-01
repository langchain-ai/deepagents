"""Tests for `/effort` reasoning effort handling."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, Mock

import pytest

from deepagents_code.app import DeepAgentsApp
from deepagents_code.config import settings
from deepagents_code.reasoning_effort import (
    current_effort_from_model_params,
    merge_effort_model_params,
    model_params_for_effort,
    supported_efforts_for_model,
    without_effort_model_params,
)
from deepagents_code.widgets.effort_selector import EffortSelectorScreen


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    original_name = settings.model_name
    original_provider = settings.model_provider
    yield
    settings.model_name = original_name
    settings.model_provider = original_provider


@pytest.mark.parametrize(
    ("model_spec", "efforts"),
    [
        ("openai:gpt-5.5", ("none", "low", "medium", "high", "xhigh")),
        ("openai_codex:gpt-5.5", ("none", "low", "medium", "high", "xhigh")),
        ("anthropic:claude-sonnet-5", ("low", "medium", "high", "xhigh", "max")),
        ("google_genai:gemini-3.5-flash", ("minimal", "low", "medium", "high")),
        (
            "fireworks:accounts/fireworks/models/deepseek-v4-pro",
            ("none", "low", "medium", "high", "xhigh", "max"),
        ),
        (
            "fireworks:accounts/fireworks/models/kimi-k2p7-code",
            ("low", "medium", "high"),
        ),
        ("fireworks:accounts/fireworks/models/glm-5p2", ("none", "high", "max")),
    ],
)
def test_supported_efforts_for_model(model_spec: str, efforts: tuple[str, ...]) -> None:
    assert supported_efforts_for_model(model_spec) == efforts


def test_model_params_for_effort_maps_provider_kwargs() -> None:
    assert model_params_for_effort("openai:gpt-5.5", "high") == {
        "reasoning": {"effort": "high", "summary": "auto"}
    }
    assert model_params_for_effort("anthropic:claude-opus-4-8", "xhigh") == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "effort": "xhigh",
    }
    assert model_params_for_effort("google_genai:gemini-3.5-flash", "minimal") == {
        "thinking_level": "minimal"
    }
    assert model_params_for_effort(
        "fireworks:accounts/fireworks/models/deepseek-v4-pro", "max"
    ) == {"model_kwargs": {"reasoning_effort": "max"}}


def test_model_params_for_effort_rejects_unsupported_effort() -> None:
    assert (
        model_params_for_effort(
            "fireworks:accounts/fireworks/models/kimi-k2p7-code", "max"
        )
        is None
    )
    assert model_params_for_effort("ollama:llama3.1", "high") is None


def test_merge_and_clear_effort_model_params_preserves_unrelated_params() -> None:
    merged = merge_effort_model_params(
        {"temperature": 0.2, "model_kwargs": {"top_p": 0.9}},
        {"model_kwargs": {"reasoning_effort": "high"}},
    )

    assert merged == {
        "temperature": 0.2,
        "model_kwargs": {"top_p": 0.9, "reasoning_effort": "high"},
    }
    assert (
        current_effort_from_model_params(
            "fireworks:accounts/fireworks/models/deepseek-v4-pro", merged
        )
        == "high"
    )
    assert without_effort_model_params(merged) == {
        "temperature": 0.2,
        "model_kwargs": {"top_p": 0.9},
    }


async def test_effort_command_sets_current_model_params() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort high")

    assert app._model_params_override == {
        "reasoning": {"effort": "high", "summary": "auto"}
    }
    assert app._mount_message.await_count == 2  # ty: ignore[unresolved-attribute]


async def test_effort_command_without_args_opens_selector() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    app._model_params_override = {"reasoning": {"effort": "medium"}}
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort")

    app.push_screen.assert_called_once()  # ty: ignore[unresolved-attribute]
    screen = app.push_screen.call_args.args[0]  # ty: ignore[unresolved-attribute]
    assert isinstance(screen, EffortSelectorScreen)
    assert screen._model_spec == "openai:gpt-5.5"
    assert screen._efforts == ("none", "low", "medium", "high", "xhigh")
    assert screen._current_effort == "medium"
    app._mount_message.assert_not_awaited()  # ty: ignore[unresolved-attribute]


async def test_effort_command_clear_removes_only_effort_params() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._model_params_override = {
        "temperature": 0.2,
        "reasoning": {"effort": "high", "summary": "auto"},
    }
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort clear")

    assert app._model_params_override == {"temperature": 0.2}


async def test_effort_command_updates_status_bar_effort() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._status_bar = Mock()  # ty: ignore
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort xhigh")

    app._status_bar.set_model.assert_called_once_with(  # ty: ignore[unresolved-attribute]
        provider="openai",
        model="gpt-5.5",
        effort="xhigh",
    )


async def test_effort_command_rejects_unsupported_effort() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    settings.model_provider = "fireworks"
    settings.model_name = "accounts/fireworks/models/kimi-k2p7-code"

    await app._handle_effort_command("/effort max")

    assert app._model_params_override is None
    assert app._mount_message.await_count == 2  # ty: ignore[unresolved-attribute]
