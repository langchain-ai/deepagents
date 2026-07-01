"""Tests for `/effort` reasoning effort handling."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import OptionList

from deepagents_code.app import DeepAgentsApp
from deepagents_code.config import settings
from deepagents_code.reasoning_effort import (
    current_effort_from_model_params,
    default_effort_for_model,
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
        ("anthropic:claude-opus-4-8", ("low", "medium", "high", "xhigh", "max")),
        ("anthropic:claude-opus-4-6", ("low", "medium", "high", "max")),
        ("anthropic:claude-opus-4-5", ("low", "medium", "high")),
        ("anthropic:claude-sonnet-5", ("low", "medium", "high", "xhigh", "max")),
        ("anthropic:claude-sonnet-4-6", ("low", "medium", "high", "max")),
        ("anthropic:claude-sonnet-4-5", ()),
        ("google_genai:gemini-3.5-flash", ("low", "medium", "high")),
        ("google_genai:gemini-3.1-pro-preview", ("low", "medium", "high")),
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


@pytest.mark.parametrize(
    ("model_spec", "default"),
    [
        ("openai:gpt-5.5", "medium"),
        ("openai_codex:gpt-5.5", "medium"),
        ("anthropic:claude-opus-4-8", "high"),
        ("anthropic:claude-sonnet-4-6", "high"),
        ("anthropic:claude-sonnet-4-5", None),
        ("google_genai:gemini-3.5-flash", "medium"),
        ("google_genai:gemini-3.1-pro-preview", "high"),
        ("fireworks:accounts/fireworks/models/deepseek-v4-pro", "high"),
        ("fireworks:accounts/fireworks/models/glm-5p2", "max"),
        ("fireworks:accounts/fireworks/models/kimi-k2p7-code", None),
        ("ollama:llama3.1", None),
    ],
)
def test_default_effort_for_model(model_spec: str, default: str | None) -> None:
    assert default_effort_for_model(model_spec) == default


def test_model_params_for_effort_maps_provider_kwargs() -> None:
    assert model_params_for_effort("openai:gpt-5.5", "high") == {
        "reasoning": {"effort": "high", "summary": "auto"}
    }
    assert model_params_for_effort("anthropic:claude-opus-4-8", "xhigh") == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "effort": "xhigh",
    }
    assert model_params_for_effort("google_genai:gemini-3.5-flash", "low") == {
        "thinking_level": "low"
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
    assert screen._default_effort == "medium"
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


@pytest.mark.parametrize("token", ["clear", "--clear", "reset"])
async def test_effort_command_clear_aliases(token: str) -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._model_params_override = {
        "temperature": 0.2,
        "reasoning": {"effort": "high", "summary": "auto"},
    }
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command(f"/effort {token}")

    assert app._model_params_override == {"temperature": 0.2}


async def test_effort_selector_reports_no_model_configured() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    settings.model_provider = None
    settings.model_name = None

    await app._handle_effort_command("/effort")

    app.push_screen.assert_not_called()  # ty: ignore[unresolved-attribute]
    assert app._mount_message.await_count == 2  # ty: ignore[unresolved-attribute]


async def test_effort_command_reports_not_configurable_model() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    settings.model_provider = "anthropic"
    settings.model_name = "claude-sonnet-4-5"

    await app._handle_effort_command("/effort high")

    assert app._model_params_override is None
    assert app._mount_message.await_count == 2  # ty: ignore[unresolved-attribute]


def test_without_effort_clears_anthropic_thinking_and_effort() -> None:
    effort_params = model_params_for_effort("anthropic:claude-opus-4-8", "xhigh")
    assert effort_params is not None
    params = merge_effort_model_params({"temperature": 0.3}, effort_params)
    assert params["effort"] == "xhigh"
    assert "thinking" in params
    assert without_effort_model_params(params) == {"temperature": 0.3}


def test_without_effort_clears_google_thinking_level() -> None:
    effort_params = model_params_for_effort("google_genai:gemini-3.5-flash", "low")
    assert effort_params is not None
    assert without_effort_model_params(effort_params) is None


@pytest.mark.parametrize(
    ("model_spec", "effort"),
    [
        ("openai:gpt-5.5", "none"),
        ("openai:gpt-5.5", "high"),
        ("anthropic:claude-opus-4-8", "xhigh"),
        ("google_genai:gemini-3.5-flash", "low"),
        ("fireworks:accounts/fireworks/models/deepseek-v4-pro", "max"),
    ],
)
def test_effort_params_round_trip_clears_to_none(model_spec: str, effort: str) -> None:
    """The clear-set must strip exactly what `model_params_for_effort` writes."""
    effort_params = model_params_for_effort(model_spec, effort)
    assert effort_params is not None
    merged = merge_effort_model_params(None, effort_params)
    assert without_effort_model_params(merged) is None


class _EffortSelectorHost(App[None]):
    """Minimal host app for mounting `EffortSelectorScreen` in tests."""


@pytest.mark.parametrize(
    ("current_effort", "default_effort", "expected_index"),
    [("medium", "low", 2), (None, "medium", 2), (None, None, 0), ("bogus", None, 0)],
)
async def test_effort_selector_highlights_current(
    current_effort: str | None, default_effort: str | None, expected_index: int
) -> None:
    app = _EffortSelectorHost()
    async with app.run_test() as pilot:
        await app.push_screen(
            EffortSelectorScreen(
                model_spec="openai:gpt-5.5",
                efforts=("none", "low", "medium", "high", "xhigh"),
                current_effort=current_effort,
                default_effort=default_effort,
            )
        )
        await pilot.pause()
        option_list = app.screen.query_one("#effort-options", OptionList)
        assert option_list.highlighted == expected_index


async def test_effort_selector_enter_selects_highlighted() -> None:
    app = _EffortSelectorHost()
    async with app.run_test() as pilot:
        results: list[str | None] = []
        await app.push_screen(
            EffortSelectorScreen(
                model_spec="openai:gpt-5.5",
                efforts=("low", "medium", "high"),
                current_effort="low",
            ),
            results.append,
        )
        await pilot.pause()
        app.screen.query_one("#effort-options", OptionList).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert results == ["low"]


async def test_effort_selector_escape_cancels() -> None:
    app = _EffortSelectorHost()
    async with app.run_test() as pilot:
        results: list[str | None] = []
        await app.push_screen(
            EffortSelectorScreen(
                model_spec="openai:gpt-5.5",
                efforts=("low", "high"),
                current_effort=None,
            ),
            results.append,
        )
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert results == [None]


def test_effort_selector_format_label_marks_current_and_default() -> None:
    screen = EffortSelectorScreen(
        model_spec="openai:gpt-5.5",
        efforts=("low", "high"),
        current_effort="high",
        default_effort="low",
    )
    assert "(current)" in str(screen._format_label("high"))
    assert "(default)" in str(screen._format_label("low"))


def test_effort_selector_format_label_combines_current_default() -> None:
    screen = EffortSelectorScreen(
        model_spec="openai:gpt-5.5",
        efforts=("low", "high"),
        current_effort="high",
        default_effort="high",
    )
    assert "(current, default)" in str(screen._format_label("high"))
