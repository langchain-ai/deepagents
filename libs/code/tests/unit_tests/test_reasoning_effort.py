"""Tests for `/effort` reasoning effort handling.

Support data comes from LangChain model profiles, so most tests mock
`get_model_profiles()` instead of relying on installed provider packages.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from textual.app import App

from deepagents_code import model_config, reasoning_effort
from deepagents_code.app import DeepAgentsApp
from deepagents_code.config import runtime_state
from deepagents_code.reasoning_effort import (
    current_effort_from_model_params,
    has_explicit_effort_model_params,
)
from deepagents_code.tui.widgets.effort_selector import EffortSelectorScreen
from deepagents_code.tui.widgets.messages import ErrorMessage


@pytest.fixture(autouse=True)
def _restore_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    original_name = runtime_state.model_name
    original_provider = runtime_state.model_provider
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "config.toml")
    model_config.clear_caches()
    yield
    runtime_state.model_name = original_name
    runtime_state.model_provider = original_provider
    model_config.clear_caches()


# Reading logic (mocked profiles, provider-agnostic)


# Contract checks against required minimum integrations.


# Compatibility reader for canonical and legacy/native model params.


def test_fireworks_duplicate_forms_fail_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model_spec = "fireworks:accounts/fireworks/models/deepseek-v4-pro"
    model_params = {
        "reasoning_effort": "high",
        "model_kwargs": {"reasoning_effort": "low"},
    }

    with caplog.at_level(logging.WARNING):
        assert current_effort_from_model_params(model_spec, model_params) is None
    assert has_explicit_effort_model_params(model_spec, model_params)
    assert "conflicting Fireworks" in caplog.text


# app.py integration (uses real profile data for openai/anthropic)


async def test_profile_override_controls_persisted_restoration() -> None:
    model_config.save_effort_for_model("openai:gpt-5.5", "custom")
    app = DeepAgentsApp(
        profile_override={
            "reasoning_output": True,
            "reasoning_effort_levels": ["custom"],
        }
    )

    await app._restore_effort_override("openai:gpt-5.5")

    assert app._model_params_override == {"reasoning_effort": "custom"}


async def test_restore_effort_override_applies_persisted_model_choice() -> None:
    model_config.save_effort_for_model("openai:gpt-5.6-luna", "max")
    app = DeepAgentsApp()
    app._model_params_override = {"temperature": 0.2}

    await app._restore_effort_override("openai:gpt-5.6-luna")

    assert app._model_params_override == {
        "temperature": 0.2,
        "reasoning_effort": "max",
    }


async def test_startup_model_params_precede_persisted_effort() -> None:
    model_config.save_effort_for_model("openai:gpt-5.5", "high")
    app = DeepAgentsApp(
        model_kwargs={
            "model_spec": "openai:gpt-5.5",
            "extra_kwargs": {"reasoning_effort": "low"},
        }
    )

    # `on_mount` restores effort before deferred model creation consumes the
    # startup kwargs. The explicit CLI value must already be active by then.
    await app._restore_effort_override("openai:gpt-5.5")

    assert app._model_params_override == {"reasoning_effort": "low"}


async def test_effort_command_save_failure_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    runtime_state.model_provider = "openai"
    runtime_state.model_name = "gpt-5.5"
    monkeypatch.setattr(
        model_config, "save_effort_for_model", lambda *_args, **_kwargs: False
    )

    await app._set_effort_override("high")

    # The effort still applies for the session, but the user is told it could
    # not be persisted, and the success message is suppressed by the early
    # return (so the only mounted message is the error).
    assert app._model_params_override == {"reasoning_effort": "high"}
    assert app._mount_message.await_count == 1  # ty: ignore[unresolved-attribute]
    message = app._mount_message.await_args.args[0]  # ty: ignore[unresolved-attribute]
    assert isinstance(message, ErrorMessage)
    assert "could not be saved" in message._content
    assert model_config.load_effort_for_model("openai:gpt-5.5") is None


class _EffortSelectorHost(App[None]):
    """Minimal host app for mounting `EffortSelectorScreen` in tests."""


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


async def test_effort_selector_dims_underlying_content() -> None:
    """The modal must inherit the translucent `ModalScreen` backdrop.

    Like the other selector modals, `/effort` should dim the content
    underneath rather than render a fully transparent overlay. The alpha is
    in (0, 1) only under a non-ansi theme, so pin `textual-dark`.
    """
    app = _EffortSelectorHost()
    async with app.run_test() as pilot:
        app.theme = "textual-dark"
        await pilot.pause()
        await app.push_screen(
            EffortSelectorScreen(
                model_spec="openai:gpt-5.5",
                efforts=("low", "high"),
                current_effort="low",
            )
        )
        await pilot.pause()
        assert 0 < app.screen.styles.background.a < 1
