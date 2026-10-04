"""Tests for `/effort` reasoning effort handling.

Support data comes from LangChain model profiles, so most tests mock
`get_model_profiles()` instead of relying on installed provider packages.
"""

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from textual.app import App
from textual.widgets import OptionList, Static

from deepagents_code import model_config, reasoning_effort
from deepagents_code.app import DeepAgentsApp, _EffortContext, _GoalApplication
from deepagents_code.config import runtime_state
from deepagents_code.reasoning_effort import (
    current_effort_from_model_params,
    has_explicit_effort_model_params,
)
from deepagents_code.tui.widgets.chat_input import ChatInput
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


def test_status_exposes_effort_when_default_is_unknown() -> None:
    app = DeepAgentsApp(
        profile_override={
            "reasoning_output": True,
            "reasoning_effort_levels": ["low", "medium", "high"],
        }
    )
    app._status_bar = Mock()
    runtime_state.model_provider = "openai"
    runtime_state.model_name = "gpt-6-astra"

    app._sync_status_model()

    app._status_bar.set_model.assert_called_once_with(
        provider="openai", model="gpt-6-astra", effort="effort?"
    )


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


@pytest.mark.parametrize("busy", [False, True])
async def test_footer_effort_selects_without_queueing_command(
    busy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp(agent=MagicMock())
    runtime_state.model_provider = "openai"
    runtime_state.model_name = "gpt-5.5"
    app._model_params_override = {"reasoning_effort": "low"}
    notify = Mock()
    monkeypatch.setattr(app, "notify", notify)
    async with app.run_test() as pilot:
        await pilot.pause()
        mount = AsyncMock()
        monkeypatch.setattr(app, "_mount_message", mount)
        app._agent_running = busy

        await app.action_open_effort_selector()
        await pilot.pause()
        assert isinstance(app.screen, EffortSelectorScreen)
        stack_size = len(app.screen_stack)
        await app.action_open_effort_selector()
        assert len(app.screen_stack) == stack_size
        assert not app._pending_messages
        assert not app._queued_widgets
        mount.assert_not_awaited()

        await pilot.press("end", "enter")
        await pilot.pause()
        if busy:
            assert app._model_params_override == {"reasoning_effort": "low"}
            assert model_config.load_effort_for_model("openai:gpt-5.5") is None
            mount.assert_not_awaited()
            assert (
                "pending until the current task completes" in notify.call_args.args[0]
            )
            app._agent_running = False
            await app._drain_deferred_actions()
        await app.workers.wait_for_complete()
        assert app._model_params_override == {"reasoning_effort": "xhigh"}
        assert model_config.load_effort_for_model("openai:gpt-5.5") == "xhigh"
        assert not app._pending_messages


async def test_effort_selected_during_startup_survives_model_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp(
        agent=MagicMock(), thread_id="resume-effort", resume_thread="resume-effort"
    )
    runtime_state.model_provider = "openai"
    runtime_state.model_name = "gpt-5.5"
    app._model_params_override = {"reasoning_effort": "low"}
    started = asyncio.Event()
    finish = asyncio.Event()
    next_turn_efforts: list[object] = []

    async def load_history(**_kwargs: object) -> None:
        started.set()
        await finish.wait()
        await app._adopt_resumed_model_if_needed(
            model_spec="openai:gpt-5.5",
            model_params={"reasoning_effort": "medium"},
        )
        assert app._model_params_override == {"reasoning_effort": "medium"}

    def send(_message: str, **_kwargs: object) -> None:
        assert app._model_params_override is not None
        next_turn_efforts.append(app._model_params_override["reasoning_effort"])

    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    monkeypatch.setattr(app, "_load_thread_history", load_history)
    monkeypatch.setattr(app, "_remote_agent", Mock(return_value=MagicMock()))
    monkeypatch.setattr(
        model_config, "get_provider_auth_status", Mock(return_value=None)
    )
    monkeypatch.setattr(app, "_run_session_start_hook", AsyncMock(return_value=True))
    monkeypatch.setattr(app, "_maybe_compact_after_resume", AsyncMock())
    monkeypatch.setattr(app, "_remount_pending_goal_rubric_review", AsyncMock())
    monkeypatch.setattr(app, "_send_to_agent", AsyncMock(side_effect=send))

    async with app.run_test() as pilot:
        app._connecting = False
        app._should_adopt_resumed_model = True
        worker = app.run_worker(app._run_session_start_sequence())
        await asyncio.wait_for(started.wait(), timeout=2)
        assert app._startup_sequence_running

        await app.action_open_effort_selector()
        await pilot.pause()
        await pilot.press("end", "enter")
        await pilot.pause()
        app.post_message(ChatInput.Submitted("next prompt", "normal"))
        await pilot.pause()
        assert len(app._pending_messages) == 1
        assert not next_turn_efforts
        assert app._model_params_override == {"reasoning_effort": "low"}
        assert model_config.load_effort_for_model("openai:gpt-5.5") is None

        finish.set()
        await worker.wait()

        assert next_turn_efforts == ["xhigh"]
        assert app._model_params_override == {"reasoning_effort": "xhigh"}
        assert model_config.load_effort_for_model("openai:gpt-5.5") == "xhigh"
        assert not app._deferred_actions
        assert not app._pending_messages


@pytest.mark.parametrize("command", ["/offload", "/compact"])
async def test_effort_selected_during_offload_applies_before_queued_prompt(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp(agent=MagicMock())
    runtime_state.model_provider = "openai"
    runtime_state.model_name = "gpt-5.5"
    app._model_params_override = {"reasoning_effort": "low"}
    started = asyncio.Event()
    finish = asyncio.Event()
    next_turn_efforts: list[object] = []

    async def offload() -> None:
        started.set()
        await finish.wait()

    def send(_message: str) -> None:
        assert app._model_params_override is not None
        next_turn_efforts.append(app._model_params_override["reasoning_effort"])

    monkeypatch.setattr(app, "_offload_impl", offload)
    monkeypatch.setattr(app, "_send_to_agent", AsyncMock(side_effect=send))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(ChatInput.Submitted(command, "command"))
        await asyncio.wait_for(started.wait(), timeout=2)
        worker = app._offload_worker
        assert worker is not None

        await app.action_open_effort_selector()
        await pilot.pause()
        await pilot.press("end", "enter")
        await pilot.pause()
        app.post_message(ChatInput.Submitted("next prompt", "normal"))
        await pilot.pause()
        assert app._model_params_override == {"reasoning_effort": "low"}
        assert len(app._pending_messages) == 1
        assert not next_turn_efforts

        finish.set()
        await worker.wait()

        assert next_turn_efforts == ["xhigh"]
        assert model_config.load_effort_for_model("openai:gpt-5.5") == "xhigh"
        assert not app._deferred_actions
        assert not app._pending_messages


@pytest.mark.parametrize("next_turn", ["prompt", "continuation", "failed_application"])
async def test_effort_selected_during_goal_reconciliation_applies_before_next_turn(
    next_turn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp(agent=MagicMock())
    runtime_state.model_provider = "openai"
    runtime_state.model_name = "gpt-5.5"
    app._model_params_override = {"reasoning_effort": "low"}
    started = asyncio.Event()
    finish = asyncio.Event()
    next_turn_efforts: list[object] = []

    async def persist(**_kwargs: object) -> bool:
        started.set()
        await finish.wait()
        if next_turn == "failed_application":
            msg = "goal persistence failed"
            raise RuntimeError(msg)
        return True

    def send(_message: str, **_kwargs: object) -> None:
        assert app._model_params_override is not None
        next_turn_efforts.append(app._model_params_override["reasoning_effort"])

    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_persist_goal_rubric_state", persist)
        monkeypatch.setattr(app, "_sync_goal_rubric_state_from_thread", AsyncMock())
        monkeypatch.setattr(app, "_send_to_agent", AsyncMock(side_effect=send))
        app._queued_goal_application = _GoalApplication(
            "ship login", "- tests pass", "create"
        )
        worker = app.run_worker(app._cleanup_agent_task())
        await asyncio.wait_for(started.wait(), timeout=2)
        assert app._agent_reconciling

        await app.action_open_effort_selector()
        await pilot.pause()
        await pilot.press("end", "enter")
        await pilot.pause()
        if next_turn != "continuation":
            app.post_message(ChatInput.Submitted("next prompt", "normal"))
            await pilot.pause()
            assert len(app._pending_messages) == 1
        assert app._model_params_override == {"reasoning_effort": "low"}
        assert not next_turn_efforts

        finish.set()
        await worker.wait()

        assert next_turn_efforts == ["xhigh"]
        assert model_config.load_effort_for_model("openai:gpt-5.5") == "xhigh"
        assert not app._deferred_actions
        assert not app._pending_messages
        assert not app._agent_reconciling


@pytest.mark.parametrize("outcome", ["apply", "interrupt", "model_change"])
async def test_pending_effort_latest_selection_and_cancellation(
    outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp(agent=MagicMock())
    runtime_state.model_provider = "openai"
    runtime_state.model_name = "gpt-5.5"
    app._model_params_override = {"reasoning_effort": "low"}
    notify = Mock()
    monkeypatch.setattr(app, "notify", notify)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._agent_running = True
        for index, keys in enumerate(
            (("end", "enter"), ("home", "enter"), ("escape",))
        ):
            app._agent_running = index == 0
            app._agent_reconciling = index > 0
            await app.action_open_effort_selector()
            await pilot.pause()
            await pilot.press(*keys)
            await pilot.pause()
        assert len(app._deferred_actions) == 1
        assert app._model_params_override == {"reasoning_effort": "low"}
        if outcome == "interrupt":
            app._discard_queue()
            assert "Cancelled the pending" in notify.call_args.args[0]
        elif outcome == "model_change":
            runtime_state.model_name = "gpt-5.5-mini"
        app._agent_running = False
        await app._drain_deferred_actions()
        if outcome == "apply":
            assert model_config.load_effort_for_model("openai:gpt-5.5") == "none"
            assert app._model_params_override == {"reasoning_effort": "none"}
        else:
            assert app._model_params_override == {"reasoning_effort": "low"}
            assert model_config.load_effort_for_model("openai:gpt-5.5") is None
        if outcome == "model_change":
            assert "Model changed" in notify.call_args.args[0]


def test_only_bare_effort_bypasses_queue() -> None:
    app = DeepAgentsApp()
    assert app._can_bypass_queue("/effort")
    assert not app._can_bypass_queue("/effort high")
    assert not app._can_bypass_queue("/effort clear")


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


async def test_effort_selector_explains_unknown_default() -> None:
    app = _EffortSelectorHost()
    async with app.run_test() as pilot:
        await app.push_screen(
            EffortSelectorScreen(
                model_spec="openai:gpt-6-astra",
                efforts=("low", "medium", "high"),
            )
        )
        await pilot.pause()

        subtitle = app.screen.query_one(".effort-selector-subtitle", Static)
        options = app.screen.query_one("#effort-options", OptionList)
        assert "Provider default unknown" in str(subtitle.render())
        assert all(
            "default" not in str(options.get_option_at_index(index).prompt)
            for index in range(options.option_count)
        )


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


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_between_tools_effort_selection(
    effort: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = "anthropic:claude-sonnet-5-5"
    app = DeepAgentsApp()
    mount = AsyncMock()
    monkeypatch.setattr(app, "_mount_message", mount)
    runtime_state.model_provider = "anthropic"
    runtime_state.model_name = "claude-sonnet-5-5"
    app._model_params_override = {"thinking": {"type": "between_tools"}}

    await app._set_effort_override(effort)

    if effort in {"xhigh", "max"}:
        assert app._model_params_override == {"thinking": {"type": "between_tools"}}
        assert model_config.load_effort_for_model(spec) is None
        assert mount.await_args is not None
        assert isinstance(mount.await_args.args[0], ErrorMessage)
        assert (
            "Supported efforts: low, medium, high" in mount.await_args.args[0]._content
        )
    else:
        assert app._model_params_override == {
            "thinking": {"type": "between_tools"},
            "reasoning_effort": effort,
        }
        assert model_config.load_effort_for_model(spec) == effort


async def test_between_tools_preserves_saved_adaptive_effort() -> None:
    spec = "anthropic:claude-sonnet-5-5"
    model_config.save_effort_for_model(spec, "max")
    app = DeepAgentsApp()
    app._model_params_override = {"thinking": {"type": "between_tools"}}

    await app._restore_effort_override(spec)

    assert app._model_params_override == {"thinking": {"type": "between_tools"}}
    assert model_config.load_effort_for_model(spec) == "max"
    app._model_params_override = {"thinking": {"type": "adaptive"}}
    await app._restore_effort_override(spec)
    assert app._model_params_override["reasoning_effort"] == "max"


def test_between_tools_config_filters_selector_and_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config.DEFAULT_CONFIG_PATH.write_text(
        '[models.providers.anthropic.params."claude-sonnet-5-5"]\n'
        'thinking = { type = "between_tools" }\n'
    )
    model_config.clear_caches()
    app = DeepAgentsApp()
    chat_input = Mock()
    monkeypatch.setattr(app, "_chat_input", chat_input)
    runtime_state.model_provider = "anthropic"
    runtime_state.model_name = "claude-sonnet-5-5"

    context = app._resolve_effort_context()
    assert isinstance(context, _EffortContext)
    assert context.efforts == ("low", "medium", "high")
    app._sync_status_model()
    chat_input.set_argument_hint_override.assert_called_with(
        "/effort", "[low|medium|high|clear]"
    )
    app._model_params_override = {"thinking": {"type": "adaptive"}}
    context = app._resolve_effort_context()
    assert isinstance(context, _EffortContext)
    assert context.efforts == (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
