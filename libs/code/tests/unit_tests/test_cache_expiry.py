"""Behavioral coverage for cache-expiry handoffs."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from deepagents_code.app import DeepAgentsApp, QueuedMessage, TextualSessionState
from deepagents_code.tui.modals.cold_cache import ColdCacheWarningScreen


def _prepare(app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "_agent", MagicMock())
    app._lc_thread_id = "source"
    app._session_state = TextualSessionState(thread_id="source")
    assert app._status_bar is not None
    app._status_bar.cache_expires_at = datetime.now(UTC) - timedelta(seconds=1)


@pytest.mark.parametrize("key", ["enter", "escape"])
async def test_modal_keys_preserve_draft_and_prompt_once(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    handoff = AsyncMock()
    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert app._chat_input is not None
        app._chat_input.value = "keep this draft"
        app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, ColdCacheWarningScreen)
        await pilot.press("shift+tab")
        assert isinstance(app.screen, ColdCacheWarningScreen)
        await pilot.press(key)
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        assert app._chat_input.value == "keep this draft"
        assert app._lc_thread_id == "source"
        assert handoff.await_count == (1 if key == "enter" else 0)
        app._check_cache_expiry()
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        if key == "escape":
            assert app._cache_expiry_bypassed is not None


@pytest.mark.parametrize(
    ("key", "action"),
    [("ctrl+c", "action_quit_or_interrupt"), ("ctrl+d", "action_quit_app")],
)
async def test_modal_preserves_app_quit_keys(
    key: str, action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    quit_action = MagicMock()
    monkeypatch.setattr(app, action, quit_action)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, ColdCacheWarningScreen)
        await pilot.press(key)
        quit_action.assert_called_once()


async def test_defers_busy_and_disabled_then_rearms_new_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        app._set_agent_running(True)
        app._check_cache_expiry()
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        app._set_agent_running(False)
        other_modal = ColdCacheWarningScreen(None, handoff=True)
        await app.push_screen(other_modal)
        app._check_cache_expiry()
        await pilot.pause()
        assert app.screen is other_modal
        assert not app._cache_expiry_seen
        await pilot.press("escape")
        await pilot.pause()
        with monkeypatch.context() as config_patch:
            config_patch.setattr(
                "deepagents_code.app._load_cache_prompt_mode",
                lambda: "off",
            )
            app._check_cache_expiry()
            await pilot.pause()
            assert not isinstance(app.screen, ColdCacheWarningScreen)
        app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, ColdCacheWarningScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert app._status_bar is not None
        app._status_bar.cache_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, ColdCacheWarningScreen)
        await pilot.press("escape")


@pytest.mark.parametrize("failure", [None, "summary", "archive", "seed", "queued"])
async def test_handoff_persists_recovery_before_switch(
    failure: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    app._lc_thread_id = "source"
    remote = MagicMock()
    remote.aoffload = AsyncMock(
        return_value={"status": "failed" if failure == "summary" else "compacted"}
    )
    remote.aensure_thread = AsyncMock()
    remote.aswitch_workspace = AsyncMock()
    remote.aupdate_state = AsyncMock(
        side_effect=RuntimeError("write failed") if failure == "seed" else None
    )
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_set_spinner", AsyncMock())
    monkeypatch.setattr(app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    monkeypatch.setattr(
        app,
        "_get_thread_state_values",
        AsyncMock(
            return_value={
                "messages": [HumanMessage("original")],
                "_summarization_event": {
                    "summary_message": HumanMessage("LLM summary"),
                    "cutoff_index": 1,
                    "file_path": None
                    if failure == "archive"
                    else "/conversation_history/source.md",
                },
            }
        ),
    )
    resume = AsyncMock()
    monkeypatch.setattr(app, "_resume_thread", resume)
    if failure == "queued":
        app._pending_messages.append(QueuedMessage(text="arrived", mode="normal"))
        await app._handoff_expired_cache("source")
        resume.assert_not_awaited()
        assert app._pending_messages[0].text == "arrived"
        assert app._lc_thread_id == "source"
        return
    if failure:
        with pytest.raises(RuntimeError):
            await app._handoff_expired_cache("source")
        resume.assert_not_awaited()
        assert app._lc_thread_id == "source"
        return
    await app._handoff_expired_cache("source")
    remote.aoffload.assert_awaited_once()
    assert remote.aoffload.await_args is not None
    assert remote.aupdate_state.await_args is not None
    assert remote.aoffload.await_args.kwargs["summarize_all"] is True
    update = remote.aupdate_state.await_args.args[1]
    content = update["messages"][0].text
    assert "LLM summary" in content
    assert "Previous thread ID: source" in content
    assert "/conversation_history/source.md" in content
    child_id = remote.aupdate_state.await_args.args[0]["configurable"]["thread_id"]
    assert child_id != "source"
    resume.assert_awaited_once_with(child_id)


@pytest.mark.parametrize("mode", ["expiry", "send", "off"])
async def test_send_timing_restores_draft_without_spending(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    process = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr("deepagents_code.app._load_cache_prompt_mode", lambda: mode)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert app._chat_input is not None
        await app._dispatch_queued_message(QueuedMessage("keep my request", "normal"))
        await pilot.pause()
        if mode == "off":
            process.assert_awaited_once_with("keep my request", "normal")
            assert not isinstance(app.screen, ColdCacheWarningScreen)
            return
        assert isinstance(app.screen, ColdCacheWarningScreen)
        assert "estimate is unavailable" in app.screen._body()
        await pilot.press("escape")
        await pilot.pause()
        process.assert_not_awaited()
        assert app._chat_input.value == "keep my request"
        app._check_cache_expiry()
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        await app._dispatch_queued_message(QueuedMessage("keep my request", "normal"))
        process.assert_awaited_once_with("keep my request", "normal")


async def test_send_mode_does_not_interrupt_idle_composer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_code.app._load_cache_prompt_mode", lambda: "send")
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        app._check_cache_expiry()
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)


@pytest.mark.parametrize("failure", [False, True])
async def test_handoff_keeps_submitted_draft_without_sending(
    failure: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    process = AsyncMock()
    handoff = AsyncMock(side_effect=RuntimeError("summary failed") if failure else None)
    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        await app._dispatch_queued_message(
            QueuedMessage("retain this request", "normal")
        )
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app._chat_input is not None
        assert app._chat_input.value == "retain this request"
        process.assert_not_awaited()
        handoff.assert_awaited_once_with("source")


@pytest.mark.parametrize("identity_changed", [False, True])
async def test_expiry_acknowledgment_does_not_hide_identity_change(
    identity_changed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    app._model_override = "openai:gpt-5.6"
    app._last_cache_model_spec = "other" if identity_changed else app._model_override
    app._last_model_request_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    app._context_tokens = 50_000
    app._cold_cache_warning_threshold_usd = 0.10
    config = MagicMock()
    config.get_effective_kwargs.return_value = {}
    config.get_base_url.return_value = None
    monkeypatch.setattr("deepagents_code.model_config.ModelConfig.load", lambda: config)
    monkeypatch.setattr(
        "deepagents_code.model_config.is_warning_suppressed", lambda *_a: False
    )
    from deepagents_code.cold_cache import RewarmEstimate

    monkeypatch.setattr(
        "deepagents_code.cold_cache.estimate_rewarm_cost",
        lambda *_a: RewarmEstimate(cold_cost_usd=1.0, incremental_cost_usd=0.8),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert app._status_bar is not None
        assert app._status_bar.cache_expires_at is not None
        app._cache_expiry_bypassed = ("source", app._status_bar.cache_expires_at)
        warning = await app._cold_cache_warning_for(QueuedMessage("next", "normal"))
        if identity_changed:
            assert warning is not None
            assert warning.reason == "identity_changed"
        else:
            assert warning is None
        advisory = await app._cold_cache_warning_for(
            QueuedMessage("", "normal"), advisory=True
        )
        assert advisory is not None
        assert "~$1.0" in ColdCacheWarningScreen(advisory, handoff=True)._body()


@pytest.mark.parametrize(
    ("mode", "legacy", "expected"),
    [
        (None, True, "expiry"),
        (None, False, "send"),
        ("off", True, "off"),
        ("expiry", False, "expiry"),
        ("send", True, "send"),
        ("invalid", True, "expiry"),
    ],
)
def test_mode_resolution_preserves_legacy_preference(
    mode: str | None, legacy: bool, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from deepagents_code.app import _load_cache_prompt_mode

    resolver = MagicMock()
    resolver.get.side_effect = lambda option: (
        SimpleNamespace(
            value=mode or "expiry", ranks=(1000,) if mode is None else (500,)
        )
        if option.key == "warnings.cache_prompt"
        else SimpleNamespace(value=legacy)
    )
    monkeypatch.setattr(
        "deepagents_code.configuration.resolver.get_config_resolver", lambda: resolver
    )
    assert _load_cache_prompt_mode() == expected
