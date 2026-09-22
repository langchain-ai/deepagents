"""Behavioral coverage for cache-expiry handoffs."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from deepagents_code.app import DeepAgentsApp, QueuedMessage, TextualSessionState
from deepagents_code.tui.modals.cache_expiry import CacheExpiryScreen


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
        assert isinstance(app.screen, CacheExpiryScreen)
        await pilot.press("shift+tab")
        assert isinstance(app.screen, CacheExpiryScreen)
        await pilot.press(key)
        await pilot.pause()
        assert not isinstance(app.screen, CacheExpiryScreen)
        assert app._chat_input.value == "keep this draft"
        assert app._lc_thread_id == "source"
        assert handoff.await_count == (1 if key == "enter" else 0)
        app._check_cache_expiry()
        await pilot.pause()
        assert not isinstance(app.screen, CacheExpiryScreen)
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
        assert isinstance(app.screen, CacheExpiryScreen)
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
        assert not isinstance(app.screen, CacheExpiryScreen)
        app._set_agent_running(False)
        other_modal = CacheExpiryScreen()
        await app.push_screen(other_modal)
        app._check_cache_expiry()
        await pilot.pause()
        assert app.screen is other_modal
        assert not app._cache_expiry_seen
        await pilot.press("escape")
        await pilot.pause()
        with monkeypatch.context() as config_patch:
            config_patch.setattr(
                "deepagents_code.app._load_bool_display_preference",
                lambda *_a, **_kw: False,
            )
            app._check_cache_expiry()
            await pilot.pause()
            assert not isinstance(app.screen, CacheExpiryScreen)
        app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, CacheExpiryScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert app._status_bar is not None
        app._status_bar.cache_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, CacheExpiryScreen)
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
