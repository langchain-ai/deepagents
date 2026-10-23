"""Idle compaction reservation and input-race regressions."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepagents_code.app import DeepAgentsApp, QueuedMessage, TextualSessionState
from deepagents_code.config_manifest import get_option
from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage


@pytest.mark.parametrize("status", ["empty", "noop", "failed"])
@pytest.mark.parametrize("idle", [False, True])
async def test_offload_feedback(
    monkeypatch: pytest.MonkeyPatch, status: str, *, idle: bool
) -> None:
    app = DeepAgentsApp()
    remote = MagicMock(aoffload=AsyncMock(return_value={"status": status}))
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr(app, "_cache_maintenance_idle", lambda **_: True)
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_agent", remote)
        app._lc_thread_id = "idle"
        assert app._status_bar is not None
        expiry = datetime.now(UTC) + timedelta(seconds=45)
        app._status_bar.cache_expires_at = expiry
        app._set_agent_running(True)
        await app._run_offload_task(idle_window=("idle", expiry) if idle else None)
        messages = "\n".join(str(widget._content) for widget in app.query(AppMessage))
        assert ("Nothing to offload" in messages) == (not idle and status != "failed")
        assert bool(app.query(ErrorMessage)) == (status == "failed")
        assert not app._agent_running
        assert app._loading_widget is None
        remote.aoffload.assert_awaited_once()


@pytest.mark.parametrize("window", [300, 3600])
@pytest.mark.parametrize(
    "blocker",
    ["none", "busy", "typing", "small", "expired", "disabled", "early", "boundary"],
)
async def test_idle_compaction_guards(
    monkeypatch: pytest.MonkeyPatch, blocker: str, window: int
) -> None:
    app = DeepAgentsApp()
    offload = AsyncMock()
    monkeypatch.setattr(app, "_offload_impl", offload)
    monkeypatch.setattr(app, "_remote_agent", lambda: MagicMock())
    monkeypatch.setattr(
        "deepagents_code.configuration.resolver.get_config_resolver",
        lambda: MagicMock(get=MagicMock(return_value=MagicMock(value=100_000))),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_agent", MagicMock())
        app._lc_thread_id = "idle"
        app._session_state = TextualSessionState(thread_id="idle")
        app._context_tokens = 100_000
        app._tokens_approximate = False
        assert app._status_bar is not None
        now = datetime.now(UTC)
        clock = MagicMock(wraps=datetime)
        clock.now.return_value = now
        monkeypatch.setattr("deepagents_code.app.datetime", clock)
        app._status_bar.set_cache_timing(now, ttl_seconds=window)
        lead = min(window * 0.2, 120)
        app._status_bar.cache_expires_at = now + timedelta(seconds=lead - 1)
        if blocker == "early":
            app._status_bar.cache_expires_at = now + timedelta(seconds=lead + 1)
        elif blocker == "boundary":
            app._status_bar.cache_expires_at = now + timedelta(seconds=lead)
        elif blocker == "busy":
            app._agent_running = True
        elif blocker == "typing":
            monkeypatch.setattr(app, "_is_user_typing", lambda: True)
        elif blocker == "small":
            app._context_tokens = 99_999
        elif blocker == "expired":
            app._status_bar.cache_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        elif blocker == "disabled":
            option = get_option("history.idle_compaction_tokens")
            assert option is not None
            monkeypatch.setattr(
                "deepagents_code.configuration.resolver.get_config_resolver",
                lambda: MagicMock(
                    get=MagicMock(return_value=MagicMock(value=option.default))
                ),
            )
        app._check_idle_compaction()
        app._check_idle_compaction()
        await pilot.pause()
        assert offload.await_count == (1 if blocker in {"none", "boundary"} else 0)
        if blocker in {"none", "boundary"}:
            assert not app._agent_running


async def test_new_input_before_worker_skips_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    offload = AsyncMock()
    drain = AsyncMock()
    monkeypatch.setattr(app, "_offload_impl", offload)
    monkeypatch.setattr(app, "_process_next_from_queue", drain)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._lc_thread_id = "idle"
        assert app._status_bar is not None
        expiry = datetime.now(UTC) + timedelta(seconds=45)
        app._status_bar.cache_expires_at = expiry
        app._set_agent_running(True)
        app._pending_messages.append(QueuedMessage(text="new prompt", mode="normal"))
        await app._run_offload_task(idle_window=("idle", expiry))
        offload.assert_not_awaited()
        assert not app._agent_running
        drain.assert_awaited()


@pytest.mark.parametrize("idle", [False, True])
async def test_new_turn_during_spinner_cleanup_stays_busy(
    monkeypatch: pytest.MonkeyPatch, *, idle: bool
) -> None:
    app = DeepAgentsApp()
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()

    async def clear_spinner(status: str | None) -> None:
        assert status is None
        cleaning.set()
        await finish_cleanup.wait()

    monkeypatch.setattr(app, "_offload_impl", AsyncMock())
    monkeypatch.setattr(app, "_set_spinner", clear_spinner)
    monkeypatch.setattr(app, "_cache_maintenance_idle", lambda **_: True)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._lc_thread_id = "idle"
        assert app._status_bar is not None
        expiry = datetime.now(UTC) + timedelta(seconds=45)
        app._status_bar.cache_expires_at = expiry
        app._set_agent_running(True)
        task = asyncio.create_task(
            app._run_offload_task(idle_window=("idle", expiry) if idle else None)
        )
        try:
            await asyncio.wait_for(cleaning.wait(), timeout=2)
            assert not app._agent_running
            app._set_agent_running(True)
            finish_cleanup.set()
            await asyncio.wait_for(task, timeout=2)
            assert app._agent_running
        finally:
            finish_cleanup.set()
            await task
            app._set_agent_running(False)


async def test_cancellation_releases_idle_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    started = asyncio.Event()

    async def offload(*, automatic: bool = False) -> None:
        assert automatic
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(app, "_offload_impl", offload)
    monkeypatch.setattr(app, "_remote_agent", lambda: MagicMock())
    monkeypatch.setattr(
        "deepagents_code.configuration.resolver.get_config_resolver",
        lambda: MagicMock(get=MagicMock(return_value=MagicMock(value=100_000))),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_agent", MagicMock())
        app._lc_thread_id = "idle"
        app._session_state = TextualSessionState(thread_id="idle")
        app._context_tokens = 100_000
        app._tokens_approximate = False
        assert app._status_bar is not None
        app._status_bar.set_cache_timing(datetime.now(UTC), ttl_seconds=300)
        app._status_bar.cache_expires_at = datetime.now(UTC) + timedelta(seconds=45)
        app._check_idle_compaction()
        await asyncio.wait_for(started.wait(), timeout=2)
        assert app._agent_running
        await pilot.press("escape")
        await pilot.pause()
        assert not app._agent_running
        assert app._offload_worker is None
