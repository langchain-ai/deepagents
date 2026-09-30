"""Behavioral coverage for cache-expiry handoffs."""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from deepagents_code.app import DeepAgentsApp, QueuedMessage, TextualSessionState
from deepagents_code.hooks.client_lifecycle import ClientHookStopError
from deepagents_code.hooks.models.domain import DcodeNotificationKind
from deepagents_code.tui.modals.cold_cache import (
    ColdCacheChoice,
    ColdCacheWarningScreen,
)
from deepagents_code.tui.widgets.messages import ErrorMessage
from deepagents_code.tui.widgets.status import StatusBar

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


@pytest.fixture(autouse=True)
def checkpoint_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepagents_code.sessions.get_db_path", lambda: tmp_path / "sessions.db"
    )


def _prepare(app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "_agent", MagicMock())
    app._lc_thread_id = "source"
    app._session_state = TextualSessionState(thread_id="source")
    assert app._status_bar is not None
    app._status_bar.cache_expires_at = datetime.now(UTC) - timedelta(seconds=1)


@pytest.mark.parametrize("remaining", [None, -1, 0, 1, 60, 61])
async def test_cache_expiring_notification_window(
    remaining: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    monkeypatch.setattr("deepagents_code.app.datetime", MagicMock(now=lambda _: now))
    app = DeepAgentsApp()
    app._lc_thread_id = "source"
    app._status_bar = StatusBar()
    app._status_bar.cache_expires_at = (
        now + timedelta(seconds=remaining) if remaining is not None else None
    )
    notify = AsyncMock()
    toast = MagicMock()
    monkeypatch.setattr(type(app._hooks), "notify", notify)
    monkeypatch.setattr(app, "notify", toast)

    await app._notify_cache_expiring()
    await app._notify_cache_expiring()

    if remaining in (1, 60):
        notify.assert_awaited_once_with(
            DcodeNotificationKind.CACHE_EXPIRING,
            "Prompt-cache retention may end within 60 seconds.",
            title="Prompt cache expiring",
        )
        toast.assert_called_once_with(
            "Prompt-cache retention may end within 60 seconds.",
            title="Prompt cache expiring",
            severity="warning",
            markup=False,
        )
    else:
        notify.assert_not_awaited()
        toast.assert_not_called()


@pytest.mark.parametrize("failure", [None, ClientHookStopError, RuntimeError])
async def test_cache_expiring_dedup_and_rearm(
    failure: type[Exception] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    app._lc_thread_id = "source"
    app._status_bar = StatusBar()
    expires_at = datetime.now(UTC) + timedelta(seconds=20)
    app._status_bar.cache_expires_at = expires_at
    notify = AsyncMock(side_effect=failure)
    toast = MagicMock()
    monkeypatch.setattr(type(app._hooks), "notify", notify)
    monkeypatch.setattr(app, "notify", toast)
    app._thread_switching = True
    await app._notify_cache_expiring()
    app._thread_switching = False
    app._exiting = True
    await app._notify_cache_expiring()
    notify.assert_not_awaited()
    toast.assert_not_called()
    app._exiting = False
    await app._notify_cache_expiring()
    app._lc_thread_id = "other"
    await app._notify_cache_expiring()
    app._lc_thread_id = "source"
    await app._notify_cache_expiring()
    assert notify.await_count == toast.call_count == 2
    app._status_bar.cache_expires_at = expires_at + timedelta(seconds=1)
    await app._notify_cache_expiring()
    assert notify.await_count == toast.call_count == 3


async def test_cache_expiring_timer_runs_without_prompt_or_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    started = asyncio.Event()
    release = asyncio.Event()

    async def notify(*_args: object, **_kwargs: object) -> None:
        started.set()
        await release.wait()

    hook = AsyncMock(side_effect=notify)
    process = AsyncMock()
    handoff = AsyncMock()
    monkeypatch.setattr(type(app._hooks), "notify", hook)
    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    monkeypatch.setattr("deepagents_code.app._load_cache_prompt_mode", lambda: "off")
    async with app.run_test(notifications=True) as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert app._status_bar is not None
        app._status_bar.cache_expires_at = datetime.now(UTC) + timedelta(seconds=20)
        app._cold_cache_suppressed_for_session = True
        app._cold_cache_warning_threshold_usd = 0
        app._set_agent_running(True)
        await asyncio.wait_for(started.wait(), timeout=5)
        app._check_cache_expiring()
        app._check_cache_expiring()
        assert hook.await_count == 1
        assert app._chat_input is not None
        app._chat_input.value = "draft remains editable"
        await pilot.pause()
        assert app._chat_input.value == "draft remains editable"
        assert app.query_one("Toast").region.height > 0
        assert any(
            notice.title == "Prompt cache expiring" for notice in app._notifications
        )
        release.set()
        assert app._cache_expiring_worker is not None
        await app._cache_expiring_worker.wait()
        app._check_cache_expiring()
        await app._cache_expiring_worker.wait()
        assert hook.await_count == 1
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        process.assert_not_awaited()
        handoff.assert_not_awaited()


async def test_cache_expiring_toast_without_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    async with app.run_test(notifications=True) as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert not app._hooks.enabled
        assert app._status_bar is not None
        app._status_bar.cache_expires_at = datetime.now(UTC) + timedelta(seconds=45)
        await app._notify_cache_expiring()
        await app._notify_cache_expiring()
        await pilot.pause()
        notices = [
            notice
            for notice in app._notifications
            if notice.title == "Prompt cache expiring"
        ]
        assert len(notices) == 1
        assert notices[0].message == "Prompt-cache retention may end within 60 seconds."
        assert app.query_one("Toast").region.height > 0


@pytest.mark.parametrize("size", [(80, 20), (60, 16)])
@pytest.mark.parametrize("allow_send", [False, True])
async def test_short_terminal_keeps_handoff_actions_visible(
    size: tuple[int, int], allow_send: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real app bindings must navigate visible choices without spending."""
    from textual.containers import Vertical, VerticalScroll

    from deepagents_code.cold_cache import (
        ColdCacheWarning,
        PromptCachePolicy,
        RewarmEstimate,
    )

    warning = ColdCacheWarning(
        policy=PromptCachePolicy(
            provider_name="OpenAI",
            window_seconds=1800,
            confidence="may_be_cold",
            minimum_tokens=1024,
            write_bucket="generic",
        ),
        estimate=RewarmEstimate(cold_cost_usd=0.42, incremental_cost_usd=0.35),
        context_tokens=84_000,
        age_seconds=11_520,
        reason="idle",
    )
    app = DeepAgentsApp()
    app._cold_cache_warning_threshold_usd = 0.10
    monkeypatch.setattr(app, "_cold_cache_estimate", AsyncMock(return_value=warning))
    process = AsyncMock()
    handoff = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert app._chat_input is not None
        app._chat_input.value = "keep this draft"
        if allow_send:
            await pilot.press("enter")
        else:
            app._check_cache_expiry()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ColdCacheWarningScreen)
        options = screen.query(".cold-cache-choice")
        for widget in screen.query(".cold-cache-choice, .cold-cache-help"):
            assert screen.find_widget(widget).visible_region == widget.region
        body = screen.query_one(VerticalScroll)
        if body.max_scroll_y:
            await pilot.press(*("pagedown",) * 10)
            await pilot.pause()
            assert body.scroll_y == body.max_scroll_y
            await pilot.press("pageup")
            await pilot.pause()
            assert body.scroll_y < body.max_scroll_y
        else:
            assert screen.query_one(Vertical).region.height < screen.size.height
        for key in ("tab",) * len(options) + ("shift+tab", "down", "up"):
            await pilot.press(key)
            await pilot.pause()
            selected = screen.query_one(".cold-cache-choice.-selected")
            assert selected.region.height > 0
            assert screen.find_widget(selected).visible_region == selected.region
            for hint in screen.query(".cold-cache-help"):
                assert screen.find_widget(hint).visible_region == hint.region
        assert screen._options[screen._selected].choice is ColdCacheChoice.CANCEL
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        assert app._chat_input.value == "keep this draft"
        process.assert_not_awaited()
        handoff.assert_not_awaited()


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


@pytest.mark.parametrize("typing_during", ["timer", "estimate"])
@pytest.mark.parametrize("submit", [False, True])
async def test_expiry_defers_typing_without_consuming_submit(
    typing_during: str, submit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    clock = MagicMock(return_value=100.0)
    monkeypatch.setattr("deepagents_code.app._monotonic", clock)
    estimating = asyncio.Event()
    release = asyncio.Event()
    estimates = 0

    async def estimate() -> None:
        nonlocal estimates
        estimates += 1
        # Pause the final estimate, after eligibility passed, to exercise
        # typing that begins after the timer scheduled the prompt.
        if typing_during == "estimate" and estimates == 2:
            estimating.set()
            await release.wait()

    monkeypatch.setattr(app, "_cold_cache_estimate", estimate)
    process = AsyncMock()
    handoff = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        if typing_during == "estimate":
            app._check_cache_expiry()
            await asyncio.wait_for(estimating.wait(), timeout=5)
        assert app._chat_input is not None
        app._chat_input.focus_input()
        await pilot.press(*"draft")
        if typing_during == "timer":
            app._check_cache_expiry()
        release.set()
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        assert app._chat_input.value == "draft"
        assert not app._modal_command_running()

        if submit:
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ColdCacheWarningScreen)
            handoff.assert_not_awaited()
            process.assert_not_awaited()
            assert app._lc_thread_id == "source"
            # Sending requires a separate, explicit choice after submission.
            await pilot.press("down", "enter")
            await pilot.pause()
            process.assert_awaited_once_with("draft", "normal")
        else:
            clock.return_value += 10
            app._check_cache_expiry()
            await pilot.pause()
            assert isinstance(app.screen, ColdCacheWarningScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert app._chat_input.value == "draft"
            process.assert_not_awaited()
        handoff.assert_not_awaited()
        assert app._lc_thread_id == "source"


@pytest.mark.parametrize("failure", ["empty", "seed", "finish", "metadata"])
async def test_handoff_failure_keeps_source_thread(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    app._lc_thread_id = "source"
    remote = MagicMock()
    remote.aoffload = AsyncMock(
        return_value={
            "status": "summarized",
            "summary": "  " if failure == "empty" else "LLM summary",
            "archive_path": "/conversation_history/source.md",
        }
    )
    remote.aensure_thread = AsyncMock()
    remote.abind_workspace = AsyncMock()
    remote.aupdate_state = AsyncMock(
        side_effect=RuntimeError("write failed") if failure == "seed" else None
    )
    if failure == "finish":
        remote.aupdate_state.side_effect = [None, RuntimeError("completion failed")]
    monkeypatch.setattr(
        "deepagents_code.sessions.thread_exists", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        "deepagents_code.sessions.set_thread_metadata",
        AsyncMock(
            side_effect=RuntimeError("metadata failed")
            if failure == "metadata"
            else None
        ),
    )
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_set_spinner", AsyncMock())
    monkeypatch.setattr(app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    resume = AsyncMock()
    monkeypatch.setattr(app, "_resume_thread", resume)
    with pytest.raises(RuntimeError):
        await app._handoff_expired_cache("source")
    resume.assert_not_awaited()
    assert app._lc_thread_id == "source"


@pytest.mark.parametrize("switch_during", ["spinner", "summary", "seed"])
async def test_handoff_preserves_source_configuration_after_thread_switch(
    switch_during: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_cwd = str(tmp_path / "source-project")
    app = DeepAgentsApp(cwd=source_cwd, assistant_id="source-agent")
    app._lc_thread_id = "source"
    app._model_override = "test:source-model"
    app._model_params_override = {"reasoning": {"effort": "high"}}

    def switch_thread(*_args: object, **_kwargs: object) -> None:
        app._lc_thread_id = "other"
        app._cwd = str(tmp_path / "other-project")
        app._assistant_id = "other-agent"
        app._model_override = "test:other-model"
        assert app._model_params_override is not None
        app._model_params_override["reasoning"]["effort"] = "low"

    remote = MagicMock()
    remote.aoffload = AsyncMock(
        return_value={
            "status": "summarized",
            "summary": "Source conversation",
            "archive_path": "/conversation_history/source.md",
        },
    )

    async def summarize(**_kwargs: object) -> dict[str, object]:  # noqa: RUF029  # mock contract
        switch_thread()
        return remote.aoffload.return_value

    if switch_during == "summary":
        remote.aoffload.side_effect = summarize
    remote.aensure_thread = AsyncMock()
    remote.abind_workspace = AsyncMock(
        side_effect=switch_thread if switch_during == "seed" else None
    )
    remote.aupdate_state = AsyncMock()
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(
        app,
        "_set_spinner",
        AsyncMock(side_effect=switch_thread if switch_during == "spinner" else None),
    )
    monkeypatch.setattr(app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    resume = AsyncMock()
    monkeypatch.setattr(app, "_resume_thread", resume)

    await app._handoff_expired_cache("source")

    assert remote.aupdate_state.await_args is not None
    assert remote.aoffload.await_args is not None
    config, values = remote.aupdate_state.await_args_list[0].args
    remote.abind_workspace.assert_awaited_once_with(config, source_cwd)
    assert values["_model_spec"] == "test:source-model"
    assert values["_model_params"] == {"reasoning": {"effort": "high"}}
    assert remote.aensure_thread.await_args is not None
    metadata = remote.aensure_thread.await_args.args[0]["metadata"]
    assert metadata["agent_name"] == "source-agent"
    assert metadata["cwd"] == source_cwd
    assert remote.aoffload.await_args.kwargs["context"]["model"] == "test:source-model"
    assert app._lc_thread_id == "other"
    resume.assert_not_awaited()


async def test_seeding_handoff_preserves_active_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.client.remote_client import RemoteAgent

    remote = RemoteAgent("http://localhost:8123")
    bindings: dict[str, str] = {}

    def bind(
        config: dict[str, dict[str, str]], cwd: str, **_kwargs: object
    ) -> tuple[dict[str, str], None]:
        thread_id = config["configurable"]["thread_id"]
        assert bindings.setdefault(thread_id, cwd) == cwd, "workspace conflict"
        return {"cwd": cwd}, None

    monkeypatch.setattr(remote, "_request_workspace", AsyncMock(side_effect=bind))
    monkeypatch.setattr(remote, "aensure_thread", AsyncMock())
    monkeypatch.setattr(remote, "aupdate_state", AsyncMock())
    monkeypatch.setattr("deepagents_code.sessions.set_thread_metadata", AsyncMock())
    # The user opened another workspace while the source was summarized.
    active_config = {"configurable": {"thread_id": "other"}}
    await remote.aswitch_workspace(active_config, "/other-project")

    child_id = await DeepAgentsApp._seed_handoff_thread(
        remote, "summary", cwd="/source-project", agent_name="agent", context={}
    )

    assert bindings[child_id] == "/source-project"
    assert await remote._workspace_for_thread(active_config) == {
        "cwd": "/other-project"
    }
    assert await remote._workspace_for_thread(
        {"configurable": {"thread_id": "next"}}
    ) == {"cwd": "/other-project"}


@pytest.mark.parametrize(
    ("queued", "assistant_id"), [(False, None), (True, "researcher")]
)
async def test_handoff_child_is_discoverable_and_resumable(
    queued: bool, assistant_id: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from deepagents_code import sessions
    from deepagents_code.app import DEFAULT_ASSISTANT_ID
    from deepagents_code.resume_state import ResumeStateMiddleware

    app = DeepAgentsApp(assistant_id=assistant_id)
    owner = assistant_id or DEFAULT_ASSISTANT_ID
    app._lc_thread_id = "source"
    remote = MagicMock()
    remote.aoffload = AsyncMock(
        return_value={
            "status": "summarized",
            "summary": "LLM summary",
            "archive_path": "/conversation_history/source.md",
        }
    )
    remote.aensure_thread = AsyncMock()
    remote.abind_workspace = AsyncMock()
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_set_spinner", AsyncMock())
    monkeypatch.setattr(app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    monkeypatch.setattr(
        DeepAgentsApp,
        "_resume_cutoff",
        lambda: (datetime.now(UTC) - timedelta(days=7), "user config", True),
    )

    async def check_resume(child_id: str) -> None:
        assert await sessions.get_thread_agent(child_id) == owner
        assert await app._thread_resume_block(child_id) is None

    resume = AsyncMock(side_effect=check_resume)
    monkeypatch.setattr(app, "_resume_thread", resume)
    if queued:
        app._pending_messages.append(QueuedMessage("arrived", "normal"))

    async with sessions.get_checkpointer() as checkpointer:
        model = FakeListChatModel(responses=["Next reply", "Unexpected extra reply"])
        graph = create_agent(
            model,
            middleware=[ResumeStateMiddleware()],
            checkpointer=checkpointer,
        )

        async def update_state(
            config: "RunnableConfig", values: dict[str, object] | None, *, as_node: str
        ) -> None:
            # The HTTP state API forwards the thread ID, but drops config metadata.
            await graph.aupdate_state(
                {"configurable": {"thread_id": config["configurable"]["thread_id"]}},
                values,
                as_node=as_node,
            )

        remote.aupdate_state = AsyncMock(side_effect=update_state)
        await app._handoff_expired_cache("source")
        assert remote.aoffload.await_args is not None
        assert remote.aoffload.await_args.kwargs["handoff"] is True
        assert remote.aupdate_state.await_args is not None
        child_id = remote.aupdate_state.await_args.args[0]["configurable"]["thread_id"]
        assert child_id != "source"
        state = await graph.aget_state({"configurable": {"thread_id": child_id}})
        content = state.values["messages"][0].text
        assert "LLM summary" in content
        assert "Previous thread ID: source" in content
        assert "/conversation_history/source.md" in content
        # Offload rejects any of these fields before attempting summarization.
        assert not state.next
        assert not state.tasks
        assert not state.interrupts
        assert model.i == 0
        assert len(state.values["messages"]) == 1
        assert await sessions.get_thread_agent(child_id) == owner
        continued = await graph.ainvoke(
            {"messages": [HumanMessage("Continue from the summary.")]},
            {
                "configurable": {"thread_id": child_id},
                "metadata": {"agent_name": owner, "cwd": app._cwd},
            },
        )
        assert [message.text for message in continued["messages"]] == [
            state.values["messages"][0].text,
            "Continue from the summary.",
            "Next reply",
        ]

    threads = await sessions.list_threads(agent_name=owner, cwd=app._cwd)
    assert [thread["thread_id"] for thread in threads] == [child_id]
    assert await sessions.get_thread_agent(child_id) == owner
    assert await app._thread_resume_block(child_id) is None
    if queued:
        resume.assert_not_awaited()
        assert app._pending_messages[0].text == "arrived"
        assert app._lc_thread_id == "source"
    else:
        resume.assert_awaited_once_with(child_id)


@pytest.mark.parametrize("switch_during_sync", [False, True])
async def test_remote_handoff_activity_refreshes_resume_age(
    switch_during_sync: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Later remote requests keep an old discovery seed resumable."""
    from deepagents_code import sessions

    app = DeepAgentsApp(agent=MagicMock(), thread_id="child")
    old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    recent = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    for thread_id in ("child", "other"):
        await sessions.save_thread_seed(
            thread_id,
            {"messages": [HumanMessage("Summary")]},
            agent_name="researcher",
            cwd=app._cwd,
        )
    async with sessions._connect() as conn:
        await conn.execute(
            "UPDATE checkpoints SET metadata = json_set(metadata, '$.updated_at', ?)",
            (old,),
        )
        await conn.commit()
    monkeypatch.setattr(
        DeepAgentsApp,
        "_resume_cutoff",
        lambda: (datetime.now(UTC) - timedelta(days=7), "user config", True),
    )
    assert await app._thread_resume_block("child") is not None

    def read_remote_state(thread_id: str) -> dict[str, object]:
        assert thread_id == "child"
        if switch_during_sync:
            app._lc_thread_id = "other"
        return {"_last_model_request_at": recent}

    monkeypatch.setattr(
        app, "_get_thread_state_values", AsyncMock(side_effect=read_remote_state)
    )
    await app._sync_session_cost_from_checkpoint()

    assert await app._thread_resume_block("child") is None
    assert await sessions.get_thread_updated_at("child") == recent
    assert await sessions.get_thread_updated_at("other") == old
    assert await sessions.get_thread_agent("child") == "researcher"
    assert await sessions.get_thread_cwd("child") == app._cwd
    # Reading an older checkpoint must never move discovery time backward.
    app._lc_thread_id = "child"
    monkeypatch.setattr(
        app,
        "_get_thread_state_values",
        AsyncMock(return_value={"_last_model_request_at": old}),
    )
    await app._sync_session_cost_from_checkpoint()
    assert await sessions.get_thread_updated_at("child") == recent


@pytest.mark.parametrize(
    ("local_database_initialized", "resume_via"),
    [(False, "automatic"), (True, "automatic"), (False, "picker"), (False, "id")],
)
async def test_handoff_with_separate_server_checkpoints(
    local_database_initialized: bool,
    resume_via: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote-only child must be registered, exposed, and opened successfully."""
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents_code import sessions
    from deepagents_code.client.remote_client import RemoteAgent
    from deepagents_code.resume_state import ResumeStateMiddleware
    from deepagents_code.tui.widgets.thread_selector import ThreadSelectorScreen

    if local_database_initialized:
        async with sessions.get_checkpointer() as checkpointer:
            await checkpointer.setup()

    app = DeepAgentsApp(assistant_id="researcher")
    app._lc_thread_id = "source"
    remote = RemoteAgent("http://server:8123")
    graph = create_agent(
        FakeListChatModel(responses=["Unused"]),
        middleware=[ResumeStateMiddleware()],
        checkpointer=InMemorySaver(),
    )

    async def update_state(
        config: "RunnableConfig", values: dict[str, object] | None, *, as_node: str
    ) -> None:
        await graph.aupdate_state(
            {"configurable": config["configurable"]}, values, as_node=as_node
        )

    transport = MagicMock()
    transport.aupdate_state = AsyncMock(side_effect=update_state)
    transport.aget_state = graph.aget_state
    registered = AsyncMock()
    transport._validate_client.return_value.threads.create = registered
    monkeypatch.setattr(remote, "_get_graph", lambda: transport)
    monkeypatch.setattr(remote, "abind_workspace", AsyncMock())

    def summarize(**_kwargs: object) -> dict[str, str]:
        if resume_via != "automatic":
            # Shell activity arriving during the summary still needs recovery.
            app._buffer_shell_for_model_context("echo later", "later result", 0)
        return {
            "status": "summarized",
            "summary": "Remote conversation summary",
            "archive_path": "/conversation_history/source.md",
        }

    monkeypatch.setattr(remote, "aoffload", AsyncMock(side_effect=summarize))
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
    monkeypatch.setattr(app, "_run_session_start_hook", AsyncMock(return_value=False))
    monkeypatch.setattr(type(app._hooks), "on_session_end", AsyncMock())
    monkeypatch.setattr(
        DeepAgentsApp,
        "_resume_cutoff",
        lambda: (datetime.now(UTC) - timedelta(days=7), "user config", True),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        monkeypatch.setattr(app, "_agent", remote)
        assert app._chat_input is not None
        app._chat_input.value = "keep this edited draft"
        child_id = await app._run_cache_handoff("source")
        assert child_id is not None
        threads = await sessions.list_threads(agent_name="researcher", cwd=app._cwd)
        assert [thread["thread_id"] for thread in threads] == [child_id]
        assert await app._thread_resume_block(child_id) is None
        assert await sessions.get_thread_cwd(child_id) == app._cwd

        if resume_via != "automatic":
            assert app._lc_thread_id == "source"
            # Verify resume fetches current server history, not the local seed.
            await graph.aupdate_state(
                {"configurable": {"thread_id": child_id}},
                {"messages": [HumanMessage("Added on the server")]},
                as_node="model",
            )
            await graph.aupdate_state(
                {"configurable": {"thread_id": child_id}}, None, as_node="__end__"
            )
            if resume_via == "picker":
                await app._handle_threads_command("/threads")
                await pilot.pause()
                assert isinstance(app.screen, ThreadSelectorScreen)
                async with asyncio.timeout(5):
                    while not app.screen._disk_load_complete:
                        await pilot.pause()
                assert [t["thread_id"] for t in app.screen._threads] == [child_id]
                await pilot.press("enter")
            else:
                await app._handle_threads_command(f"/threads -r {child_id}")
            async with asyncio.timeout(5):
                while app._lc_thread_id != child_id or app._thread_switching:
                    await pilot.pause()

        assert app._lc_thread_id == child_id
        assert app._session_state is not None
        assert app._session_state.thread_id == child_id
        assert app._chat_input.value == "keep this edited draft"
        displayed = [m.content for m in app._message_store.get_all_messages()]
        assert any("Remote conversation summary" in content for content in displayed)
        if resume_via != "automatic":
            assert "Added on the server" in displayed
        state = await graph.aget_state({"configurable": {"thread_id": child_id}})
        assert not state.next


@pytest.mark.parametrize(
    "outcome",
    [
        "write_failure",
        "cancel",
        "lost_response",
        "summary_failure",
        "during_save",
        "during_summary",
        "running_shell",
    ],
)
async def test_handoff_preserves_shell_context(
    outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from deepagents_code import sessions
    from deepagents_code.resume_state import ResumeStateMiddleware

    app = DeepAgentsApp()
    app._lc_thread_id = "source"
    app._buffer_shell_for_model_context("echo important", "important result", 0)
    remote = MagicMock()
    remote.aensure_thread = AsyncMock()
    remote.abind_workspace = AsyncMock()
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_set_spinner", AsyncMock())
    monkeypatch.setattr(app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    source_config: RunnableConfig = {"configurable": {"thread_id": "source"}}
    archive: list[BaseMessage] = []

    async with sessions.get_checkpointer() as checkpointer:
        graph = create_agent(
            FakeListChatModel(responses=["original reply"]),
            middleware=[ResumeStateMiddleware()],
            checkpointer=checkpointer,
        )
        await graph.ainvoke({"messages": [HumanMessage("original")]}, source_config)

        async def update_state(
            config: "RunnableConfig",
            values: dict[str, object] | None,
            *,
            as_node: str | None = None,
            recovery: bool = False,
        ) -> None:
            del recovery
            if outcome == "write_failure":
                msg = "checkpoint down"
                raise RuntimeError(msg)
            if outcome == "cancel":
                raise asyncio.CancelledError
            # Cross the same serialization boundary as the HTTP client: the graph
            # must not assign IDs back onto the app's buffered message objects.
            await graph.aupdate_state(config, deepcopy(values), as_node=as_node)
            if outcome == "lost_response":
                msg = "response lost"
                raise RuntimeError(msg)
            if outcome == "during_save" and config == source_config:
                app._buffer_shell_for_model_context("echo later", "later result", 0)

        async def summarize(**_kwargs: object) -> dict[str, object]:
            state = await graph.aget_state(source_config)
            # The offload endpoint rejects any checkpoint with pending work.
            assert not state.next
            assert not state.tasks
            archive[:] = state.values["messages"]
            if outcome == "during_summary":
                app._buffer_shell_for_model_context("echo later", "later result", 0)
            if outcome == "running_shell":
                app._shell_running = True
            if outcome == "summary_failure":
                return {
                    "status": "failed",
                    "error": "summary failed",
                    "archive_path": None,
                }
            return {
                "status": "summarized",
                "summary": "\n".join(m.text for m in archive),
                "archive_path": "/conversation_history/source.md",
            }

        def resume(child_id: str) -> None:
            # Resuming clears this buffer along with the previous transcript.
            app._pending_shell_messages.clear()
            app._lc_thread_id = child_id

        remote.aupdate_state = AsyncMock(side_effect=update_state)
        remote.aoffload = AsyncMock(side_effect=summarize)
        monkeypatch.setattr(app, "_resume_thread", AsyncMock(side_effect=resume))

        if outcome in {"write_failure", "cancel", "lost_response", "summary_failure"}:
            error = asyncio.CancelledError if outcome == "cancel" else RuntimeError
            with pytest.raises(error):
                await app._handoff_expired_cache("source")
            assert app._lc_thread_id == "source"
            if outcome != "summary_failure":
                remote.aoffload.assert_not_awaited()
                assert len(app._pending_shell_messages) == 1
                assert "important result" in app._pending_shell_messages[0].text
            outcome = "success"

        await app._handoff_expired_cache("source")
        original = await graph.aget_state(source_config)
        assert remote.aupdate_state.await_args is not None
        child_id = remote.aupdate_state.await_args.args[0]["configurable"]["thread_id"]
        child = await graph.aget_state({"configurable": {"thread_id": child_id}})
        assert not original.next
        assert not original.tasks
        assert len(original.values["messages"]) == 3
        assert "important result" in original.values["messages"][2].text
        assert "important result" in archive[2].text
        assert "important result" in child.values["messages"][0].text
        if outcome in {"during_save", "during_summary", "running_shell"}:
            assert app._lc_thread_id == "source"
            if outcome != "running_shell":
                assert len(app._pending_shell_messages) == 1
                assert "later result" in app._pending_shell_messages[0].text
        else:
            assert app._lc_thread_id == child_id
            assert app._pending_shell_messages == []


@pytest.mark.parametrize(
    ("mode", "keys"),
    [("expiry", ("escape",)), ("send", ("shift+tab", "enter")), ("off", ())],
)
async def test_send_timing_restores_draft_without_spending(
    mode: str, keys: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    process = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr("deepagents_code.app._load_cache_prompt_mode", lambda: mode)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert app._chat_input is not None
        if mode != "expiry":
            app._check_cache_expiry()
            await pilot.pause()
            assert not isinstance(app.screen, ColdCacheWarningScreen)
        await app._dispatch_queued_message(QueuedMessage("keep my request", "normal"))
        await pilot.pause()
        if mode == "off":
            process.assert_awaited_once_with("keep my request", "normal")
            assert not isinstance(app.screen, ColdCacheWarningScreen)
            return
        assert isinstance(app.screen, ColdCacheWarningScreen)
        assert "estimate isn't available" in app.screen._body()
        await pilot.press(*keys)
        await pilot.pause()
        process.assert_not_awaited()
        assert app._chat_input.value == "keep my request"
        app._check_cache_expiry()
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        await app._dispatch_queued_message(QueuedMessage("keep my request", "normal"))
        process.assert_awaited_once_with("keep my request", "normal")


@pytest.mark.parametrize(
    ("mode", "submit"), [("expiry", False), ("expiry", True), ("send", True)]
)
@pytest.mark.parametrize(
    ("delta", "debug_forced", "expect_prompt"),
    [
        (0.01, False, False),
        (10.0, False, True),
        (12.0, False, True),
        (0.01, True, True),
    ],
)
async def test_handoff_honors_cost_threshold(
    mode: str,
    submit: bool,
    delta: float,
    debug_forced: bool,
    expect_prompt: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.cold_cache import (
        ColdCacheWarning,
        PromptCachePolicy,
        RewarmEstimate,
    )

    warning = ColdCacheWarning(
        policy=PromptCachePolicy(
            provider_name="OpenAI",
            window_seconds=1800,
            confidence="may_be_cold",
            minimum_tokens=1024,
            write_bucket="generic",
        ),
        estimate=RewarmEstimate(cold_cost_usd=25.0, incremental_cost_usd=delta),
        context_tokens=84_000,
        age_seconds=11_520,
        reason="idle",
    )
    app = DeepAgentsApp()
    app._cold_cache_warning_threshold_usd = 10.0
    monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG_COLD_CACHE", str(int(debug_forced)))
    process = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr(app, "_cold_cache_estimate", AsyncMock(return_value=warning))
    monkeypatch.setattr("deepagents_code.app._load_cache_prompt_mode", lambda: mode)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert app._chat_input is not None
        app._chat_input.value = "send this request"
        if submit:
            await pilot.press("enter")
        else:
            app._check_cache_expiry()
        await pilot.pause()
        if expect_prompt:
            assert isinstance(app.screen, ColdCacheWarningScreen)
            process.assert_not_awaited()
            await pilot.press("tab", "enter")
            await pilot.pause()
        else:
            assert not isinstance(app.screen, ColdCacheWarningScreen)
            if submit:
                assert not app._cache_expiry_seen
        if submit:
            process.assert_awaited_once_with("send this request", "normal")
        else:
            process.assert_not_awaited()
            assert app._chat_input.value == "send this request"


async def test_idle_handoff_estimate_failure_preserves_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    app._cold_cache_warning_threshold_usd = 10.0
    monkeypatch.setattr(
        app,
        "_cold_cache_estimate",
        AsyncMock(side_effect=RuntimeError("pricing unavailable")),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, ColdCacheWarningScreen)
        assert "estimate isn't available" in app.screen._body()
        await pilot.press("escape")


@pytest.mark.parametrize("allow_send", [False, True])
@pytest.mark.parametrize("cache_state", ["warm", "below_minimum", "empty", "unpriced"])
async def test_handoff_distinguishes_cache_skips_from_missing_prices(
    cache_state: str, allow_send: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired footer alone must not offer a needless paid handoff."""
    app = DeepAgentsApp()
    app._model_override = "openai:gpt-5.6"
    app._context_tokens = {"below_minimum": 500, "empty": 0}.get(cache_state, 50_000)
    app._cold_cache_warning_threshold_usd = 0.10
    config = MagicMock()
    config.get_effective_kwargs.return_value = {}
    config.get_base_url.return_value = None
    monkeypatch.setattr("deepagents_code.model_config.ModelConfig.load", lambda: config)
    monkeypatch.setattr(
        "deepagents_code.model_config.is_warning_suppressed", lambda *_a: False
    )
    monkeypatch.setattr(
        "deepagents_code.cold_cache.estimate_rewarm_cost", lambda *_a: None
    )
    process = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    old = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    recent = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        app._sync_cache_state_from_state(
            {
                "_last_model_request_at": recent if cache_state == "warm" else old,
                "_last_cache_model_spec": app._model_override,
                "_last_cache_params": {},
                "_last_cache_endpoint": "default",
                "_last_cache_use": {
                    "requested_at": old,
                    "model_spec": app._model_override,
                    "endpoint": "default",
                    "params": {},
                },
            }
        )
        await app._refresh_cache_timing()
        assert app._status_bar is not None
        assert app._status_bar.cache_expires_at is not None
        assert app._status_bar.cache_expires_at < datetime.now(UTC)
        if allow_send:
            await pilot.press(*"continue", "enter")
        else:
            app._check_cache_expiry()
        await pilot.pause()
        if cache_state == "unpriced":
            assert isinstance(app.screen, ColdCacheWarningScreen)
            assert "estimate isn't available" in app.screen._body()
            process.assert_not_awaited()
            await pilot.press("escape")
        else:
            assert not isinstance(app.screen, ColdCacheWarningScreen)
            if allow_send:
                process.assert_awaited_once_with("continue", "normal")
            else:
                process.assert_not_awaited()


def _record_errors(app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    errors: list[str] = []
    mount = app._mount_message

    async def record(widget: object) -> None:
        if isinstance(widget, ErrorMessage):
            errors.append(str(widget._content))
        await mount(widget)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(app, "_mount_message", record)
    return errors


@pytest.mark.parametrize("outcome", ["sent", "failure", "thread_changed"])
async def test_expiry_send_action_dispatches_once_in_current_thread(
    outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()

    async def send(_text: str, _mode: str) -> None:
        assert app._lc_thread_id == "source"
        if outcome == "failure":
            msg = "send failed"
            raise RuntimeError(msg)
        await app._set_spinner("Thinking")

    process = AsyncMock(side_effect=send)
    handoff = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    monkeypatch.setattr("deepagents_code.app._load_cache_prompt_mode", lambda: "send")
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        errors = _record_errors(app, monkeypatch)
        await app._dispatch_queued_message(QueuedMessage("send this request", "normal"))
        await pilot.pause()
        assert isinstance(app.screen, ColdCacheWarningScreen)
        assert app._chat_input is not None
        if outcome == "thread_changed":
            app._lc_thread_id = "other"
            app._chat_input.value = "unrelated draft"
        await pilot.press("tab", "enter")
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        handoff.assert_not_awaited()
        if outcome == "thread_changed":
            process.assert_not_awaited()
            assert app._chat_input.value == "unrelated draft"
        else:
            process.assert_awaited_once_with("send this request", "normal")
            assert app._lc_thread_id == "source"
            if outcome == "failure":
                assert app._chat_input.value == "send this request"
                assert errors
            else:
                assert app._chat_input.value == ""
                assert app._loading_widget is not None
                assert not errors


@pytest.mark.parametrize("outcome", ["success", "failure"])
@pytest.mark.parametrize("submitted", [False, True])
async def test_handoff_pauses_submission_but_keeps_draft_editable(
    outcome: str, submitted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    started = asyncio.Event()
    release = asyncio.Event()

    async def handoff(_thread_id: str) -> str:
        started.set()
        await release.wait()
        if outcome == "failure":
            msg = "summary failed"
            raise RuntimeError(msg)
        app._lc_thread_id = "child"
        assert app._status_bar is not None
        app._status_bar.cache_expires_at = None
        return "child"

    process = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        errors = _record_errors(app, monkeypatch)
        assert app._chat_input is not None
        app._chat_input.value = "draft"
        if submitted:
            await pilot.press("enter")
        else:
            app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, ColdCacheWarningScreen)
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), timeout=5)
        await pilot.pause()
        assert app._chat_input.value == "draft"
        app._chat_input.focus_input()
        await pilot.press("end", "space", "e", "d", "i", "t")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app._chat_input.value == "draft edit"
        assert not app._pending_messages
        process.assert_not_awaited()

        release.set()
        async with asyncio.timeout(5):
            while app._modal_command_running():
                await pilot.pause()
        assert app._chat_input.value == "draft edit"
        assert app._lc_thread_id == ("child" if outcome == "success" else "source")
        assert not app._pending_messages
        process.assert_not_awaited()
        if outcome == "failure":
            assert len(errors) == 1
            assert "summary failed" in errors[0]
            assert "Your original conversation is still available." in errors[0]
        else:
            assert errors == []
        await pilot.press("enter")
        await pilot.pause()
        process.assert_awaited_once_with("draft edit", "normal")


@pytest.mark.parametrize("cancel", ["escape", "force-clear"])
@pytest.mark.parametrize("submitted", [False, True])
async def test_cancel_handoff_restores_submission(
    cancel: str, submitted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    started = asyncio.Event()
    release = asyncio.Event()

    async def handoff(_thread_id: str) -> str:
        await app._set_spinner("Summarizing")
        started.set()
        await release.wait()
        return "child"

    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
    monkeypatch.setattr(app, "_run_session_start_hook", AsyncMock(return_value=True))
    monkeypatch.setattr(type(app._hooks), "on_session_end", AsyncMock())
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert app._chat_input is not None
        if submitted:
            await pilot.press(*"draft", "enter")
        else:
            app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, ColdCacheWarningScreen)
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), timeout=5)
        await pilot.pause()
        task = app._modal_command_tasks["cache-expiry"]
        assert app._chat_input.submission_block_reason
        app._chat_input.focus_input()
        if cancel == "force-clear":
            await pilot.press(
                "home", "shift+end", "backspace", *"/force-clear", "enter"
            )
        else:
            await pilot.press("escape")
        await pilot.pause()
        assert app._session_state is not None
        thread_id = app._lc_thread_id
        assert thread_id == app._session_state.thread_id
        assert (thread_id != "source") == (cancel == "force-clear")
        assert app._chat_input.submission_block_reason is None
        assert task.cancelled()
        assert not app._modal_command_running()
        assert app._loading_widget is None

        process = AsyncMock()
        monkeypatch.setattr(app, "_process_message", process)
        await pilot.press("home", "shift+end", "backspace", *"next", "enter")
        await pilot.pause()
        process.assert_awaited_once_with("next", "normal")
        assert not app._pending_messages
        assert app._lc_thread_id == thread_id


@pytest.mark.parametrize("other_draft", ["", "unrelated draft"])
async def test_handoff_does_not_restore_draft_into_unrelated_thread(
    other_draft: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()

    def summarize(**_kwargs: object) -> dict[str, str]:
        # The user leaves the source while its summary is being generated.
        app._lc_thread_id = "other"
        assert app._chat_input is not None
        app._chat_input.value = other_draft
        return {
            "status": "summarized",
            "summary": "source summary",
            "archive_path": "/conversation_history/source.md",
        }

    remote = MagicMock()
    remote.aoffload = AsyncMock(side_effect=summarize)
    remote.aensure_thread = AsyncMock()
    remote.abind_workspace = AsyncMock()
    remote.aupdate_state = AsyncMock()
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr("deepagents_code.sessions.set_thread_metadata", AsyncMock())
    process = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        await app._dispatch_queued_message(QueuedMessage("source draft", "normal"))
        task = app._modal_command_tasks["cache-expiry"]
        await pilot.pause()
        await pilot.press("enter")
        await asyncio.wait_for(task, timeout=5)
        await pilot.pause()

        assert remote.aupdate_state.await_args is not None
        child_id = remote.aupdate_state.await_args.args[0]["configurable"]["thread_id"]
        assert child_id not in {"source", "other"}
        assert app._lc_thread_id == "other"
        assert app._chat_input is not None
        assert app._chat_input.value == other_draft
        process.assert_not_awaited()
        assert not app._modal_command_running()


@pytest.mark.parametrize("cancel", ["escape", "shutdown", "force-clear"])
@pytest.mark.parametrize("pause_at", ["prefetch", "history"])
async def test_handoff_switch_cancellation_preserves_conversation(
    cancel: str, pause_at: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import _ThreadHistoryPayload
    from deepagents_code.tui.widgets.message_store import MessageData, MessageType

    app = DeepAgentsApp()
    paused = asyncio.Event()
    release = asyncio.Event()
    source = MessageData(type=MessageType.USER, content="original conversation")
    child = MessageData(type=MessageType.USER, content="summarized conversation")
    payloads = {
        "source": _ThreadHistoryPayload([source], 100, "", session_cost_usd=1.25),
        "child": _ThreadHistoryPayload([child], 10, ""),
    }
    load = app._load_thread_history

    async def fetch(thread_id: str) -> _ThreadHistoryPayload:
        if thread_id == "child" and pause_at == "prefetch":
            paused.set()
            await release.wait()
        return payloads[thread_id]

    async def load_history(
        *,
        thread_id: str | None = None,
        preloaded_payload: _ThreadHistoryPayload | None = None,
        resolve_pending_goal: bool = True,
    ) -> None:
        if thread_id == "child" and pause_at == "history":
            paused.set()
            await release.wait()
        await load(
            thread_id=thread_id,
            preloaded_payload=preloaded_payload,
            resolve_pending_goal=resolve_pending_goal,
        )

    monkeypatch.setattr(app, "_fetch_thread_history_data", fetch)
    monkeypatch.setattr(app, "_load_thread_history", load_history)
    monkeypatch.setattr(app, "_thread_resume_block", AsyncMock(return_value=None))
    monkeypatch.setattr(
        app, "_offer_thread_cwd_switch", AsyncMock(return_value="continue")
    )
    monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
    monkeypatch.setattr(app, "_run_session_start_hook", AsyncMock(return_value=False))
    monkeypatch.setattr(type(app._hooks), "on_session_end", AsyncMock())
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        assert app._session_state is not None
        app._session_state.previous_thread_id = "earlier"
        await load(thread_id="source")
        task = app._schedule_off_message_pump(
            app._switch_to_handoff("source", "child"), context="cache-expiry"
        )
        assert task is not None
        await asyncio.wait_for(paused.wait(), timeout=5)
        if cancel == "force-clear":
            assert app._chat_input is not None
            app._chat_input.focus_input()
            await pilot.press(*"/force-clear", "enter")
            await pilot.pause()
            assert task.cancelled()
            assert app._lc_thread_id == app._session_state.thread_id
            assert app._lc_thread_id not in {"source", "child"}
            assert app._message_store.get_message(source.id) is None
            assert app._message_store.get_message(child.id) is None
            assert app._session_cost_usd == pytest.approx(0.0)
            assert not app._thread_switching
            assert not app._modal_command_running()
            return
        if cancel == "escape":
            await pilot.press("escape", "escape")
            release.set()
            await task
            expected, previous, cost = child, "source", 0.0
        else:
            app._cancel_modal_command_tasks()
            with pytest.raises(asyncio.CancelledError):
                await task
            expected, previous, cost = source, "earlier", 1.25
        await pilot.pause()
        assert (
            app._lc_thread_id
            == app._session_state.thread_id
            == ("child" if cancel == "escape" else "source")
        )
        assert app._session_state.previous_thread_id == previous
        assert app._message_store.get_message(expected.id) is not None
        assert app._session_cost_usd == cost
        assert not app._thread_switching
        assert not app._modal_command_running()


async def test_unanswered_prompt_stays_without_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    handoff = AsyncMock()
    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    monkeypatch.setattr("deepagents_code.app._MODAL_WATCHDOG_TIMEOUT_SECONDS", 0.05)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        errors = _record_errors(app, monkeypatch)
        assert app._chat_input is not None
        app._chat_input.value = "keep this draft"
        app._check_cache_expiry()
        await pilot.pause(0.2)
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        assert errors == []
        handoff.assert_not_awaited()
        assert app._chat_input.value == "keep this draft"
        app._check_cache_expiry()
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)


@pytest.mark.parametrize("scope", ["session", "persistent"])
async def test_cold_cache_opt_out_suppresses_handoff(
    scope: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.cold_cache import COLD_CACHE_WARNING_KEY
    from deepagents_code.model_config import suppress_warning

    app = DeepAgentsApp()
    process = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    if scope == "persistent":
        suppress_warning(COLD_CACHE_WARNING_KEY)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        app._cold_cache_suppressed_for_session = scope == "session"
        await app._dispatch_queued_message(QueuedMessage("send it", "normal"))
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        process.assert_awaited_once_with("send it", "normal")
        app._check_cache_expiry()
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)


@pytest.mark.parametrize(
    ("mode", "submit"), [("expiry", False), ("expiry", True), ("send", True)]
)
async def test_zero_threshold_disables_handoff_prompts(
    mode: str, submit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.configuration.resolver import reset_config_resolver
    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(
        f'[warnings]\ncache_prompt = "{mode}"\ncold_cache_min_delta_usd = 0\n'
    )
    reset_config_resolver()
    try:
        app = DeepAgentsApp()
        process = AsyncMock()
        monkeypatch.setattr(app, "_process_message", process)
        async with app.run_test() as pilot:
            await pilot.pause()
            _prepare(app, monkeypatch)
            if submit:
                await app._dispatch_queued_message(QueuedMessage("send it", "normal"))
            else:
                app._check_cache_expiry()
            await pilot.pause()
            assert not isinstance(app.screen, ColdCacheWarningScreen)
            assert not app._cache_expiry_seen
            if submit:
                process.assert_awaited_once_with("send it", "normal")
            else:
                process.assert_not_awaited()
    finally:
        reset_config_resolver()


async def test_resumed_thread_with_lapsed_window_does_not_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()

    async def restore_timing() -> None:  # noqa: RUF029  # mock contract
        # The checkpoint's last request is well past its retention window.
        assert app._status_bar is not None
        app._status_bar.cache_expires_at = datetime.now(UTC) - timedelta(hours=2)

    monkeypatch.setattr(app, "_refresh_cache_timing", restore_timing)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        payload = app._goal_rubric_payload_from_state(
            {"_last_model_request_at": "2026-01-01T00:00:00+00:00"},
            messages=[],
            context_tokens=0,
            model_spec="",
        )
        await app._load_thread_history(preloaded_payload=payload)
        app._check_cache_expiry()
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        # A window that lapses during the session still prompts.
        assert app._status_bar is not None
        app._status_bar.cache_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        app._check_cache_expiry()
        await pilot.pause()
        assert isinstance(app.screen, ColdCacheWarningScreen)
        await pilot.press("escape")


@pytest.mark.parametrize(
    "message",
    [
        QueuedMessage("/help", "command"),
        QueuedMessage("ls", "shell"),
        QueuedMessage("continue the goal", "normal", origin="external"),
    ],
    ids=["command", "shell", "external"],
)
async def test_expiry_never_blocks_non_interactive_dispatch(
    message: QueuedMessage, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    process = AsyncMock()
    monkeypatch.setattr(app, "_process_message", process)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        await app._dispatch_queued_message(message)
        await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)
        process.assert_awaited_once_with(message.text, message.mode)


@pytest.mark.parametrize("dismissal", ["stay", "timeout"])
async def test_expiry_acknowledgment_expires_after_a_cold_request(
    dismissal: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request without a cache hit must not inherit an earlier bypass."""
    from deepagents_code.cold_cache import RewarmEstimate

    app = DeepAgentsApp()
    app._model_override = "openai:gpt-5.6"
    app._context_tokens = 50_000
    app._cold_cache_warning_threshold_usd = 0.10
    config = MagicMock()
    config.get_effective_kwargs.return_value = {}
    config.get_base_url.return_value = None
    monkeypatch.setattr("deepagents_code.model_config.ModelConfig.load", lambda: config)
    monkeypatch.setattr(
        "deepagents_code.model_config.is_warning_suppressed", lambda *_a: False
    )
    monkeypatch.setattr(
        "deepagents_code.cold_cache.estimate_rewarm_cost",
        lambda *_a: RewarmEstimate(cold_cost_usd=1.0, incremental_cost_usd=0.8),
    )
    if dismissal == "timeout":
        monkeypatch.setattr(
            app, "_push_screen_wait", AsyncMock(side_effect=TimeoutError)
        )
    requested_at = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    cache_use = {
        "requested_at": requested_at,
        "model_spec": app._model_override,
        "endpoint": "default",
        "params": {},
    }
    state = {
        "_last_model_request_at": requested_at,
        "_last_cache_model_spec": app._model_override,
        "_last_cache_model_params": {},
        "_last_cache_endpoint": "default",
        "_last_cache_use": cache_use,
    }
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        app._sync_cache_state_from_state(state)
        await app._refresh_cache_timing()
        assert app._status_bar is not None
        expires_at = app._status_bar.cache_expires_at
        assert expires_at is not None
        app._check_cache_expiry()
        await pilot.pause()
        if dismissal == "stay":
            assert isinstance(app.screen, ColdCacheWarningScreen)
            assert "~$1.0" in app.screen._body()
            await pilot.press("escape")
            await pilot.pause()
        assert not isinstance(app.screen, ColdCacheWarningScreen)

        # Re-reading the same checkpoint preserves the user's choice.
        app._sync_cache_state_from_state(state)
        message = QueuedMessage("next", "normal")
        assert await app._cold_cache_warning_for(message) is None

        # A later successful cold request advances the request time but leaves
        # cache activity unchanged for providers that report only cache reads.
        state["_last_model_request_at"] = (
            datetime.now(UTC) - timedelta(hours=1)
        ).isoformat()
        app._sync_cache_state_from_state(state)
        await app._refresh_cache_timing()
        assert app._last_cache_use == cache_use
        assert app._status_bar.cache_expires_at == expires_at
        warning = await app._cold_cache_warning_for(message)
        assert warning is not None
        assert warning.reason == "idle"


async def test_expiry_acknowledgment_does_not_hide_identity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    app._model_override = "openai:gpt-5.6"
    app._last_cache_model_spec = "other"
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
        app._cache_expiry_bypassed = (
            "source",
            app._status_bar.cache_expires_at,
            app._last_model_request_at,
        )
        warning = await app._cold_cache_warning_for(QueuedMessage("next", "normal"))
        assert warning is not None
        assert warning.reason == "identity_changed"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (None, "expiry"),
        ('cache_prompt = "send"', "send"),
        ('cache_prompt = "Off"', "off"),
        ('cache_prompt = "never"', "send"),
    ],
)
def test_mode_resolution_rejects_unknown_values(
    line: str | None, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import _load_cache_prompt_mode
    from deepagents_code.configuration.resolver import reset_config_resolver
    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(f"[warnings]\n{line or ''}\n")
    reset_config_resolver()
    monkeypatch.setattr("deepagents_code.app._warn_invalid_cache_prompt", MagicMock())
    try:
        assert _load_cache_prompt_mode() == expected
    finally:
        reset_config_resolver()
