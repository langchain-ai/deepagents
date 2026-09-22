"""Behavioral coverage for cache-expiry handoffs."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from deepagents_code.app import DeepAgentsApp, QueuedMessage, TextualSessionState
from deepagents_code.tui.modals.cold_cache import ColdCacheWarningScreen
from deepagents_code.tui.widgets.messages import ErrorMessage

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


@pytest.mark.parametrize(
    "failure", [None, "summary", "empty", "seed", "metadata", "queued"]
)
async def test_handoff_persists_recovery_before_switch(
    failure: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    app._lc_thread_id = "source"
    remote = MagicMock()
    remote.aoffload = AsyncMock(
        return_value={
            "status": "failed" if failure == "summary" else "summarized",
            "summary": "  " if failure == "empty" else "LLM summary",
            "archive_path": "/conversation_history/source.md",
            "error": "transcript not saved" if failure == "summary" else None,
        }
    )
    remote.aensure_thread = AsyncMock()
    remote.aswitch_workspace = AsyncMock()
    remote.aupdate_state = AsyncMock(
        side_effect=RuntimeError("write failed") if failure == "seed" else None
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
    assert remote.aoffload.await_args.kwargs["handoff"] is True
    update = remote.aupdate_state.await_args.args[1]
    content = update["messages"][0].text
    assert "LLM summary" in content
    assert "Previous thread ID: source" in content
    assert "/conversation_history/source.md" in content
    child_id = remote.aupdate_state.await_args.args[0]["configurable"]["thread_id"]
    assert child_id != "source"
    resume.assert_awaited_once_with(child_id)


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("assistant_id", [None, "researcher"])
async def test_handoff_child_is_discoverable_and_resumable(
    queued: bool, assistant_id: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langgraph.graph import END, StateGraph

    from deepagents_code import sessions
    from deepagents_code.app import DEFAULT_ASSISTANT_ID

    @dataclass
    class State:
        messages: list[HumanMessage]

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
    remote.aswitch_workspace = AsyncMock()
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
        builder = StateGraph(State)
        builder.add_node("model", lambda state: {"messages": state.messages})
        builder.set_entry_point("model")
        builder.add_edge("model", END)
        graph = builder.compile(checkpointer=checkpointer)

        async def update_state(
            config: "RunnableConfig", values: dict[str, object], *, as_node: str
        ) -> None:
            # The HTTP state API forwards the thread ID, but drops config metadata.
            await graph.aupdate_state(
                {"configurable": {"thread_id": config["configurable"]["thread_id"]}},
                values,
                as_node=as_node,
            )

        remote.aupdate_state = AsyncMock(side_effect=update_state)
        await app._handoff_expired_cache("source")
        assert remote.aupdate_state.await_args is not None
        child_id = remote.aupdate_state.await_args.args[0]["configurable"]["thread_id"]
        state = await graph.aget_state({"configurable": {"thread_id": child_id}})
        assert "LLM summary" in state.values["messages"][0].text

    threads = await sessions.list_threads(agent_name=owner, cwd=app._cwd)
    assert [thread["thread_id"] for thread in threads] == [child_id]
    assert await sessions.get_thread_agent(child_id) == owner
    assert await app._thread_resume_block(child_id) is None
    if queued:
        resume.assert_not_awaited()
    else:
        resume.assert_awaited_once_with(child_id)


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
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
    from langgraph.graph import END, StateGraph

    from deepagents_code import sessions

    @dataclass
    class State:
        messages: Annotated[list[BaseMessage], add_messages]

    app = DeepAgentsApp()
    app._lc_thread_id = "source"
    app._buffer_shell_for_model_context("echo important", "important result", 0)
    remote = MagicMock()
    remote.aensure_thread = AsyncMock()
    remote.aswitch_workspace = AsyncMock()
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_set_spinner", AsyncMock())
    monkeypatch.setattr(app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    source_config: RunnableConfig = {"configurable": {"thread_id": "source"}}
    archive: list[BaseMessage] = []

    async with sessions.get_checkpointer() as checkpointer:
        builder = StateGraph(State)
        builder.add_node("model", lambda state: {"messages": state.messages})
        builder.set_entry_point("model")
        builder.add_edge("model", END)
        graph = builder.compile(checkpointer=checkpointer)
        await graph.aupdate_state(
            source_config, {"messages": [HumanMessage("original")]}, as_node="model"
        )

        async def update_state(
            config: "RunnableConfig",
            values: dict[str, object],
            *,
            as_node: str,
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
        assert len(original.values["messages"]) == 2
        assert "important result" in original.values["messages"][1].text
        assert "important result" in archive[1].text
        assert "important result" in child.values["messages"][0].text
        if outcome in {"during_save", "during_summary", "running_shell"}:
            assert app._lc_thread_id == "source"
            if outcome != "running_shell":
                assert len(app._pending_shell_messages) == 1
                assert "later result" in app._pending_shell_messages[0].text
        else:
            assert app._lc_thread_id == child_id
            assert app._pending_shell_messages == []


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


def _record_errors(app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    errors: list[str] = []
    mount = app._mount_message

    async def record(widget: object) -> None:
        if isinstance(widget, ErrorMessage):
            errors.append(str(widget._content))
        await mount(widget)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(app, "_mount_message", record)
    return errors


@pytest.mark.parametrize("failure", [False, True])
async def test_handoff_keeps_submitted_draft_without_sending(
    failure: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    process = AsyncMock()

    async def handoff(_thread_id: str) -> None:  # noqa: RUF029  # mock contract
        if failure:
            msg = "summary failed"
            raise RuntimeError(msg)
        # A real handoff resumes the child thread, which clears the composer.
        app._lc_thread_id = "child"
        assert app._chat_input is not None
        app._chat_input.value = ""

    monkeypatch.setattr(app, "_process_message", process)
    monkeypatch.setattr(app, "_handoff_expired_cache", handoff)
    async with app.run_test() as pilot:
        await pilot.pause()
        _prepare(app, monkeypatch)
        errors = _record_errors(app, monkeypatch)
        await app._dispatch_queued_message(
            QueuedMessage("retain this request", "normal")
        )
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app._chat_input is not None
        assert app._chat_input.value == "retain this request"
        process.assert_not_awaited()
        assert app._lc_thread_id == ("source" if failure else "child")
        if failure:
            assert len(errors) == 1
            assert "summary failed" in errors[0]
            assert "original thread is unchanged" in errors[0]
        else:
            assert errors == []


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
    monkeypatch.setattr(app, "_cold_cache_warning_for", AsyncMock(return_value=None))
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
