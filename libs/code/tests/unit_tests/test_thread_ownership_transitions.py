"""Deletion and concurrent thread-switch ownership regressions."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepagents_code import sessions
from deepagents_code.app import DeepAgentsApp, _ThreadsResumeTarget
from deepagents_code.thread_ownership import held_lease, release_all, try_acquire

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel

    from deepagents_code.output import OutputFormat


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(sessions, "get_db_path", lambda: tmp_path / "sessions.db")
    yield
    release_all()


@pytest.mark.usefixtures("isolated_state")
@pytest.mark.parametrize("transition", ["clear", "switch", "failed_switch"])
async def test_prestarted_server_releases_initial_thread_after_transition(
    monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    from deepagents_code.app import TextualSessionState
    from deepagents_code.client.remote_client import RemoteAgent

    remote = RemoteAgent("http://test:0", local_ownership=True)
    graph = MagicMock()
    graph._validate_client.return_value.threads.create = AsyncMock()
    remote._graph = graph
    await remote.aensure_thread({"configurable": {"thread_id": "current"}})
    initial = held_lease("current")
    assert initial is not None
    # The app accepts RemoteAgent at runtime but still annotates agent as Pregel.
    app = DeepAgentsApp(
        agent=cast("Pregel", remote), thread_id="current", server_proc=MagicMock()
    )
    app._session_state = TextualSessionState(thread_id="current")
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
    monkeypatch.setattr(app, "_run_session_start_hook", AsyncMock(return_value=True))
    monkeypatch.setattr("deepagents_code.app._new_thread_id", lambda: "target")

    def switch(thread_id: str) -> None:
        if transition == "failed_switch":
            msg = "history failed"
            raise RuntimeError(msg)
        app._lc_thread_id = thread_id

    monkeypatch.setattr(app, "_resume_owned_thread", AsyncMock(side_effect=switch))
    async with app.run_test():
        if transition == "clear":
            await app._handle_command("/clear")
        elif transition == "failed_switch":
            with pytest.raises(RuntimeError, match="history failed"):
                await app._resume_thread("target")
        else:
            await app._resume_thread("target")

        if transition == "failed_switch":
            assert app._lc_thread_id == "current"
            assert held_lease("current") is initial
            assert try_acquire("current") is None
            assert try_acquire("target") is not None
        else:
            assert app._lc_thread_id == "target"
            assert held_lease("current") is None
            assert try_acquire("current") is not None
            assert try_acquire("target") is None


@pytest.mark.usefixtures("isolated_state")
@pytest.mark.parametrize("failure", [None, "stop", "start", "cancel"])
async def test_cwd_server_replacement_fences_previous_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from langgraph.checkpoint.base import empty_checkpoint

    from deepagents_code.client.launch import server_manager
    from deepagents_code.client.remote_client import RemoteAgent
    from deepagents_code.thread_ownership import ThreadOwnershipError, owned_saver_class

    app = DeepAgentsApp(thread_id="current", server_kwargs={})
    app._reserve_thread("current")
    lease = held_lease("current")
    assert lease is not None
    old_client = RemoteAgent("http://old", local_ownership=True)
    new_client = RemoteAgent("http://new", local_ownership=True)
    config = {"configurable": {"thread_id": "current", "checkpoint_ns": ""}}
    old_config: RunnableConfig = {
        "configurable": old_client._prepare_mutation(config)["configurable"]
    }
    app._agent = old_client
    old_server = MagicMock()
    if failure == "stop":
        old_server.stop.side_effect = RuntimeError("old server survived")
    app._server_proc = old_server
    app._mcp_preload_kwargs = None
    start = AsyncMock(return_value=(new_client, MagicMock(), None))
    if failure == "start":
        start.side_effect = RuntimeError("replacement failed")
    elif failure == "cancel":
        start.side_effect = asyncio.CancelledError()
    monkeypatch.setattr(server_manager, "start_server_and_get_agent", start)
    for name in (
        "_sync_status_connection",
        "_preserve_launch_relative_server_paths",
        "on_deep_agents_app_server_ready",
        "notify",
    ):
        monkeypatch.setattr(app, name, MagicMock())
    monkeypatch.setattr(app, "_switch_process_cwd", AsyncMock())

    async with owned_saver_class(db_path=sessions.get_db_path()).from_conn_string(
        str(sessions.get_db_path())
    ) as saver:
        await saver.aput(old_config, empty_checkpoint(), {}, {})
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await app._replace_server_after_cwd_switch(tmp_path)
        else:
            result = await app._replace_server_after_cwd_switch(tmp_path)
            assert result == ("abort" if failure == "start" else "continue")

        assert held_lease("current") is lease
        assert try_acquire("current") is None
        with pytest.raises(ThreadOwnershipError, match="ownership changed"):
            await saver.aput(old_config, empty_checkpoint(), {}, {})

        active_client = new_client
        if failure in {"start", "cancel"}:
            assert app._agent is old_client
            assert app._server_proc is old_server
            active_client = old_client
        current_config: RunnableConfig = {
            "configurable": active_client._prepare_mutation(config)["configurable"]
        }
        checkpoint = empty_checkpoint()
        saved = await saver.aput(current_config, checkpoint, {}, {})
        restored = await saver.aget_tuple(saved)
        assert restored is not None
        assert restored.checkpoint["id"] == checkpoint["id"]


@pytest.mark.usefixtures("isolated_state")
async def test_delete_rejects_reserved_history(monkeypatch: pytest.MonkeyPatch) -> None:
    from deepagents_code import offload

    cleanup = MagicMock()
    monkeypatch.setattr(offload, "delete_offloaded_history", cleanup)
    with closing(sqlite3.connect(sessions.get_db_path())) as conn, conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
        conn.execute("INSERT INTO checkpoints VALUES ('busy')")
    lease = try_acquire("busy")
    assert lease is not None
    with pytest.raises(BlockingIOError, match="Close it there before deleting"):
        await sessions.delete_thread("busy")
    cleanup.assert_not_called()
    assert await sessions.thread_exists("busy")
    lease.release()
    assert await sessions.delete_thread("busy")
    cleanup.assert_called_once_with("busy")
    assert not await sessions.thread_exists("busy")
    assert try_acquire("busy") is not None


@pytest.mark.usefixtures("isolated_state")
async def test_cancelled_delete_holds_reservation_until_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code import btw_cost, offload

    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def delayed_cost(*_args: object) -> None:
        entered.set()
        await proceed.wait()

    cleanup = MagicMock(side_effect=lambda _thread: assert_reserved())

    def assert_reserved() -> None:
        assert try_acquire("target") is None

    monkeypatch.setattr(btw_cost, "delete_cost", delayed_cost)
    monkeypatch.setattr(offload, "delete_offloaded_history", cleanup)
    task = asyncio.create_task(sessions.delete_thread("target"))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert try_acquire("target") is None
    assert not task.done()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    cleanup.assert_called_once_with("target")
    assert try_acquire("target") is not None


@pytest.mark.usefixtures("isolated_state")
@pytest.mark.parametrize("cross_agent", [False, True])
async def test_overlapping_switch_does_not_release_destination(
    monkeypatch: pytest.MonkeyPatch, cross_agent: bool
) -> None:
    app = DeepAgentsApp(thread_id="current", server_kwargs={})
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    app._reserve_thread("current")
    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def switch(_target: object) -> None:
        entered.set()
        await proceed.wait()
        app._lc_thread_id = "target"

    target = _ThreadsResumeTarget(thread_id="target", agent_name="other")
    method = (
        "_confirm_then_resume_owned_cross_agent_thread"
        if cross_agent
        else "_resume_owned_thread"
    )
    monkeypatch.setattr(app, method, switch)

    async def resume() -> None:
        if cross_agent:
            await app._confirm_then_resume_cross_agent_thread(target)
        else:
            await app._resume_thread("target")

    task = asyncio.create_task(resume())
    await entered.wait()
    lease = held_lease("target")
    assert lease is not None
    await resume()
    assert held_lease("target") is lease
    assert try_acquire("target") is None
    proceed.set()
    await task
    assert app._lc_thread_id == "target"
    assert held_lease("target") is lease
    assert held_lease("current") is None
    assert try_acquire("current") is not None


@pytest.mark.usefixtures("isolated_state")
async def test_revisited_thread_binds_before_history_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import TextualSessionState
    from deepagents_code.client.remote_client import RemoteAgent
    from deepagents_code.thread_ownership import OWNER_KEY

    app = DeepAgentsApp(thread_id="current", server_kwargs={})
    agent = RemoteAgent("http://test:0", local_ownership=True)
    app._agent = agent
    app._session_state = TextualSessionState(thread_id="current")
    app._reserve_thread("current")
    old = try_acquire("target")
    assert old is not None
    agent.bind_thread_ownership("target")
    old.release()
    app._reserve_thread("target")
    current = held_lease("target")
    assert current is not None
    assert current.token != old.token
    for name in (
        "_set_spinner",
        "_clear_messages",
        "_reload_hooks",
        "_mount_previous_thread_hint",
        "_remount_pending_goal_rubric_review",
    ):
        monkeypatch.setattr(app, name, AsyncMock())
    for name in (
        "_update_status",
        "_sync_status_queued",
        "_update_tokens",
        "_reset_thread_usage",
        "_update_welcome_banner",
    ):
        monkeypatch.setattr(app, name, MagicMock())
    monkeypatch.setattr(app, "_thread_resume_block", AsyncMock(return_value=None))
    monkeypatch.setattr(
        app, "_offer_thread_cwd_switch", AsyncMock(return_value="continue")
    )
    monkeypatch.setattr(
        app, "_fetch_thread_history_data", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(app, "_run_session_start_hook", AsyncMock(return_value=False))
    monkeypatch.setattr(type(app._hooks), "on_session_end", AsyncMock())

    def load(**_kwargs: object) -> None:
        prepared = agent._prepare_mutation({"configurable": {"thread_id": "target"}})
        assert prepared["configurable"][OWNER_KEY] == current.token

    loaded = AsyncMock(side_effect=load)
    monkeypatch.setattr(app, "_load_thread_history", loaded)
    await app._resume_thread("target")
    loaded.assert_awaited_once()
    assert app._lc_thread_id == "target"


@pytest.mark.usefixtures("isolated_state")
@pytest.mark.parametrize("failure", ["ensure", "metadata", None])
async def test_handoff_seed_releases_only_failed_reservations(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from deepagents_code.thread_ownership import ensure_owned

    monkeypatch.setattr("uuid.uuid4", lambda: "child")
    remote = MagicMock()

    def ensure(_config: object) -> None:
        ensure_owned("child")
        if failure == "ensure":
            msg = "seed failed"
            raise RuntimeError(msg)

    remote.aensure_thread = AsyncMock(side_effect=ensure)
    remote.abind_workspace = AsyncMock()
    remote.aupdate_state = AsyncMock()
    monkeypatch.setattr(sessions, "thread_exists", AsyncMock(return_value=True))
    monkeypatch.setattr(
        sessions,
        "set_thread_metadata",
        AsyncMock(
            side_effect=RuntimeError("seed failed") if failure == "metadata" else None
        ),
    )
    if failure:
        with pytest.raises(RuntimeError, match="seed failed"):
            await DeepAgentsApp._seed_handoff_thread(
                remote, "summary", cwd="/tmp", agent_name="agent", context={}
            )
        assert held_lease("child") is None
        assert try_acquire("child") is not None
    else:
        assert (
            await DeepAgentsApp._seed_handoff_thread(
                remote, "summary", cwd="/tmp", agent_name="agent", context={}
            )
            == "child"
        )
        assert held_lease("child") is not None
        assert try_acquire("child") is None


@pytest.mark.usefixtures("isolated_state")
async def test_clear_preserves_session_when_reservation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import TextualSessionState

    app = DeepAgentsApp(thread_id="current", server_kwargs={})
    app._session_state = TextualSessionState(thread_id="current")
    app._reserve_thread("current")
    outgoing = held_lease("current")
    clear = AsyncMock()
    message = AsyncMock()
    monkeypatch.setattr(app, "_clear_messages", clear)
    monkeypatch.setattr(app, "_mount_message", message)
    monkeypatch.setattr(
        app, "_reserve_thread", MagicMock(side_effect=OSError("read-only state"))
    )
    await app._handle_command("/clear")
    clear.assert_not_awaited()
    assert app._lc_thread_id == "current"
    assert app._session_state.thread_id == "current"
    assert held_lease("current") is outgoing
    message.assert_awaited_once()
    assert "read-only state" in str(message.call_args.args[0]._content)


@pytest.mark.usefixtures("isolated_state")
@pytest.mark.parametrize("output_format", ["text", "json"])
async def test_delete_command_reports_occupied_thread(
    output_format: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from rich.console import Console

    from deepagents_code import config

    monkeypatch.setattr(config, "console", Console(width=200, color_system=None))
    lease = try_acquire("busy")
    assert lease is not None
    mode: OutputFormat = "json" if output_format == "json" else "text"
    with pytest.raises(SystemExit) as exited:
        await sessions.delete_thread_command("busy", output_format=mode)
    assert exited.value.code == 1
    output = capsys.readouterr().out
    assert "Close it there before deleting" in output
    assert held_lease("busy") is lease
    if mode == "json":
        payload = json.loads(output)
        assert payload["command"] == "threads delete"
        assert payload["data"]["deleted"] is False
        assert payload["data"]["thread_id"] == "busy"
