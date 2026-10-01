"""Cross-process thread ownership and stale-server fencing tests."""

import asyncio
import importlib.util
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from deepagents_code.thread_ownership import (
    OWNER_KEY,
    ThreadLease,
    ThreadOwnershipError,
    ensure_owned,
    owned_saver_class,
    release_all,
    try_acquire,
    writer_guard,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

_CLIENT = """
import sys
from pathlib import Path
from deepagents_code.thread_ownership import try_acquire
lease = try_acquire(sys.argv[2], db_path=Path(sys.argv[1]))
print(lease.token if lease else "busy", flush=True)
if lease:
    sys.stdin.readline()
    lease.release()
"""


def _client(db_path, thread_id) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _CLIENT, str(db_path), thread_id],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _stop(process) -> None:
    process.kill()
    process.communicate(timeout=10)


def test_reservations_exclude_other_processes_and_allow_distinct_threads(tmp_path):
    db_path = tmp_path / "sessions.db"
    lease = try_acquire("one", db_path=db_path)
    assert lease is not None
    process = _client(db_path, "one")
    try:
        output, error = process.communicate(timeout=10)
        assert process.returncode == 0, error
        assert output.strip() == "busy"
        other = try_acquire("two", db_path=db_path)
        assert other is not None
        other.release()
    finally:
        lease.release()
        if process.poll() is None:
            _stop(process)
    replacement = try_acquire("one", db_path=db_path)
    assert replacement is not None
    replacement.release()


async def test_client_crash_fences_orphaned_server(tmp_path):
    db_path = tmp_path / "sessions.db"
    process = _client(db_path, "thread")
    try:
        assert process.stdout is not None
        token = await asyncio.to_thread(process.stdout.readline)
        token = token.strip()
        assert token
        assert token != "busy"
        async with writer_guard("thread", db_path=db_path, token=token):
            assert try_acquire("thread", db_path=db_path) is None
        _stop(process)
        with pytest.raises(ThreadOwnershipError, match="client exited"):
            async with writer_guard("thread", db_path=db_path, token=token):
                pytest.fail("An orphan server must not write after client death")
        successor = try_acquire("thread", db_path=db_path)
        assert successor is not None
        try:
            with pytest.raises(ThreadOwnershipError, match="ownership changed"):
                async with writer_guard("thread", db_path=db_path, token=token):
                    pytest.fail("An old writer must not write after takeover")
            async with writer_guard("thread", db_path=db_path, token=successor.token):
                pass
        finally:
            successor.release()
    finally:
        if process.poll() is None:
            _stop(process)


async def test_inflight_writer_blocks_takeover_after_client_release(tmp_path):
    db_path = tmp_path / "sessions.db"
    lease = try_acquire("thread", db_path=db_path)
    assert lease is not None
    async with writer_guard("thread", db_path=db_path, token=lease.token):
        lease.release()
        assert try_acquire("thread", db_path=db_path) is None
    replacement = try_acquire("thread", db_path=db_path)
    assert replacement is not None
    replacement.release()


async def test_saver_fences_checkpoints_writes_and_deletion(tmp_path):
    db_path = tmp_path / "sessions.db"
    lease = try_acquire("thread", db_path=db_path)
    assert lease is not None
    saver_class = owned_saver_class(db_path=db_path)
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread",
            "checkpoint_ns": "",
            OWNER_KEY: lease.token,
        }
    }
    checkpoint = empty_checkpoint()
    try:
        async with saver_class.from_conn_string(str(db_path)) as saver:
            saved = await saver.aput(config, checkpoint, {}, {})
            await saver.aput_writes(saved, [("result", "first")], "task")
            with pytest.raises(ThreadOwnershipError, match="open elsewhere"):
                await saver.adelete_thread("thread")
            lease.release()
            with pytest.raises(ThreadOwnershipError):
                await saver.aput(config, empty_checkpoint(), {}, {})
            with pytest.raises(ThreadOwnershipError):
                await saver.aput_writes(saved, [("result", "stale")], "task")
            result = await saver.aget_tuple(saved)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]
            assert result.pending_writes == [("task", "result", "first")]
            await saver.adelete_thread("thread")
            assert await saver.aget_tuple(saved) is None
            successor = try_acquire("thread", db_path=db_path)
            assert successor is not None
            successor.release()
    finally:
        lease.release()


async def test_cancelled_delete_acquisition_releases_reservation(tmp_path, monkeypatch):
    import threading

    from deepagents_code import thread_ownership

    db_path = tmp_path / "sessions.db"
    acquired = threading.Event()
    finish = threading.Event()
    acquire = thread_ownership.try_acquire

    def delayed(thread_id, *, db_path) -> ThreadLease | None:
        lease = acquire(thread_id, db_path=db_path)
        acquired.set()
        assert finish.wait(timeout=5)
        return lease

    monkeypatch.setattr(thread_ownership, "try_acquire", delayed)
    async with owned_saver_class(db_path=db_path).from_conn_string(
        str(db_path)
    ) as saver:
        task = asyncio.create_task(saver.adelete_thread("thread"))
        try:
            assert await asyncio.to_thread(acquired.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            successor = acquire("thread", db_path=db_path)
            assert successor is not None
            successor.release()
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_cancelled_delete_holds_reservation_until_database_finishes(
    tmp_path, monkeypatch
):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = tmp_path / "sessions.db"
    entered = asyncio.Event()
    finish = asyncio.Event()
    delete = AsyncSqliteSaver.adelete_thread

    async def delayed(self, thread_id) -> None:
        entered.set()
        await finish.wait()
        await delete(self, thread_id)

    monkeypatch.setattr(AsyncSqliteSaver, "adelete_thread", delayed)
    async with owned_saver_class(db_path=db_path).from_conn_string(
        str(db_path)
    ) as saver:
        await saver.setup()
        task = asyncio.create_task(saver.adelete_thread("thread"))
        try:
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert try_acquire("thread", db_path=db_path) is None
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            successor = try_acquire("thread", db_path=db_path)
            assert successor is not None
            successor.release()
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_generated_server_checkpointer_enforces_ownership(tmp_path, monkeypatch):
    from deepagents_code import sessions
    from deepagents_code.client.launch.server_manager import _write_checkpointer

    db_path = tmp_path / "sessions.db"
    monkeypatch.setattr(sessions, "get_db_path", lambda: db_path)
    monkeypatch.setenv("DEEPAGENTS_CODE_SERVER_DB_PATH", "")
    _write_checkpointer(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "ownership_checkpointer", tmp_path / "checkpointer.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread", "checkpoint_ns": ""}
    }
    async with module.create_checkpointer() as saver:
        with pytest.raises(ThreadOwnershipError):
            await saver.aput(config, empty_checkpoint(), {}, {})
        lease = try_acquire("thread", db_path=db_path)
        assert lease is not None
        try:
            config["configurable"][OWNER_KEY] = lease.token
            saved = await saver.aput(config, empty_checkpoint(), {}, {})
            assert await saver.aget_tuple(saved) is not None
        finally:
            lease.release()


def test_thread_ids_cannot_escape_state_directory(tmp_path):
    lease = try_acquire("../../outside", db_path=tmp_path / "sessions.db")
    assert lease is not None
    try:
        assert set(tmp_path.iterdir()) == {tmp_path / "sessions.db.owners"}
    finally:
        lease.release()


async def test_same_client_reacquisition_fences_old_computation(tmp_path):
    db_path = tmp_path / "sessions.db"
    first = ensure_owned("thread", db_path=db_path)
    assert ensure_owned("thread", db_path=db_path) is first
    assert try_acquire("thread", db_path=db_path) is None
    first.release()
    second = ensure_owned("thread", db_path=db_path)
    try:
        assert first.token != second.token
        with pytest.raises(ThreadOwnershipError, match="ownership changed"):
            async with writer_guard("thread", db_path=db_path, token=first.token):
                pytest.fail("Reacquisition must fence old computations")
        first.release()
        assert ensure_owned("thread", db_path=db_path) is second
        async with writer_guard("thread", db_path=db_path, token=second.token):
            pass
    finally:
        second.release()


async def test_local_remote_agent_propagates_tokens_without_claiming_reads(
    tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock, MagicMock

    from deepagents_code import sessions
    from deepagents_code.client.remote_client import RemoteAgent

    monkeypatch.setattr(sessions, "get_db_path", lambda: tmp_path / "sessions.db")
    agent = RemoteAgent("http://localhost", local_ownership=True)
    graph = MagicMock()
    graph.aget_state = AsyncMock(return_value=None)
    graph.aupdate_state = AsyncMock()
    agent._graph = graph
    config = {"configurable": {"thread_id": "thread"}}
    await agent.aget_state(config)
    lease = try_acquire("thread")
    assert lease is not None
    try:
        await agent.aupdate_state(config, {"messages": []})
        prepared = graph.aupdate_state.call_args.args[0]
        assert prepared["configurable"][OWNER_KEY] == lease.token
        assert graph.aupdate_state.call_args.kwargs["headers"][OWNER_KEY] == lease.token
        assert OWNER_KEY not in config["configurable"]
        lease.release()
        successor = ensure_owned("thread")
        try:
            with pytest.raises(ThreadOwnershipError, match="reservation changed"):
                await agent.aupdate_state(prepared, {})
        finally:
            successor.release()
    finally:
        release_all()


async def test_graph_preserves_ownership_across_checkpoints(tmp_path):
    from langgraph.graph import END, START, StateGraph
    from pydantic import BaseModel

    class State(BaseModel):
        value: int

    def increment(state: State) -> dict[str, int]:
        return {"value": state.value + 1}

    db_path = tmp_path / "sessions.db"
    lease = ensure_owned("thread", db_path=db_path)
    try:
        async with owned_saver_class(db_path=db_path).from_conn_string(
            str(db_path)
        ) as saver:
            graph = StateGraph(State)
            graph.add_node("increment", increment)
            graph.add_edge(START, "increment")
            graph.add_edge("increment", END)
            agent = graph.compile(checkpointer=saver)
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread", OWNER_KEY: lease.token}
            }
            assert await agent.ainvoke(State(value=1), config) == {"value": 2}
            await agent.aupdate_state(config, {"value": 5})
            assert (await agent.aget_state(config)).values == {"value": 5}
    finally:
        lease.release()


def test_remote_client_does_not_upgrade_stale_thread_only_config(tmp_path, monkeypatch):
    from deepagents_code import sessions
    from deepagents_code.client.remote_client import RemoteAgent

    monkeypatch.setattr(sessions, "get_db_path", lambda: tmp_path / "sessions.db")
    agent = RemoteAgent("http://localhost", local_ownership=True)
    config = {"configurable": {"thread_id": "thread"}}
    old = agent._prepare_mutation(config)
    lease = ensure_owned("thread")
    lease.release()
    try:
        with pytest.raises(ThreadOwnershipError, match="reservation changed"):
            agent._prepare_mutation(config)
        successor = try_acquire("thread")
        assert successor is not None
        with pytest.raises(ThreadOwnershipError, match="reservation changed"):
            agent._prepare_mutation(config)
        agent.bind_thread_ownership("thread")
        assert (
            agent._prepare_mutation(config)["configurable"][OWNER_KEY]
            == successor.token
        )
        with pytest.raises(ThreadOwnershipError, match="reservation changed"):
            agent._prepare_mutation(old)
    finally:
        release_all()


async def test_rotation_waits_for_inflight_write_without_releasing_reservation(
    tmp_path,
):
    db_path = tmp_path / "sessions.db"
    lease = ensure_owned("thread", db_path=db_path)
    old_token = lease.token
    try:
        async with writer_guard("thread", db_path=db_path, token=old_token):
            rotation = asyncio.create_task(lease.rotate())
            await asyncio.sleep(0)
            assert not rotation.done()
            assert try_acquire("thread", db_path=db_path) is None
        await asyncio.wait_for(rotation, timeout=2)
        assert lease.token != old_token
        assert try_acquire("thread", db_path=db_path) is None
        with pytest.raises(ThreadOwnershipError, match="ownership changed"):
            async with writer_guard("thread", db_path=db_path, token=old_token):
                pytest.fail("Old writer survived rotation")
        async with writer_guard("thread", db_path=db_path, token=lease.token):
            pass
    finally:
        lease.release()


async def test_cancellation_during_lock_acquisition_releases_acquired_gate(
    tmp_path, monkeypatch
):
    import threading

    from deepagents_code import thread_ownership

    db_path = tmp_path / "sessions.db"
    lease = ensure_owned("thread", db_path=db_path)
    acquired = threading.Event()
    finish = threading.Event()
    acquire = thread_ownership._acquire

    def delayed(lock) -> None:
        acquire(lock)
        if str(lock.lock_file).endswith(".writer"):
            acquired.set()
            assert finish.wait(timeout=5)

    async def write() -> None:
        async with writer_guard("thread", db_path=db_path, token=lease.token):
            pytest.fail("Cancelled acquisition must not enter the mutation")

    monkeypatch.setattr(thread_ownership, "_acquire", delayed)
    task = asyncio.create_task(write())
    try:
        assert await asyncio.to_thread(acquired.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        monkeypatch.setattr(thread_ownership, "_acquire", acquire)
        lease.release()
        successor = try_acquire("thread", db_path=db_path)
        assert successor is not None
        successor.release()
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        lease.release()


async def test_cancellation_while_waiting_for_busy_gate_stops_retrying(tmp_path):
    db_path = tmp_path / "sessions.db"
    lease = ensure_owned("thread", db_path=db_path)

    async def write() -> None:
        async with writer_guard("thread", db_path=db_path, token=lease.token):
            pytest.fail("Cancelled waiter must not acquire the gate")

    try:
        async with writer_guard("thread", db_path=db_path, token=lease.token):
            task = asyncio.create_task(write())
            await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
    finally:
        lease.release()


async def test_failed_server_restart_still_fences_old_writer(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from deepagents_code import sessions
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.client.remote_client import RemoteAgent

    db_path = tmp_path / "sessions.db"
    monkeypatch.setattr(sessions, "get_db_path", lambda: db_path)
    app = DeepAgentsApp(thread_id="thread", server_kwargs={})
    app._reserve_thread("thread")
    old_client = RemoteAgent("http://localhost", local_ownership=True)
    config = {"configurable": {"thread_id": "thread"}}
    token = old_client._prepare_mutation(config)["configurable"][OWNER_KEY]
    app._agent = old_client
    app._server_proc = MagicMock()
    monkeypatch.setattr(app, "_sync_status_connection", MagicMock())
    monkeypatch.setattr(app, "post_message", MagicMock())
    monkeypatch.setattr(
        app,
        "_restart_server_process",
        AsyncMock(side_effect=RuntimeError("stop failed")),
    )
    try:
        result = await app._respawn_server(
            log_message="restart failed",
            mcp_failure_log="mcp failed",
            mcp_failure_toast="mcp failed",
        )
        assert not result.restarted
        assert try_acquire("thread") is None
        with pytest.raises(ThreadOwnershipError, match="ownership changed"):
            async with writer_guard("thread", db_path=db_path, token=token):
                pytest.fail("Orphaned old writer survived failed restart")
        with pytest.raises(ThreadOwnershipError, match="reservation changed"):
            old_client._prepare_mutation(config)
    finally:
        release_all()
