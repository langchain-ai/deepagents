from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepagents_talon.archive import ArchiveScope
from deepagents_talon.archive_saver import ConversationSaver
from tests.archive_helpers import open_archive

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata

    from deepagents_talon.archive import ArchiveEntry

SCOPE = ArchiveScope(talon_history_channel="whatsapp", talon_history_chat="chat")
OTHER = ArchiveScope(talon_history_channel="telegram", talon_history_chat="chat")


def _checkpoint(text="orchard"):
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": [HumanMessage(text, id="message")]}
    checkpoint["channel_versions"] = {"messages": "1"}
    return checkpoint


def _config(session="session", scope=SCOPE, namespace=""):
    return {
        "configurable": {"thread_id": session, "checkpoint_ns": namespace},
        "metadata": scope,
    }


async def _save(saver, session="session", scope=SCOPE, namespace=""):
    checkpoint = _checkpoint()
    return await saver.aput(
        _config(session, scope, namespace), checkpoint, {}, checkpoint["channel_versions"]
    )


async def test_backend_delete_failure_can_retry_after_archive_reopens(tmp_path, monkeypatch):
    backend = InMemorySaver()
    path = str(tmp_path / "archive.sqlite")
    async with open_archive(path) as archive:
        saver = ConversationSaver(backend, archive=archive)
        owned = await _save(saver)
        child = await _save(saver, namespace="worker")
        other = await _save(saver, "other", OTHER)
        deletion = backend.adelete_thread

        async def fail_delete(_thread):
            msg = "backend unavailable"
            raise OSError(msg)

        monkeypatch.setattr(backend, "adelete_thread", fail_delete)
        with pytest.raises(OSError, match="backend unavailable"):
            await saver.clear_history(SCOPE)
        assert await archive.sessions(SCOPE) == ["session"]
        assert await archive.entries(SCOPE)
        assert await saver.aget(owned)
    monkeypatch.setattr(backend, "adelete_thread", deletion)
    async with open_archive(path) as archive:
        saver = ConversationSaver(backend, archive=archive)
        await saver.clear_history(SCOPE)
        await saver.clear_history(SCOPE)
        assert await saver.aget(owned) is None
        assert await saver.aget(child) is None
        assert await archive.sessions(SCOPE) == []
        assert await archive.entries(SCOPE) == []
        assert await saver.aget(other)
        assert await archive.entries(OTHER)


async def test_scope_reassignment_rejected_before_checkpoint_mutation(tmp_path):
    async with open_archive(str(tmp_path / "archive.sqlite")) as archive:
        saver = ConversationSaver(InMemorySaver(), archive=archive)
        original = await _save(saver)
        with pytest.raises(ValueError, match="another scope"):
            await _save(saver, scope=OTHER)
        checkpoints = [item async for item in saver.alist(_config())]
        assert len(checkpoints) == 1
        assert checkpoints[0].config == original
        assert await archive.entries(OTHER, session_id="session") == []
        assert len(await archive.entries(SCOPE)) == 1


async def test_failed_checkpoint_does_not_archive_uncommitted_messages(tmp_path, monkeypatch):
    backend = InMemorySaver()
    async with open_archive(str(tmp_path / "archive.sqlite")) as archive:
        saver = ConversationSaver(backend, archive=archive)
        put = backend.aput

        async def fail_put(*_args: object):
            msg = "backend unavailable"
            raise OSError(msg)

        monkeypatch.setattr(backend, "aput", fail_put)
        with pytest.raises(OSError, match="backend unavailable"):
            await _save(saver)
        assert await archive.entries(SCOPE) == []
        assert await backend.aget(_config()) is None
        monkeypatch.setattr(backend, "aput", put)
        await _save(saver)
        assert len(await archive.entries(SCOPE)) == 1


@pytest.mark.parametrize("backend", [InMemorySaver, AsyncSqliteSaver])
async def test_archive_failure_retries_exact_checkpoint_after_reopen(
    tmp_path, monkeypatch, backend
):
    async with aiosqlite.connect(str(tmp_path / "checkpoints.sqlite")) as connection:
        backend = backend(connection) if backend is AsyncSqliteSaver else backend()
        path = str(tmp_path / "archive.sqlite")
        checkpoint = _checkpoint()
        async with open_archive(path) as archive:
            saver = ConversationSaver(backend, archive=archive)

            async def fail_message(*_args: object):
                msg = "archive unavailable"
                raise OSError(msg)

            monkeypatch.setattr(archive, "_append_chunk", fail_message)
            with pytest.raises(OSError, match="archive unavailable"):
                await saver.aput(_config(), checkpoint, {}, checkpoint["channel_versions"])
            assert await backend.aget(_config())
            assert await archive.entries(SCOPE) == []
        async with open_archive(path) as archive:
            saver = ConversationSaver(backend, archive=archive)
            for _ in range(2):
                await saver.aput(_config(), checkpoint, {}, checkpoint["channel_versions"])
            assert len(await archive.entries(SCOPE)) == 1
            assert len([item async for item in saver.alist(_config())]) == 1


async def test_unscoped_and_nested_writes_do_not_enter_archive(tmp_path):
    async with open_archive(str(tmp_path / "archive.sqlite")) as archive:
        saver = ConversationSaver(InMemorySaver(), archive=archive)
        await _save(saver, "cron", {})
        await _save(saver, "nested", namespace="worker")
        assert await archive.entries(SCOPE) == []
        assert await archive.sessions(SCOPE) == []
        assert await saver.aget(_config("cron"))
        assert await saver.aget(_config("nested", namespace="worker"))


@pytest.mark.parametrize("pause_at", ["checkpoint", "archive"])
@pytest.mark.parametrize("reset", [False, True])
async def test_cancelled_write_finishes_archiving_before_return_or_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pause_at: str, *, reset: bool
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    observed: list[ArchiveEntry] = []
    async with (
        AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as backend,
        open_archive(str(tmp_path / "archive.sqlite")) as archive,
    ):
        saver = ConversationSaver(backend, archive=archive)
        put, append, delete = backend.aput, archive.append, backend.adelete_thread

        async def paused_put(
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            result = await put(config, checkpoint, metadata, new_versions)
            if pause_at == "checkpoint":
                entered.set()
                await release.wait()
            return result

        async def paused_append(
            scope: ArchiveScope, session: str, timestamp: str, messages: Sequence[BaseMessage]
        ) -> None:
            if messages and pause_at == "archive":
                entered.set()
                await release.wait()
            await append(scope, session, timestamp, messages)

        async def observe_delete(thread: str) -> None:
            observed.extend(await archive.entries(SCOPE))
            await delete(thread)

        monkeypatch.setattr(backend, "aput", paused_put)
        monkeypatch.setattr(archive, "append", paused_append)
        monkeypatch.setattr(backend, "adelete_thread", observe_delete)
        write = asyncio.create_task(_save(saver))
        clearing = None
        try:
            await entered.wait()
            assert await backend.aget(_config())
            for _ in range(2):
                write.cancel()
                await asyncio.sleep(0)
                assert not write.done()
            if reset:
                clearing = asyncio.create_task(saver.clear_history(SCOPE))
                await asyncio.sleep(0)
                assert not clearing.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await write
            if clearing:
                await clearing
                assert [item["text"] for item in observed] == ["orchard"]
                assert await archive.entries(SCOPE) == []
                assert await backend.aget(_config()) is None
            else:
                assert [item["text"] for item in await archive.entries(SCOPE)] == ["orchard"]
        finally:
            release.set()
            await asyncio.gather(write, *([clearing] if clearing else []), return_exceptions=True)


async def test_idless_occurrences_survive_checkpoint_retries_and_reopening(tmp_path: Path) -> None:
    path = str(tmp_path / "archive.sqlite")
    backend = InMemorySaver()
    first, second = _checkpoint(), _checkpoint()
    messages = [HumanMessage("yes"), HumanMessage("yes"), HumanMessage("noted", id="stable")]
    first["channel_values"]["messages"] = messages
    second["channel_values"]["messages"] = messages
    for checkpoint in (first, first, second, second):
        async with open_archive(path) as archive:
            saver = ConversationSaver(backend, archive=archive)
            await saver.aput(_config(), checkpoint, {}, checkpoint["channel_versions"])
    async with open_archive(path) as archive:
        entries = await archive.entries(SCOPE, session_id="session")
        assert [item["text"] for item in entries] == ["yes", "yes", "noted", "yes", "yes"]
        assert len({item["message_id"] for item in entries}) == 5
        assert (await archive.conversations(SCOPE))[0]["message_count"] == 5
    assert [message.id for message in messages] == [None, None, "stable"]
