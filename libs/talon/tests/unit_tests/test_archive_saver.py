from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_talon.archive import ArchiveScope, SQLiteConversationArchive
from deepagents_talon.archive_saver import ConversationSaver

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
    async with SQLiteConversationArchive.from_conn_string(path) as archive:
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
    async with SQLiteConversationArchive.from_conn_string(path) as archive:
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
    async with SQLiteConversationArchive.from_conn_string(
        str(tmp_path / "archive.sqlite")
    ) as archive:
        saver = ConversationSaver(InMemorySaver(), archive=archive)
        original = await _save(saver)
        with pytest.raises(ValueError, match="different channel or chat"):
            await _save(saver, scope=OTHER)
        checkpoints = [item async for item in saver.alist(_config())]
        assert len(checkpoints) == 1
        assert checkpoints[0].config == original
        assert await archive.entries(OTHER, session_id="session") == []
        assert len(await archive.entries(SCOPE)) == 1


async def test_failed_checkpoint_does_not_archive_uncommitted_messages(tmp_path, monkeypatch):
    backend = InMemorySaver()
    async with SQLiteConversationArchive.from_conn_string(
        str(tmp_path / "archive.sqlite")
    ) as archive:
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


async def test_archive_failure_retries_exact_checkpoint_after_reopen(tmp_path, monkeypatch):
    backend = InMemorySaver()
    path = str(tmp_path / "archive.sqlite")
    checkpoint = _checkpoint()
    async with SQLiteConversationArchive.from_conn_string(path) as archive:
        saver = ConversationSaver(backend, archive=archive)

        async def fail_message(*_args: object):
            msg = "archive unavailable"
            raise OSError(msg)

        monkeypatch.setattr(archive, "_index_message", fail_message)
        with pytest.raises(OSError, match="archive unavailable"):
            await saver.aput(_config(), checkpoint, {}, checkpoint["channel_versions"])
        assert await backend.aget(_config())
        assert await archive.entries(SCOPE) == []
    async with SQLiteConversationArchive.from_conn_string(path) as archive:
        saver = ConversationSaver(backend, archive=archive)
        for _ in range(2):
            await saver.aput(_config(), checkpoint, {}, checkpoint["channel_versions"])
        assert len(await archive.entries(SCOPE)) == 1
        assert len([item async for item in saver.alist(_config())]) == 1


async def test_unscoped_and_nested_writes_do_not_enter_archive(tmp_path):
    async with SQLiteConversationArchive.from_conn_string(
        str(tmp_path / "archive.sqlite")
    ) as archive:
        saver = ConversationSaver(InMemorySaver(), archive=archive)
        await _save(saver, "cron", {})
        await _save(saver, "nested", namespace="worker")
        assert await archive.entries(SCOPE) == []
        assert await archive.sessions(SCOPE) == []
        assert await saver.aget(_config("cron"))
        assert await saver.aget(_config("nested", namespace="worker"))
