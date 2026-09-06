from __future__ import annotations

import argparse
from contextlib import asynccontextmanager, nullcontext

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents_talon.__main__ import _run_host
from deepagents_talon.archive_saver import ConversationSaver
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore
from deepagents_talon.history import ArchiveScope, conversation_tools
from deepagents_talon.history_store import HistoryStorage, history_checkpointer
from deepagents_talon.store_archive import StoreConversationArchive

SCOPE = ArchiveScope(talon_history_channel="test", talon_history_chat="one")
OTHER = ArchiveScope(talon_history_channel="test", talon_history_chat="two")


async def save(saver, scope, session):
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": [HumanMessage("remember this", id="message")]}
    return await saver.aput(
        {"configurable": {"thread_id": session, "checkpoint_ns": ""}, "metadata": scope},
        checkpoint,
        {},
        {},
    )


@pytest.mark.parametrize("fail_host", [False, True])
async def test_custom_host_persists_reads_and_resets_without_sqlite(
    tmp_path, monkeypatch, fail_host
):
    config = TalonConfig.from_env({"AGENT_MODEL": "test:model"}, base_home=tmp_path)
    backend = InMemorySaver()
    store = InMemoryStore()
    archive = StoreConversationArchive(store, namespace=("test",))
    store_closed = False

    @asynccontextmanager
    async def metadata(_config):
        nonlocal store_closed
        try:
            yield store
        finally:
            store_closed = True

    @asynccontextmanager
    async def archives(config):
        async with metadata(config) as supplied_store:
            assert supplied_store is store
            async with archive.open():
                yield archive
                assert not store_closed

    async def agent_runtime(_config, cron_store=None, checkpointer=None):
        assert cron_store is cron
        return checkpointer

    async def run_host(_args, _config, _cron, _channels, saver):
        owned = await save(saver, SCOPE, "owned")
        other = await save(saver, OTHER, "other")
        tools = {tool.name: tool for tool in conversation_tools(saver.archive, lambda: SCOPE)}
        page = await tools["search_conversations"].ainvoke({"query": "remember"})
        assert [entry["session_id"] for entry in page] == ["owned"]
        assert len(await tools["list_conversations"].ainvoke({})) == 1
        assert len(await tools["read_conversation"].ainvoke({"session_id": "owned"})) == 1
        assert not await tools["read_conversation"].ainvoke({"session_id": "other"})
        await saver.clear_history(SCOPE)
        assert await backend.aget(owned) is None
        assert await backend.aget(other) is not None
        assert not await archive.entries(SCOPE)
        assert await archive.entries(OTHER)
        if fail_host:
            msg = "host failed"
            raise RuntimeError(msg)

    monkeypatch.setattr("deepagents_talon.__main__._agent_runtime", agent_runtime)
    monkeypatch.setattr("deepagents_talon.__main__._run_host_with_agent", run_host)
    cron = CronJobStore(assistant_id=config.assistant_id, cron_dir=config.cron_dir)
    with pytest.raises(RuntimeError, match="host failed") if fail_host else nullcontext():
        await _run_host(
            argparse.Namespace(once=True),
            config,
            cron,
            (),
            checkpointer=backend,
            history_storage=HistoryStorage(archive_factory=archives),
        )
    with pytest.raises(RuntimeError, match="closed"):
        await archive.setup()
    assert store_closed
    assert not config.checkpoint_path.exists()


async def test_archive_startup_failure_closes_store(tmp_path):
    closed = False

    @asynccontextmanager
    async def metadata(_config):
        nonlocal closed
        try:
            yield InMemoryStore()
        finally:
            closed = True

    async def initialize():
        msg = "archive initialization failed"
        raise RuntimeError(msg)

    @asynccontextmanager
    async def archives(config):
        async with metadata(config):
            yield await initialize()

    storage = HistoryStorage(archive_factory=archives)
    with pytest.raises(RuntimeError, match="initialization failed"):
        async with storage.open(TalonConfig.from_env({}, base_home=tmp_path)):
            pytest.fail("host must not start")
    assert closed


async def test_explicit_storage_rejects_double_wrapping_and_boolean_savers(tmp_path):
    config = TalonConfig.from_env({}, base_home=tmp_path)
    for backend in (
        True,
        ConversationSaver(
            InMemorySaver(), archive=StoreConversationArchive(InMemoryStore(), namespace=("test",))
        ),
    ):
        with pytest.raises(TypeError, match="unwrapped checkpoint saver"):
            async with history_checkpointer(config, checkpointer=backend, storage=HistoryStorage()):
                pytest.fail("must reject ambiguous ownership")
    assert not config.checkpoint_path.exists()


async def test_custom_archive_does_not_open_default_sqlite(tmp_path):
    config = TalonConfig.from_env({}, base_home=tmp_path)
    archive = StoreConversationArchive(InMemoryStore(), namespace=("test",))

    @asynccontextmanager
    async def archives(_config):
        async with archive.open():
            yield archive

    async with history_checkpointer(
        config, checkpointer=InMemorySaver(), storage=HistoryStorage(archive_factory=archives)
    ) as saver:
        await save(saver, SCOPE, "owned")
        assert await archive.entries(SCOPE)
    assert not config.checkpoint_path.exists()
