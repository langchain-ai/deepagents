from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, convert_to_messages
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_talon.archive_saver import ConversationSaver
from deepagents_talon.history_backends import _sqlite_store
from deepagents_talon.host import TalonHost, _Turn
from deepagents_talon.interfaces import AgentRequest, AgentResult, SendResult
from deepagents_talon.runtime import DeepAgentRuntime
from deepagents_talon.store_archive import StoreConversationArchive
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config
from tests.unit_tests.test_history_content import CountingEmbeddings, reopen
from tests.unit_tests.test_history_vectors import SCOPE, settled, vector_store

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_archive_keeps_internal_text_but_embeds_only_user_and_delivered_reply(
    tmp_path: Path,
    backend: str,
) -> None:
    embed = CountingEmbeddings()
    async with vector_store(backend, tmp_path / "vectors.sqlite", embed) as store:
        async with reopen(tmp_path / "archive.sqlite", store) as archive:
            saver = ConversationSaver(InMemorySaver(), archive=archive)
            checkpoint = empty_checkpoint()
            checkpoint["channel_versions"] = {"messages": "1"}
            checkpoint["channel_values"] = {
                "messages": [
                    HumanMessage(
                        "actual", id="user", additional_kwargs={"talon_history_source": "user"}
                    ),
                    HumanMessage("legacy unknown", id="legacy"),
                    HumanMessage(
                        "raw subagent",
                        id="subagent",
                        additional_kwargs={"talon_history_source": "subagent"},
                    ),
                    HumanMessage(
                        "synthetic",
                        id="synthetic",
                        additional_kwargs={"talon_history_source": "internal"},
                    ),
                    ToolMessage("raw output", id="tool", tool_call_id="call"),
                    AIMessage(
                        "intermediate",
                        id="intermediate",
                        additional_kwargs={"talon_history_source": "delivered"},
                    ),
                    AIMessage("[SILENT]", id="silent"),
                    AIMessage("final", id="final"),
                ]
            }
            await saver.aput(
                {"configurable": {"thread_id": "session", "checkpoint_ns": ""}, "metadata": SCOPE},
                checkpoint,
                {},
                {"messages": "1"},
            )
            await settled(archive)
            assert embed.documents == ["actual"]
            assert len(await archive.entries(SCOPE, limit=20)) == 8
            await archive.record_delivery(SCOPE, "session", "final")
            await settled(archive)
            assert embed.documents == ["actual", "final"]
            assert len(await archive.entries(SCOPE, limit=20)) == 8
            assert (await archive.entries(SCOPE, query="raw output"))[0]["text"] == "raw output"
        async with reopen(tmp_path / "archive.sqlite", store) as archive:
            await archive.record_delivery(SCOPE, "session", "final")
            await settled(archive)
            assert embed.documents == ["actual", "final"]
            await archive.delete_session("session")
            records = await archive.records.store.asearch(archive.records.namespace, limit=100)
            assert not any(item.key.startswith(("reply:", "vector-content:")) for item in records)


@pytest.mark.parametrize("metadata", [{}, {"trigger": "cron"}, {"background_delivery": True}])
async def test_runtime_marks_real_inputs_separately_from_subagents_and_nudges(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    embed = CountingEmbeddings()
    calls = []

    async def invoke(
        payload: dict[str, list[dict[str, object]]], config: dict[str, object]
    ) -> dict[str, list[AIMessage]]:
        del config
        calls.append(payload["messages"])
        return {"messages": [AIMessage("" if len(calls) == 1 else "done")]}

    runtime = DeepAgentRuntime(model="test:model", env={})
    runtime._graph = SimpleNamespace(ainvoke=invoke)
    token = runtime._pending_results.set({"subagent": "raw result"})
    try:
        assert (
            await runtime._invoke_until_text(AgentRequest("session", "actual", metadata), None)
            == "done"
        )
    finally:
        runtime._pending_results.reset(token)
    async with (
        vector_store("memory", tmp_path / "unused", embed) as store,
        reopen(tmp_path / "archive.sqlite", store) as archive,
    ):
        for messages in calls:
            await archive.append(SCOPE, "session", "time", convert_to_messages(messages))
        await settled(archive)
        assert embed.documents == ([] if metadata else ["actual"])


@pytest.mark.parametrize("outcome", ["delivered", "failed", "suppressed", "superseded"])
async def test_host_indexes_only_acknowledged_final_delivery(tmp_path: Path, outcome: str) -> None:
    embed = CountingEmbeddings()
    async with (
        vector_store("memory", tmp_path / "unused", embed) as store,
        reopen(tmp_path / "archive.sqlite", store) as archive,
    ):

        class Agent(BlockingAgent):
            async def record_delivered_reply(
                self, session: str, channel: str, chat: str, text: str
            ) -> None:
                assert (session, channel, chat) == ("session", "whatsapp", "one")
                await archive.record_delivery(SCOPE, session, text)

        class Channel(RecordingChannel):
            async def send_message(self, conversation_id: str, text: str) -> SendResult:
                self.sent.append((conversation_id, text))
                return SendResult(success=outcome != "failed")

        channel = Channel(provider="whatsapp")
        host = TalonHost(config=_config(tmp_path), agent=Agent(), channels=[channel])
        host._generations["session"] = 2 if outcome == "superseded" else 1
        await archive.append(SCOPE, "session", "time", [AIMessage("final", id="reply")])
        await settled(archive)
        await host._settle_agent_turn(
            _Turn("session", "session", "whatsapp", 1, recovery_degraded=False),
            AgentResult("final"),
            channel=channel,
            reply_conversation_id="one",
            suppress_result=outcome == "suppressed",
        )
        await settled(archive)
        assert embed.documents == (["final"] if outcome == "delivered" else [])


async def test_policy_upgrade_removes_old_vectors_without_losing_keyword_history(
    tmp_path: Path,
) -> None:
    embed = CountingEmbeddings()
    async with vector_store("sqlite", tmp_path / "vectors.sqlite", embed) as store:
        async with reopen(tmp_path / "archive.sqlite", store) as archive:
            await archive.append(SCOPE, "session", "time", [HumanMessage("legacy raw", id="old")])
            await settled(archive)
            namespace = archive.vectors.namespace("whatsapp", "one")
            entry = (await archive.entries(SCOPE))[0]
            async with archive.records.access():
                record = await archive.records.get(str(entry["cursor"]))
                record.pop("indexable")
                root = await archive.records.root()
                root.pop("semantic_policy")
                await archive.records.commit([(str(entry["cursor"]), record), ("root", root)])
            assert await archive.vectors.archive.semantic(SCOPE, [str(entry["cursor"])]) == []
        async with reopen(tmp_path / "archive.sqlite", store) as archive:
            await settled(archive)
            assert not await store.asearch(namespace)
            assert embed.documents == ["legacy raw"]
            assert (await archive.entries(SCOPE, query="legacy raw"))[0]["text"] == "legacy raw"
            await archive.record_delivery(SCOPE, "session", "legacy raw")
            await settled(archive)
            assert embed.documents == ["legacy raw", "legacy raw"]


async def test_delivery_promotion_survives_restart_after_cursor_passed_reply(
    tmp_path: Path,
) -> None:
    embed = CountingEmbeddings()
    path = tmp_path / "archive.sqlite"
    async with vector_store("sqlite", tmp_path / "vectors.sqlite", embed) as store:
        async with reopen(path, store) as archive:
            await archive.append(SCOPE, "session", "time", [AIMessage("final", id="reply")])
            await settled(archive)
            assert embed.documents == []
        async with _sqlite_store(path.as_uri()) as metadata:
            archive = StoreConversationArchive(metadata, namespace=("content",))
            await archive.record_delivery(SCOPE, "session", "final")
        async with reopen(path, store) as archive:
            await settled(archive)
            assert embed.documents == ["final"]
            assert len(await archive.entries(SCOPE)) == 1
