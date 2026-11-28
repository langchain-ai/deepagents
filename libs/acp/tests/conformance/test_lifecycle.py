"""Conformance: `session/list`, `resume`, `close`, and `delete`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from acp.exceptions import RequestError
from acp.schema import ClientCapabilities, UserMessageChunk
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.conformance.harness import (
    Harness,
    RecordingClient,
    call,
    deep_agent,
    reply,
    scripted_model,
)


def _server(checkpointer: MemorySaver, *, client: RecordingClient | None = None) -> Harness:
    def build(context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model("ok", "ok", "ok"), checkpointer=checkpointer)

    server = AgentServerACP(agent=build, load_sessions=True, checkpointer=checkpointer)
    recording = client or RecordingClient()
    server.on_connect(recording)
    return Harness(server=server, client=recording)


async def test_capabilities_advertise_the_lifecycle_methods() -> None:
    harness = _server(MemorySaver())

    response = await harness.server.initialize(
        protocol_version=1, client_capabilities=ClientCapabilities()
    )

    sessions = response.agent_capabilities.session_capabilities
    assert sessions is not None
    assert all(
        capability is not None
        for capability in (sessions.list, sessions.resume, sessions.close, sessions.delete)
    )


async def test_list_shows_acp_sessions_for_the_cwd_and_each_one_loads() -> None:
    checkpointer = MemorySaver()
    first = _server(checkpointer)
    here = await first.new_session(cwd="/repo")
    await first.prompt(here, "Rename the loader")
    elsewhere = await first.new_session(cwd="/other")
    await first.prompt(elsewhere, "Something else")
    await first.settle()
    # A thread written by another client must not be listed.
    await deep_agent(scripted_model("hi"), checkpointer=checkpointer).ainvoke(
        {"messages": [{"role": "user", "content": "tui"}]},
        {"configurable": {"thread_id": "tui-thread"}},
    )

    restarted = _server(checkpointer)
    listed = await restarted.server.list_sessions(cwd="/repo")

    assert [(info.session_id, info.title) for info in listed.sessions] == [
        (here, "Rename the loader")
    ]
    everything = await restarted.server.list_sessions()
    assert {info.session_id for info in everything.sessions} == {here, elsewhere}
    for info in everything.sessions:
        await restarted.load_session(info.session_id, cwd=info.cwd)


async def test_resume_restores_without_replaying() -> None:
    checkpointer = MemorySaver()
    first = _server(checkpointer)
    session_id = await first.new_session(cwd="/repo")
    await first.prompt(session_id, "Hello")

    restarted = _server(checkpointer)
    await restarted.server.resume_session(session_id=session_id, cwd="/repo", mcp_servers=[])

    replayed = restarted.client.session_updates(session_id)
    assert not any(isinstance(update, UserMessageChunk) for update in replayed)
    assert (await restarted.prompt(session_id, "Again")).stop_reason == "end_turn"


async def test_close_cancels_a_running_turn_and_releases_the_session() -> None:
    started = asyncio.Event()
    closed: list[str] = []

    async def wait_forever(reason: str) -> str:
        """Block until cancelled."""
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    tool = StructuredTool.from_function(coroutine=wait_forever, name="wait_forever")

    class Server(AgentServerACP):
        async def session_closed(self, session_id: str) -> None:
            closed.append(session_id)

    model = scripted_model(reply("", call("wait_forever", {"reason": "test"})))
    harness = Harness.start(deep_agent(model, tools=[tool]), server_class=Server)
    session_id = await harness.new_session()

    turn = asyncio.create_task(harness.prompt(session_id, "Wait"))
    await asyncio.wait_for(started.wait(), timeout=5)
    await harness.server.close_session(session_id=session_id)

    assert (await asyncio.wait_for(turn, timeout=2)).stop_reason == "cancelled"
    assert closed == [session_id]
    assert session_id not in harness.server._sessions


async def test_delete_removes_acp_sessions_only() -> None:
    checkpointer = MemorySaver()
    harness = _server(checkpointer)
    session_id = await harness.new_session(cwd="/repo")
    await harness.prompt(session_id, "Hello")
    await deep_agent(scripted_model("hi"), checkpointer=checkpointer).ainvoke(
        {"messages": [{"role": "user", "content": "tui"}]},
        {"configurable": {"thread_id": "tui-thread"}},
    )

    await harness.server.delete_session(session_id=session_id)
    with pytest.raises(RequestError):
        await harness.server.delete_session(session_id="tui-thread")

    assert (await harness.server.list_sessions()).sessions == []
    assert await checkpointer.aget_tuple({"configurable": {"thread_id": "tui-thread"}})
