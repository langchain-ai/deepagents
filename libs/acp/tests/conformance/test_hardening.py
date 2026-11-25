"""Conformance: what a session reveals and allows while it opens, fails, or changes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from acp.exceptions import RequestError
from acp.schema import SessionMode, SessionModeState, TextContentBlock, UserMessageChunk
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.conformance.harness import (
    Harness,
    RecordingClient,
    call,
    deep_agent,
    reply,
    scripted_model,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


async def test_prompts_wait_until_a_restore_finishes() -> None:
    checkpointer = MemorySaver()
    in_check = asyncio.Event()
    release = asyncio.Event()
    pausing = {"value": False}

    class Server(AgentServerACP):
        async def check_ready(self, context: AgentSessionContext) -> None:
            if pausing["value"]:
                in_check.set()
                await release.wait()

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model("ok"), checkpointer=checkpointer)

    server = Harness.start(
        build, server_class=Server, load_sessions=True, checkpointer=checkpointer
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    pausing["value"] = True
    resume = asyncio.create_task(
        server.resume_session(session_id=session.session_id, cwd="/tmp", mcp_servers=[])
    )
    await asyncio.wait_for(in_check.wait(), timeout=2)
    with pytest.raises(RequestError):
        await server.prompt(
            [TextContentBlock(type="text", text="hi")], session_id=session.session_id
        )
    release.set()
    await resume


async def test_a_saved_mode_no_longer_offered_falls_back_to_the_first_mode() -> None:
    checkpointer = MemorySaver()
    everything = SessionModeState(
        current_mode_id="auto",
        available_modes=[
            SessionMode(id="manual", name="Manual"),
            SessionMode(id="auto", name="Auto"),
            SessionMode(id="yolo", name="YOLO"),
        ],
    )
    without_yolo = SessionModeState(
        current_mode_id="auto", available_modes=everything.available_modes[:2]
    )

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model(), checkpointer=checkpointer)

    first = Harness.start(
        build, modes=everything, load_sessions=True, checkpointer=checkpointer
    ).server
    session = await first.new_session(cwd="/tmp", mcp_servers=[])
    await first.set_session_mode(mode_id="yolo", session_id=session.session_id)

    restarted = Harness.start(
        build, modes=without_yolo, load_sessions=True, checkpointer=checkpointer
    ).server
    loaded = await restarted.load_session(cwd="/tmp", session_id=session.session_id, mcp_servers=[])

    assert loaded.modes is not None
    assert loaded.modes.current_mode_id == "manual"


async def test_a_failed_new_session_releases_what_it_acquired() -> None:
    released: list[str] = []

    class Server(AgentServerACP):
        async def session_closed(self, session_id: str) -> None:
            released.append(session_id)

    def build(_context: AgentSessionContext) -> Any:
        msg = "graph construction failed"
        raise RuntimeError(msg)

    server = Harness.start(
        build, server_class=Server, load_sessions=True, checkpointer=MemorySaver()
    ).server

    with pytest.raises(RuntimeError):
        await server.new_session(cwd="/tmp", mcp_servers=[])

    assert len(released) == 1
    assert server._sessions == {}


async def test_an_unsupported_interrupt_error_does_not_echo_its_value() -> None:
    @tool
    def private_prompt() -> str:
        """Interrupt with private data."""
        return str(interrupt({"type": "custom", "detail": "private-value"}))

    model = scripted_model(reply("", call("private_prompt")))
    harness = Harness.start(deep_agent(model, tools=[private_prompt]))
    session_id = await harness.new_session()

    with pytest.raises(RequestError) as excinfo:
        await harness.prompt(session_id, "Go")

    assert excinfo.value.data is None


async def test_a_title_uses_the_model_the_turn_ran_with() -> None:
    titled_with: list[str | None] = []
    models = [{"value": "a", "name": "A"}, {"value": "b", "name": "B"}]
    started = asyncio.Event()
    release = asyncio.Event()

    @tool
    async def slow() -> str:
        """Wait to be released."""
        started.set()
        await release.wait()
        return "done"

    class Server(AgentServerACP):
        async def session_title(
            self, context: AgentSessionContext, messages: Sequence[Any]
        ) -> str | None:
            titled_with.append(context.model)
            return "Title"

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model(reply("", call("slow")), "ok"), tools=[slow])

    harness = Harness.start(build, server_class=Server, models=models)
    session_id = await harness.new_session()
    turn = asyncio.create_task(harness.prompt(session_id, "Go"))
    await asyncio.wait_for(started.wait(), timeout=2)
    await harness.server.set_config_option(config_id="model", session_id=session_id, value="b")
    release.set()
    await turn
    await harness.settle()

    assert titled_with == ["a"]


async def test_a_permission_request_names_the_real_path() -> None:
    model = scripted_model(
        reply("", call("write_file", {"file_path": "notes.txt", "content": "hi"})), "ok"
    )
    client = RecordingClient(permission_replies=["approve"])
    harness = Harness.start(deep_agent(model, interrupt_on={"write_file": True}), client=client)
    session_id = await harness.new_session(cwd="/repo")

    await harness.prompt(session_id, "Write")

    [request] = client.permission_requests
    assert request.tool_call.title == "Write `/notes.txt`"
    assert [location.path for location in request.tool_call.locations or []] == ["/notes.txt"]


async def test_prompts_wait_until_load_replay_finishes() -> None:
    checkpointer = MemorySaver()
    replaying = asyncio.Event()
    release = asyncio.Event()

    class PausingClient(RecordingClient):
        pause = False

        async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
            if self.pause and isinstance(update, UserMessageChunk):
                replaying.set()
                await release.wait()
            await super().session_update(session_id, update, **kwargs)

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model("ok", "ok"), checkpointer=checkpointer)

    client = PausingClient()
    server = Harness.start(
        build, client=client, load_sessions=True, checkpointer=checkpointer
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])
    await server.prompt([TextContentBlock(type="text", text="hi")], session_id=session.session_id)

    client.pause = True
    load = asyncio.create_task(
        server.load_session(cwd="/tmp", session_id=session.session_id, mcp_servers=[])
    )
    await asyncio.wait_for(replaying.wait(), timeout=2)
    with pytest.raises(RequestError):
        await server.prompt(
            [TextContentBlock(type="text", text="again")], session_id=session.session_id
        )
    client.pause = False
    release.set()
    await load
    response = await server.prompt(
        [TextContentBlock(type="text", text="again")], session_id=session.session_id
    )
    assert response.stop_reason == "end_turn"


async def test_a_delete_names_the_real_path() -> None:
    model = scripted_model(reply("", call("delete", {"file_path": "notes.txt"})), "ok")
    harness = Harness.start(deep_agent(model))
    session_id = await harness.new_session(cwd="/repo")

    await harness.prompt(session_id, "Delete")

    start = harness.client.tool_call(session_id, "call-delete")[0]
    assert (start.kind, start.title) == ("delete", "Delete `/notes.txt`")
    assert [location.path for location in start.locations or []] == ["/notes.txt"]


async def test_always_answers_do_not_carry_into_a_reopened_session() -> None:
    checkpointer = MemorySaver()
    model = scripted_model(
        reply("", call("echo", {"text": "a"}, call_id="first")),
        "ok",
        reply("", call("echo", {"text": "b"}, call_id="second")),
        "ok",
    )

    @tool
    def echo(text: str) -> str:
        """Echo."""
        return text

    graph = deep_agent(model, tools=[echo], checkpointer=checkpointer, interrupt_on={"echo": True})
    client = RecordingClient(permission_replies=["approve_always", "approve"])
    harness = Harness.start(graph, client=client, load_sessions=True, checkpointer=checkpointer)
    session_id = await harness.new_session()
    await harness.prompt(session_id, "One")

    await harness.server.resume_session(session_id=session_id, cwd="/tmp", mcp_servers=[])
    await harness.prompt(session_id, "Two")

    assert len(client.permission_requests) == 2


async def test_a_failed_new_session_leaves_nothing_in_history() -> None:
    checkpointer = MemorySaver()

    class Server(AgentServerACP):
        def extra_config_options(self, context: AgentSessionContext) -> list[Any]:
            msg = "option hook failed"
            raise RuntimeError(msg)

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model(), checkpointer=checkpointer)

    server = Harness.start(
        build, server_class=Server, load_sessions=True, checkpointer=checkpointer
    ).server
    with pytest.raises(RuntimeError):
        await server.new_session(cwd="/tmp", mcp_servers=[])

    assert (await server.list_sessions()).sessions == []


async def test_a_load_whose_replay_fails_releases_the_session() -> None:
    checkpointer = MemorySaver()
    released: list[str] = []

    class FailingClient(RecordingClient):
        fail = False

        async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
            if self.fail and isinstance(update, UserMessageChunk):
                msg = "client went away mid-replay"
                raise ConnectionError(msg)
            await super().session_update(session_id, update, **kwargs)

    class Server(AgentServerACP):
        async def session_closed(self, session_id: str) -> None:
            released.append(session_id)

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model("ok"), checkpointer=checkpointer)

    client = FailingClient()
    server = Harness.start(
        build,
        client=client,
        server_class=Server,
        load_sessions=True,
        checkpointer=checkpointer,
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])
    await server.prompt([TextContentBlock(type="text", text="hi")], session_id=session.session_id)

    client.fail = True
    with pytest.raises(ConnectionError):
        await server.load_session(cwd="/tmp", session_id=session.session_id, mcp_servers=[])

    assert released == [session.session_id]
    assert session.session_id not in server._sessions


async def test_a_session_cannot_reopen_while_a_turn_awaits_permission() -> None:
    checkpointer = MemorySaver()
    asked = asyncio.Event()
    answer = asyncio.Event()

    @tool
    def echo(text: str) -> str:
        """Echo."""
        return text

    class WaitingClient(RecordingClient):
        async def request_permission(self, *args: Any, **kwargs: Any) -> Any:
            asked.set()
            await answer.wait()
            return await super().request_permission(*args, **kwargs)

    model = scripted_model(reply("", call("echo", {"text": "a"})), "ok")
    graph = deep_agent(model, tools=[echo], checkpointer=checkpointer, interrupt_on={"echo": True})
    client = WaitingClient(permission_replies=["approve_always"])
    harness = Harness.start(graph, client=client, load_sessions=True, checkpointer=checkpointer)
    session_id = await harness.new_session()

    turn = asyncio.create_task(harness.prompt(session_id, "Echo"))
    await asyncio.wait_for(asked.wait(), timeout=2)
    with pytest.raises(RequestError):
        await harness.server.resume_session(session_id=session_id, cwd="/tmp", mcp_servers=[])
    answer.set()

    assert (await turn).stop_reason == "end_turn"


async def test_a_close_cancelled_while_releasing_still_forgets_the_session() -> None:
    releases: list[str] = []

    class Server(AgentServerACP):
        async def session_closed(self, session_id: str) -> None:
            releases.append(session_id)
            if len(releases) == 1:
                raise asyncio.CancelledError

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model())

    server = Harness.start(build, server_class=Server).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    with pytest.raises(asyncio.CancelledError):
        await server.close_session(session_id=session.session_id)
    await server.close_session(session_id=session.session_id)

    assert releases == [session.session_id, session.session_id]


async def test_a_load_closed_mid_replay_sends_nothing_more() -> None:
    checkpointer = MemorySaver()
    replaying = asyncio.Event()
    release = asyncio.Event()

    class PausingClient(RecordingClient):
        pause = False

        async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
            if self.pause and isinstance(update, UserMessageChunk):
                self.pause = False
                replaying.set()
                await release.wait()
            await super().session_update(session_id, update, **kwargs)

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model("ok"), checkpointer=checkpointer)

    client = PausingClient()
    server = Harness.start(
        build, client=client, load_sessions=True, checkpointer=checkpointer
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])
    await server.prompt([TextContentBlock(type="text", text="hi")], session_id=session.session_id)
    sent_before = len(client.session_updates(session.session_id))

    client.pause = True
    load = asyncio.create_task(
        server.load_session(cwd="/tmp", session_id=session.session_id, mcp_servers=[])
    )
    await asyncio.wait_for(replaying.wait(), timeout=2)
    await server.close_session(session_id=session.session_id)
    release.set()
    with pytest.raises(RequestError):
        await load

    replayed = client.session_updates(session.session_id)[sent_before:]
    assert [type(update) for update in replayed] == [UserMessageChunk]
