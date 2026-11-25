"""Conformance: option changes, builds, and close don't race each other."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from acp.exceptions import RequestError
from acp.schema import (
    ConfigOptionUpdate,
    SessionMode,
    SessionModeState,
    TextContentBlock,
    UserMessageChunk,
)
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp import server as server_module
from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.conformance.harness import (
    Harness,
    RecordingClient,
    call,
    deep_agent,
    reply,
    scripted_model,
)

_MODES = SessionModeState(
    current_mode_id="manual",
    available_modes=[
        SessionMode(id="manual", name="Manual"),
        SessionMode(id="auto", name="Auto"),
    ],
)


async def test_set_mode_rejects_a_mode_that_was_not_offered() -> None:
    harness = Harness.start(lambda _context: deep_agent(scripted_model()), modes=_MODES)
    server, client = harness.server, harness.client
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    with pytest.raises(RequestError):
        await server.set_session_mode(mode_id="yolo", session_id=session.session_id)
    await server.set_session_mode(mode_id="auto", session_id=session.session_id)

    [announced] = [
        update
        for update in client.session_updates(session.session_id)
        if isinstance(update, ConfigOptionUpdate)
    ]
    [mode] = [option for option in announced.config_options if option.id == "mode"]
    assert mode.current_value == "auto"


async def test_a_slow_old_build_never_replaces_a_newer_choice() -> None:
    checkpointer = MemorySaver()
    auto_started = asyncio.Event()
    release_auto = asyncio.Event()
    built: list[str] = []

    async def build(context: AgentSessionContext) -> Any:
        if context.mode == "auto":
            auto_started.set()
            await release_auto.wait()
        built.append(context.mode)
        return deep_agent(scripted_model("ok"), checkpointer=checkpointer)

    server = Harness.start(
        build, modes=_MODES, load_sessions=True, checkpointer=checkpointer
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    await server.set_session_mode(mode_id="auto", session_id=session.session_id)
    turn = asyncio.create_task(
        server.prompt([TextContentBlock(type="text", text="hi")], session_id=session.session_id)
    )
    await asyncio.wait_for(auto_started.wait(), timeout=2)
    await server.set_session_mode(mode_id="manual", session_id=session.session_id)
    release_auto.set()
    await turn

    # The auto build finished after the switch, so it was thrown away and redone.
    assert built[-2:] == ["auto", "manual"]


async def test_prompts_are_refused_while_a_session_closes() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Server(AgentServerACP):
        async def session_closed(self, session_id: str) -> None:
            entered.set()
            await release.wait()

    server = Harness.start(deep_agent(scripted_model("ok")), server_class=Server).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    closing = asyncio.create_task(server.close_session(session_id=session.session_id))
    await asyncio.wait_for(entered.wait(), timeout=2)
    with pytest.raises(RequestError):
        await server.prompt(
            [TextContentBlock(type="text", text="hi")], session_id=session.session_id
        )
    release.set()
    await closing

    with pytest.raises(RequestError):
        await server.prompt(
            [TextContentBlock(type="text", text="hi")], session_id=session.session_id
        )


async def test_a_turn_that_ignores_cancel_is_retired_and_the_session_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_module, "_CLOSE_DEADLINE_SECONDS", 0)
    checkpointer = MemorySaver()
    started = asyncio.Event()
    ignored = asyncio.Event()
    stop = asyncio.Event()

    async def stubborn(reason: str) -> str:
        """Ignore the first cancellation."""
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            ignored.set()
            await stop.wait()
        return "late"

    tool = StructuredTool.from_function(coroutine=stubborn, name="stubborn")
    models = iter(
        [scripted_model(reply("", call("stubborn", {"reason": "x"}))), scripted_model("ok")]
    )

    def build(context: AgentSessionContext) -> Any:
        return deep_agent(next(models), tools=[tool], checkpointer=checkpointer)

    server = Harness.start(build, load_sessions=True, checkpointer=checkpointer).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    first = asyncio.create_task(
        server.prompt([TextContentBlock(type="text", text="go")], session_id=session.session_id)
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    await server.close_session(session_id=session.session_id)
    await server.resume_session(session_id=session.session_id, cwd="/tmp", mcp_servers=[])

    # The reopened session takes prompts while the retired turn is still stuck.
    again = await server.prompt(
        [TextContentBlock(type="text", text="again")], session_id=session.session_id
    )
    assert again.stop_reason == "end_turn"
    await asyncio.wait_for(ignored.wait(), timeout=5)
    stop.set()
    await first


async def test_a_bare_slash_is_an_ordinary_prompt() -> None:
    server = Harness.start(deep_agent(scripted_model("ok"))).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    response = await server.prompt(
        [TextContentBlock(type="text", text="/")], session_id=session.session_id
    )

    assert response.stop_reason == "end_turn"


async def test_a_build_paused_across_close_neither_blocks_nor_pollutes_a_reopen() -> None:
    checkpointer = MemorySaver()
    entered = asyncio.Event()
    release = asyncio.Event()
    built: list[tuple[str, str]] = []

    async def build(context: AgentSessionContext) -> Any:
        if context.mode == "auto" and not entered.is_set():
            entered.set()
            await release.wait()
        built.append((context.cwd, context.mode))
        return deep_agent(scripted_model("ok"), checkpointer=checkpointer)

    server = Harness.start(
        build, modes=_MODES, load_sessions=True, checkpointer=checkpointer
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    await server.set_session_mode(mode_id="auto", session_id=session.session_id)
    # The next use builds the auto agent; pause that build across the close.
    change = asyncio.create_task(server._agent_for(session.session_id))
    await asyncio.wait_for(entered.wait(), timeout=2)
    await server.close_session(session_id=session.session_id)
    await asyncio.wait_for(
        server.resume_session(session_id=session.session_id, cwd="/tmp", mcp_servers=[]),
        timeout=2,
    )
    release.set()

    with pytest.raises(RequestError):
        await change
    assert all(cwd == "/tmp" for cwd, _mode in built)
    response = await server.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert response.stop_reason == "end_turn"


async def test_load_and_resume_are_refused_while_a_session_closes() -> None:
    checkpointer = MemorySaver()
    entered = asyncio.Event()
    release = asyncio.Event()

    class Server(AgentServerACP):
        async def session_closed(self, session_id: str) -> None:
            entered.set()
            await release.wait()

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model("ok"), checkpointer=checkpointer)

    server = Harness.start(
        build, load_sessions=True, checkpointer=checkpointer, server_class=Server
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    closing = asyncio.create_task(server.close_session(session_id=session.session_id))
    await asyncio.wait_for(entered.wait(), timeout=2)
    with pytest.raises(RequestError):
        await server.resume_session(session_id=session.session_id, cwd="/tmp", mcp_servers=[])
    with pytest.raises(RequestError):
        await server.load_session(cwd="/tmp", session_id=session.session_id, mcp_servers=[])
    release.set()
    await closing

    await server.resume_session(session_id=session.session_id, cwd="/tmp", mcp_servers=[])
    response = await server.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert response.stop_reason == "end_turn"


async def test_a_resume_in_flight_when_close_starts_fails_instead_of_vanishing() -> None:
    checkpointer = MemorySaver()
    in_check = asyncio.Event()
    release_check = asyncio.Event()
    pausing = {"value": False}

    class Server(AgentServerACP):
        async def check_ready(self, context: AgentSessionContext) -> None:
            if pausing["value"]:
                in_check.set()
                await release_check.wait()

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model("ok"), checkpointer=checkpointer)

    server = Harness.start(
        build, load_sessions=True, checkpointer=checkpointer, server_class=Server
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    pausing["value"] = True
    resume = asyncio.create_task(
        server.resume_session(session_id=session.session_id, cwd="/tmp", mcp_servers=[])
    )
    await asyncio.wait_for(in_check.wait(), timeout=2)
    closing = asyncio.create_task(server.close_session(session_id=session.session_id))
    await asyncio.sleep(0)
    release_check.set()

    with pytest.raises(RequestError):
        await resume
    await closing
    pausing["value"] = False
    await server.resume_session(session_id=session.session_id, cwd="/tmp", mcp_servers=[])
    response = await server.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert response.stop_reason == "end_turn"


async def test_a_load_replaying_when_close_starts_fails_instead_of_vanishing() -> None:
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
    await server.close_session(session_id=session.session_id)
    release.set()

    with pytest.raises(RequestError):
        await load


async def test_close_with_a_pending_mode_change_never_builds_the_new_agent() -> None:
    checkpointer = MemorySaver()
    started = asyncio.Event()

    async def wait_forever(reason: str) -> str:
        """Block until cancelled."""
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    tool = StructuredTool.from_function(coroutine=wait_forever, name="wait_forever")

    async def build(context: AgentSessionContext) -> Any:
        if context.mode == "auto":
            await asyncio.Event().wait()  # building the new mode's agent would hang close
        model = scripted_model(reply("", call("wait_forever", {"reason": "x"})))
        return deep_agent(model, tools=[tool], checkpointer=checkpointer)

    server = Harness.start(
        build, modes=_MODES, load_sessions=True, checkpointer=checkpointer
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])
    turn = asyncio.create_task(
        server.prompt([TextContentBlock(type="text", text="go")], session_id=session.session_id)
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    await server.set_session_mode(mode_id="auto", session_id=session.session_id)

    await asyncio.wait_for(server.close_session(session_id=session.session_id), timeout=3)
    assert (await turn).stop_reason == "cancelled"


async def test_a_failed_resume_releases_what_the_session_acquired() -> None:
    checkpointer = MemorySaver()
    released: list[str] = []
    builds = {"count": 0}

    class Server(AgentServerACP):
        async def session_closed(self, session_id: str) -> None:
            released.append(session_id)

    def build(_context: AgentSessionContext) -> Any:
        builds["count"] += 1
        if builds["count"] > 1:
            msg = "graph construction failed"
            raise RuntimeError(msg)
        return deep_agent(scripted_model("ok"), checkpointer=checkpointer)

    server = Harness.start(
        build, load_sessions=True, checkpointer=checkpointer, server_class=Server
    ).server
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    with pytest.raises(RuntimeError):
        await server.resume_session(session_id=session.session_id, cwd="/tmp", mcp_servers=[])

    assert released == [session.session_id]
    assert session.session_id not in server._sessions
