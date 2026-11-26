"""Conformance: session load, and isolation between concurrent sessions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import AgentServerACP, AgentSessionContext, ToolCallDisplay
from tests.conformance.harness import (
    Harness,
    RecordingClient,
    call,
    check_tool_call,
    deep_agent,
    echo,
    reply,
    scripted_model,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langgraph.graph.state import CompiledStateGraph


async def test_load_session_replays_history() -> None:
    checkpointer = MemorySaver()
    model = scripted_model(reply("", call("echo", {"text": "hi"})), reply("Done"))
    first = Harness.start(
        deep_agent(model, tools=[echo], checkpointer=checkpointer), load_sessions=True
    )
    session_id = await first.new_session()
    await first.prompt(session_id, "Echo hi")

    restarted = Harness.start(
        deep_agent(scripted_model(), tools=[echo], checkpointer=checkpointer),
        load_sessions=True,
    )
    await restarted.load_session(session_id)

    assert restarted.trace(session_id) == [
        ("user_message_chunk", "Echo hi"),
        ("tool_call", "call-echo", "pending"),
        ("tool_call_update", "call-echo", "completed"),
        ("agent_message_chunk", "Done"),
    ]


async def test_replay_shows_tool_calls_and_plans_as_the_live_turn_did() -> None:
    checkpointer = MemorySaver()
    todos = [{"content": "Echo hi", "status": "in_progress"}]
    model = scripted_model(
        reply("", call("write_todos", {"todos": todos})),
        reply("", call("echo", {"text": "hi"})),
        reply("Done"),
    )
    first = Harness.start(
        deep_agent(model, tools=[echo], checkpointer=checkpointer), load_sessions=True
    )
    session_id = await first.new_session()
    await first.prompt(session_id, "Plan, then echo hi")

    restarted = Harness.start(
        deep_agent(scripted_model(), tools=[echo], checkpointer=checkpointer),
        load_sessions=True,
    )
    await restarted.load_session(session_id)

    for tool_call_id in ("call-write_todos", "call-echo"):
        replayed = restarted.client.tool_call(session_id, tool_call_id)
        assert replayed == first.client.tool_call(session_id, tool_call_id)
        assert len(replayed) == 2
    assert ("plan", (("Echo hi", "in_progress"),)) in restarted.trace(session_id)


class _DescribingServer(AgentServerACP):
    def tool_call_display(
        self, context: AgentSessionContext, *, name: str, args: Mapping[str, Any]
    ) -> ToolCallDisplay | None:
        if name != "echo":
            return None
        return ToolCallDisplay(title=f"Echo {args['text']}", meta={"origin": "tests"})


async def test_a_described_tool_call_is_shown_the_same_live_for_approval_and_replayed() -> None:
    checkpointer = MemorySaver()
    model = scripted_model(reply("", call("echo", {"text": "hi"})), reply("Done"))
    first = Harness.start(
        deep_agent(model, tools=[echo], checkpointer=checkpointer, interrupt_on={"echo": True}),
        client=RecordingClient(permission_replies=["approve"]),
        load_sessions=True,
        server_class=_DescribingServer,
    )
    session_id = await first.new_session()
    await first.prompt(session_id, "Echo hi")

    restarted = Harness.start(
        deep_agent(scripted_model(), tools=[echo], checkpointer=checkpointer),
        load_sessions=True,
        server_class=_DescribingServer,
    )
    await restarted.load_session(session_id)

    [request] = first.client.permission_requests
    shown = [
        first.client.tool_call(session_id, "call-echo")[0],
        request.tool_call,
        restarted.client.tool_call(session_id, "call-echo")[0],
    ]
    assert [(tool_call.title, tool_call.field_meta) for tool_call in shown] == [
        ("Echo hi", {"origin": "tests"})
    ] * 3


async def test_replay_keeps_tool_calls_whose_ids_repeat_across_turns() -> None:
    checkpointer = MemorySaver()
    model = scripted_model(
        reply("", call("echo", {"text": "a"})),
        reply("One"),
        reply("", call("echo", {"text": "b"})),
        reply("Two"),
    )
    first = Harness.start(
        deep_agent(model, tools=[echo], checkpointer=checkpointer), load_sessions=True
    )
    session_id = await first.new_session()
    await first.prompt(session_id, "First")
    await first.prompt(session_id, "Second")

    restarted = Harness.start(
        deep_agent(scripted_model(), tools=[echo], checkpointer=checkpointer),
        load_sessions=True,
    )
    await restarted.load_session(session_id)

    trace = restarted.trace(session_id)
    assert [entry for entry in trace if entry[0] != "agent_message_chunk"] == [
        ("user_message_chunk", "First"),
        ("tool_call", "call-echo", "pending"),
        ("tool_call_update", "call-echo", "completed"),
        ("user_message_chunk", "Second"),
        ("tool_call", "call-echo", "pending"),
        ("tool_call_update", "call-echo", "completed"),
    ]


async def test_concurrent_sessions_each_see_their_own_cwd() -> None:
    both_running = asyncio.Barrier(2)
    models = {
        cwd: scripted_model(reply("", call("pwd", {"reason": "check"})), reply("ok"))
        for cwd in ("/repo-a", "/repo-b")
    }
    checkpointer = MemorySaver()

    def build_agent(context: AgentSessionContext) -> CompiledStateGraph:
        async def pwd(reason: str) -> str:
            """Return the working directory."""
            await both_running.wait()
            return context.cwd

        tool = StructuredTool.from_function(coroutine=pwd, name="pwd")
        return create_agent(models[context.cwd], tools=[tool], checkpointer=checkpointer)

    harness = Harness.start(build_agent)
    session_a = await harness.new_session(cwd="/repo-a")
    session_b = await harness.new_session(cwd="/repo-b")

    await asyncio.wait_for(
        asyncio.gather(harness.prompt(session_a, "pwd"), harness.prompt(session_b, "pwd")),
        timeout=5,
    )

    for session_id, cwd in ((session_a, "/repo-a"), (session_b, "/repo-b")):
        _, final = harness.client.tool_call(session_id, "call-pwd")
        assert final.content[0].content.text == cwd


async def test_replay_finishes_a_call_a_cancelled_turn_left_open() -> None:
    checkpointer = MemorySaver()
    started = asyncio.Event()

    async def wait_forever(reason: str) -> str:
        """Block until cancelled."""
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    tool = StructuredTool.from_function(coroutine=wait_forever, name="wait_forever")
    model = scripted_model(reply("", call("wait_forever", {"reason": "x"})), "ok")
    graph = deep_agent(model, tools=[tool], checkpointer=checkpointer)
    harness = Harness.start(graph, load_sessions=True, checkpointer=checkpointer)
    session_id = await harness.new_session()

    turn = asyncio.create_task(harness.prompt(session_id, "Wait"))
    await asyncio.wait_for(started.wait(), timeout=5)
    await harness.cancel(session_id)
    assert (await turn).stop_reason == "cancelled"
    # The next turn's PatchToolCalls answers the dangling call after this user message.
    await harness.prompt(session_id, "Again")
    await harness.server.close_session(session_id=session_id)

    replay = RecordingClient()
    reloaded = Harness.start(graph, client=replay, load_sessions=True, checkpointer=checkpointer)
    await reloaded.load_session(session_id)

    check_tool_call(replay.tool_call(session_id, "call-wait_forever"), "failed")


async def test_replay_closes_a_call_that_never_got_a_result() -> None:
    checkpointer = MemorySaver()
    started = asyncio.Event()

    async def wait_forever(reason: str) -> str:
        """Block until cancelled."""
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    tool = StructuredTool.from_function(coroutine=wait_forever, name="wait_forever")
    model = scripted_model(reply("", call("wait_forever", {"reason": "x"})))
    graph = deep_agent(model, tools=[tool], checkpointer=checkpointer)
    harness = Harness.start(graph, load_sessions=True, checkpointer=checkpointer)
    session_id = await harness.new_session()

    turn = asyncio.create_task(harness.prompt(session_id, "Wait"))
    await asyncio.wait_for(started.wait(), timeout=5)
    await harness.server.close_session(session_id=session_id)
    assert (await turn).stop_reason == "cancelled"

    replay = RecordingClient()
    reloaded = Harness.start(graph, client=replay, load_sessions=True, checkpointer=checkpointer)
    await reloaded.load_session(session_id)

    check_tool_call(replay.tool_call(session_id, "call-wait_forever"), "failed")
