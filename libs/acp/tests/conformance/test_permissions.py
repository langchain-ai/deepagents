"""Conformance: permission requests show the tool call they gate, and resume the right interrupt."""

from __future__ import annotations

from typing import Any

from acp.schema import TextContentBlock
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt, interrupt

from deepagents_acp.server import AgentServerACP, AgentSessionContext
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


async def test_permission_links_to_the_streamed_tool_call() -> None:
    model = scripted_model(reply("", call("echo", {"text": "hi"})), reply("Done"))
    client = RecordingClient(permission_replies=["approve"])
    harness = Harness.start(
        deep_agent(model, tools=[echo], interrupt_on={"echo": True}), client=client
    )
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Echo hi")

    [request] = client.permission_requests
    assert request.tool_call.tool_call_id == "call-echo"
    assert {option.kind for option in request.options} == {
        "allow_once",
        "allow_always",
        "reject_once",
        "reject_always",
    }
    check_tool_call(client.tool_call(session_id, "call-echo"), "completed")


async def test_identical_calls_get_unlinked_permission_requests() -> None:
    # Two calls with the same name and args: guessing could show the wrong one.
    first = call("echo", {"text": "hi"}, call_id="call-1")
    second = call("echo", {"text": "hi"}, call_id="call-2")
    model = scripted_model(reply("", first, second), reply("Done"))
    client = RecordingClient(permission_replies=["approve", "approve"])
    harness = Harness.start(
        deep_agent(model, tools=[echo], interrupt_on={"echo": True}), client=client
    )
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Echo twice")

    ids = {request.tool_call.tool_call_id for request in client.permission_requests}
    assert len(ids) == 2
    assert ids.isdisjoint({"call-1", "call-2"})


async def test_reject_always_rejects_later_calls_without_asking() -> None:
    model = scripted_model(
        reply("", call("echo", {"text": "a"}, call_id="call-a")),
        reply("", call("echo", {"text": "b"}, call_id="call-b")),
        reply("Gave up"),
    )
    client = RecordingClient(permission_replies=["reject_always"])
    harness = Harness.start(
        deep_agent(model, tools=[echo], interrupt_on={"echo": True}), client=client
    )
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Echo")

    assert len(client.permission_requests) == 1
    check_tool_call(client.tool_call(session_id, "call-a"), "failed")
    check_tool_call(client.tool_call(session_id, "call-b"), "failed")


async def test_sessions_keep_their_own_working_directory() -> None:
    seen: list[str] = []

    def build(context: Any) -> Any:
        seen.append(context.cwd)
        return deep_agent(scripted_model("ok"))

    harness = Harness.start(build)
    first = await harness.new_session(cwd="/repo/one")
    second = await harness.new_session(cwd="/repo/two")

    await harness.prompt(first, "Hi")
    await harness.prompt(second, "Hi")

    assert seen == ["/repo/one", "/repo/two"]


async def test_locations_show_the_path_the_tool_uses() -> None:
    # Clients approve edits by location, so a relative path must not look like it
    # is under the session cwd: the filesystem tools resolve it from `/`.
    model = scripted_model(
        reply("", call("read_file", {"file_path": "notes.txt"}, call_id="relative")),
        reply("", call("read_file", {"file_path": "../etc/passwd"}, call_id="traversal")),
        "ok",
    )
    harness = Harness.start(deep_agent(model))
    session_id = await harness.new_session(cwd="/repo")

    await harness.prompt(session_id, "Read")

    relative = harness.client.tool_call(session_id, "relative")[0]
    traversal = harness.client.tool_call(session_id, "traversal")[0]
    assert [location.path for location in relative.locations] == ["/notes.txt"]
    assert traversal.locations is None


async def test_custom_interrupts_resume_through_handle_interrupt() -> None:
    @tool
    def confirm(question: str) -> str:
        """Ask the user a yes/no question."""
        return str(interrupt({"type": "confirm", "question": question}))

    class Server(AgentServerACP):
        async def handle_interrupt(self, session_id: str, pending: Interrupt) -> Any:
            assert pending.value == {"type": "confirm", "question": "Ship it?"}
            return "yes"

    model = scripted_model(reply("", call("confirm", {"question": "Ship it?"})), reply("Shipped"))
    harness = Harness.start(deep_agent(model, tools=[confirm]), server_class=Server)
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Ask")

    _, final = check_tool_call(harness.client.tool_call(session_id, "call-confirm"), "completed")
    assert final.content[0].content.text == "yes"


async def test_consecutive_interrupts_in_one_tool_are_each_answered() -> None:
    @tool
    def survey() -> str:
        """Ask two questions in a row."""
        first = interrupt({"type": "confirm", "question": "First?"})
        second = interrupt({"type": "confirm", "question": "Second?"})
        return f"{first}/{second}"

    asked: list[str] = []

    class Server(AgentServerACP):
        async def handle_interrupt(self, session_id: str, pending: Interrupt) -> Any:
            asked.append(pending.value["question"])
            return pending.value["question"].lower()

    model = scripted_model(reply("", call("survey")), reply("Done"))
    harness = Harness.start(deep_agent(model, tools=[survey]), server_class=Server)
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Ask")

    assert asked == ["First?", "Second?"]
    _, final = check_tool_call(harness.client.tool_call(session_id, "call-survey"), "completed")
    assert final.content[0].content.text == "first?/second?"


async def test_config_change_while_awaiting_permission_waits_for_the_next_turn() -> None:
    checkpointer = MemorySaver()
    models = [{"value": "a", "name": "A"}, {"value": "b", "name": "B"}]
    built: list[str | None] = []

    def build(context: AgentSessionContext) -> Any:
        built.append(context.model)
        model = scripted_model(reply("", call("echo", {"text": "hi"})), reply("Done"))
        return deep_agent(
            model, tools=[echo], checkpointer=checkpointer, interrupt_on={"echo": True}
        )

    class SwitchingClient(RecordingClient):
        async def request_permission(self, session_id: str, *args: Any, **kwargs: Any) -> Any:
            await server.set_config_option(config_id="model", session_id=session_id, value="b")
            return await super().request_permission(session_id, *args, **kwargs)

    client = SwitchingClient(permission_replies=["approve"])
    server = AgentServerACP(agent=build, models=models, load_sessions=True)
    server.on_connect(client)
    session = await server.new_session(cwd="/tmp", mcp_servers=[])

    response = await server.prompt(
        [TextContentBlock(type="text", text="Echo hi")], session_id=session.session_id
    )

    assert response.stop_reason == "end_turn"
    check_tool_call(client.tool_call(session.session_id, "call-echo"), "completed")
    assert [
        entry for entry in client.trace(session.session_id) if entry[0] == "agent_message_chunk"
    ] == [("agent_message_chunk", "Done")]
