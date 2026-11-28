"""Conformance: slash commands are announced and run through `command_input`."""

from __future__ import annotations

import asyncio
from typing import Any

from acp.schema import AvailableCommand, AvailableCommandsUpdate

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.conformance.harness import Harness, deep_agent, scripted_model


class _CommandServer(AgentServerACP):
    def available_commands(self, context: AgentSessionContext) -> list[AvailableCommand]:
        return [AvailableCommand(name="review", description="Review the diff")]

    async def command_input(
        self, context: AgentSessionContext, name: str, args: str
    ) -> dict[str, Any] | None:
        return {"messages": [{"role": "user", "content": f"Run the {name} skill on: {args}"}]}


async def test_commands_are_announced_after_session_new() -> None:
    harness = Harness.start(deep_agent(scripted_model()), server_class=_CommandServer)
    session_id = await harness.new_session()
    await harness.settle()

    [update] = harness.client.session_updates(session_id)
    assert isinstance(update, AvailableCommandsUpdate)
    assert [command.name for command in update.available_commands] == ["review"]


async def test_advertised_command_becomes_its_message_and_others_stay_text() -> None:
    model = scripted_model("ok", "ok")
    graph = deep_agent(model)
    harness = Harness.start(graph, server_class=_CommandServer)
    session_id = await harness.new_session()

    await harness.prompt(session_id, "/review src/app.py")
    await harness.prompt(session_id, "/unknown thing")

    state = await graph.aget_state({"configurable": {"thread_id": session_id}})
    seen = [m.text for m in state.values["messages"] if m.type == "human"]
    assert seen == ["Run the review skill on: src/app.py", "/unknown thing"]
