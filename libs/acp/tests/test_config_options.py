"""Extra config options: offered per model, passed to the agent, kept across load."""

from __future__ import annotations

from typing import Any

import pytest
from acp.exceptions import RequestError
from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption, TextContentBlock
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.conformance.harness import RecordingClient, deep_agent, scripted_model

_MODELS = [{"value": "fast", "name": "Fast"}, {"value": "deep", "name": "Deep"}]


class _EffortServer(AgentServerACP):
    def extra_config_options(self, context: AgentSessionContext) -> list[SessionConfigOptionSelect]:
        if context.model != "deep":
            return []
        return [
            SessionConfigOptionSelect(
                id="effort",
                name="Effort",
                category="thought_level",
                current_value=context.option("effort") or "medium",
                options=[
                    SessionConfigSelectOption(value=value, name=value)
                    for value in ("low", "medium", "high")
                ],
            )
        ]


def _server(built: list[AgentSessionContext], checkpointer: MemorySaver) -> _EffortServer:
    def build(context: AgentSessionContext) -> Any:
        built.append(context)
        return deep_agent(scripted_model("ok"), checkpointer=checkpointer)

    server = _EffortServer(
        agent=build, models=_MODELS, load_sessions=True, checkpointer=checkpointer
    )
    server.on_connect(RecordingClient())
    return server


def _ids(options: list[Any] | None) -> list[str]:
    return [option.id for option in options or []]


async def test_effort_follows_the_model_and_reaches_the_agent() -> None:
    built: list[AgentSessionContext] = []
    server = _server(built, MemorySaver())
    session = await server.new_session(cwd="/tmp", mcp_servers=[])
    assert _ids(session.config_options) == ["model"]

    switched = await server.set_config_option(
        config_id="model", session_id=session.session_id, value="deep"
    )
    assert _ids(switched.config_options) == ["model", "effort"]

    picked = await server.set_config_option(
        config_id="effort", session_id=session.session_id, value="high"
    )
    assert picked.config_options[-1].current_value == "high"
    await server.prompt([TextContentBlock(type="text", text="hi")], session_id=session.session_id)
    assert built[-1].option("effort") == "high"

    with pytest.raises(RequestError):
        await server.set_config_option(
            config_id="effort", session_id=session.session_id, value="extreme"
        )


async def test_effort_is_restored_on_load() -> None:
    checkpointer = MemorySaver()
    first = _server([], checkpointer)
    session = await first.new_session(cwd="/tmp", mcp_servers=[])
    await first.set_config_option(config_id="model", session_id=session.session_id, value="deep")
    await first.set_config_option(config_id="effort", session_id=session.session_id, value="low")

    built: list[AgentSessionContext] = []
    restarted = _server(built, checkpointer)
    loaded = await restarted.load_session(cwd="/tmp", session_id=session.session_id, mcp_servers=[])

    assert loaded.config_options[-1].current_value == "low"
    assert built[-1].option("effort") == "low"
