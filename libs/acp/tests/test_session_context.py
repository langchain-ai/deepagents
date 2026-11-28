"""The agent factory sees its session's id and the client's MCP servers, and may be async."""

from __future__ import annotations

from typing import Any

from acp.schema import HttpMcpServer, TextContentBlock

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.conformance.harness import RecordingClient, deep_agent, reply, scripted_model


async def test_async_factory_gets_session_id_and_client_mcp_servers() -> None:
    seen: list[AgentSessionContext] = []

    async def build(context: AgentSessionContext) -> Any:
        seen.append(context)
        return deep_agent(scripted_model(reply("ok")))

    server = AgentServerACP(agent=build)
    server.on_connect(RecordingClient())
    t3 = HttpMcpServer(type="http", name="t3-code", url="http://127.0.0.1:1/mcp", headers=[])
    session = await server.new_session(cwd="/tmp", mcp_servers=[t3])

    response = await server.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )

    assert response.stop_reason == "end_turn"
    assert seen[0].session_id == session.session_id
    assert seen[0].mcp_servers == (t3,)
