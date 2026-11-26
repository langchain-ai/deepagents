"""End-to-end ACP handshake over stdio against the installed `agent-client-protocol`."""

from __future__ import annotations

import sys
import textwrap
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process
from acp.interfaces import Client
from acp.schema import ClientCapabilities, Implementation

# Spawning a real process catches SDK API drift that in-process tests miss,
# such as helpers removed from the `acp` namespace (#7617).
_SERVER = textwrap.dedent(
    """
    import asyncio

    from acp import run_agent
    from deepagents_acp.server import AgentServerACP


    def build_agent(context):
        raise AssertionError("the handshake must not build the agent")


    asyncio.run(run_agent(AgentServerACP(agent=build_agent)))
    """
)


class _Client(Client):
    async def request_permission(self, *args: Any, **kwargs: Any) -> Any:
        msg = "unexpected permission request"
        raise AssertionError(msg)

    async def session_update(self, *args: Any, **kwargs: Any) -> None:
        return None


async def test_initialize_and_new_session_over_stdio(tmp_path) -> None:
    async with spawn_agent_process(_Client(), sys.executable, "-c", _SERVER) as (conn, _proc):
        init = await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
            client_info=Implementation(name="test", version="0"),
        )
        assert init.protocol_version == PROTOCOL_VERSION
        assert init.agent_capabilities is not None

        session = await conn.new_session(cwd=str(tmp_path), mcp_servers=[])
        assert session.session_id
