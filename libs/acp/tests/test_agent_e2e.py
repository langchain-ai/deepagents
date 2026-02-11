from __future__ import annotations

from typing import Any

from acp import text_block, update_agent_message
from acp.schema import TextContentBlock
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.agent import ACPDeepAgent
from tests.chat_model import GenericFakeChatModel


class FakeACPClient:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, source: str) -> None:
        self.updates.append({"session_id": session_id, "update": update, "source": source})


async def test_acp_agent_prompt_streams_text() -> None:
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Hello!")]), stream_delimiter=r"(\s)"
    )
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = ACPDeepAgent(agent=graph, mode="auto", root_dir="/tmp")
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])
    session_id = session.session_id

    resp = await agent.prompt([TextContentBlock(type="text", text="Hi")], session_id=session_id)
    assert resp.stop_reason == "end_turn"

    texts: list[str] = []
    for entry in client.updates:
        update = entry["update"]
        if update == update_agent_message(text_block("Hello!")):
            texts.append("Hello!")
    assert texts == ["Hello!"]


async def test_acp_agent_sends_tool_call_updates() -> None:
    def build_agent(_: str) -> Any:
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Done")]),
            stream_delimiter=None,
        )
        return create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = ACPDeepAgent(agent=build_agent, mode="auto", root_dir="/tmp")
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])
    session_id = session.session_id

    resp = await agent.prompt([TextContentBlock(type="text", text="Read")], session_id=session_id)
    assert resp.stop_reason == "end_turn"

    tool_updates = [
        u["update"]
        for u in client.updates
        if hasattr(u["update"], "session_update")
        and getattr(u["update"], "session_update") == "tool_call_update"
    ]
    assert tool_updates == []
