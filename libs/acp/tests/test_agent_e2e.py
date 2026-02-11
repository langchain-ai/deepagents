from __future__ import annotations

from typing import Any

from acp import text_block, update_agent_message
from acp.schema import (
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)
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


async def test_acp_agent_cancel_stops_prompt() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Should not appear")]))
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = ACPDeepAgent(agent=graph, mode="auto", root_dir="/tmp")
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    async def cancel_during_prompt() -> None:
        await agent.cancel(session_id=session.session_id)

    import asyncio

    task = asyncio.create_task(
        agent.prompt([TextContentBlock(type="text", text="Hi")], session_id=session.session_id)
    )
    await asyncio.sleep(0)
    await cancel_during_prompt()
    resp = await task
    assert resp.stop_reason in {"cancelled", "end_turn"}


async def test_acp_agent_initialize_and_modes() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="OK")]), stream_delimiter=None)
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = ACPDeepAgent(agent=graph, mode="auto", root_dir="/tmp")
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    init = await agent.initialize(protocol_version=1)
    assert init.agent_capabilities.prompt_capabilities.image is True

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])
    assert session.session_id
    assert session.modes.current_mode_id == "auto"
    assert {m.id for m in session.modes.available_modes} == {"ask_before_edits", "auto"}

    await agent.set_session_mode(mode_id="ask_before_edits", session_id=session.session_id)
    session2 = await agent.new_session(cwd="/tmp", mcp_servers=[])
    assert session2.modes.current_mode_id == "ask_before_edits"


async def test_acp_agent_multimodal_prompt_blocks_do_not_error() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]), stream_delimiter=None)
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = ACPDeepAgent(agent=graph, mode="auto", root_dir="/root")
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/root", mcp_servers=[])

    blocks = [
        TextContentBlock(type="text", text="hi"),
        ImageContentBlock(type="image", mime_type="image/png", data="AAAA"),
        ResourceContentBlock(
            type="resource_link",
            name="file",
            uri="file:///root/a.txt",
            description="d",
            mime_type="text/plain",
        ),
        EmbeddedResourceContentBlock(
            type="resource",
            resource=TextResourceContents(
                mime_type="text/plain",
                text="hello",
                uri="file:///mem.txt",
            ),
        ),
    ]

    resp = await agent.prompt(blocks, session_id=session.session_id)
    assert resp.stop_reason == "end_turn"
