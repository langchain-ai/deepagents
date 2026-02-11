from __future__ import annotations

from typing import Any

from acp import text_block, update_agent_message
from acp.interfaces import Client
from acp.schema import (
    AllowedOutcome,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    PermissionOption,
    RequestPermissionResponse,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
    ToolCallUpdate,
)
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.agent import ACPDeepAgent
from tests.chat_model import GenericFakeChatModel


class FakeACPClient(Client):
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.permission_requests: list[dict[str, Any]] = []
        self.next_permission: str = "approve"

    async def session_update(self, session_id: str, update: Any, source: str) -> None:
        self.updates.append({"session_id": session_id, "update": update, "source": source})

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        self.permission_requests.append(
            {"session_id": session_id, "tool_call": tool_call, "options": options}
        )
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=self.next_permission)
        )


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


async def test_acp_agent_ask_before_edits_requests_permission_for_plan() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]), stream_delimiter=None)
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = ACPDeepAgent(agent=graph, mode="ask_before_edits", root_dir="/root")
    client = FakeACPClient()
    client.next_permission = "approve"
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/root", mcp_servers=[])

    interrupt_value = {
        "action_requests": [
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "a", "status": "pending"}]},
            }
        ]
    }

    class _Interrupt:
        id = "tc_1"
        value = interrupt_value

    class _State:
        next = ("x",)
        interrupts = [_Interrupt()]

    current_state = _State()

    decisions = await agent._handle_interrupts(
        current_state=current_state,
        session_id=session.session_id,
        active_tool_calls={},
    )
    assert decisions == [{"type": "approve"}]

    assert len(client.permission_requests) == 1
    assert client.permission_requests[0]["tool_call"].title == "Review Plan"
    assert [o.option_id for o in client.permission_requests[0]["options"]] == [
        "approve",
        "reject",
    ]


async def test_acp_agent_tool_call_chunk_starts_tool_call() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]), stream_delimiter=None)
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = ACPDeepAgent(agent=graph, mode="auto", root_dir="/tmp")
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    class ToolChunkCarrier:
        tool_call_chunks = [
            {
                "id": "call_123",
                "name": "read_file",
                "args": '{"file_path": "/tmp/x.txt"}',
                "index": 0,
            }
        ]
        content = ""

    active_tool_calls: dict[str, Any] = {}
    tool_call_accumulator: dict[int, Any] = {}

    await agent._process_tool_call_chunks(
        session_id=session.session_id,
        message_chunk=ToolChunkCarrier(),
        active_tool_calls=active_tool_calls,
        tool_call_accumulator=tool_call_accumulator,
    )

    assert active_tool_calls == {
        "call_123": {"name": "read_file", "args": {"file_path": "/tmp/x.txt"}}
    }


async def test_acp_agent_tool_result_completes_tool_call() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]), stream_delimiter=None)
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = ACPDeepAgent(agent=graph, mode="auto", root_dir="/tmp")
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    msg = ToolMessage(content="result", tool_call_id="call_1")
    agent._cancelled = True

    async def one_chunk(*args: Any, **kwargs: Any):
        yield (msg, {})

    class Graph:
        astream = one_chunk

        async def aget_state(self, config: Any) -> Any:
            class S:
                next = ()
                interrupts: list[Any] = []

            return S()

    agent._deepagent = Graph()  # type: ignore[assignment]

    resp = await agent.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"


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
