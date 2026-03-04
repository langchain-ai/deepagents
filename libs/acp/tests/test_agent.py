from __future__ import annotations

import asyncio
from typing import Any, Literal

import pytest
from acp import text_block, update_agent_message
from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import (
    AllowedOutcome,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    PermissionOption,
    RequestPermissionResponse,
    ResourceContentBlock,
    SessionMode,
    SessionModeState,
    TextContentBlock,
    TextResourceContents,
    ToolCallUpdate,
)
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.chat_model import GenericFakeChatModel


class FakeACPClient(Client):
    def __init__(
        self, *, permission_outcomes: list[Literal["approve", "reject"]] | None = None
    ) -> None:
        self.events: list[dict[str, Any]] = []
        self.permission_outcomes: list[Literal["approve", "reject"]] = list(
            permission_outcomes or ["approve"]
        )

    async def session_update(self, session_id: str, update: Any, source: str) -> None:
        self.events.append(
            {
                "type": "session_update",
                "session_id": session_id,
                "update": update,
                "source": source,
            }
        )

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        self.events.append(
            {
                "type": "request_permission",
                "session_id": session_id,
                "tool_call": tool_call,
                "options": options,
            }
        )
        outcome: Literal["approve", "reject"] = (
            self.permission_outcomes.pop(0) if self.permission_outcomes else "approve"
        )
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=outcome)
        )


async def test_acp_agent_prompt_streams_text() -> None:
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Hello!")]), stream_delimiter=r"(\s)"
    )
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])
    session_id = session.session_id

    resp = await agent.prompt([TextContentBlock(type="text", text="Hi")], session_id=session_id)
    assert resp.stop_reason == "end_turn"

    texts: list[str] = []
    for entry in client.events:
        if entry["type"] != "session_update":
            continue
        update = entry["update"]
        if update == update_agent_message(text_block("Hello!")):
            texts.append("Hello!")
    assert texts == ["Hello!"]


async def test_acp_agent_cancel_stops_prompt() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Should not appear")]))
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    async def cancel_during_prompt() -> None:
        await agent.cancel(session_id=session.session_id)

    task = asyncio.create_task(
        agent.prompt([TextContentBlock(type="text", text="Hi")], session_id=session.session_id)
    )
    await asyncio.sleep(0)
    await cancel_during_prompt()
    resp = await task
    assert resp.stop_reason in {"cancelled", "end_turn"}


async def test_acp_agent_prompt_streams_list_content_blocks() -> None:
    class ListContentMessage:
        content = [
            {"type": "text", "text": "Hello"},
            " ",
            {"type": "text", "text": "world"},
        ]
        tool_call_chunks: list[dict[str, Any]] = []

    async def astream(*args: Any, **kwargs: Any):
        yield (ListContentMessage(), {})

    class Graph:
        @staticmethod
        async def astream(*args: Any, **kwargs: Any):
            yield ((), "messages", (ListContentMessage(), {}))

        async def aget_state(self, config: Any) -> Any:
            class S:
                next = ()
                interrupts: list[Any] = []

            return S()

    agent = AgentServerACP(
        agent=create_deep_agent(
            model=GenericFakeChatModel(
                messages=iter([AIMessage(content="ok")]), stream_delimiter=None
            ),
            checkpointer=MemorySaver(),
        ),
    )
    agent._agent = Graph()  # type: ignore[assignment]
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])
    resp = await agent.prompt(
        [TextContentBlock(type="text", text="Hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    assert any(
        e["update"] == update_agent_message(text_block("Hello world"))
        for e in client.events
        if e["type"] == "session_update"
    )


async def test_acp_agent_initialize_and_modes() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="OK")]), stream_delimiter=None)
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    init = await agent.initialize(protocol_version=1)
    assert init.agent_capabilities.prompt_capabilities.image is True

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])
    assert session.session_id
    assert session.modes is None


@tool(description="Write a file")
def write_file_tool(file_path: str, content: str) -> str:
    return "ok"


async def test_acp_agent_hitl_requests_permission_via_public_api() -> None:
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file_tool",
                            "args": {"file_path": "/tmp/x.txt", "content": "hi"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        stream_delimiter=None,
    )
    graph = create_deep_agent(
        model=model,
        tools=[write_file_tool],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file_tool": True})],
        checkpointer=MemorySaver(),
    )

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient(permission_outcomes=["approve"])
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    resp = await agent.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    permission_requests = [e for e in client.events if e["type"] == "request_permission"]
    assert permission_requests
    assert permission_requests[0]["tool_call"].title == "write_file_tool"


async def test_acp_deep_agent_hitl_interrupt_on_edit_file_requests_permission() -> None:
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/tmp/x.txt",
                                "old_string": "a",
                                "new_string": "b",
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        stream_delimiter=None,
    )
    graph = create_deep_agent(
        model=model,
        checkpointer=MemorySaver(),
        interrupt_on={"edit_file": True},
    )

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient(permission_outcomes=["approve"])
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    resp = await agent.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    permission_requests = [e for e in client.events if e["type"] == "request_permission"]
    assert permission_requests
    assert permission_requests[0]["tool_call"].title == "Edit `/tmp/x.txt`"


async def test_acp_agent_tool_call_chunk_starts_tool_call() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]), stream_delimiter=None)
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    msg = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "id": "call_123",
                "name": "read_file",
                "args": '{"file_path": "/tmp/x.txt"}',
                "index": 0,
            }
        ],
    )

    active_tool_calls: dict[str, Any] = {}
    tool_call_accumulator: dict[int, Any] = {}

    await agent._process_tool_call_chunks(
        session_id=session.session_id,
        message_chunk=msg,
        active_tool_calls=active_tool_calls,
        tool_call_accumulator=tool_call_accumulator,
    )

    assert active_tool_calls == {
        "call_123": {"name": "read_file", "args": {"file_path": "/tmp/x.txt"}}
    }


async def test_acp_agent_tool_result_completes_tool_call() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]), stream_delimiter=None)
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    tool_start = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "id": "call_1",
                "name": "execute",
                "args": '{"command": "echo hi"}',
                "index": 0,
            }
        ],
    )
    tool_result = ToolMessage(
        content="hi\n[Command succeeded with exit code 0]",
        tool_call_id="call_1",
    )

    async def graph_astream(*args: Any, **kwargs: Any):
        yield ((), "messages", (tool_start, {}))
        yield ((), "messages", (tool_result, {}))

    class Graph:
        astream = graph_astream

        async def aget_state(self, config: Any) -> Any:
            class S:
                next = ()
                interrupts: list[Any] = []

            return S()

    agent._agent = Graph()  # type: ignore[assignment]

    resp = await agent.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    tool_call_events = [
        e
        for e in client.events
        if e["type"] == "session_update" and getattr(e["update"], "tool_call_id", None) == "call_1"
    ]
    assert tool_call_events


async def test_acp_agent_multimodal_prompt_blocks_do_not_error() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]), stream_delimiter=None)
    graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent = AgentServerACP(agent=graph)
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


async def test_acp_agent_end_to_end_clears_plan() -> None:
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_todos",
                            "args": {
                                "todos": [
                                    {"content": "a", "status": "in_progress"},
                                    {"content": "b", "status": "pending"},
                                ]
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        stream_delimiter=None,
    )
    graph = create_deep_agent(
        model=model,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_todos": True})],
        checkpointer=MemorySaver(),
    )

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient(permission_outcomes=["reject"])
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    resp = await agent.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    permission_requests = [e for e in client.events if e["type"] == "request_permission"]
    assert permission_requests
    assert permission_requests[0]["tool_call"].title == "Review Plan"

    plan_updates = [
        e["update"]
        for e in client.events
        if e["type"] == "session_update" and getattr(e["update"], "session_update", None) == "plan"
    ]
    assert plan_updates

    plan_clear_updates = [u for u in plan_updates if getattr(u, "entries", None) == []]
    assert plan_clear_updates


async def test_acp_agent_hitl_approve_always_execute_auto_approves_next_time() -> None:
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute",
                            "args": {"command": "python -m pytest -q"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        stream_delimiter=None,
    )
    graph = create_deep_agent(
        model=model,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"execute": True})],
        checkpointer=MemorySaver(),
    )

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient(permission_outcomes=["approve_always"])
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    resp = await agent.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    permission_requests = [e for e in client.events if e["type"] == "request_permission"]
    assert len(permission_requests) == 1
    assert permission_requests[0]["tool_call"].title.startswith("Execute:")

    assert session.session_id in agent._allowed_command_types
    assert ("execute", "python -m pytest") in agent._allowed_command_types[session.session_id]

    client.events = []
    state = type("S", (), {"next": ("x",), "interrupts": []})()
    interrupt = type(
        "I",
        (),
        {
            "id": "int_2",
            "value": {
                "action_requests": [{"name": "execute", "args": {"command": "python -m pytest -q"}}]
            },
        },
    )()
    state.interrupts = [interrupt]
    decisions = await agent._handle_interrupts(current_state=state, session_id=session.session_id)
    assert decisions == [{"type": "approve"}]
    assert [e for e in client.events if e["type"] == "request_permission"] == []


async def test_acp_agent_hitl_approve_always_tool_auto_approves_next_time() -> None:
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "/tmp/x.txt", "content": "hi"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        stream_delimiter=None,
    )
    graph = create_deep_agent(
        model=model,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
        checkpointer=MemorySaver(),
    )

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient(permission_outcomes=["approve_always"])
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    resp = await agent.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    assert session.session_id in agent._allowed_command_types
    assert ("write_file", None) in agent._allowed_command_types[session.session_id]

    client.events = []
    state = type("S", (), {"next": ("x",), "interrupts": []})()
    interrupt = type(
        "I",
        (),
        {"id": "int_2", "value": {"action_requests": [{"name": "write_file", "args": {}}]}},
    )()
    state.interrupts = [interrupt]
    decisions = await agent._handle_interrupts(current_state=state, session_id=session.session_id)
    assert decisions == [{"type": "approve"}]
    assert [e for e in client.events if e["type"] == "request_permission"] == []


async def test_acp_agent_hitl_client_cancel_raises_request_error() -> None:
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_todos",
                            "args": {
                                "todos": [
                                    {"content": "a", "status": "in_progress"},
                                    {"content": "b", "status": "pending"},
                                ]
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        stream_delimiter=None,
    )
    graph = create_deep_agent(
        model=model,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_todos": True})],
        checkpointer=MemorySaver(),
    )

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient(permission_outcomes=[])

    async def request_permission_cancel(*args: Any, **kwargs: Any) -> RequestPermissionResponse:
        raise RequestError(400, "cancelled")

    client.request_permission = request_permission_cancel  # type: ignore[assignment]
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    with pytest.raises(RequestError):
        await agent.prompt(
            [TextContentBlock(type="text", text="hi")], session_id=session.session_id
        )


async def test_acp_agent_nested_agent_tool_call_returns_final_text() -> None:
    subagent_model = GenericFakeChatModel(messages=iter([AIMessage(content="blue")]))
    subagent = create_deep_agent(model=subagent_model, checkpointer=MemorySaver())

    @tool
    async def call_agent1(question: str) -> str:
        """Invoke subagent1 with the provided question."""
        resp = await subagent.ainvoke({"messages": [{"role": "user", "content": question}]})
        return resp["messages"][-1].content

    outer_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "call_agent1",
                            "args": {"question": "what is bobs favorite color"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="blue"),
            ]
        ),
        stream_delimiter=None,
    )
    outer = create_deep_agent(model=outer_model, tools=[call_agent1], checkpointer=MemorySaver())

    agent = AgentServerACP(agent=outer)
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    resp = await agent.prompt(
        [TextContentBlock(type="text", text="hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    assert any(
        e["update"] == update_agent_message(text_block("blue"))
        for e in client.events
        if e["type"] == "session_update"
    )


async def test_acp_agent_with_prebuilt_langchain_agent_end_to_end() -> None:
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Hello!")]), stream_delimiter=r"(\s)"
    )
    prebuilt = create_agent(model, tools=[], checkpointer=MemorySaver())

    agent = AgentServerACP(agent=prebuilt)
    client = FakeACPClient()
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])
    resp = await agent.prompt(
        [TextContentBlock(type="text", text="Hi")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    assert any(
        e["update"] == update_agent_message(text_block("Hello!"))
        for e in client.events
        if e["type"] == "session_update"
    )


async def test_acp_langchain_create_agent_nested_agent_tool_call_messages() -> None:
    @tool
    async def ask_user(question: str, runtime: ToolRuntime) -> str:
        """Ask the user a question via interrupt."""
        return interrupt(question)

    subagent_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {"question": "what is bobs favorite color?"},
                            "id": "call_ask",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="blue"),
            ]
        ),
        stream_delimiter=None,
    )
    subagent = create_agent(subagent_model, tools=[ask_user], checkpointer=MemorySaver())

    @tool
    async def call_agent1(question: str, runtime: ToolRuntime) -> str:
        """Invoke subagent1 with the provided question."""
        resp = await subagent.ainvoke({"messages": [{"role": "user", "content": question}]})
        return resp["messages"][-1].content

    outer_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "call_agent1",
                            "args": {"question": "what is bobs favorite color"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="blue"),
            ]
        ),
        stream_delimiter=None,
    )
    outer = create_agent(outer_model, tools=[call_agent1], checkpointer=MemorySaver())

    agent = AgentServerACP(agent=outer)
    client = FakeACPClient(permission_outcomes=["approve"])
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    with pytest.raises(RequestError):
        await agent.prompt(
            [TextContentBlock(type="text", text="hi")], session_id=session.session_id
        )


async def test_set_session_mode_resets_agent_with_new_mode() -> None:
    """Test that changing session mode properly resets the agent with the new mode context."""
    # Track which mode was used to create each agent instance
    created_agents: list[dict[str, Any]] = []

    def agent_factory(context: AgentSessionContext) -> CompiledStateGraph:
        """Factory that tracks the mode used to create each agent instance."""
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content=f"Response in {context.mode} mode")]),
            stream_delimiter=None,
        )
        agent = create_deep_agent(model=model, checkpointer=MemorySaver())
        created_agents.append({"mode": context.mode, "cwd": context.cwd, "agent": agent})
        return agent

    modes = SessionModeState(
        current_mode_id="mode_a",
        available_modes=[
            SessionMode(id="mode_a", name="Mode A", description="First mode"),
            SessionMode(id="mode_b", name="Mode B", description="Second mode"),
        ],
    )

    agent_server = AgentServerACP(agent=agent_factory, modes=modes)
    client = FakeACPClient()
    agent_server.on_connect(client)  # type: ignore[arg-type]

    # Create a new session - should use default mode "mode_a"
    session = await agent_server.new_session(cwd="/tmp", mcp_servers=[])
    session_id = session.session_id
    assert session.modes is not None
    assert session.modes.current_mode_id == "mode_a"

    # Trigger agent creation with first prompt
    await agent_server.prompt(
        [TextContentBlock(type="text", text="Test in mode A")], session_id=session_id
    )

    # Verify first agent was created with mode_a
    assert len(created_agents) == 1
    assert created_agents[0]["mode"] == "mode_a"
    assert created_agents[0]["cwd"] == "/tmp"

    # Change the session mode to mode_b
    await agent_server.set_session_mode(mode_id="mode_b", session_id=session_id)

    # Verify that changing mode created a new agent instance with mode_b
    assert len(created_agents) == 2
    assert created_agents[1]["mode"] == "mode_b"
    assert created_agents[1]["cwd"] == "/tmp"

    # Verify the agent was actually reset (not the same instance)
    assert created_agents[0]["agent"] is not created_agents[1]["agent"]

    # Make another prompt to ensure the new agent is being used
    await agent_server.prompt(
        [TextContentBlock(type="text", text="Test in mode B")], session_id=session_id
    )

    # Should still be 2 agents (no new one created, just using the existing mode_b agent)
    assert len(created_agents) == 2


async def test_reset_agent_with_compiled_state_graph() -> None:
    """Test that _reset_agent works correctly when agent_factory is a CompiledStateGraph."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]), stream_delimiter=None)
    compiled_graph = create_deep_agent(model=model, checkpointer=MemorySaver())

    agent_server = AgentServerACP(agent=compiled_graph)
    client = FakeACPClient()
    agent_server.on_connect(client)  # type: ignore[arg-type]

    session = await agent_server.new_session(cwd="/tmp", mcp_servers=[])
    session_id = session.session_id

    # Initially agent is None
    assert agent_server._agent is None

    # Call _reset_agent
    agent_server._reset_agent(session_id)

    # Agent should now be set to the compiled graph
    assert agent_server._agent is compiled_graph


async def test_reset_agent_preserves_session_cwd() -> None:
    """Test that _reset_agent uses the correct cwd from session context."""
    created_cwds: list[str] = []

    def agent_factory(context: AgentSessionContext) -> CompiledStateGraph:
        """Factory that tracks the cwd used to create each agent instance."""
        created_cwds.append(context.cwd)
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="OK")]), stream_delimiter=None
        )
        return create_deep_agent(model=model, checkpointer=MemorySaver())

    modes = SessionModeState(
        current_mode_id="default",
        available_modes=[SessionMode(id="default", name="Default", description="Default mode")],
    )

    agent_server = AgentServerACP(agent=agent_factory, modes=modes)
    client = FakeACPClient()
    agent_server.on_connect(client)  # type: ignore[arg-type]

    # Create session with custom cwd
    session = await agent_server.new_session(cwd="/custom/path", mcp_servers=[])
    session_id = session.session_id

    # Trigger agent creation
    await agent_server.prompt([TextContentBlock(type="text", text="Test")], session_id=session_id)

    # Verify the agent was created with the correct cwd
    assert len(created_cwds) == 1
    assert created_cwds[0] == "/custom/path"

    # Change mode (which calls _reset_agent)
    await agent_server.set_session_mode(mode_id="default", session_id=session_id)

    # Verify the new agent was also created with the same cwd
    assert len(created_cwds) == 2
    assert created_cwds[1] == "/custom/path"


async def test_acp_agent_hitl_requests_permission_only_once() -> None:
    """Test that tools requiring approval only prompt the user once, not twice.

    This is a regression test for a bug where _handle_interrupts was called twice
    for the same interrupt: once during streaming when an __interrupt__ update was
    detected (line ~515), and again after the stream ended (line ~593). The fix
    removes the redundant post-stream call since all interrupts are properly handled
    during streaming via __interrupt__ updates.
    """
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "/tmp/test.txt", "content": "hello"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="File written successfully"),
            ]
        ),
        stream_delimiter=None,
    )
    graph = create_deep_agent(
        model=model,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
        checkpointer=MemorySaver(),
    )

    agent = AgentServerACP(agent=graph)
    client = FakeACPClient(permission_outcomes=["approve"])
    agent.on_connect(client)  # type: ignore[arg-type]

    session = await agent.new_session(cwd="/tmp", mcp_servers=[])

    resp = await agent.prompt(
        [TextContentBlock(type="text", text="Write a test file")], session_id=session.session_id
    )
    assert resp.stop_reason == "end_turn"

    # Count permission requests - should be exactly 1, not 2
    permission_requests = [e for e in client.events if e["type"] == "request_permission"]
    assert len(permission_requests) == 1, (
        f"Expected exactly 1 permission request, got {len(permission_requests)}. "
        "This indicates the double approval bug has regressed."
    )
    assert permission_requests[0]["tool_call"].title == "Write `/tmp/test.txt`"
