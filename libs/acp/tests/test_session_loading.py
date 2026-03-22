"""Tests for session loading and listing functionality."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from deepagents_acp.server import AgentServerACP, AgentSessionContext


class State(TypedDict):
    """Test state schema."""

    messages: list


def simple_node(state: State) -> State:
    """Simple node that does nothing (passthrough)."""
    # Just return the state as-is without modifying
    return state


@pytest.mark.asyncio
async def test_load_session_capability_advertised_with_checkpointer():
    """Test that loadSession capability is advertised when checkpointer is provided."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

            def build_agent(context: AgentSessionContext):
                workflow = StateGraph(State)
                workflow.add_node("chat", simple_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)
                return workflow.compile(checkpointer=checkpointer)

            server = AgentServerACP(agent=build_agent, checkpointer=checkpointer)

            # Test capability is advertised
            response = await server.initialize(protocol_version=1)
            assert response.agent_capabilities.load_session is True

    finally:
        db_file = Path(db_path)
        if db_file.exists():  # noqa: ASYNC240  # Cleanup code, not critical path
            db_file.unlink()  # noqa: ASYNC240  # Cleanup code, not critical path


@pytest.mark.asyncio
async def test_load_session_capability_not_advertised_without_checkpointer():
    """Test that loadSession capability is False when no checkpointer is provided."""

    def build_agent(context: AgentSessionContext):
        workflow = StateGraph(State)
        workflow.add_node("chat", simple_node)
        workflow.add_edge(START, "chat")
        workflow.add_edge("chat", END)
        return workflow.compile()

    server = AgentServerACP(agent=build_agent)

    # Test capability is not advertised
    response = await server.initialize(protocol_version=1)
    assert response.agent_capabilities.load_session is False


@pytest.mark.asyncio
async def test_load_session_replays_conversation_history():
    """Test that load_session replays the complete conversation history."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

            def build_agent(context: AgentSessionContext):
                workflow = StateGraph(State)
                workflow.add_node("chat", simple_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)
                return workflow.compile(checkpointer=checkpointer)

            # Create server and session
            server = AgentServerACP(agent=build_agent, checkpointer=checkpointer)
            mock_client = AsyncMock()
            server.on_connect(mock_client)

            # Create a new session
            new_session = await server.new_session(cwd="/tmp/test")
            session_id = new_session.session_id

            # Initialize agent and add conversation history
            server._reset_agent(session_id)
            agent = server._agent
            if agent.checkpointer is None:
                agent.checkpointer = checkpointer

            config = {"configurable": {"thread_id": session_id}}

            # Add messages to the session by invoking the graph
            # First exchange
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Hello, world!")]}, config
            )
            # Manually add AI response to state
            messages = result1["messages"]
            messages.append(AIMessage(content="Hi there!"))

            # Second exchange
            messages.append(HumanMessage(content="How are you?"))
            result2 = await agent.ainvoke({"messages": messages}, config)
            # Manually add AI response
            final_messages = result2["messages"]
            final_messages.append(AIMessage(content="I'm doing well!"))

            # Final invoke to save complete state
            await agent.ainvoke({"messages": final_messages}, config)

            # Reset mock to clear previous calls
            mock_client.reset_mock()

            # Load the session
            load_response = await server.load_session(
                session_id=session_id, cwd="/tmp/test", mcp_servers=[]
            )

            # Verify load response
            assert load_response is not None

            # Verify messages were replayed via session_update
            update_calls = [
                call for call in mock_client.method_calls if call[0] == "session_update"
            ]

            # Should have 4 messages (2 user + 2 agent)
            assert len(update_calls) == 4

            # Verify message order and content
            call_args = [call[2] for call in update_calls]  # Get kwargs

            # First message should be user message
            assert call_args[0]["update"].session_update == "user_message_chunk"
            assert call_args[0]["update"].content.text == "Hello, world!"

            # Second message should be agent message
            assert call_args[1]["update"].session_update == "agent_message_chunk"
            assert call_args[1]["update"].content.text == "Hi there!"

            # Third message should be user message
            assert call_args[2]["update"].session_update == "user_message_chunk"
            assert call_args[2]["update"].content.text == "How are you?"

            # Fourth message should be agent message
            assert call_args[3]["update"].session_update == "agent_message_chunk"
            assert call_args[3]["update"].content.text == "I'm doing well!"

    finally:
        db_file = Path(db_path)
        if db_file.exists():  # noqa: ASYNC240  # Cleanup code, not critical path
            db_file.unlink()  # noqa: ASYNC240  # Cleanup code, not critical path


@pytest.mark.asyncio
async def test_load_session_with_empty_history():
    """Test that load_session handles sessions with no conversation history gracefully."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

            def build_agent(context: AgentSessionContext):
                workflow = StateGraph(State)
                workflow.add_node("chat", simple_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)
                return workflow.compile(checkpointer=checkpointer)

            server = AgentServerACP(agent=build_agent, checkpointer=checkpointer)
            mock_client = AsyncMock()
            server.on_connect(mock_client)

            # Load a non-existent session
            load_response = await server.load_session(
                session_id="non-existent-session", cwd="/tmp/test", mcp_servers=[]
            )

            # Should return successfully
            assert load_response is not None

            # Should not have sent any updates
            update_calls = [
                call for call in mock_client.method_calls if call[0] == "session_update"
            ]
            assert len(update_calls) == 0

    finally:
        db_file = Path(db_path)
        if db_file.exists():  # noqa: ASYNC240  # Cleanup code, not critical path
            db_file.unlink()  # noqa: ASYNC240  # Cleanup code, not critical path


@pytest.mark.asyncio
async def test_load_session_persists_across_server_restarts():
    """Test that sessions can be loaded after server restart (simulated)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        session_id = None

        # First server instance - create session and add messages
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

            def build_agent(context: AgentSessionContext):
                workflow = StateGraph(State)
                workflow.add_node("chat", simple_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)
                return workflow.compile(checkpointer=checkpointer)

            server1 = AgentServerACP(agent=build_agent, checkpointer=checkpointer)
            mock_client1 = AsyncMock()
            server1.on_connect(mock_client1)

            # Create session
            new_session = await server1.new_session(cwd="/tmp/test")
            session_id = new_session.session_id

            # Add messages
            server1._reset_agent(session_id)
            agent = server1._agent
            if agent.checkpointer is None:
                agent.checkpointer = checkpointer

            config = {"configurable": {"thread_id": session_id}}
            # Create a simple conversation
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="Test message")]}, config
            )
            messages = result["messages"]
            messages.append(AIMessage(content="Test response"))
            await agent.ainvoke({"messages": messages}, config)

        # Second server instance - load the session (simulating restart)
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

            def build_agent2(context: AgentSessionContext):
                workflow = StateGraph(State)
                workflow.add_node("chat", simple_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)
                return workflow.compile(checkpointer=checkpointer)

            server2 = AgentServerACP(agent=build_agent2, checkpointer=checkpointer)
            mock_client2 = AsyncMock()
            server2.on_connect(mock_client2)

            # Load the session from the first server
            load_response = await server2.load_session(
                session_id=session_id, cwd="/tmp/test", mcp_servers=[]
            )

            # Verify session was loaded
            assert load_response is not None

            # Verify messages were replayed
            update_calls = [
                call for call in mock_client2.method_calls if call[0] == "session_update"
            ]
            assert len(update_calls) == 2

    finally:
        db_file = Path(db_path)
        if db_file.exists():  # noqa: ASYNC240  # Cleanup code, not critical path
            db_file.unlink()  # noqa: ASYNC240  # Cleanup code, not critical path


@pytest.mark.asyncio
async def test_list_sessions_capability_advertised_with_checkpointer():
    """Test that sessionCapabilities.list is advertised when checkpointer is provided."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

            def build_agent(context: AgentSessionContext):
                workflow = StateGraph(State)
                workflow.add_node("chat", simple_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)
                return workflow.compile(checkpointer=checkpointer)

            server = AgentServerACP(agent=build_agent, checkpointer=checkpointer)

            # Test capability is advertised
            response = await server.initialize(protocol_version=1)
            assert response.agent_capabilities.session_capabilities is not None
            assert response.agent_capabilities.session_capabilities.list is not None

    finally:
        db_file = Path(db_path)
        if db_file.exists():  # noqa: ASYNC240
            db_file.unlink()  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_list_sessions_returns_empty_without_checkpointer():
    """Test that list_sessions returns empty list when no checkpointer is provided."""

    def build_agent(context: AgentSessionContext):
        workflow = StateGraph(State)
        workflow.add_node("chat", simple_node)
        workflow.add_edge(START, "chat")
        workflow.add_edge("chat", END)
        return workflow.compile()

    server = AgentServerACP(agent=build_agent)

    # List sessions should return empty
    response = await server.list_sessions()
    assert response.sessions == []


@pytest.mark.asyncio
async def test_list_sessions_returns_created_sessions():
    """Test that list_sessions returns sessions that were created."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

            def build_agent(context: AgentSessionContext):
                workflow = StateGraph(State)
                workflow.add_node("chat", simple_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)
                return workflow.compile(checkpointer=checkpointer)

            server = AgentServerACP(agent=build_agent, checkpointer=checkpointer)
            mock_client = AsyncMock()
            server.on_connect(mock_client)

            # Create multiple sessions
            session1 = await server.new_session(cwd="/tmp/project1")
            session2 = await server.new_session(cwd="/tmp/project2")

            # Add messages to each session
            for session_id in [session1.session_id, session2.session_id]:
                server._reset_agent(session_id)
                agent = server._agent
                if agent.checkpointer is None:
                    agent.checkpointer = checkpointer

                config = {"configurable": {"thread_id": session_id}}
                await agent.ainvoke({"messages": [HumanMessage(content="Test")]}, config)

            # List all sessions
            response = await server.list_sessions()

            # Should return both sessions
            assert len(response.sessions) == 2

            # Check session IDs are present
            session_ids = {s.session_id for s in response.sessions}
            assert session1.session_id in session_ids
            assert session2.session_id in session_ids

            # Check cwds are correct
            session_cwds = {s.session_id: s.cwd for s in response.sessions}
            assert session_cwds[session1.session_id] == "/tmp/project1"
            assert session_cwds[session2.session_id] == "/tmp/project2"

    finally:
        db_file = Path(db_path)
        if db_file.exists():  # noqa: ASYNC240
            db_file.unlink()  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_list_sessions_filters_by_cwd():
    """Test that list_sessions can filter by working directory."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

            def build_agent(context: AgentSessionContext):
                workflow = StateGraph(State)
                workflow.add_node("chat", simple_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)
                return workflow.compile(checkpointer=checkpointer)

            server = AgentServerACP(agent=build_agent, checkpointer=checkpointer)
            mock_client = AsyncMock()
            server.on_connect(mock_client)

            # Create sessions with different cwds
            session1 = await server.new_session(cwd="/tmp/project1")
            session2 = await server.new_session(cwd="/tmp/project2")
            session3 = await server.new_session(cwd="/tmp/project1")

            # Add messages to each session
            for session_id in [session1.session_id, session2.session_id, session3.session_id]:
                server._reset_agent(session_id)
                agent = server._agent
                if agent.checkpointer is None:
                    agent.checkpointer = checkpointer

                config = {"configurable": {"thread_id": session_id}}
                await agent.ainvoke({"messages": [HumanMessage(content="Test")]}, config)

            # List sessions filtered by cwd
            response = await server.list_sessions(cwd="/tmp/project1")

            # Should return only sessions with matching cwd
            assert len(response.sessions) == 2
            for session in response.sessions:
                assert session.cwd == "/tmp/project1"

            # Check correct session IDs
            session_ids = {s.session_id for s in response.sessions}
            assert session1.session_id in session_ids
            assert session3.session_id in session_ids
            assert session2.session_id not in session_ids

    finally:
        db_file = Path(db_path)
        if db_file.exists():  # noqa: ASYNC240
            db_file.unlink()  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_list_sessions_includes_timestamps():
    """Test that list_sessions includes updated_at timestamps."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

            def build_agent(context: AgentSessionContext):
                workflow = StateGraph(State)
                workflow.add_node("chat", simple_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)
                return workflow.compile(checkpointer=checkpointer)

            server = AgentServerACP(agent=build_agent, checkpointer=checkpointer)
            mock_client = AsyncMock()
            server.on_connect(mock_client)

            # Create a session
            session1 = await server.new_session(cwd="/tmp/test")

            # Add message
            server._reset_agent(session1.session_id)
            agent = server._agent
            if agent.checkpointer is None:
                agent.checkpointer = checkpointer

            config = {"configurable": {"thread_id": session1.session_id}}
            await agent.ainvoke({"messages": [HumanMessage(content="Test")]}, config)

            # List sessions
            response = await server.list_sessions()

            # Should have timestamp
            assert len(response.sessions) == 1
            assert response.sessions[0].updated_at is not None

    finally:
        db_file = Path(db_path)
        if db_file.exists():  # noqa: ASYNC240
            db_file.unlink()  # noqa: ASYNC240
