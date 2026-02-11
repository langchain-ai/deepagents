"""Comprehensive tests for ACPAgentServer and serve_acp_stdio with the new ACP architecture."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp.schema import TextContentBlock
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import ACPAgentServer, serve_acp_stdio


@tool(description="Read a file from the filesystem")
def read_file_tool(file_path: str) -> str:
    """Read a file.

    Args:
        file_path: Path to the file to read

    Returns:
        File contents
    """
    return f"Contents of {file_path}"


class MockClient:
    """Mock ACP Client for testing."""

    def __init__(self):
        self.session_updates = []

    async def session_update(self, session_id: str, update: Any, source: str):
        """Track session updates."""
        self.session_updates.append(
            {
                "session_id": session_id,
                "update": update,
                "source": source,
            }
        )


class TestRunAgent:
    """Test suite for the serve_acp_stdio entry point function."""

    @pytest.mark.asyncio
    async def test_run_agent_initializes_in_ask_before_edits_mode(self) -> None:
        """Test that serve_acp_stdio starts with ask_before_edits mode by default."""
        with patch("deepagents_acp.server.run_acp_agent", new_callable=AsyncMock) as mock_run:
            with patch("deepagents_acp.server.ACPAgentServer") as mock_agent_class:
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent

                await serve_acp_stdio("/test/root")

                # Verify ACPAgentServer was initialized with ask_before_edits mode
                mock_agent_class.assert_called_once()
                call_kwargs = mock_agent_class.call_args[1]
                assert call_kwargs["root_dir"] == "/test/root"
                assert call_kwargs["mode"] == "ask_before_edits"
                # MemorySaver is actually InMemorySaver in langgraph
                assert "Saver" in call_kwargs["checkpointer"].__class__.__name__

                # Verify run_acp_agent was called with the agent
                mock_run.assert_called_once_with(mock_agent)

    @pytest.mark.asyncio
    async def test_run_agent_creates_new_memory_saver(self) -> None:
        """Test that serve_acp_stdio creates a fresh MemorySaver for each invocation."""
        checkpointers = []

        with patch("deepagents_acp.server.run_acp_agent", new_callable=AsyncMock):
            with patch("deepagents_acp.server.ACPAgentServer") as mock_agent_class:
                mock_agent_class.return_value = MagicMock()

                def capture_checkpointer(**kwargs):
                    checkpointers.append(id(kwargs["checkpointer"]))
                    return MagicMock()

                mock_agent_class.side_effect = capture_checkpointer

                await serve_acp_stdio("/test/root1")
                await serve_acp_stdio("/test/root2")

                # Verify each call creates a unique MemorySaver instance
                assert len(checkpointers) == 2
                assert checkpointers[0] != checkpointers[1]


class TestACPAgentServerModes:
    """Test suite for ACPAgentServer mode configurations."""

    def test_ask_before_edits_mode_config(self) -> None:
        """Test interrupt configuration for ask_before_edits mode."""
        config = ACPAgentServer._get_interrupt_config("ask_before_edits")

        assert config == {
            "edit_file": {"allowed_decisions": ["approve", "reject"]},
            "write_file": {"allowed_decisions": ["approve", "reject"]},
            "write_todos": {"allowed_decisions": ["approve", "reject"]},
        }

    def test_auto_mode_config(self) -> None:
        """Test interrupt configuration for auto mode."""
        config = ACPAgentServer._get_interrupt_config("auto")

        # Auto mode only asks for permission on todos, not file operations
        assert config == {
            "write_todos": {"allowed_decisions": ["approve", "reject"]},
        }
        assert "edit_file" not in config
        assert "write_file" not in config

    def test_unknown_mode_returns_empty_config(self) -> None:
        """Test that unknown mode returns no interrupts."""
        config = ACPAgentServer._get_interrupt_config("unknown_mode")

        assert config == {}


class TestACPAgentServerInitialization:
    """Test suite for ACPAgentServer initialization."""

    def test_initialization_sets_attributes(self) -> None:
        """Test that ACPAgentServer initialization sets all required attributes."""
        with patch("deepagents_acp.server.create_deep_agent") as mock_create:
            mock_create.return_value = MagicMock()
            checkpointer = MemorySaver()

            agent = ACPAgentServer(
                root_dir="/test/root",
                mode="ask_before_edits",
                checkpointer=checkpointer,
            )

            assert agent._root_dir == "/test/root"
            assert agent._mode == "ask_before_edits"
            assert agent._checkpointer is checkpointer
            assert agent._cancelled is False
            assert agent._deepagent is not None

    @pytest.mark.skip(reason="test not working yet.")
    def test_create_deepagent_uses_filesystem_backend_with_virtual_mode(self) -> None:
        """Test that _create_deepagent creates a FilesystemBackend with virtual_mode=True."""
        with patch("deepagents_acp.server.FilesystemBackend") as mock_backend_class:
            with patch("deepagents_acp.server.create_deep_agent") as mock_create:
                mock_backend = MagicMock()
                mock_backend_class.return_value = mock_backend
                mock_create.return_value = MagicMock()
                checkpointer = MemorySaver()

                ACPAgentServer(
                    root_dir="/test/root",
                    mode="ask_before_edits",
                    checkpointer=checkpointer,
                )

                # Verify FilesystemBackend was created with virtual_mode=True
                mock_backend_class.assert_called_once_with(root_dir="/test/root", virtual_mode=True)

                # Verify create_deep_agent was called with correct args
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs["checkpointer"] is checkpointer
                assert call_kwargs["backend"] is mock_backend
                assert call_kwargs["interrupt_on"] == {
                    "edit_file": {"allowed_decisions": ["approve", "reject"]},
                    "write_file": {"allowed_decisions": ["approve", "reject"]},
                    "write_todos": {"allowed_decisions": ["approve", "reject"]},
                }

    def test_initialization_with_auto_mode(self) -> None:
        """Test that ACPAgentServer can be initialized with auto mode."""
        with patch("deepagents_acp.server.create_deep_agent") as mock_create:
            mock_create.return_value = MagicMock()
            checkpointer = MemorySaver()

            agent = ACPAgentServer(
                root_dir="/test/root",
                mode="auto",
                checkpointer=checkpointer,
            )

            assert agent._mode == "auto"

            # Verify create_deep_agent was called with auto mode interrupt config
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["interrupt_on"] == {
                "write_todos": {"allowed_decisions": ["approve", "reject"]},
            }

    def test_on_connect_sets_connection(self):
        """Test that on_connect sets the client connection."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            mock_client = MockClient()
            agent.on_connect(mock_client)

            assert agent._conn is mock_client

    @pytest.mark.asyncio
    async def test_initialize_returns_capabilities(self):
        """Test that initialize returns correct protocol version and capabilities."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            response = await agent.initialize(protocol_version=1)

            assert response.protocol_version == 1
            assert response.agent_capabilities is not None
            assert response.agent_capabilities.prompt_capabilities.image is True

    @pytest.mark.asyncio
    async def test_new_session_returns_available_modes(self):
        """Test that new_session returns session ID and available modes."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            response = await agent.new_session(cwd="/tmp", mcp_servers=[])

            # Verify session ID is generated
            assert response.session_id is not None
            assert len(response.session_id) > 0

            # Verify modes are returned
            assert response.modes is not None
            assert len(response.modes.available_modes) == 2
            assert response.modes.current_mode_id == "ask_before_edits"

            # Verify mode details
            mode_ids = [m.id for m in response.modes.available_modes]
            assert "ask_before_edits" in mode_ids
            assert "auto" in mode_ids

    @pytest.mark.asyncio
    async def test_set_session_mode_changes_mode(self):
        """Test that set_session_mode recreates the agent with new mode."""
        with patch("deepagents_acp.server.create_deep_agent") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            assert agent._mode == "ask_before_edits"

            # Change mode
            await agent.set_session_mode(mode_id="auto", session_id="test-session")

            # Verify mode changed
            assert agent._mode == "auto"

            # Verify agent was recreated with new mode
            assert mock_create.call_count >= 2  # Once for init, once for mode change


class TestACPAgentServerPromptHandling:
    """Test ACPAgentServer prompt processing."""

    @pytest.mark.asyncio
    async def test_prompt_with_text_content(self):
        """Test processing a simple text prompt."""
        with patch("deepagents_acp.server.create_deep_agent") as mock_create:
            # Mock the deep agent graph to return a simple response
            mock_graph = MagicMock()

            async def mock_astream(*args, **kwargs):
                # Yield message chunks
                yield ("Hello", {})
                yield (" ", {})
                yield ("world!", {})

            mock_graph.astream = mock_astream
            # Mock aget_state to return a state with no interrupts
            mock_state = MagicMock()
            mock_state.interrupts = []
            mock_graph.aget_state = AsyncMock(return_value=mock_state)
            mock_create.return_value = mock_graph

            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            mock_client = MockClient()
            agent.on_connect(mock_client)

            # Send prompt
            response = await agent.prompt(
                prompt=[TextContentBlock(text="Hello", type="text")],
                session_id="test-session",
            )

            # Verify response
            assert response is not None

            # Verify session updates were sent
            assert len(mock_client.session_updates) > 0

            # Check that text was logged
            text_updates = [
                u for u in mock_client.session_updates if hasattr(u["update"], "content")
            ]
            assert len(text_updates) > 0


class TestACPAgentServerToolHandling:
    """Test ACPAgentServer tool call handling."""

    @pytest.mark.asyncio
    async def test_tool_call_update_sent(self):
        """Test that tool call updates are sent to client."""
        with patch("deepagents_acp.server.create_deep_agent") as mock_create:
            mock_graph = MagicMock()

            # Create a mock message with tool call chunks
            # The chunk needs id, name, and complete args to trigger tool call
            mock_message = MagicMock()
            mock_message.tool_call_chunks = [
                {
                    "id": "call_123",
                    "name": "read_file",
                    "args": '{"file_path": "/test.py"}',
                    "index": 0,
                },
            ]

            async def mock_astream(*args, **kwargs):
                yield (mock_message, {})

            mock_graph.astream = mock_astream
            mock_state = MagicMock()
            mock_state.interrupts = []
            mock_graph.aget_state = AsyncMock(return_value=mock_state)
            mock_create.return_value = mock_graph

            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            mock_client = MockClient()
            agent.on_connect(mock_client)

            # Send prompt
            await agent.prompt(
                prompt=[TextContentBlock(text="Read test.py", type="text")],
                session_id="test-session",
            )

            # Verify tool call update was sent
            tool_updates = [
                u for u in mock_client.session_updates if hasattr(u["update"], "tool_call_id")
            ]
            assert len(tool_updates) > 0

            # Verify tool call details
            tool_update = tool_updates[0]["update"]
            assert tool_update.tool_call_id == "call_123"
            assert "Read" in tool_update.title
            assert tool_update.status == "pending"


class TestACPAgentServerTodoHandling:
    """Test ACPAgentServer todo/plan handling."""

    @pytest.mark.asyncio
    async def test_write_todos_sends_plan_update(self):
        """Test that write_todos tool sends a plan update."""
        with patch("deepagents_acp.server.create_deep_agent") as mock_create:
            mock_graph = MagicMock()

            # Create mock message with write_todos tool call
            mock_message = MagicMock()
            todos_json = '{"todos": [{"content": "Task 1", "status": "pending"}, {"content": "Task 2", "status": "in_progress"}]}'  # noqa
            mock_message.tool_call_chunks = [
                {"id": "call_todos", "name": "write_todos", "args": todos_json, "index": 0},
            ]

            async def mock_astream(*args, **kwargs):
                yield (mock_message, {})

            mock_graph.astream = mock_astream
            mock_state = MagicMock()
            mock_state.interrupts = []
            mock_graph.aget_state = AsyncMock(return_value=mock_state)
            mock_create.return_value = mock_graph

            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            mock_client = MockClient()
            agent.on_connect(mock_client)

            # Send prompt
            await agent.prompt(
                prompt=[TextContentBlock(text="Create a plan", type="text")],
                session_id="test-session",
            )

            # Verify plan update was sent
            plan_updates = [
                u
                for u in mock_client.session_updates
                if hasattr(u["update"], "session_update") and u["update"].session_update == "plan"
            ]
            assert len(plan_updates) > 0

            # Verify plan entries
            plan_update = plan_updates[0]["update"]
            assert len(plan_update.entries) == 2
            assert plan_update.entries[0].content == "Task 1"
            assert plan_update.entries[0].status == "pending"
            assert plan_update.entries[1].content == "Task 2"
            assert plan_update.entries[1].status == "in_progress"

    @pytest.mark.asyncio
    async def test_clear_plan_sends_empty_update(self):
        """Test that _clear_plan sends an empty plan update."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            mock_client = MockClient()
            agent.on_connect(mock_client)

            # Clear plan
            await agent._clear_plan("test-session")

            # Verify empty plan update was sent
            assert len(mock_client.session_updates) == 1
            update = mock_client.session_updates[0]
            assert update["update"].session_update == "plan"
            assert len(update["update"].entries) == 0


class TestACPAgentServerToolCallFormatting:
    """Test tool call update formatting for different tools."""

    def test_create_tool_call_update_for_read_file(self):
        """Test tool call update creation for read_file."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            update = agent._create_tool_call_update(
                tool_id="call_123",
                tool_name="read_file",
                tool_args={"file_path": "/test/file.py"},
            )

            assert update.tool_call_id == "call_123"
            assert "Read" in update.title
            assert "`/test/file.py`" in update.title
            assert update.kind == "read"
            assert update.status == "pending"

    def test_create_tool_call_update_for_edit_file(self):
        """Test tool call update creation for edit_file."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            update = agent._create_tool_call_update(
                tool_id="call_123",
                tool_name="edit_file",
                tool_args={
                    "file_path": "/test/file.py",
                    "old_string": "old code",
                    "new_string": "new code",
                },
            )

            assert update.tool_call_id == "call_123"
            assert "Edit" in update.title
            assert "`/test/file.py`" in update.title

    def test_create_tool_call_update_for_write_file(self):
        """Test tool call update creation for write_file."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            update = agent._create_tool_call_update(
                tool_id="call_123",
                tool_name="write_file",
                tool_args={"file_path": "/test/file.py"},
            )

            assert update.tool_call_id == "call_123"
            assert "Write" in update.title
            assert "`/test/file.py`" in update.title
            assert update.kind == "edit"

    def test_create_tool_call_update_for_search_tools(self):
        """Test tool call update creation for search tools."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            for tool_name in ["ls", "glob", "grep"]:
                update = agent._create_tool_call_update(
                    tool_id="call_123",
                    tool_name=tool_name,
                    tool_args={},
                )

                assert update.tool_call_id == "call_123"
                assert update.title == tool_name
                assert update.kind == "search"


class TestACPAgentServerEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_mode_switch_affects_interrupt_behavior(self):
        """Test that switching modes changes interrupt configuration."""
        with patch("deepagents_acp.server.create_deep_agent") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            # Initially in ask_before_edits mode
            assert agent._mode == "ask_before_edits"

            # Verify initial interrupt config
            initial_calls = mock_create.call_count
            initial_interrupt_on = mock_create.call_args[1]["interrupt_on"]
            assert "edit_file" in initial_interrupt_on
            assert "write_file" in initial_interrupt_on
            assert "write_todos" in initial_interrupt_on

            # Switch to auto mode
            await agent.set_session_mode(mode_id="auto", session_id="test-session")

            # Verify mode switched
            assert agent._mode == "auto"

            # Verify new agent was created with different interrupt config
            assert mock_create.call_count == initial_calls + 1
            new_interrupt_on = mock_create.call_args[1]["interrupt_on"]
            assert "edit_file" not in new_interrupt_on
            assert "write_file" not in new_interrupt_on
            assert "write_todos" in new_interrupt_on  # Only todos in auto mode


class TestACPAgentServerPlanApproval:
    """Test ACPAgentServer plan auto-approval behavior."""

    @pytest.mark.asyncio
    async def test_initial_plan_requires_approval(self):
        """Test that initial plan requires user approval and is stored after approval."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.outcome.outcome = "selected"
            mock_response.outcome.option_id = "approve"
            mock_client.request_permission = AsyncMock(return_value=mock_response)
            mock_client.session_update = AsyncMock()
            agent.on_connect(mock_client)

            session_id = "test-session"
            todos = [{"content": "Task 1", "status": "pending"}]

            # Setup interrupt
            mock_interrupt = MagicMock()
            mock_interrupt.id = "call_todos"
            mock_interrupt.value = {
                "action_requests": [{"name": "write_todos", "args": {"todos": todos}}]
            }
            mock_state = MagicMock()
            mock_state.next = ("some_node",)
            mock_state.interrupts = [mock_interrupt]

            # Process interrupt
            decisions = await agent._handle_interrupts(
                current_state=mock_state,
                session_id=session_id,
                active_tool_calls={},
            )

            # Verify user approval was requested
            mock_client.request_permission.assert_called_once()
            assert decisions[0]["type"] == "approve"

            # Verify plan was stored after approval
            assert session_id in agent._session_plans
            assert agent._session_plans[session_id] == todos

    @pytest.mark.asyncio
    async def test_plan_updates_auto_approved_when_in_progress(self):
        """Test that plan updates are auto-approved when plan is still in progress."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            mock_client = MagicMock()
            mock_client.request_permission = AsyncMock()
            mock_client.session_update = AsyncMock()
            agent.on_connect(mock_client)

            session_id = "test-session"

            # Set existing in-progress plan
            agent._session_plans[session_id] = [
                {"content": "Task 1", "status": "completed"},
                {"content": "Task 2", "status": "in_progress"},
            ]

            # Setup interrupt with updated plan
            updated_todos = [
                {"content": "Task 1", "status": "completed"},
                {"content": "Task 2 - updated", "status": "completed"},
            ]
            mock_interrupt = MagicMock()
            mock_interrupt.id = "call_todos"
            mock_interrupt.value = {
                "action_requests": [{"name": "write_todos", "args": {"todos": updated_todos}}]
            }
            mock_state = MagicMock()
            mock_state.next = ("some_node",)
            mock_state.interrupts = [mock_interrupt]

            # Process interrupt
            decisions = await agent._handle_interrupts(
                current_state=mock_state,
                session_id=session_id,
                active_tool_calls={},
            )

            # Verify auto-approval (no permission request)
            mock_client.request_permission.assert_not_called()
            assert decisions[0]["type"] == "approve"

            # Verify plan was updated
            assert agent._session_plans[session_id] == updated_todos

    @pytest.mark.asyncio
    async def test_new_plan_requires_approval_after_completion(self):
        """Test that new plans require approval after all tasks are completed."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.outcome.outcome = "selected"
            mock_response.outcome.option_id = "approve"
            mock_client.request_permission = AsyncMock(return_value=mock_response)
            mock_client.session_update = AsyncMock()
            agent.on_connect(mock_client)

            session_id = "test-session"

            # Set existing completed plan
            agent._session_plans[session_id] = [
                {"content": "Task 1", "status": "completed"},
                {"content": "Task 2", "status": "completed"},
            ]

            # Setup interrupt with new plan
            new_todos = [{"content": "New Task", "status": "pending"}]
            mock_interrupt = MagicMock()
            mock_interrupt.id = "call_todos"
            mock_interrupt.value = {
                "action_requests": [{"name": "write_todos", "args": {"todos": new_todos}}]
            }
            mock_state = MagicMock()
            mock_state.next = ("some_node",)
            mock_state.interrupts = [mock_interrupt]

            # Process interrupt
            decisions = await agent._handle_interrupts(
                current_state=mock_state,
                session_id=session_id,
                active_tool_calls={},
            )

            # Verify user approval was requested (not auto-approved)
            mock_client.request_permission.assert_called_once()
            assert decisions[0]["type"] == "approve"

            # Verify new plan was stored
            assert agent._session_plans[session_id] == new_todos

    @pytest.mark.asyncio
    async def test_clear_plan_removes_from_session_plans(self):
        """Test that clearing a plan removes it from session_plans."""
        with patch("deepagents_acp.server.create_deep_agent"):
            agent = ACPAgentServer(
                root_dir="/test",
                mode="ask_before_edits",
                checkpointer=MemorySaver(),
            )

            mock_client = MockClient()
            agent.on_connect(mock_client)

            session_id = "test-session"
            agent._session_plans[session_id] = [{"content": "Task 1", "status": "pending"}]

            await agent._clear_plan(session_id)

            assert agent._session_plans[session_id] == []


[]
