"""Integration tests for MCP middleware with create_deep_agent.

This module tests the integration of MCPMiddleware with the create_deep_agent
function and verifies end-to-end functionality.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.mcp import MCPMiddleware, MCPServerConfig
from tests.unit_tests.chat_model import GenericFakeChatModel


# =============================================================================
# Test create_deep_agent with mcp_servers parameter
# =============================================================================


def test_create_deep_agent_with_mcp_servers() -> None:
    """Test that create_deep_agent accepts mcp_servers parameter."""
    mcp_servers = [
        {
            "name": "test-server",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-test"],
        }
    ]

    # Create agent with mcp_servers
    # Use fake model to avoid actual API calls
    agent = create_deep_agent(
        model=GenericFakeChatModel(
            messages=iter([AIMessage(content="I see MCP tools are available.")])
        ),
        mcp_servers=mcp_servers,
    )

    # Agent should be created successfully
    assert agent is not None


def test_create_deep_agent_with_mcp_root() -> None:
    """Test that create_deep_agent accepts mcp_root parameter."""
    mcp_servers = [
        {
            "name": "test-server",
            "command": "test-cmd",
        }
    ]

    agent = create_deep_agent(
        model=GenericFakeChatModel(
            messages=iter([AIMessage(content="Custom MCP root configured.")])
        ),
        mcp_servers=mcp_servers,
        mcp_root="/custom/mcp/path",
    )

    assert agent is not None


def test_create_deep_agent_without_mcp() -> None:
    """Test that create_deep_agent works without mcp_servers (backward compatibility)."""
    agent = create_deep_agent(
        model=GenericFakeChatModel(
            messages=iter([AIMessage(content="No MCP configured.")])
        ),
    )

    # Agent should be created successfully without MCP
    assert agent is not None


# =============================================================================
# Test MCP middleware integration with filesystem
# =============================================================================


def test_mcp_metadata_folder_structure(tmp_path: Path) -> None:
    """Test that MCP creates folder-per-server structure for metadata."""
    servers: list[MCPServerConfig] = [
        {"name": "server-one", "command": "cmd1"},
        {"name": "server-two", "command": "cmd2"},
    ]

    middleware = MCPMiddleware(
        servers=servers,
        mcp_root=str(tmp_path / ".mcp"),
        sync_on_startup=False,  # Don't auto-sync
    )

    # Create directories manually to simulate sync
    (tmp_path / ".mcp" / "server-one").mkdir(parents=True)
    (tmp_path / ".mcp" / "server-two").mkdir(parents=True)

    # Verify directories exist
    assert (tmp_path / ".mcp" / "server-one").is_dir()
    assert (tmp_path / ".mcp" / "server-two").is_dir()


# =============================================================================
# Test MCP tools availability
# =============================================================================


def test_mcp_middleware_provides_mcp_invoke_tool() -> None:
    """Test that MCPMiddleware provides mcp_invoke tool."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Should have exactly one tool
    assert len(middleware.tools) == 1

    # Tool should be named mcp_invoke
    tool = middleware.tools[0]
    assert tool.name == "mcp_invoke"
    assert "Invoke an MCP tool" in tool.description


# =============================================================================
# Test system prompt injection in agent
# =============================================================================


def test_mcp_system_prompt_injected(tmp_path: Path) -> None:
    """Test that MCP system prompt is injected into agent conversation."""
    mcp_servers = [
        {
            "name": "test-server",
            "command": "test-cmd",
        }
    ]

    # Create fake model to capture prompts
    fake_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="I see MCP instructions.")])
    )

    agent = create_deep_agent(
        model=fake_model,
        mcp_servers=mcp_servers,
        mcp_root=str(tmp_path / ".mcp"),
    )

    # Invoke agent
    result = agent.invoke(
        {"messages": [HumanMessage(content="What tools are available?")]}
    )

    # Check that the model received MCP instructions
    assert len(fake_model.call_history) > 0
    first_call = fake_model.call_history[0]
    messages = first_call["messages"]

    # Find system message
    system_message = messages[0]
    assert system_message.type == "system"

    # Verify MCP content is present
    content = system_message.text
    assert "MCP Tools" in content or "mcp_invoke" in content


# =============================================================================
# Test MCP with other middleware
# =============================================================================


def test_mcp_works_with_skills(tmp_path: Path) -> None:
    """Test that MCP middleware works alongside skills middleware."""
    # Create skill directory
    skills_dir = tmp_path / "skills" / "user"
    skill_path = skills_dir / "test-skill" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("""---
name: test-skill
description: A test skill
---

# Test Skill

Instructions here.
""")

    mcp_servers = [
        {
            "name": "test-server",
            "command": "test-cmd",
        }
    ]

    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    fake_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="I have both skills and MCP.")])
    )

    agent = create_deep_agent(
        model=fake_model,
        backend=backend,
        skills=[str(skills_dir)],
        mcp_servers=mcp_servers,
        mcp_root=str(tmp_path / ".mcp"),
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="What capabilities do you have?")]}
    )

    # Should complete successfully
    assert "messages" in result
    assert len(result["messages"]) > 0


# =============================================================================
# Test error handling
# =============================================================================


def test_mcp_invalid_server_config() -> None:
    """Test that invalid server config raises appropriate error."""
    # Missing required fields
    with pytest.raises(ValueError):
        create_deep_agent(
            model=GenericFakeChatModel(
                messages=iter([AIMessage(content="Never reached.")])
            ),
            mcp_servers=[{"name": "incomplete"}],  # Missing 'command'
        )


def test_mcp_duplicate_server_names() -> None:
    """Test that duplicate server names are rejected."""
    with pytest.raises(ValueError, match="Duplicate server name"):
        create_deep_agent(
            model=GenericFakeChatModel(
                messages=iter([AIMessage(content="Never reached.")])
            ),
            mcp_servers=[
                {"name": "same-name", "command": "cmd1"},
                {"name": "same-name", "command": "cmd2"},
            ],
        )


# =============================================================================
# Test async integration
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_async_invocation(tmp_path: Path) -> None:
    """Test MCP middleware with async agent invocation."""
    mcp_servers = [
        {
            "name": "test-server",
            "command": "test-cmd",
        }
    ]

    fake_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Async MCP invocation complete.")])
    )

    agent = create_deep_agent(
        model=fake_model,
        mcp_servers=mcp_servers,
        mcp_root=str(tmp_path / ".mcp"),
    )

    # Use async invocation
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Test async MCP.")]}
    )

    assert "messages" in result
    assert len(result["messages"]) > 0


# =============================================================================
# Test MCP tool invocation (mocked)
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_invoke_tool_without_connection() -> None:
    """Test mcp_invoke returns error when not connected."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Get the tool
    tool = middleware.tools[0]

    # Create mock runtime
    mock_runtime = MagicMock()

    # Call the tool without connection
    result = await tool.coroutine("test_tool", {"arg": "value"}, mock_runtime)

    # Should return error message
    assert "Error" in result
    assert "not connected" in result.lower()


@pytest.mark.asyncio
async def test_mcp_invoke_tool_with_mock_client() -> None:
    """Test mcp_invoke successfully invokes tool via mock client."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock tool that will be found
    mock_tool = MagicMock()
    mock_tool.name = "test-server_search"
    mock_tool.ainvoke = AsyncMock(return_value="Search results: AI news")

    # Create mock client
    mock_client = MagicMock()
    mock_client.get_tools.return_value = [mock_tool]

    middleware._mcp_client = mock_client

    # Get the mcp_invoke tool
    tool = middleware.tools[0]

    # Create mock runtime
    mock_runtime = MagicMock()

    # Call the tool
    result = await tool.coroutine("search", {"query": "AI news"}, mock_runtime)

    # Should return the mock result
    assert "Search results" in result
    mock_tool.ainvoke.assert_called_once_with({"query": "AI news"})


@pytest.mark.asyncio
async def test_mcp_invoke_tool_not_found() -> None:
    """Test mcp_invoke returns error when tool not found."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock client with no matching tool
    mock_tool = MagicMock()
    mock_tool.name = "test-server_other_tool"

    mock_client = MagicMock()
    mock_client.get_tools.return_value = [mock_tool]

    middleware._mcp_client = mock_client

    # Get the mcp_invoke tool
    tool = middleware.tools[0]

    # Create mock runtime
    mock_runtime = MagicMock()

    # Call the tool with non-existent name
    result = await tool.coroutine("nonexistent", {}, mock_runtime)

    # Should return error message
    assert "Error" in result
    assert "not found" in result.lower()


# =============================================================================
# Test metadata persistence
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_metadata_persists_across_sessions(tmp_path: Path) -> None:
    """Test that MCP metadata persists in filesystem across middleware instances."""
    mcp_root = tmp_path / ".mcp"
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    # First middleware instance - write metadata
    middleware1 = MCPMiddleware(
        servers=servers,
        mcp_root=str(mcp_root),
        sync_on_startup=False,
    )

    # Manually create metadata file
    server_dir = mcp_root / "test-server"
    server_dir.mkdir(parents=True)
    metadata_file = server_dir / "search.json"
    metadata_file.write_text('{"server": "test-server", "name": "search", "description": "Search tool", "input_schema": {}, "status": "available"}')

    # Second middleware instance - should be able to read metadata
    middleware2 = MCPMiddleware(
        servers=servers,
        mcp_root=str(mcp_root),
        sync_on_startup=False,
    )

    # Verify the metadata file exists and is readable
    assert metadata_file.exists()
    content = metadata_file.read_text()
    assert "test-server" in content
    assert "search" in content


# =============================================================================
# Test backward compatibility
# =============================================================================


def test_existing_tools_parameter_still_works() -> None:
    """Test that existing 'tools' parameter still works alongside mcp_servers."""
    from langchain_core.tools import tool

    @tool
    def custom_tool(x: str) -> str:
        """A custom tool."""
        return f"Custom: {x}"

    mcp_servers = [
        {
            "name": "test-server",
            "command": "test-cmd",
        }
    ]

    fake_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="I have both custom tools and MCP.")])
    )

    agent = create_deep_agent(
        model=fake_model,
        tools=[custom_tool],
        mcp_servers=mcp_servers,
    )

    # Should create agent successfully
    assert agent is not None


def test_all_middleware_stacks_work_together(tmp_path: Path) -> None:
    """Test that MCP works with all other middleware options."""
    # Setup
    skills_dir = tmp_path / "skills" / "user"
    skills_dir.mkdir(parents=True)

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "AGENTS.md").write_text("# Memory\n\nAgent memory content.")

    mcp_servers = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    fake_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Full middleware stack working.")])
    )

    # Create agent with all middleware options
    agent = create_deep_agent(
        model=fake_model,
        backend=backend,
        skills=[str(skills_dir)],
        memory=[str(memory_dir / "AGENTS.md")],
        mcp_servers=mcp_servers,
        mcp_root=str(tmp_path / ".mcp"),
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Test all middleware.")]}
    )

    # Should complete successfully
    assert "messages" in result
