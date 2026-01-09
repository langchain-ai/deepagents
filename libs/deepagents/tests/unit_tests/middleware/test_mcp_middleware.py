"""Unit tests for MCP middleware.

This module tests the MCPMiddleware class and helper functions for progressive
MCP tool discovery and execution.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.mcp import (
    MCP_SYSTEM_PROMPT,
    MCPMiddleware,
    MCPServerConfig,
    MCPState,
    MCPToolMetadata,
)


# =============================================================================
# Test Server Validation
# =============================================================================


def test_validate_servers_valid() -> None:
    """Test that valid server configurations are accepted."""
    servers: list[MCPServerConfig] = [
        {
            "name": "brave-search",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": "test-key"},
        },
        {
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
        },
    ]

    # Should not raise
    middleware = MCPMiddleware(servers=servers)
    assert middleware.servers == servers


def test_validate_servers_empty() -> None:
    """Test that empty server list raises ValueError."""
    with pytest.raises(ValueError, match="At least one MCP server"):
        MCPMiddleware(servers=[])


def test_validate_servers_missing_name() -> None:
    """Test that missing 'name' field raises ValueError."""
    servers = [
        {
            "command": "npx",
        }
    ]

    with pytest.raises(ValueError, match="requires 'name' and 'command'"):
        MCPMiddleware(servers=servers)  # type: ignore[arg-type]


def test_validate_servers_missing_command() -> None:
    """Test that missing 'command' field raises ValueError."""
    servers = [
        {
            "name": "test-server",
        }
    ]

    with pytest.raises(ValueError, match="requires 'name' and 'command'"):
        MCPMiddleware(servers=servers)  # type: ignore[arg-type]


def test_validate_servers_duplicate_names() -> None:
    """Test that duplicate server names raise ValueError."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "cmd1"},
        {"name": "test-server", "command": "cmd2"},
    ]

    with pytest.raises(ValueError, match="Duplicate server name"):
        MCPMiddleware(servers=servers)


# =============================================================================
# Test Middleware Initialization
# =============================================================================


def test_middleware_initialization() -> None:
    """Test middleware initialization with valid configuration."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(
        servers=servers,
        mcp_root="/custom/mcp/root",
        sync_on_startup=False,
    )

    assert middleware.servers == servers
    assert middleware.mcp_root == "/custom/mcp/root"
    assert middleware.sync_on_startup is False
    assert middleware._mcp_client is None
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "mcp_invoke"


def test_middleware_default_values() -> None:
    """Test middleware uses correct default values."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    assert middleware.mcp_root == "/tmp/.mcp"
    assert middleware.sync_on_startup is True


def test_middleware_creates_filesystem_backend(tmp_path: Path) -> None:
    """Test that middleware creates FilesystemBackend for metadata storage."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(
        servers=servers,
        mcp_root=str(tmp_path / ".mcp"),
    )

    assert isinstance(middleware._mcp_backend, FilesystemBackend)


# =============================================================================
# Test Tool Creation
# =============================================================================


def test_mcp_invoke_tool_created() -> None:
    """Test that mcp_invoke tool is created during initialization."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.name == "mcp_invoke"
    assert "MCP tool" in tool.description
    assert "ls /.mcp/" in tool.description


def test_mcp_invoke_tool_has_closure() -> None:
    """Test that mcp_invoke tool captures middleware instance via closure."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)
    tool = middleware.tools[0]

    # The tool should have access to middleware via closure
    # We can verify this by checking the coroutine exists
    assert tool.coroutine is not None


# =============================================================================
# Test Before Agent Hooks
# =============================================================================


def test_before_agent_sets_initialized() -> None:
    """Test that before_agent returns state update marking as initialized."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Call before_agent with empty state
    result = middleware.before_agent({}, None, {})  # type: ignore

    assert result is not None
    assert result["mcp_initialized"] is True


def test_before_agent_skips_if_initialized() -> None:
    """Test that before_agent returns None if already initialized."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Call before_agent with state that has mcp_initialized=True
    state: MCPState = {"mcp_initialized": True}  # type: ignore
    result = middleware.before_agent(state, None, {})  # type: ignore

    assert result is None


@pytest.mark.asyncio
async def test_abefore_agent_sets_initialized() -> None:
    """Test that abefore_agent returns state update marking as initialized."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Mock connect to avoid actually connecting
    middleware.connect = AsyncMock()  # type: ignore

    # Call abefore_agent with empty state
    result = await middleware.abefore_agent({}, None, {})  # type: ignore

    assert result is not None
    assert result["mcp_initialized"] is True


@pytest.mark.asyncio
async def test_abefore_agent_skips_if_initialized() -> None:
    """Test that abefore_agent returns None if already initialized."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Call abefore_agent with state that has mcp_initialized=True
    state: MCPState = {"mcp_initialized": True}  # type: ignore
    result = await middleware.abefore_agent(state, None, {})  # type: ignore

    assert result is None


@pytest.mark.asyncio
async def test_abefore_agent_auto_connects() -> None:
    """Test that abefore_agent automatically connects if not connected."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Mock connect
    connect_mock = AsyncMock()
    middleware.connect = connect_mock  # type: ignore

    # Call abefore_agent
    await middleware.abefore_agent({}, None, {})  # type: ignore

    # Verify connect was called
    connect_mock.assert_called_once()


@pytest.mark.asyncio
async def test_abefore_agent_handles_connect_failure() -> None:
    """Test that abefore_agent handles connection failure gracefully."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Mock connect to raise an exception
    middleware.connect = AsyncMock(side_effect=Exception("Connection failed"))  # type: ignore

    # Should not raise, but should still return initialized=True
    result = await middleware.abefore_agent({}, None, {})  # type: ignore

    assert result is not None
    assert result["mcp_initialized"] is True


# =============================================================================
# Test System Prompt Injection
# =============================================================================


def test_inject_mcp_prompt_with_existing_prompt() -> None:
    """Test that MCP prompt is appended to existing system prompt."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock request with existing system prompt
    mock_request = MagicMock()
    mock_request.system_prompt = "Existing system prompt"
    mock_request.override = MagicMock(return_value=mock_request)

    result = middleware._inject_mcp_prompt(mock_request)

    # Verify override was called with combined prompt
    mock_request.override.assert_called_once()
    call_kwargs = mock_request.override.call_args[1]
    assert "Existing system prompt" in call_kwargs["system_prompt"]
    assert "MCP Tools" in call_kwargs["system_prompt"]


def test_inject_mcp_prompt_without_existing_prompt() -> None:
    """Test that MCP prompt is used directly when no existing prompt."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock request with no system prompt
    mock_request = MagicMock()
    mock_request.system_prompt = None
    mock_request.override = MagicMock(return_value=mock_request)

    result = middleware._inject_mcp_prompt(mock_request)

    # Verify override was called with MCP prompt only
    mock_request.override.assert_called_once()
    call_kwargs = mock_request.override.call_args[1]
    assert call_kwargs["system_prompt"] == MCP_SYSTEM_PROMPT


def test_wrap_model_call_injects_prompt() -> None:
    """Test that wrap_model_call injects MCP prompt."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock request and handler
    mock_request = MagicMock()
    mock_request.system_prompt = "Original prompt"
    mock_request.override = MagicMock(return_value=mock_request)

    mock_handler = MagicMock(return_value="response")

    # Call wrap_model_call
    result = middleware.wrap_model_call(mock_request, mock_handler)

    # Verify handler was called
    mock_handler.assert_called_once()


@pytest.mark.asyncio
async def test_awrap_model_call_injects_prompt() -> None:
    """Test that awrap_model_call injects MCP prompt."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock request and handler
    mock_request = MagicMock()
    mock_request.system_prompt = "Original prompt"
    mock_request.override = MagicMock(return_value=mock_request)

    mock_handler = AsyncMock(return_value="response")

    # Call awrap_model_call
    result = await middleware.awrap_model_call(mock_request, mock_handler)

    # Verify handler was called
    mock_handler.assert_called_once()


# =============================================================================
# Test Lifecycle Management
# =============================================================================


@pytest.mark.asyncio
async def test_connect_requires_langchain_mcp_adapters() -> None:
    """Test that connect raises ImportError if langchain-mcp-adapters not installed."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Mock the import to fail
    with patch.dict("sys.modules", {"langchain_mcp_adapters": None, "langchain_mcp_adapters.client": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="langchain-mcp-adapters is required"):
                await middleware.connect()


@pytest.mark.asyncio
async def test_close_cleans_up_client() -> None:
    """Test that close() cleans up MCP client."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Mock the client
    mock_client = AsyncMock()
    middleware._mcp_client = mock_client

    # Call close
    await middleware.close()

    # Verify cleanup
    mock_client.__aexit__.assert_called_once()
    assert middleware._mcp_client is None


@pytest.mark.asyncio
async def test_close_handles_errors() -> None:
    """Test that close() handles errors gracefully."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Mock the client to raise an error on exit
    mock_client = AsyncMock()
    mock_client.__aexit__ = AsyncMock(side_effect=Exception("Cleanup failed"))
    middleware._mcp_client = mock_client

    # Should not raise
    await middleware.close()

    # Client should still be cleaned up
    assert middleware._mcp_client is None


@pytest.mark.asyncio
async def test_context_manager() -> None:
    """Test async context manager support."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "test-cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Mock connect and close
    middleware.connect = AsyncMock()  # type: ignore
    middleware.close = AsyncMock()  # type: ignore

    async with middleware as mcp:
        assert mcp is middleware
        middleware.connect.assert_called_once()

    middleware.close.assert_called_once()


# =============================================================================
# Test Server Name Extraction
# =============================================================================


def test_extract_server_name_with_prefix() -> None:
    """Test server name extraction from prefixed tool name."""
    servers: list[MCPServerConfig] = [
        {"name": "brave-search", "command": "cmd"},
        {"name": "filesystem", "command": "cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock tool with prefixed name
    mock_tool = MagicMock()
    mock_tool.name = "brave-search_search"

    result = middleware._extract_server_name(mock_tool)
    assert result == "brave-search"


def test_extract_server_name_fallback() -> None:
    """Test server name extraction falls back to first server."""
    servers: list[MCPServerConfig] = [
        {"name": "default-server", "command": "cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock tool without matching prefix
    mock_tool = MagicMock()
    mock_tool.name = "some_tool"

    result = middleware._extract_server_name(mock_tool)
    assert result == "default-server"


def test_extract_tool_name_removes_prefix() -> None:
    """Test tool name extraction removes server prefix."""
    servers: list[MCPServerConfig] = [
        {"name": "brave-search", "command": "cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock tool
    mock_tool = MagicMock()
    mock_tool.name = "brave-search_search"

    result = middleware._extract_tool_name(mock_tool, "brave-search")
    assert result == "search"


def test_extract_tool_name_no_prefix() -> None:
    """Test tool name extraction when no prefix present."""
    servers: list[MCPServerConfig] = [
        {"name": "brave-search", "command": "cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock tool without prefix
    mock_tool = MagicMock()
    mock_tool.name = "search"

    result = middleware._extract_tool_name(mock_tool, "brave-search")
    assert result == "search"


# =============================================================================
# Test Schema Extraction
# =============================================================================


def test_get_tool_schema_with_args_schema() -> None:
    """Test schema extraction from tool with args_schema."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock tool with args_schema
    mock_schema = MagicMock()
    mock_schema.schema.return_value = {"type": "object", "properties": {"query": {"type": "string"}}}

    mock_tool = MagicMock()
    mock_tool.args_schema = mock_schema

    result = middleware._get_tool_schema(mock_tool)
    assert result == {"type": "object", "properties": {"query": {"type": "string"}}}


def test_get_tool_schema_fallback() -> None:
    """Test schema extraction fallback when no schema available."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Create mock tool without args_schema
    mock_tool = MagicMock()
    mock_tool.args_schema = None

    result = middleware._get_tool_schema(mock_tool)
    assert result == {"type": "object", "properties": {}}


# =============================================================================
# Test Metadata Sync
# =============================================================================


@pytest.mark.asyncio
async def test_sync_metadata_writes_files(tmp_path: Path) -> None:
    """Test that _sync_metadata writes tool metadata to filesystem."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "cmd"},
    ]

    middleware = MCPMiddleware(
        servers=servers,
        mcp_root=str(tmp_path),
    )

    # Create mock client with tools
    mock_tool = MagicMock()
    mock_tool.name = "test-server_test_tool"
    mock_tool.description = "A test tool"
    mock_tool.args_schema = None

    mock_client = MagicMock()
    mock_client.get_tools.return_value = [mock_tool]

    middleware._mcp_client = mock_client

    # Call sync
    await middleware._sync_metadata()

    # Verify file was created
    metadata_file = tmp_path / "test-server" / "test_tool.json"
    assert metadata_file.exists()

    # Verify content
    import json

    content = json.loads(metadata_file.read_text())
    assert content["server"] == "test-server"
    assert content["name"] == "test_tool"
    assert content["description"] == "A test tool"
    assert content["status"] == "available"


@pytest.mark.asyncio
async def test_sync_metadata_handles_no_client() -> None:
    """Test that _sync_metadata handles missing client gracefully."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # No client set - should not raise
    await middleware._sync_metadata()


@pytest.mark.asyncio
async def test_sync_metadata_handles_errors() -> None:
    """Test that _sync_metadata handles errors gracefully."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    # Mock client that raises
    mock_client = MagicMock()
    mock_client.get_tools.side_effect = Exception("Failed to get tools")

    middleware._mcp_client = mock_client

    # Should not raise
    await middleware._sync_metadata()


# =============================================================================
# Test State Schema
# =============================================================================


def test_state_schema_is_mcp_state() -> None:
    """Test that middleware uses MCPState as state schema."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "command": "cmd"},
    ]

    middleware = MCPMiddleware(servers=servers)

    assert middleware.state_schema is MCPState


# =============================================================================
# Test System Prompt Content
# =============================================================================


def test_system_prompt_contains_discovery_pattern() -> None:
    """Test that MCP system prompt contains discovery instructions."""
    assert "ls /.mcp/" in MCP_SYSTEM_PROMPT
    assert "read_file" in MCP_SYSTEM_PROMPT or "read tool" in MCP_SYSTEM_PROMPT.lower()
    assert "mcp_invoke" in MCP_SYSTEM_PROMPT


def test_system_prompt_contains_example() -> None:
    """Test that MCP system prompt contains example workflow."""
    assert "Example" in MCP_SYSTEM_PROMPT
    assert "brave-search" in MCP_SYSTEM_PROMPT or "search" in MCP_SYSTEM_PROMPT


def test_system_prompt_contains_best_practices() -> None:
    """Test that MCP system prompt contains best practices."""
    assert "Best Practices" in MCP_SYSTEM_PROMPT
    assert "progressive disclosure" in MCP_SYSTEM_PROMPT.lower()


# =============================================================================
# Test Type Definitions
# =============================================================================


def test_mcp_server_config_type() -> None:
    """Test MCPServerConfig TypedDict structure."""
    config: MCPServerConfig = {
        "name": "test-server",
        "command": "test-cmd",
        "args": ["arg1", "arg2"],
        "env": {"KEY": "value"},
    }

    assert config["name"] == "test-server"
    assert config["command"] == "test-cmd"
    assert config["args"] == ["arg1", "arg2"]
    assert config["env"] == {"KEY": "value"}


def test_mcp_tool_metadata_type() -> None:
    """Test MCPToolMetadata TypedDict structure."""
    metadata: MCPToolMetadata = {
        "server": "test-server",
        "name": "test-tool",
        "description": "A test tool",
        "input_schema": {"type": "object"},
        "status": "available",
    }

    assert metadata["server"] == "test-server"
    assert metadata["name"] == "test-tool"
    assert metadata["description"] == "A test tool"
    assert metadata["input_schema"] == {"type": "object"}
    assert metadata["status"] == "available"
