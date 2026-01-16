"""Unit tests for MCP middleware."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol, WriteResult
from deepagents.middleware.mcp import (
    MCP_SYSTEM_PROMPT,
    MCPMiddleware,
    MCPServerConfig,
    MCPState,
    MCPToolMetadata,
)


def test_validate_servers_valid() -> None:
    """Test that valid HTTP server configurations are accepted."""
    servers: list[MCPServerConfig] = [
        {"name": "math-server", "url": "http://localhost:3000/mcp"},
        {"name": "weather-server", "url": "http://localhost:3001/mcp", "headers": {"Authorization": "Bearer test-token"}},
    ]
    middleware = MCPMiddleware(servers=servers)
    assert middleware.servers == servers


def test_validate_servers_empty() -> None:
    """Test that empty server list raises ValueError."""
    with pytest.raises(ValueError, match="At least one MCP server"):
        MCPMiddleware(servers=[])


def test_validate_servers_missing_name() -> None:
    """Test that missing 'name' field raises ValueError."""
    with pytest.raises(ValueError, match="requires 'name'"):
        MCPMiddleware(servers=[{"url": "http://localhost:3000/mcp"}])  # type: ignore[list-item]


def test_validate_servers_missing_url() -> None:
    """Test that missing 'url' field raises ValueError."""
    with pytest.raises(ValueError, match="requires 'url'"):
        MCPMiddleware(servers=[{"name": "test-server"}])  # type: ignore[list-item]


def test_validate_servers_duplicate_names() -> None:
    """Test that duplicate server names raise ValueError."""
    servers: list[MCPServerConfig] = [
        {"name": "test-server", "url": "http://localhost:3000/mcp"},
        {"name": "test-server", "url": "http://localhost:3001/mcp"},
    ]
    with pytest.raises(ValueError, match="Duplicate server name"):
        MCPMiddleware(servers=servers)


def test_middleware_initialization() -> None:
    """Test middleware initialization with valid configuration."""
    servers: list[MCPServerConfig] = [{"name": "test-server", "url": "http://localhost:3000/mcp"}]
    middleware = MCPMiddleware(servers=servers, mcp_prefix="/custom/.mcp", sync_on_startup=False)

    assert middleware.servers == servers
    assert middleware.mcp_prefix == "/custom/.mcp"
    assert middleware.sync_on_startup is False
    assert middleware._mcp_client is None
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "mcp_invoke"


def test_middleware_default_values() -> None:
    """Test middleware uses correct default values."""
    middleware = MCPMiddleware(servers=[{"name": "test-server", "url": "http://localhost:3000/mcp"}])
    assert middleware.mcp_prefix == "/.mcp"
    assert middleware.sync_on_startup is True


def test_middleware_with_headers() -> None:
    """Test middleware initialization with authentication headers."""
    servers: list[MCPServerConfig] = [
        {
            "name": "authenticated-server",
            "url": "https://api.example.com/mcp",
            "headers": {"Authorization": "Bearer sk-xxx", "X-Custom-Header": "custom-value"},
        },
    ]
    middleware = MCPMiddleware(servers=servers)
    assert middleware.servers[0]["headers"] == {"Authorization": "Bearer sk-xxx", "X-Custom-Header": "custom-value"}


def test_mcp_invoke_tool_created() -> None:
    """Test that mcp_invoke tool is created during initialization."""
    middleware = MCPMiddleware(servers=[{"name": "test-server", "url": "http://localhost:3000/mcp"}])

    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.name == "mcp_invoke"
    assert "MCP tool" in tool.description
    assert "ls /.mcp/" in tool.description


def test_extract_server_name_with_prefix() -> None:
    """Test server name extraction from prefixed tool name."""
    servers: list[MCPServerConfig] = [
        {"name": "brave-search", "url": "http://localhost:3000/mcp"},
        {"name": "filesystem", "url": "http://localhost:3001/mcp"},
    ]
    middleware = MCPMiddleware(servers=servers)

    mock_tool = MagicMock()
    mock_tool.name = "brave-search_search"
    assert middleware._extract_server_name(mock_tool) == "brave-search"


def test_extract_server_name_fallback() -> None:
    """Test server name extraction falls back to first server."""
    middleware = MCPMiddleware(servers=[{"name": "default-server", "url": "http://localhost:3000/mcp"}])

    mock_tool = MagicMock()
    mock_tool.name = "some_tool"
    assert middleware._extract_server_name(mock_tool) == "default-server"


def test_extract_tool_name_removes_prefix() -> None:
    """Test tool name extraction removes server prefix."""
    middleware = MCPMiddleware(servers=[{"name": "brave-search", "url": "http://localhost:3000/mcp"}])

    mock_tool = MagicMock()
    mock_tool.name = "brave-search_search"
    assert middleware._extract_tool_name(mock_tool, "brave-search") == "search"


def test_extract_tool_name_no_prefix() -> None:
    """Test tool name extraction when no prefix present."""
    middleware = MCPMiddleware(servers=[{"name": "brave-search", "url": "http://localhost:3000/mcp"}])

    mock_tool = MagicMock()
    mock_tool.name = "search"
    assert middleware._extract_tool_name(mock_tool, "brave-search") == "search"


def test_get_tool_schema_with_args_schema() -> None:
    """Test schema extraction from tool with args_schema."""
    middleware = MCPMiddleware(servers=[{"name": "test-server", "url": "http://localhost:3000/mcp"}])

    mock_schema = MagicMock()
    mock_schema.schema.return_value = {"type": "object", "properties": {"query": {"type": "string"}}}

    mock_tool = MagicMock()
    mock_tool.args_schema = mock_schema

    assert middleware._get_tool_schema(mock_tool) == {"type": "object", "properties": {"query": {"type": "string"}}}


def test_get_tool_schema_fallback() -> None:
    """Test schema extraction fallback when no schema available."""
    middleware = MCPMiddleware(servers=[{"name": "test-server", "url": "http://localhost:3000/mcp"}])

    mock_tool = MagicMock()
    mock_tool.args_schema = None
    mock_tool.get_input_schema.side_effect = Exception("No schema")

    assert middleware._get_tool_schema(mock_tool) == {"type": "object", "properties": {}}


def test_system_prompt_contains_discovery_pattern() -> None:
    """Test that MCP system prompt contains discovery instructions."""
    assert "ls /.mcp/" in MCP_SYSTEM_PROMPT
    assert "mcp_invoke" in MCP_SYSTEM_PROMPT


def test_system_prompt_contains_example() -> None:
    """Test that MCP system prompt contains example workflow."""
    assert "Example" in MCP_SYSTEM_PROMPT


def test_system_prompt_contains_best_practices() -> None:
    """Test that MCP system prompt contains best practices."""
    assert "Best Practices" in MCP_SYSTEM_PROMPT
    assert "progressive disclosure" in MCP_SYSTEM_PROMPT.lower()


def test_mcp_server_config_type() -> None:
    """Test MCPServerConfig TypedDict structure."""
    config: MCPServerConfig = {
        "name": "test-server",
        "url": "http://localhost:3000/mcp",
        "headers": {"Authorization": "Bearer token"},
    }
    assert config["name"] == "test-server"
    assert config["url"] == "http://localhost:3000/mcp"
    assert config["headers"] == {"Authorization": "Bearer token"}


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


def test_mcp_state_has_files_key() -> None:
    """Test that MCPState has 'files' key from FilesystemState inheritance."""
    assert "files" in MCPState.__annotations__
    assert "mcp_initialized" in MCPState.__annotations__


def test_state_schema_is_mcp_state() -> None:
    """Test that middleware uses MCPState as state schema."""
    middleware = MCPMiddleware(servers=[{"name": "test-server", "url": "http://localhost:3000/mcp"}])
    assert middleware.state_schema is MCPState


def test_before_agent_sets_initialized() -> None:
    """Test sync before_agent hook."""
    middleware = MCPMiddleware(servers=[{"name": "test", "url": "http://localhost:3000/mcp"}])
    result = middleware.before_agent({}, None, {})  # type: ignore[arg-type]
    assert result == {"mcp_initialized": True}


def test_before_agent_skips_if_initialized() -> None:
    """Test that before_agent returns None if already initialized."""
    middleware = MCPMiddleware(servers=[{"name": "test", "url": "http://localhost:3000/mcp"}])
    result = middleware.before_agent({"mcp_initialized": True}, None, {})  # type: ignore[arg-type]
    assert result is None


@pytest.mark.asyncio
async def test_sync_metadata_handles_no_client() -> None:
    """Test that _sync_metadata handles missing client gracefully."""
    middleware = MCPMiddleware(servers=[{"name": "test", "url": "http://localhost:9999/mcp"}])
    # Create a mock backend - won't be used since client is not connected
    mock_backend = MagicMock(spec=BackendProtocol)
    files_update = await middleware._sync_metadata(mock_backend)
    assert files_update == {}


@pytest.mark.asyncio
async def test_invoke_without_connection() -> None:
    """Test invoking tool without connection returns error."""
    middleware = MCPMiddleware(servers=[{"name": "test", "url": "http://localhost:9999/mcp"}])
    tool = middleware.tools[0]
    result = await tool.coroutine("test_tool", {}, MagicMock())
    assert "Error" in result
    assert "not connected" in result.lower()


@pytest.mark.asyncio
async def test_connect_requires_langchain_mcp_adapters() -> None:
    """Test that connect raises ImportError if dependency missing."""
    middleware = MCPMiddleware(servers=[{"name": "test", "url": "http://localhost:3000/mcp"}])

    with patch.dict("sys.modules", {"langchain_mcp_adapters": None, "langchain_mcp_adapters.client": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="langchain-mcp-adapters"):
                await middleware.connect()


@pytest.mark.asyncio
async def test_abefore_agent_skips_if_initialized() -> None:
    """Test that abefore_agent returns None if already initialized."""
    middleware = MCPMiddleware(servers=[{"name": "test", "url": "http://localhost:3000/mcp"}])
    result = await middleware.abefore_agent({"mcp_initialized": True}, None, {})  # type: ignore[arg-type]
    assert result is None


# Tests that require real MCP servers use fixtures from conftest.py


@pytest.mark.asyncio
async def test_connect_to_server(math_server: dict[str, Any]) -> None:
    """Test connecting to a real MCP server."""
    middleware = MCPMiddleware(servers=[{"name": "math", "url": math_server["url"]}])
    await middleware.connect()
    assert middleware._mcp_client is not None
    await middleware.close()


@pytest.mark.asyncio
async def test_context_manager(math_server: dict[str, Any]) -> None:
    """Test async context manager with real server."""
    middleware = MCPMiddleware(servers=[{"name": "math", "url": math_server["url"]}])
    async with middleware:
        assert middleware._mcp_client is not None
    assert middleware._mcp_client is None


@pytest.mark.asyncio
async def test_close_cleans_up_client(math_server: dict[str, Any]) -> None:
    """Test that close() properly cleans up."""
    middleware = MCPMiddleware(servers=[{"name": "math", "url": math_server["url"]}])
    await middleware.connect()
    await middleware.close()
    assert middleware._mcp_client is None


@pytest.mark.asyncio
async def test_sync_metadata_returns_files_update(math_server: dict[str, Any]) -> None:
    """Test that _sync_metadata returns correct file structure when using StateBackend."""
    middleware = MCPMiddleware(servers=[{"name": "math", "url": math_server["url"]}], mcp_prefix="/.mcp")
    await middleware.connect()

    # Create a mock backend that simulates StateBackend behavior
    written_files: dict[str, Any] = {}

    async def mock_awrite(file_path: str, content: str) -> WriteResult:
        from deepagents.backends.utils import create_file_data

        file_data = create_file_data(content)
        written_files[file_path] = file_data
        return WriteResult(path=file_path, files_update={file_path: file_data})

    mock_backend = MagicMock(spec=BackendProtocol)
    mock_backend.awrite = mock_awrite

    files_update = await middleware._sync_metadata(mock_backend)

    assert "/.mcp/math/add.json" in files_update
    assert "/.mcp/math/multiply.json" in files_update
    assert "/.mcp/math/divide.json" in files_update

    add_file = files_update["/.mcp/math/add.json"]
    assert "content" in add_file
    assert "created_at" in add_file

    await middleware.close()


@pytest.mark.asyncio
async def test_abefore_agent_connects_and_syncs(math_server: dict[str, Any]) -> None:
    """Test that abefore_agent auto-connects and syncs metadata."""
    middleware = MCPMiddleware(servers=[{"name": "math", "url": math_server["url"]}], sync_on_startup=True)

    # Create a mock runtime with required attributes for _get_backend
    mock_runtime = MagicMock()
    mock_runtime.context = {}
    mock_runtime.stream_writer = None
    mock_runtime.store = None

    result = await middleware.abefore_agent({}, mock_runtime, {})  # type: ignore[arg-type]

    assert result is not None
    assert result["mcp_initialized"] is True
    assert "files" in result
    assert "/.mcp/math/add.json" in result["files"]


@pytest.mark.asyncio
async def test_invoke_add_tool(math_server: dict[str, Any]) -> None:
    """Test invoking add tool on real server."""
    middleware = MCPMiddleware(servers=[{"name": "math", "url": math_server["url"]}])
    await middleware.connect()

    tool = middleware.tools[0]
    result = await tool.coroutine("add", {"a": 5, "b": 3}, MagicMock())

    assert "8" in str(result)
    await middleware.close()


@pytest.mark.asyncio
async def test_invoke_multiply_tool(math_server: dict[str, Any]) -> None:
    """Test invoking multiply tool on real server."""
    middleware = MCPMiddleware(servers=[{"name": "math", "url": math_server["url"]}])
    await middleware.connect()

    tool = middleware.tools[0]
    result = await tool.coroutine("multiply", {"a": 4, "b": 7}, MagicMock())

    assert "28" in str(result)
    await middleware.close()


@pytest.mark.asyncio
async def test_invoke_tool_not_found(math_server: dict[str, Any]) -> None:
    """Test invoking non-existent tool."""
    middleware = MCPMiddleware(servers=[{"name": "math", "url": math_server["url"]}])
    await middleware.connect()

    tool = middleware.tools[0]
    result = await tool.coroutine("nonexistent", {}, MagicMock())

    assert "Error" in result
    assert "not found" in result.lower()
    await middleware.close()


@pytest.mark.asyncio
async def test_connect_multiple_servers(math_server: dict[str, Any], weather_server: dict[str, Any]) -> None:
    """Test connecting to multiple servers simultaneously."""
    middleware = MCPMiddleware(
        servers=[
            {"name": "math", "url": math_server["url"]},
            {"name": "weather", "url": weather_server["url"]},
        ]
    )
    await middleware.connect()

    # Create a mock backend that simulates StateBackend behavior
    async def mock_awrite(file_path: str, content: str) -> WriteResult:
        from deepagents.backends.utils import create_file_data

        file_data = create_file_data(content)
        return WriteResult(path=file_path, files_update={file_path: file_data})

    mock_backend = MagicMock(spec=BackendProtocol)
    mock_backend.awrite = mock_awrite

    files_update = await middleware._sync_metadata(mock_backend)

    assert any("math" in path for path in files_update)
    assert any("weather" in path for path in files_update)

    await middleware.close()
