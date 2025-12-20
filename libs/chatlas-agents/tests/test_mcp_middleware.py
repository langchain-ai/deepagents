"""Tests for MCPMiddleware."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chatlas_agents.config import MCPServerConfig
from chatlas_agents.middleware import MCPMiddleware

# Constant for the patch path used throughout tests
MCP_LOAD_TOOLS_PATCH = 'chatlas_agents.middleware.mcp.create_mcp_client_and_load_tools'


@pytest.fixture
def mcp_config():
    """Create a test MCP configuration."""
    return MCPServerConfig(
        url="http://test-mcp-server/mcp",
        timeout=10,
    )


@pytest.fixture
def mock_tool():
    """Create a mock LangChain tool."""
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "A test tool for unit testing"
    return tool


@pytest.mark.asyncio
async def test_mcp_middleware_create(mcp_config, mock_tool):
    """Test MCPMiddleware.create() factory method."""
    # Mock the create_mcp_client_and_load_tools function
    with patch(MCP_LOAD_TOOLS_PATCH) as mock_load:
        mock_load.return_value = [mock_tool]
        
        # Create middleware
        middleware = await MCPMiddleware.create(mcp_config)
        
        # Verify
        assert middleware is not None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "test_tool"
        assert middleware.config == mcp_config
        mock_load.assert_called_once_with(mcp_config)


@pytest.mark.asyncio
async def test_mcp_middleware_create_connection_error(mcp_config):
    """Test MCPMiddleware.create() handles connection errors gracefully."""
    # Mock the create_mcp_client_and_load_tools to raise an error
    with patch(MCP_LOAD_TOOLS_PATCH) as mock_load:
        mock_load.side_effect = ConnectionError("Connection refused")
        
        # Verify that a ConnectionError is raised with helpful message
        with pytest.raises(ConnectionError) as exc_info:
            await MCPMiddleware.create(mcp_config)
        
        # Check error message contains helpful diagnostics
        error_msg = str(exc_info.value)
        assert "Failed to connect to MCP server" in error_msg
        assert mcp_config.url in error_msg
        assert "verify" in error_msg.lower()


@pytest.mark.asyncio
async def test_mcp_middleware_load_tools(mcp_config, mock_tool):
    """Test MCPMiddleware.load_tools() method."""
    # Create middleware without pre-loaded tools
    middleware = MCPMiddleware(config=mcp_config, tools=None)
    assert len(middleware.tools) == 0
    
    # Mock the load function
    with patch(MCP_LOAD_TOOLS_PATCH) as mock_load:
        mock_load.return_value = [mock_tool]
        
        # Load tools
        await middleware.load_tools()
        
        # Verify
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "test_tool"
        mock_load.assert_called_once_with(mcp_config)


def test_mcp_middleware_format_tools_description(mcp_config, mock_tool):
    """Test formatting of tools description for system prompt."""
    middleware = MCPMiddleware(config=mcp_config, tools=[mock_tool])
    
    description = middleware._format_tools_description()
    
    assert "test_tool" in description
    assert "A test tool for unit testing" in description
    assert "**Available MCP Tools:**" in description


def test_mcp_middleware_custom_system_prompt(mcp_config, mock_tool):
    """Test custom system prompt template."""
    custom_template = "Custom MCP Section:\n{tools_description}"
    middleware = MCPMiddleware(
        config=mcp_config, 
        tools=[mock_tool],
        system_prompt_template=custom_template
    )
    
    assert middleware.system_prompt_template == custom_template


def test_mcp_middleware_format_tools_description_empty(mcp_config):
    """Test formatting when no tools are available."""
    middleware = MCPMiddleware(config=mcp_config, tools=[])
    
    description = middleware._format_tools_description()
    
    assert "No MCP tools are currently available" in description


def test_mcp_middleware_before_agent(mcp_config, mock_tool):
    """Test before_agent hook stores tools in state."""
    middleware = MCPMiddleware(config=mcp_config, tools=[mock_tool])
    
    # Mock state and runtime
    state = {}
    runtime = MagicMock()
    
    # Call before_agent
    update = middleware.before_agent(state, runtime)
    
    # Verify state update
    assert update is not None
    assert "mcp_tools" in update
    assert len(update["mcp_tools"]) == 1
    assert update["mcp_tools"][0].name == "test_tool"


def test_mcp_middleware_wrap_model_call_with_injection(mcp_config, mock_tool):
    """Test wrap_model_call injects MCP tools into system prompt."""
    middleware = MCPMiddleware(config=mcp_config, tools=[mock_tool], inject_prompt=True)
    
    # Mock request and handler
    mock_request = MagicMock()
    mock_request.system_prompt = "Base system prompt"
    mock_request.override = MagicMock(return_value=mock_request)
    
    mock_handler = MagicMock(return_value="response")
    
    # Call wrap_model_call
    result = middleware.wrap_model_call(mock_request, mock_handler)
    
    # Verify system prompt was modified
    mock_request.override.assert_called_once()
    call_args = mock_request.override.call_args
    modified_prompt = call_args[1]["system_prompt"]
    
    assert "Base system prompt" in modified_prompt
    assert "MCP Server Tools" in modified_prompt
    assert "test_tool" in modified_prompt


def test_mcp_middleware_wrap_model_call_without_injection(mcp_config, mock_tool):
    """Test wrap_model_call when prompt injection is disabled."""
    middleware = MCPMiddleware(config=mcp_config, tools=[mock_tool], inject_prompt=False)
    
    # Mock request and handler
    mock_request = MagicMock()
    mock_request.system_prompt = "Base system prompt"
    
    mock_handler = MagicMock(return_value="response")
    
    # Call wrap_model_call
    result = middleware.wrap_model_call(mock_request, mock_handler)
    
    # Verify handler called with original request
    mock_handler.assert_called_once_with(mock_request)
    # Verify request.override was NOT called
    mock_request.override.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_middleware_awrap_model_call_with_injection(mcp_config, mock_tool):
    """Test awrap_model_call injects MCP tools into system prompt."""
    middleware = MCPMiddleware(config=mcp_config, tools=[mock_tool], inject_prompt=True)
    
    # Mock request and handler
    mock_request = MagicMock()
    mock_request.system_prompt = "Base system prompt"
    mock_request.override = MagicMock(return_value=mock_request)
    
    mock_handler = AsyncMock(return_value="response")
    
    # Call awrap_model_call
    result = await middleware.awrap_model_call(mock_request, mock_handler)
    
    # Verify system prompt was modified
    mock_request.override.assert_called_once()
    call_args = mock_request.override.call_args
    modified_prompt = call_args[1]["system_prompt"]
    
    assert "Base system prompt" in modified_prompt
    assert "MCP Server Tools" in modified_prompt
    assert "test_tool" in modified_prompt


@pytest.mark.asyncio
async def test_mcp_middleware_integration_scenario(mcp_config):
    """Test a complete integration scenario."""
    # Mock tools
    tool1 = MagicMock()
    tool1.name = "search_chatlas"
    tool1.description = "Search ChATLAS documentation"
    
    tool2 = MagicMock()
    tool2.name = "query_database"
    tool2.description = "Query the ATLAS database"
    
    # Create middleware with mock
    with patch(MCP_LOAD_TOOLS_PATCH) as mock_load:
        mock_load.return_value = [tool1, tool2]
        
        middleware = await MCPMiddleware.create(mcp_config)
        
        # Verify tools loaded
        assert len(middleware.tools) == 2
        
        # Verify before_agent populates state
        state = {}
        runtime = MagicMock()
        update = middleware.before_agent(state, runtime)
        assert len(update["mcp_tools"]) == 2
        
        # Verify system prompt contains both tools
        description = middleware._format_tools_description()
        assert "search_chatlas" in description
        assert "query_database" in description
        assert "Search ChATLAS documentation" in description
        assert "Query the ATLAS database" in description
