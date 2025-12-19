"""Tests for MCP client."""

import pytest
from chatlas_agents.config import MCPServerConfig
from chatlas_agents.mcp import create_mcp_client, create_mcp_client_and_load_tools
from langchain_mcp_adapters.client import MultiServerMCPClient


def test_create_mcp_client():
    """Test creating MCP client."""
    config = MCPServerConfig(
        url="https://chatlas-mcp.app.cern.ch/mcp",
        timeout=30,
    )
    client = create_mcp_client(config)
    assert isinstance(client, MultiServerMCPClient)
    assert client is not None


def test_mcp_config_properties():
    """Test MCP config properties are correctly passed."""
    config = MCPServerConfig(
        url="https://chatlas-mcp.app.cern.ch/mcp",
        timeout=30,
        max_retries=3,
    )
    assert config.url == "https://chatlas-mcp.app.cern.ch/mcp"
    assert config.timeout == 30
    assert config.max_retries == 3


# Note: Integration tests with actual MCP server would require
# the server to be running and accessible. The create_mcp_client_and_load_tools
# function creates sessions on-demand when tools are invoked.
