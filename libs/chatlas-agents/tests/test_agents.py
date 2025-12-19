"""Tests for ChATLAS agents."""

import pytest
from chatlas_agents.config import AgentConfig, LLMConfig, LLMProvider, MCPServerConfig
from chatlas_agents.agents import DeepAgentWrapper


def test_agent_initialization():
    """Test basic agent initialization."""
    config = AgentConfig(
        name="test-agent",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI, 
            model="gpt-4",
            api_key="test-key"
        ),
        mcp=MCPServerConfig(url="https://chatlas-mcp.app.cern.ch/mcp"),
    )
    agent = DeepAgentWrapper(config)
    assert agent.config.name == "test-agent"
    assert agent.mcp_client is not None


@pytest.mark.asyncio
async def test_agent_cleanup():
    """Test agent cleanup."""
    config = AgentConfig(
        name="test-agent",
        llm=LLMConfig(api_key="test-key")
    )
    agent = DeepAgentWrapper(config)
    await agent.close()
    # Should complete without errors


# Note: Full integration tests would require:
# - Access to actual MCP server
# - Valid API keys for LLM providers
# - Mock implementations for testing
