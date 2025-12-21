"""Tests for ChATLAS agents configuration."""

import pytest
from chatlas_agents.config import (
    AgentConfig,
    LLMConfig,
    LLMProvider,
    MCPServerConfig,
    SubAgentConfig,
    Settings,
)


def test_llm_config_default():
    """Test default LLM configuration."""
    config = LLMConfig()
    assert config.provider == LLMProvider.OPENAI
    assert config.model == "gpt-5-mini"
    assert config.temperature == 0.7
    assert config.streaming is False


def test_llm_config_custom():
    """Test custom LLM configuration."""
    config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        temperature=0.5,
        max_tokens=1000,
    )
    assert config.provider == LLMProvider.ANTHROPIC
    assert config.model == "claude-3-5-sonnet-20241022"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000


def test_mcp_config_default():
    """Test default MCP server configuration."""
    config = MCPServerConfig()
    assert config.url == "https://chatlas-mcp.app.cern.ch/mcp"
    assert config.timeout == 120  # Updated default timeout
    assert config.max_retries == 3


def test_agent_config_default():
    """Test default agent configuration."""
    config = AgentConfig()
    assert config.name == "chatlas-agent"
    assert config.llm.provider == LLMProvider.OPENAI
    assert config.mcp.url == "https://chatlas-mcp.app.cern.ch/mcp"
    assert config.max_iterations == 10
    assert config.verbose is False


def test_agent_config_with_sub_agents():
    """Test agent configuration with sub-agents."""
    config = AgentConfig(
        name="test-agent",
        sub_agents=[
            SubAgentConfig(name="sub1", description="Sub-agent 1"),
            SubAgentConfig(name="sub2", description="Sub-agent 2", enabled=False),
        ],
    )
    assert len(config.sub_agents) == 2
    assert config.sub_agents[0].name == "sub1"
    assert config.sub_agents[0].enabled is True
    assert config.sub_agents[1].enabled is False


def test_settings_to_agent_config():
    """Test converting settings to agent configuration."""
    settings = Settings(
        llm_provider="anthropic",
        llm_model="claude-3-5-sonnet-20241022",
        agent_name="test-agent",
    )
    config = settings.to_agent_config()
    assert config.name == "test-agent"
    assert config.llm.provider == LLMProvider.ANTHROPIC
    assert config.llm.model == "claude-3-5-sonnet-20241022"
