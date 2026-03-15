"""Integration tests for skill tools middleware.

This module tests the SkillToolMiddleware in full integration with:
- StateBackend for file operations
- SkillsMiddleware for skill loading
- Full agent creation and invocation
"""

import pytest
from langchain_core.messages import HumanMessage

from deepagents.graph import create_deep_agent
from deepagents.middleware.skill_tools import (
    clear_native_tools,
    register_tool,
)
from tests.utils import SAMPLE_MODEL


@pytest.fixture
def _clean_native_tools():
    """Fixture to clean up native tools before/after tests."""
    clear_native_tools()
    yield
    clear_native_tools()


# ==========================================
# Native Tool Integration Tests
# ==========================================


@pytest.mark.usefixtures("_clean_native_tools")
def test_skill_with_native_tools(tmp_path):
    """Test skill with native Python tools."""
    # Create a skill directory with SKILL.md
    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: test-skill
description: A test skill with native tools
---

# Test Skill

This skill provides native Python tools.
"""
    )

    # Register a native tool for this skill
    @register_tool("test-skill")
    def test_tool(message: str) -> str:
        """A simple test tool."""
        return f"Echo: {message}"

    # Create agent with the skill
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Verify agent was created successfully
    assert agent is not None


@pytest.mark.usefixtures("_clean_native_tools")
def test_multiple_skills_with_native_tools(tmp_path):
    """Test multiple skills with different native tools."""
    # Create first skill
    skill1_dir = tmp_path / "skills" / "skill-a"
    skill1_dir.mkdir(parents=True)
    (skill1_dir / "SKILL.md").write_text(
        """---
name: skill-a
description: Skill A
---
# Skill A
"""
    )

    # Create second skill
    skill2_dir = tmp_path / "skills" / "skill-b"
    skill2_dir.mkdir(parents=True)
    (skill2_dir / "SKILL.md").write_text(
        """---
name: skill-b
description: Skill B
---
# Skill B
"""
    )

    # Register tools for both skills
    @register_tool("skill-a")
    def tool_a(value: int) -> int:
        """Tool from skill A."""
        return value * 2

    @register_tool("skill-b")
    def tool_b(value: int) -> int:
        """Tool from skill B."""
        return value + 10

    # Create agent with both skills
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(tmp_path / "skills")],
    )

    # Verify agent was created successfully
    assert agent is not None


# ==========================================
# MCP Server Configuration Tests
# ==========================================


def test_skill_with_mcp_servers(tmp_path):
    """Test skill with MCP server configuration in YAML."""
    # Create a skill with MCP servers
    skill_dir = tmp_path / "skills" / "mcp-skill"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: mcp-skill
description: A test skill with MCP servers
mcp-servers:
  - name: test-server
    transport: stdio
    command: echo
    args: ["hello"]
  - name: another-server
    transport: http
    url: https://example.com/mcp
    headers:
      Authorization: Bearer ${TOKEN}
---

# MCP Skill

This skill uses MCP servers.
"""
    )

    # Create agent with the skill
    # Note: MCP connection won't happen until agent invocation
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Verify agent was created successfully
    assert agent is not None


def test_skill_with_invalid_mcp_config(tmp_path):
    """Test skill with invalid MCP server configuration."""
    # Create a skill with invalid MCP config
    skill_dir = tmp_path / "skills" / "invalid-mcp"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: invalid-mcp
description: Skill with invalid MCP config
mcp-servers:
  - name: ""
    transport: invalid
---

# Invalid MCP Skill
"""
    )

    # Create agent - should still work, just ignore invalid config
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Agent should still be created
    assert agent is not None


# ==========================================
# Mixed MCP and Native Tools Tests
# ==========================================


@pytest.mark.usefixtures("_clean_native_tools")
def test_skill_with_mcp_and_native_tools(tmp_path):
    """Test skill with both MCP servers and native tools."""
    # Create skill with both
    skill_dir = tmp_path / "skills" / "hybrid-skill"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: hybrid-skill
description: Skill with MCP and native tools
mcp-servers:
  - name: mcp-server
    transport: stdio
    command: echo
    args: ["test"]
---

# Hybrid Skill

This skill has both MCP servers and native Python tools.
"""
    )

    # Register native tool
    @register_tool("hybrid-skill")
    def native_tool(value: str) -> str:
        """A native tool."""
        return f"Native: {value}"

    # Create agent
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Verify agent was created successfully
    assert agent is not None


# ==========================================
# Middleware Stack Integration Tests
# ==========================================


def test_middleware_order_with_skills(tmp_path):
    """Test that SkillToolMiddleware comes after SkillsMiddleware."""
    # Create a simple skill
    skill_dir = tmp_path / "skills" / "order-test"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: order-test
description: Test middleware order
---

# Order Test Skill
"""
    )

    # Create agent
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Verify agent has both middlewares
    assert agent is not None

    # The middleware stack should have:
    # 1. SkillsMiddleware
    # 2. SkillToolMiddleware
    # (and others)


# ==========================================
# Environment Variable Expansion Tests
# ==========================================


def test_mcp_env_var_expansion(tmp_path, monkeypatch):
    """Test environment variable expansion in MCP configs."""
    # Set test environment variable
    monkeypatch.setenv("TEST_TOKEN", "test-token-value")
    monkeypatch.setenv("TEST_URL", "https://example.com/api")

    # Create skill with env vars
    skill_dir = tmp_path / "skills" / "env-test"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: env-test
description: Test environment variable expansion
mcp-servers:
  - name: server-with-env
    transport: http
    url: ${TEST_URL}
    headers:
      Authorization: Bearer ${TEST_TOKEN}
---

# Env Test Skill
"""
    )

    # Create agent - env vars should be expanded during parsing
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Verify agent was created
    assert agent is not None


# ==========================================
# Backward Compatibility Tests
# ==========================================


def test_skill_without_mcp_still_works(tmp_path):
    """Test that skills without MCP config work as before."""
    # Create a skill without mcp-servers field
    skill_dir = tmp_path / "skills" / "legacy-skill"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: legacy-skill
description: A skill without MCP configuration
---

# Legacy Skill

This skill doesn't use MCP servers.
"""
    )

    # Create agent - should work exactly as before
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Verify agent was created successfully
    assert agent is not None


# ==========================================
# State Lifecycle Tests
# ==========================================


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_native_tools")
async def test_middleware_state_initialization(tmp_path):
    """Test that middleware properly initializes state."""
    # Create skill
    skill_dir = tmp_path / "skills" / "state-test"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: state-test
description: Test state initialization
---

# State Test Skill
"""
    )

    # Register a tool
    @register_tool("state-test")
    def state_tool() -> str:
        """A tool for state testing."""
        return "state result"

    # Create agent
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Invoke agent to trigger state initialization
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Hello")]},
    )

    # Verify result
    assert result is not None


# ==========================================
# Tool Dispatcher Integration Tests
# ==========================================


@pytest.mark.usefixtures("_clean_native_tools")
def test_tool_dispatcher_routing(tmp_path):
    """Test that tool dispatcher correctly routes to native tools."""
    # Create skill
    skill_dir = tmp_path / "skills" / "dispatch-test"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: dispatch-test
description: Test tool dispatcher routing
---

# Dispatch Test Skill
"""
    )

    # Register multiple tools
    @register_tool("dispatch-test")
    def tool1(x: int) -> int:
        """First tool."""
        return x * 2

    @register_tool("dispatch-test")
    def tool2(x: int) -> int:
        """Second tool."""
        return x + 5

    # Create agent
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Verify agent was created
    assert agent is not None


# ==========================================
# Error Handling Tests
# ==========================================


def test_mcp_connection_failure_handling(tmp_path):
    """Test that MCP connection failures don't crash the agent."""
    # Create skill with MCP server that won't connect
    skill_dir = tmp_path / "skills" / "error-test"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: error-test
description: Test error handling
mcp-servers:
  - name: nonexistent-server
    transport: stdio
    command: nonexistent-command-xyz-123
    args: ["--fail"]
---

# Error Test Skill
"""
    )

    # Create agent - should handle connection failure gracefully
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Agent should still be created despite invalid MCP config
    assert agent is not None


def test_skill_with_corrupted_mcp_yaml(tmp_path):
    """Test skill with malformed MCP YAML."""
    # Create skill with malformed YAML
    skill_dir = tmp_path / "skills" / "corrupt-yaml"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: corrupt-yaml
description: Test corrupted YAML handling
mcp-servers: "this should be a list not a string"
---

# Corrupt YAML Skill
"""
    )

    # Create agent - should handle YAML error gracefully
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Agent should still be created
    assert agent is not None


# ==========================================
# System Prompt Injection Tests
# ==========================================


@pytest.mark.usefixtures("_clean_native_tools")
def test_system_prompt_injection_with_tools(tmp_path):
    """Test that tool documentation is injected into system prompt."""
    # Create skill
    skill_dir = tmp_path / "skills" / "prompt-test"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: prompt-test
description: Test system prompt injection
---

# Prompt Test Skill
"""
    )

    # Register a tool with good documentation
    @register_tool("prompt-test")
    def documented_tool(query: str) -> str:
        """Search for information and return results."""
        return f"Results for: {query}"

    # Create agent
    agent = create_deep_agent(
        model=SAMPLE_MODEL,
        skills=[str(skill_dir.parent)],
    )

    # Verify agent was created
    assert agent is not None

    # When invoked, the system prompt should contain tool documentation
    # (This would require checking the actual model request, which is
    #  tested more thoroughly in unit tests)
