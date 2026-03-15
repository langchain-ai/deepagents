"""Unit tests for skill tools middleware.

This module tests the SkillToolMiddleware and related functions including
native tool registration, MCP server configuration parsing, and the
tool dispatcher.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents.backends.state import StateBackend
from deepagents.middleware.skill_tools import (
    MCPServerConfig,
    SkillToolMiddleware,
    _expand_env_vars,
    _expand_env_vars_recursive,
    clear_native_tools,
    get_native_tools,
    register_tool,
)

# ==========================================
# Native Tool Registry Tests
# ==========================================


def test_register_tool() -> None:
    """Test registering a native tool for a skill."""

    @register_tool("test-skill")
    def my_tool(arg: str) -> str:
        """Test tool."""
        return f"Got: {arg}"

    # Verify tool is registered
    tools = get_native_tools("test-skill")
    assert "my_tool" in tools
    assert tools["my_tool"]("hello") == "Got: hello"


def test_register_multiple_tools() -> None:
    """Test registering multiple tools for the same skill."""

    @register_tool("multi-tool-skill")
    def tool1(x: int) -> int:
        """First tool."""
        return x * 2

    @register_tool("multi-tool-skill")
    def tool2(x: int) -> int:
        """Second tool."""
        return x + 10

    tools = get_native_tools("multi-tool-skill")
    assert len(tools) == 2
    assert tools["tool1"](5) == 10
    assert tools["tool2"](5) == 15


def test_register_different_skills() -> None:
    """Test registering tools for different skills."""

    @register_tool("skill-a")
    def tool_a() -> str:
        """Tool A."""
        return "a"

    @register_tool("skill-b")
    def tool_b() -> str:
        """Tool B."""
        return "b"

    assert get_native_tools("skill-a")["tool_a"]() == "a"
    assert get_native_tools("skill-b")["tool_b"]() == "b"


def test_get_native_tools_empty() -> None:
    """Test getting tools for non-existent skill."""
    tools = get_native_tools("non-existent-skill")
    assert tools == {}


def test_clear_native_tools_specific_skill() -> None:
    """Test clearing tools for a specific skill."""

    @register_tool("clear-test-skill")
    def tool1() -> str:
        """Tool 1."""
        return "1"

    @register_tool("other-skill")
    def tool2() -> str:
        """Tool 2."""
        return "2"

    # Clear only the first skill
    clear_native_tools("clear-test-skill")

    # First skill should be empty
    assert get_native_tools("clear-test-skill") == {}

    # Second skill should still have tools
    assert "tool2" in get_native_tools("other-skill")


def test_clear_all_native_tools() -> None:
    """Test clearing all native tools."""

    @register_tool("skill-1")
    def tool1() -> str:
        """Tool 1."""
        return "1"

    @register_tool("skill-2")
    def tool2() -> str:
        """Tool 2."""
        return "2"

    # Clear all
    clear_native_tools()

    # Both should be empty
    assert get_native_tools("skill-1") == {}
    assert get_native_tools("skill-2") == {}


# ==========================================
# Environment Variable Expansion Tests
# ==========================================


def test_expand_env_vars_simple() -> None:
    """Test expanding a simple environment variable."""
    os.environ["TEST_VAR"] = "test_value"

    result = _expand_env_vars("${TEST_VAR}")
    assert result == "test_value"


def test_expand_env_vars_missing() -> None:
    """Test expanding a missing environment variable."""
    result = _expand_env_vars("${NONEXISTENT_VAR}")
    assert result == "${NONEXISTENT_VAR}"  # No change if not set


def test_expand_env_vars_mixed() -> None:
    """Test expanding environment variables in a mixed string."""
    os.environ["PREFIX"] = "pre"
    os.environ["SUFFIX"] = "post"

    result = _expand_env_vars("${PREFIX}-middle-${SUFFIX}")
    assert result == "pre-middle-post"


def test_expand_env_vars_recursive_dict() -> None:
    """Test recursively expanding environment variables in a dict."""
    os.environ["VAR1"] = "value1"
    os.environ["VAR2"] = "value2"

    input_dict = {
        "key1": "${VAR1}",
        "nested": {
            "key2": "${VAR2}",
            "key3": "static",
        },
    }

    result = _expand_env_vars_recursive(input_dict)

    assert result["key1"] == "value1"
    assert result["nested"]["key2"] == "value2"
    assert result["nested"]["key3"] == "static"


def test_expand_env_vars_recursive_list() -> None:
    """Test recursively expanding environment variables in a list."""
    os.environ["ITEM1"] = "first"
    os.environ["ITEM2"] = "second"

    input_list = ["${ITEM1}", "${ITEM2}", "static"]

    result = _expand_env_vars_recursive(input_list)

    assert result == ["first", "second", "static"]


# ==========================================
# MCP Server Configuration Tests
# ==========================================


def test_mcp_server_config_stdio_complete() -> None:
    """Test creating a complete stdio MCP server config."""
    config: MCPServerConfig = {
        "name": "test-server",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-test"],
        "env": {"API_KEY": "${API_KEY}"},
    }

    assert config["name"] == "test-server"
    assert config["transport"] == "stdio"
    assert config["command"] == "npx"
    assert config["args"] == ["-y", "@modelcontextprotocol/server-test"]
    assert config["env"] == {"API_KEY": "${API_KEY}"}


def test_mcp_server_config_http_complete() -> None:
    """Test creating a complete HTTP MCP server config."""
    config: MCPServerConfig = {
        "name": "http-server",
        "transport": "http",
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer ${TOKEN}"},
    }

    assert config["name"] == "http-server"
    assert config["transport"] == "http"
    assert config["url"] == "https://example.com/mcp"
    assert config["headers"] == {"Authorization": "Bearer ${TOKEN}"}


def test_mcp_server_config_minimal() -> None:
    """Test creating a minimal MCP server config (only name required)."""
    config: MCPServerConfig = {"name": "minimal-server"}

    assert config["name"] == "minimal-server"
    # Transport defaults to "stdio" in the implementation


# ==========================================
# Middleware Initialization Tests
# ==========================================


def test_middleware_initialization_state_backend() -> None:
    """Test initializing middleware with StateBackend."""
    middleware = SkillToolMiddleware(backend=StateBackend)
    assert middleware is not None
    assert middleware._backend == StateBackend


def test_middleware_initialization_factory() -> None:
    """Test initializing middleware with a factory function."""

    def backend_factory(rt):
        return StateBackend(rt)

    middleware = SkillToolMiddleware(backend=backend_factory)
    assert middleware is not None


# ==========================================
# Backend Resolution Tests
# ==========================================


def test_get_backend_direct_instance() -> None:
    """Test resolving backend from direct instance."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    # Mock minimal required objects
    mock_state = {}
    mock_runtime = MagicMock()
    mock_runtime.context = MagicMock()
    mock_runtime.stream_writer = MagicMock()
    mock_runtime.store = MagicMock()
    mock_config = MagicMock()

    backend = middleware._get_backend(mock_state, mock_runtime, mock_config)

    # StateBackend is callable, so it's treated as a factory and an instance is created
    assert isinstance(backend, StateBackend)


def test_get_backend_factory() -> None:
    """Test resolving backend from factory function."""

    def factory(rt):
        return StateBackend(rt)

    middleware = SkillToolMiddleware(backend=factory)

    # Mock minimal required objects
    mock_state = {}
    mock_runtime = MagicMock()
    mock_runtime.context = MagicMock()
    mock_runtime.stream_writer = MagicMock()
    mock_runtime.store = MagicMock()
    mock_config = MagicMock()

    backend = middleware._get_backend(mock_state, mock_runtime, mock_config)

    assert backend is not None
    assert isinstance(backend, StateBackend)


# ==========================================
# Active Skills Tests
# ==========================================


def test_get_active_skills_from_metadata() -> None:
    """Test extracting active skill names from skills_metadata."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    state = {
        "skills_metadata": [
            {"name": "skill-a", "description": "Skill A"},
            {"name": "skill-b", "description": "Skill B"},
            {"name": "skill-c", "description": "Skill C"},
        ]
    }

    active_skills = middleware._get_active_skills(state)

    assert active_skills == {"skill-a", "skill-b", "skill-c"}


def test_get_active_skills_empty_metadata() -> None:
    """Test getting active skills when metadata is empty."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    state = {"skills_metadata": []}

    active_skills = middleware._get_active_skills(state)

    assert active_skills == set()


def test_get_active_skills_no_metadata() -> None:
    """Test getting active skills when metadata is missing."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    state = {}

    active_skills = middleware._get_active_skills(state)

    assert active_skills == set()


# ==========================================
# Tool Dispatcher Tests
# ==========================================


@pytest.mark.asyncio
async def test_tool_dispatcher_found() -> None:
    """Test tool dispatcher finding and executing a tool."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    # Mock state with active tools
    mock_tool = MagicMock()
    mock_tool.ainvoke = AsyncMock(return_value="tool result")
    state = {
        "active_skill_tools": {
            "test-skill": {
                "test_tool": mock_tool,
            }
        }
    }

    dispatcher = middleware._create_tool_dispatcher(state)

    result = await dispatcher.ainvoke(
        {
            "skill_name": "test-skill",
            "tool_name": "test_tool",
            "arguments": {"arg1": "value1"},
        }
    )

    assert result == "tool result"
    mock_tool.ainvoke.assert_called_once_with({"arg1": "value1"})


@pytest.mark.asyncio
async def test_tool_dispatcher_not_found() -> None:
    """Test tool dispatcher when tool is not found."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    state = {
        "active_skill_tools": {
            "test-skill": {
                "existing_tool": MagicMock(),
            }
        }
    }

    dispatcher = middleware._create_tool_dispatcher(state)

    result = await dispatcher.ainvoke(
        {
            "skill_name": "test-skill",
            "tool_name": "nonexistent_tool",
            "arguments": {},
        }
    )

    # Should return JSON error
    error_data = json.loads(result)
    assert "error" in error_data
    assert "not found" in error_data["error"]


@pytest.mark.asyncio
async def test_tool_dispatcher_fuzzy_match() -> None:
    """Test tool dispatcher with fuzzy matching (MCP prefix)."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    # Mock tool with server prefix (typical MCP naming)
    mock_tool = MagicMock()
    mock_tool.ainvoke = AsyncMock(return_value="mcp result")
    state = {
        "active_skill_tools": {
            "test-skill": {
                "server_test_tool": mock_tool,  # MCP tool with prefix
            }
        }
    }

    dispatcher = middleware._create_tool_dispatcher(state)

    # Try to call without prefix
    result = await dispatcher.ainvoke(
        {
            "skill_name": "test-skill",
            "tool_name": "test_tool",
            "arguments": {},
        }
    )

    assert result == "mcp result"
    mock_tool.ainvoke.assert_called_once_with({})


# ==========================================
# System Prompt Injection Tests
# ==========================================


def test_format_tools_documentation() -> None:
    """Test formatting tools documentation for system prompt."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    # Create mock tools
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool1"
    mock_tool1.description = "First tool"

    mock_tool2 = MagicMock()
    mock_tool2.name = "tool2"
    mock_tool2.description = "Second tool"

    active_skill_tools = {
        "skill-a": {
            "tool1": mock_tool1,
        },
        "skill-b": {
            "tool2": mock_tool2,
        },
    }

    doc = middleware._format_tools_documentation(active_skill_tools)

    assert "## Available Skill Tools" in doc
    assert "skill_tool_call" in doc
    assert "### skill-a" in doc
    assert "### skill-b" in doc
    assert "**tool1**: First tool" in doc
    assert "**tool2**: Second tool" in doc


def test_format_tools_documentation_empty() -> None:
    """Test formatting documentation when no tools are available."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    doc = middleware._format_tools_documentation({})

    assert doc == ""


# ==========================================
# modify_request Tests
# ==========================================


def test_modify_request_with_tools() -> None:
    """Test modifying request with active tools."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    # Mock request with system message and active tools
    mock_system_message = MagicMock()
    mock_system_message.content = "Original system prompt"

    mock_request = MagicMock()
    mock_request.system_message = mock_system_message
    mock_request.state = {
        "active_skill_tools": {
            "test-skill": {
                "tool1": MagicMock(name="tool1", description="Test tool"),
            }
        }
    }
    mock_request.override = MagicMock(return_value=mock_request)

    # Patch append_to_system_message
    with patch("deepagents.middleware.skill_tools.append_to_system_message") as mock_append:
        mock_append.return_value = mock_system_message

        middleware.modify_request(mock_request)

        # Verify override was called
        mock_request.override.assert_called_once()
        # Verify append was called
        mock_append.assert_called_once()


def test_modify_request_no_tools() -> None:
    """Test modifying request when no tools are active."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    # Mock request without active tools
    mock_request = MagicMock()
    mock_request.state = {"active_skill_tools": {}}
    mock_request.override = MagicMock(return_value=mock_request)

    middleware.modify_request(mock_request)

    # Override should not be called when no tools
    mock_request.override.assert_not_called()


# ==========================================
# Lifecycle Tests
# ==========================================


@pytest.mark.asyncio
async def test_abefore_agent_initialization() -> None:
    """Test abefore_agent initializes state on first call."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    # Mock minimal required objects
    mock_state = {"skills_metadata": [{"name": "test", "description": "Test"}]}
    mock_runtime = MagicMock()
    mock_runtime.context = MagicMock()
    mock_runtime.stream_writer = MagicMock()
    mock_runtime.store = MagicMock()
    mock_config = MagicMock()

    result = await middleware.abefore_agent(mock_state, mock_runtime, mock_config)

    # Should return state update
    assert result is not None
    assert "mcp_clients" in result or "active_skill_tools" in result


@pytest.mark.asyncio
async def test_abefore_agent_skip_if_initialized() -> None:
    """Test abefore_agent skips if already initialized."""
    middleware = SkillToolMiddleware(backend=StateBackend)

    # Mock state that's already initialized
    mock_state = {
        "skills_metadata": [{"name": "test", "description": "Test"}],
        "mcp_clients": {},  # Already initialized
        "active_skill_tools": {},
    }
    mock_runtime = MagicMock()
    mock_runtime.context = MagicMock()
    mock_runtime.stream_writer = MagicMock()
    mock_runtime.store = MagicMock()
    mock_config = MagicMock()

    result = await middleware.abefore_agent(mock_state, mock_runtime, mock_config)

    # Should return None (no update)
    assert result is None
