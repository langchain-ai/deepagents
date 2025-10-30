"""Tests for AgentMemoryMiddleware."""

import pytest
from langchain.tools import ToolRuntime

from deepagents.middleware.agent_memory import (
    AgentMemoryMiddleware,
    _strip_line_numbers,
)
from deepagents.backends.state import StateBackend
from deepagents.backends.utils import create_file_data


class TestStripLineNumbers:
    """Test the _strip_line_numbers helper function."""

    def test_strip_simple_line_numbers(self):
        """Test stripping basic line numbers."""
        content = "     1\t# My Agent\n     2\tYou are helpful\n     3\tAlways be polite"
        result = _strip_line_numbers(content)
        assert result == "# My Agent\nYou are helpful\nAlways be polite"

    def test_strip_continuation_line_numbers(self):
        """Test stripping continuation line numbers (e.g., 5.1, 5.2)."""
        content = "     1\tFirst line\n     2\tSecond line\n   2.1\tcontinuation\n     3\tThird line"
        result = _strip_line_numbers(content)
        assert result == "First line\nSecond line\ncontinuation\nThird line"

    def test_strip_empty_content(self):
        """Test stripping from empty content."""
        content = ""
        result = _strip_line_numbers(content)
        assert result == ""

    def test_strip_content_without_line_numbers(self):
        """Test that content without line numbers is unchanged."""
        content = "# My Agent\nYou are helpful"
        result = _strip_line_numbers(content)
        assert result == "# My Agent\nYou are helpful"

    def test_strip_preserves_indentation(self):
        """Test that indentation after line numbers is preserved."""
        content = "     1\t  # Indented heading\n     2\t    More indentation"
        result = _strip_line_numbers(content)
        assert result == "  # Indented heading\n    More indentation"


class TestAgentMemoryMiddleware:
    """Test the AgentMemoryMiddleware class."""

    def test_before_agent_strips_line_numbers(self):
        """Test that before_agent strips line numbers from memory content."""
        # Set up backend with memory file
        runtime = ToolRuntime(
            state={
                "files": {
                    "/agent.md": create_file_data("# My Agent\nYou are helpful\nAlways be polite")
                }
            },
            context=None,
            tool_call_id="",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        backend = StateBackend(runtime)

        # Create middleware
        middleware = AgentMemoryMiddleware(backend=backend, memory_path="/memories/")

        # Call before_agent
        state = {}
        result = middleware.before_agent(state, None)

        # Verify line numbers are stripped
        assert result is not None
        assert "agent_memory" in result
        # Should NOT contain line numbers
        assert "\t" not in result["agent_memory"]
        # Should contain the actual content
        assert "# My Agent" in result["agent_memory"]
        assert "You are helpful" in result["agent_memory"]
        assert "Always be polite" in result["agent_memory"]

    def test_before_agent_handles_missing_file(self):
        """Test that before_agent handles missing memory file gracefully."""
        # Set up backend without memory file
        runtime = ToolRuntime(
            state={"files": {}},
            context=None,
            tool_call_id="",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        backend = StateBackend(runtime)

        # Create middleware
        middleware = AgentMemoryMiddleware(backend=backend, memory_path="/memories/")

        # Call before_agent
        state = {}
        result = middleware.before_agent(state, None)

        # Verify error message is returned
        assert result is not None
        assert "agent_memory" in result
        assert "Error" in result["agent_memory"]

    def test_before_agent_only_loads_once(self):
        """Test that before_agent doesn't reload memory if already loaded."""
        # Set up backend with memory file
        runtime = ToolRuntime(
            state={
                "files": {
                    "/agent.md": create_file_data("# My Agent")
                }
            },
            context=None,
            tool_call_id="",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        backend = StateBackend(runtime)

        # Create middleware
        middleware = AgentMemoryMiddleware(backend=backend, memory_path="/memories/")

        # Call before_agent with pre-existing memory
        state = {"agent_memory": "Already loaded"}
        result = middleware.before_agent(state, None)

        # Verify memory is not reloaded
        assert result is None or result.get("agent_memory") is None
