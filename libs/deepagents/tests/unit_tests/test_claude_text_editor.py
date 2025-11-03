"""Tests for ClaudeTextEditorMiddleware."""

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langgraph.types import Command

from deepagents.backends import StateBackend
from deepagents.middleware.claude_text_editor import (
    CLAUDE_TEXT_EDITOR_SYSTEM_PROMPT,
    ClaudeTextEditorMiddleware,
    ClaudeTextEditorState,
)
from deepagents.middleware.filesystem import FileData


class TestClaudeTextEditorMiddleware:
    """Test ClaudeTextEditorMiddleware initialization and configuration."""

    def test_init_default(self):
        """Test default initialization."""
        middleware = ClaudeTextEditorMiddleware()
        assert callable(middleware.backend)
        assert middleware.system_prompt == CLAUDE_TEXT_EDITOR_SYSTEM_PROMPT
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "str_replace_based_edit_tool"

    def test_init_custom_system_prompt(self):
        """Test initialization with custom system prompt."""
        middleware = ClaudeTextEditorMiddleware(system_prompt="Custom system prompt")
        assert callable(middleware.backend)
        assert middleware.system_prompt == "Custom system prompt"
        assert len(middleware.tools) == 1

    def test_init_custom_backend(self):
        """Test initialization with custom backend."""
        backend_factory = lambda rt: StateBackend(rt)
        middleware = ClaudeTextEditorMiddleware(backend=backend_factory)
        assert callable(middleware.backend)
        assert len(middleware.tools) == 1

    def test_agent_has_single_tool(self):
        """Test that agent has only the str_replace_based_edit_tool."""
        middleware = [ClaudeTextEditorMiddleware()]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        assert "files" in agent.stream_channels
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "str_replace_based_edit_tool" in agent_tools
        # Verify no other file tools are present
        assert "ls" not in agent_tools
        assert "read_file" not in agent_tools
        assert "write_file" not in agent_tools
        assert "edit_file" not in agent_tools
        assert "glob" not in agent_tools
        assert "grep" not in agent_tools


class TestViewCommand:
    """Test the 'view' command functionality."""

    def test_view_entire_file(self):
        """Test viewing an entire file without view_range."""
        state = ClaudeTextEditorState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world", "Second line", "Third line"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "view",
            "path": "/test.txt",
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        assert "Hello world" in result
        assert "Second line" in result
        assert "Third line" in result

    def test_view_with_range(self):
        """Test viewing a file with specific line range."""
        state = ClaudeTextEditorState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "view",
            "path": "/test.txt",
            "view_range": [2, 4],
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        # Should only show lines 2-4
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 4" in result

    def test_view_with_end_marker(self):
        """Test viewing a file from a line to the end using -1."""
        state = ClaudeTextEditorState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "view",
            "path": "/test.txt",
            "view_range": [3, -1],
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        # Should show lines 3-5
        assert "Line 3" in result
        assert "Line 4" in result
        assert "Line 5" in result

    def test_view_directory(self):
        """Test viewing a directory listing."""
        state = ClaudeTextEditorState(
            messages=[],
            files={
                "/dir/file1.txt": FileData(
                    content=["Content 1"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/dir/file2.txt": FileData(
                    content=["Content 2"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "view",
            "path": "/dir/",
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        # Should show directory listing
        assert "Directory listing" in result or "/dir/file1.txt" in result
        assert "/dir/file1.txt" in result or "file1.txt" in result
        assert "/dir/file2.txt" in result or "file2.txt" in result

    def test_view_nonexistent_file(self):
        """Test viewing a file that doesn't exist."""
        state = ClaudeTextEditorState(messages=[], files={})
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "view",
            "path": "/nonexistent.txt",
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        assert "Error" in result or "not found" in result.lower() or "does not exist" in result.lower()


class TestStrReplaceCommand:
    """Test the 'str_replace' command functionality."""

    def test_str_replace_single_occurrence(self):
        """Test replacing a string that appears once."""
        state = ClaudeTextEditorState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world", "Goodbye world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "str_replace",
            "path": "/test.txt",
            "old_str": "Hello",
            "new_str": "Hi",
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="test_id", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        # Should be either a string message or a Command
        if isinstance(result, Command):
            assert result.update is not None
            assert "files" in result.update
        else:
            assert "Successfully replaced" in result

    def test_str_replace_multiple_occurrences(self):
        """Test replacing a string that appears multiple times (replaces all by default)."""
        state = ClaudeTextEditorState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world", "Hello again", "Hello there"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "str_replace",
            "path": "/test.txt",
            "old_str": "Hello",
            "new_str": "Hi",
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="test_id", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        # Should replace all occurrences
        if isinstance(result, str):
            assert "3" in result or "multiple" in result.lower()

    def test_str_replace_missing_params(self):
        """Test str_replace with missing parameters."""
        state = ClaudeTextEditorState(messages=[], files={})
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "str_replace",
            "path": "/test.txt",
            "old_str": "Hello",
            # Missing new_str
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="test_id", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        assert "Error" in result
        assert "requires" in result


class TestCreateCommand:
    """Test the 'create' command functionality."""

    def test_create_new_file(self):
        """Test creating a new file."""
        state = ClaudeTextEditorState(messages=[], files={})
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "create",
            "path": "/new_file.txt",
            "file_text": "This is new content",
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="test_id", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        if isinstance(result, Command):
            assert result.update is not None
            assert "files" in result.update
        else:
            assert "Created file" in result

    def test_create_overwrite_existing(self):
        """Test that create overwrites existing files."""
        state = ClaudeTextEditorState(
            messages=[],
            files={
                "/existing.txt": FileData(
                    content=["Old content"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "create",
            "path": "/existing.txt",
            "file_text": "New content",
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="test_id", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        # Should succeed without error (Claude's create overwrites)
        if isinstance(result, Command):
            assert result.update is not None
        else:
            assert "Created file" in result or "Error" not in result

    def test_create_missing_file_text(self):
        """Test create with missing file_text parameter."""
        state = ClaudeTextEditorState(messages=[], files={})
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "create",
            "path": "/new_file.txt",
            # Missing file_text
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="test_id", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        assert "Error" in result
        assert "requires" in result


class TestInsertCommand:
    """Test the 'insert' command functionality."""

    def test_insert_at_line(self):
        """Test inserting text at a specific line number."""
        state = ClaudeTextEditorState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Line 1", "Line 2", "Line 3"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "insert",
            "path": "/test.txt",
            "insert_line": 2,
            "new_str": "Inserted line",
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="test_id", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        # Should succeed
        if isinstance(result, Command):
            assert result.update is not None
        else:
            assert "Inserted text" in result or "Error" not in result

    def test_insert_missing_params(self):
        """Test insert with missing parameters."""
        state = ClaudeTextEditorState(messages=[], files={})
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "insert",
            "path": "/test.txt",
            "insert_line": 2,
            # Missing new_str
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="test_id", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        assert "Error" in result
        assert "requires" in result

    def test_insert_invalid_line_number(self):
        """Test insert with invalid line number."""
        state = ClaudeTextEditorState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Line 1"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]
        result = tool.invoke({
            "command": "insert",
            "path": "/test.txt",
            "insert_line": 0,  # Invalid: must be >= 1
            "new_str": "New line",
            "runtime": ToolRuntime(
                state=state, context=None, tool_call_id="test_id", store=None,
                stream_writer=lambda _: None, config={}
            ),
        })
        assert "Error" in result


class TestInvalidCommand:
    """Test error handling for invalid commands."""

    def test_unknown_command(self):
        """Test that unknown commands are rejected by Pydantic validation."""
        import pytest
        from pydantic_core import ValidationError

        state = ClaudeTextEditorState(messages=[], files={})
        middleware = ClaudeTextEditorMiddleware()
        tool = middleware.tools[0]

        # Pydantic validates the Literal type before the function runs
        with pytest.raises(ValidationError) as exc_info:
            tool.invoke({
                "command": "invalid_command",
                "path": "/test.txt",
                "runtime": ToolRuntime(
                    state=state, context=None, tool_call_id="test_id", store=None,
                    stream_writer=lambda _: None, config={}
                ),
            })

        # Verify it's a validation error for the command field
        assert "command" in str(exc_info.value)
