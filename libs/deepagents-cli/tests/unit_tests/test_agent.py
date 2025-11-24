"""Unit tests for agent formatting functions."""

import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from deepagents_cli.agent import (
    _format_edit_file_description,
    _format_execute_description,
    _format_fetch_url_description,
    _format_shell_description,
    _format_task_description,
    _format_web_search_description,
    _format_write_file_description,
    create_agent_with_config,
)
from deepagents_cli.config import create_model


def test_format_write_file_description_create_new_file(tmp_path: Path) -> None:
    """Test write_file description for creating a new file."""
    new_file = tmp_path / "new_file.py"
    tool_call = {
        "name": "write_file",
        "args": {
            "file_path": str(new_file),
            "content": "def hello():\n    return 'world'\n",
        },
        "id": "call-1",
    }

    state = Mock()
    runtime = Mock()

    description = _format_write_file_description(tool_call, state, runtime)

    assert f"File: {new_file}" in description
    assert "Action: Create file" in description
    assert "Lines: 2" in description


def test_format_write_file_description_overwrite_existing_file(tmp_path: Path) -> None:
    """Test write_file description for overwriting an existing file."""
    existing_file = tmp_path / "existing.py"
    existing_file.write_text("old content")

    tool_call = {
        "name": "write_file",
        "args": {
            "file_path": str(existing_file),
            "content": "line1\nline2\nline3\n",
        },
        "id": "call-2",
    }

    state = Mock()
    runtime = Mock()

    description = _format_write_file_description(tool_call, state, runtime)

    assert f"File: {existing_file}" in description
    assert "Action: Overwrite file" in description
    assert "Lines: 3" in description


def test_format_edit_file_description_single_occurrence():
    """Test edit_file description for single occurrence replacement."""
    tool_call = {
        "name": "edit_file",
        "args": {
            "file_path": "/path/to/file.py",
            "old_string": "foo",
            "new_string": "bar",
            "replace_all": False,
        },
        "id": "call-3",
    }

    state = Mock()
    runtime = Mock()

    description = _format_edit_file_description(tool_call, state, runtime)

    assert "File: /path/to/file.py" in description
    assert "Action: Replace text (single occurrence)" in description


def test_format_edit_file_description_all_occurrences():
    """Test edit_file description for replacing all occurrences."""
    tool_call = {
        "name": "edit_file",
        "args": {
            "file_path": "/path/to/file.py",
            "old_string": "foo",
            "new_string": "bar",
            "replace_all": True,
        },
        "id": "call-4",
    }

    state = Mock()
    runtime = Mock()

    description = _format_edit_file_description(tool_call, state, runtime)

    assert "File: /path/to/file.py" in description
    assert "Action: Replace text (all occurrences)" in description


def test_format_web_search_description():
    """Test web_search description formatting."""
    tool_call = {
        "name": "web_search",
        "args": {
            "query": "python async programming",
            "max_results": 10,
        },
        "id": "call-5",
    }

    state = Mock()
    runtime = Mock()

    description = _format_web_search_description(tool_call, state, runtime)

    assert "Query: python async programming" in description
    assert "Max results: 10" in description
    assert "⚠️  This will use Tavily API credits" in description


def test_format_web_search_description_default_max_results():
    """Test web_search description with default max_results."""
    tool_call = {
        "name": "web_search",
        "args": {
            "query": "langchain tutorial",
        },
        "id": "call-6",
    }

    state = Mock()
    runtime = Mock()

    description = _format_web_search_description(tool_call, state, runtime)

    assert "Query: langchain tutorial" in description
    assert "Max results: 5" in description


def test_format_fetch_url_description():
    """Test fetch_url description formatting."""
    tool_call = {
        "name": "fetch_url",
        "args": {
            "url": "https://example.com/docs",
            "timeout": 60,
        },
        "id": "call-7",
    }

    state = Mock()
    runtime = Mock()

    description = _format_fetch_url_description(tool_call, state, runtime)

    assert "URL: https://example.com/docs" in description
    assert "Timeout: 60s" in description
    assert "⚠️  Will fetch and convert web content to markdown" in description


def test_format_fetch_url_description_default_timeout():
    """Test fetch_url description with default timeout."""
    tool_call = {
        "name": "fetch_url",
        "args": {
            "url": "https://api.example.com",
        },
        "id": "call-8",
    }

    state = Mock()
    runtime = Mock()

    description = _format_fetch_url_description(tool_call, state, runtime)

    assert "URL: https://api.example.com" in description
    assert "Timeout: 30s" in description


def test_format_task_description():
    """Test task (subagent) description formatting."""
    tool_call = {
        "name": "task",
        "args": {
            "description": "Analyze code structure and identify the main components.",
            "subagent_type": "general-purpose",
        },
        "id": "call-9",
    }

    state = Mock()
    runtime = Mock()

    description = _format_task_description(tool_call, state, runtime)

    assert "Subagent Type: general-purpose" in description
    assert "Task Instructions:" in description
    assert "Analyze code structure and identify the main components." in description
    assert "⚠️  Subagent will have access to file operations and shell commands" in description


def test_format_task_description_truncates_long_description():
    """Test task description truncates long descriptions."""
    long_description = "x" * 600  # 600 characters
    tool_call = {
        "name": "task",
        "args": {
            "description": long_description,
            "subagent_type": "general-purpose",
        },
        "id": "call-10",
    }

    state = Mock()
    runtime = Mock()

    description = _format_task_description(tool_call, state, runtime)

    assert "Subagent Type: general-purpose" in description
    assert "..." in description
    # Description should be truncated to 500 chars + "..."
    assert len(description) < len(long_description) + 300


def test_format_shell_description():
    """Test shell command description formatting."""
    tool_call = {
        "name": "shell",
        "args": {
            "command": "ls -la /tmp",
        },
        "id": "call-11",
    }

    state = Mock()
    runtime = Mock()

    description = _format_shell_description(tool_call, state, runtime)

    assert "Shell Command: ls -la /tmp" in description
    assert "Working Directory:" in description


def test_format_execute_description():
    """Test execute command description formatting."""
    tool_call = {
        "name": "execute",
        "args": {
            "command": "python script.py",
        },
        "id": "call-12",
    }

    state = Mock()
    runtime = Mock()

    description = _format_execute_description(tool_call, state, runtime)

    assert "Execute Command: python script.py" in description
    assert "Location: Remote Sandbox" in description


def test_run_command_calls_shell_tool():
    """Test that 'run make format' calls shell tool with 'make format' command.
    
    This test verifies that when a user says "run make format", the agent
    correctly interprets this as a shell command and calls the shell tool
    with just "make format" (not including the word "run").
    
    The test stops at the interrupt (HITL approval point) before the shell
    tool is actually executed, to verify the correct command is being passed.
    """
    # Create agent with shell tool enabled (no auto-approve to trigger interrupt)
    model = create_model()
    checkpointer = MemorySaver()
    agent, _backend = create_agent_with_config(
        model=model,
        assistant_id="test_agent",
        tools=[],
        sandbox=None,
        sandbox_type=None,
    )
    agent.checkpointer = checkpointer
    
    # Create config with thread_id
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Invoke agent with "run make format" command
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "run make format",
                }
            ]
        },
        config=config,
    )
    
    # Verify that an interrupt occurred (shell tool requires approval)
    assert "__interrupt__" in result, "Expected shell tool to trigger HITL interrupt"
    assert result["__interrupt__"] is not None
    
    # Extract interrupt information
    interrupts = result["__interrupt__"]
    assert len(interrupts) > 0, "Expected at least one interrupt"
    
    interrupt_value = interrupts[0].value
    action_requests = interrupt_value.get("action_requests", [])
    
    # Verify that a shell tool call is present
    shell_calls = [req for req in action_requests if req.get("name") == "shell"]
    assert len(shell_calls) > 0, "Expected at least one shell tool call"
    
    # Verify the shell command is "make format" (not "run make format")
    shell_call = shell_calls[0]
    command = shell_call.get("args", {}).get("command", "")
    assert command == "make format", f"Expected shell command to be 'make format', got: {command}"
