"""Test execute tool result formatting."""

from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import ACPAgentServer


def test_format_execute_result_success():
    """Test formatting of successful execute command."""
    server = ACPAgentServer(
        root_dir="/tmp",
        checkpointer=MemorySaver(),
        mode="accept_edits",
    )

    command = "pwd"
    result = "/Users/jacoblee/langchain/deepagents\n[Command succeeded with exit code 0]"

    formatted = server._format_execute_result(command, result)

    # Check that formatted result contains expected sections
    assert "**Command:**" in formatted
    assert "```bash" in formatted
    assert "pwd" in formatted
    assert "**Output:**" in formatted
    assert "/Users/jacoblee/langchain/deepagents" in formatted
    assert "**Status:**" in formatted
    assert "Command succeeded with exit code 0" in formatted


def test_format_execute_result_failed():
    """Test formatting of failed execute command."""
    server = ACPAgentServer(
        root_dir="/tmp",
        checkpointer=MemorySaver(),
        mode="accept_edits",
    )

    command = "nonexistent-command"
    result = "Error: command not found\n[Command failed with exit code 127]"

    formatted = server._format_execute_result(command, result)

    # Check that formatted result contains expected sections
    assert "**Command:**" in formatted
    assert "nonexistent-command" in formatted
    assert "**Output:**" in formatted
    assert "Error: command not found" in formatted
    assert "**Status:**" in formatted
    assert "Command failed with exit code 127" in formatted


def test_format_execute_result_empty_output():
    """Test formatting of execute command with empty output."""
    server = ACPAgentServer(
        root_dir="/tmp",
        checkpointer=MemorySaver(),
        mode="accept_edits",
    )

    command = "echo -n"
    result = "[Command succeeded with exit code 0]"

    formatted = server._format_execute_result(command, result)

    # Check that formatted result handles empty output
    assert "**Command:**" in formatted
    assert "echo -n" in formatted
    assert "**Output:** _(empty)_" in formatted
    assert "**Status:**" in formatted
    assert "Command succeeded with exit code 0" in formatted


def test_format_execute_result_truncated():
    """Test formatting of truncated execute output."""
    server = ACPAgentServer(
        root_dir="/tmp",
        checkpointer=MemorySaver(),
        mode="accept_edits",
    )

    command = "cat large-file.txt"
    result = (
        "Line 1\nLine 2\nLine 3\n"
        "[Command succeeded with exit code 0]\n"
        "[Output was truncated due to size limits]"
    )

    formatted = server._format_execute_result(command, result)

    # Check that formatted result shows truncation warning
    assert "**Command:**" in formatted
    assert "cat large-file.txt" in formatted
    assert "**Output:**" in formatted
    assert "Line 1" in formatted
    assert "**Status:**" in formatted
    assert "Command succeeded with exit code 0" in formatted
    assert "Output was truncated due to size limits" in formatted


def test_format_execute_result_multiline_output():
    """Test formatting of execute command with multiline output."""
    server = ACPAgentServer(
        root_dir="/tmp",
        checkpointer=MemorySaver(),
        mode="accept_edits",
    )

    command = "ls -la"
    result = (
        "total 24\n"
        "drwxr-xr-x  5 user  staff  160 Jan 1 12:00 .\n"
        "drwxr-xr-x  10 user  staff  320 Jan 1 12:00 ..\n"
        "-rw-r--r--  1 user  staff  100 Jan 1 12:00 file.txt\n"
        "[Command succeeded with exit code 0]"
    )

    formatted = server._format_execute_result(command, result)

    # Check that multiline output is preserved
    assert "**Command:**" in formatted
    assert "ls -la" in formatted
    assert "**Output:**" in formatted
    assert "total 24" in formatted
    assert "drwxr-xr-x" in formatted
    assert "file.txt" in formatted
    assert "**Status:**" in formatted
    assert "Command succeeded with exit code 0" in formatted
