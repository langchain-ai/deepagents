"""Unit tests for `ShellMiddleware`."""

import pytest
from langchain_core.tools.base import ToolException

from deepagents_cli.shell import _DEFAULT_SHELL_TIMEOUT, ShellMiddleware


class TestShellMiddlewareInit:
    """Tests for `ShellMiddleware` initialization."""

    def test_default_timeout(self, tmp_path: str) -> None:
        """Test that default `timeout` is used when not specified."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        assert mw._default_timeout == _DEFAULT_SHELL_TIMEOUT

    def test_custom_timeout(self, tmp_path: str) -> None:
        """Test that custom `timeout` is accepted."""
        mw = ShellMiddleware(workspace_root=str(tmp_path), timeout=300)
        assert mw._default_timeout == 300

    def test_rejects_invalid_timeout(self, tmp_path: str) -> None:
        """Test that zero/negative `timeout` raises `ValueError`."""
        with pytest.raises(ValueError, match="positive"):
            ShellMiddleware(workspace_root=str(tmp_path), timeout=0)

    def test_tools_list_populated(self, tmp_path: str) -> None:
        """Test that `tools` list contains the shell tool."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "shell"


class TestRunShellCommand:
    """Tests for `_run_shell_command`."""

    def test_basic_command(self, tmp_path: str) -> None:
        """Test running a basic `echo` command."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        result = mw._run_shell_command("echo hello", tool_call_id="test")
        assert "hello" in result.content
        assert result.status == "success"

    def test_command_with_stderr(self, tmp_path: str) -> None:
        """Test that `stderr` is captured."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        result = mw._run_shell_command("echo error >&2", tool_call_id="test")
        assert "[stderr]" in result.content
        assert "error" in result.content

    def test_nonzero_exit_code(self, tmp_path: str) -> None:
        """Test that non-zero exit codes are reported."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        result = mw._run_shell_command("exit 1", tool_call_id="test")
        assert "Exit code: 1" in result.content
        assert result.status == "error"

    def test_empty_command_raises(self, tmp_path: str) -> None:
        """Test that empty command raises `ToolException`."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        with pytest.raises(ToolException, match="non-empty"):
            mw._run_shell_command("", tool_call_id="test")

    def test_per_command_timeout_override(self, tmp_path: str) -> None:
        """Test that per-command timeout overrides default."""
        mw = ShellMiddleware(workspace_root=str(tmp_path), timeout=1)
        # Command should succeed with longer timeout
        result = mw._run_shell_command("sleep 0.1 && echo done", tool_call_id="test", timeout=5)
        assert "done" in result.content
        assert result.status == "success"

    def test_timeout_triggers(self, tmp_path: str) -> None:
        """Test that timeout triggers for long-running commands."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        result = mw._run_shell_command("sleep 10", tool_call_id="test", timeout=1)
        assert "timed out" in result.content
        assert "1 seconds" in result.content
        assert result.status == "error"

    def test_timeout_error_message_suggests_parameter(self, tmp_path: str) -> None:
        """Test that timeout error suggests using timeout parameter."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        result = mw._run_shell_command("sleep 10", tool_call_id="test", timeout=1)
        assert "timeout parameter" in result.content

    def test_zero_per_command_timeout_raises(self, tmp_path: str) -> None:
        """Test that zero per-command timeout raises `ToolException`."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        with pytest.raises(ToolException, match="positive"):
            mw._run_shell_command("echo test", tool_call_id="test", timeout=0)

    def test_negative_per_command_timeout_raises(self, tmp_path: str) -> None:
        """Test that negative per-command timeout raises `ToolException`."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        with pytest.raises(ToolException, match="positive"):
            mw._run_shell_command("echo test", tool_call_id="test", timeout=-5)

    def test_output_truncation(self, tmp_path: str) -> None:
        """Test that long output is truncated."""
        mw = ShellMiddleware(workspace_root=str(tmp_path), max_output_bytes=100)
        result = mw._run_shell_command("echo " + "x" * 200, tool_call_id="test")
        assert "truncated" in result.content
        assert len(result.content) < 250  # Some overhead for message

    def test_workspace_root_used(self, tmp_path: str) -> None:
        """Test that commands run in `workspace_root`."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        result = mw._run_shell_command("pwd", tool_call_id="test")
        assert str(tmp_path) in result.content


class TestTimeoutRetryWorkflow:
    """Tests simulating the model retrying with increased `timeout`."""

    def test_command_fails_with_short_timeout_succeeds_with_longer(self, tmp_path: str) -> None:
        """Test that a command timing out can succeed with increased `timeout`.

        This simulates the workflow where:

        1. Model runs a long command with default `timeout`
        2. Command times out
        3. Model retries with a longer `timeout`
        4. Command succeeds
        """
        mw = ShellMiddleware(workspace_root=str(tmp_path), timeout=1)

        # First attempt: times out with short default
        result1 = mw._run_shell_command("sleep 2 && echo done", tool_call_id="attempt1")
        assert result1.status == "error"
        assert "timed out" in result1.content

        # Second attempt: model increases timeout, command succeeds
        result2 = mw._run_shell_command("sleep 2 && echo done", tool_call_id="attempt2", timeout=5)
        assert result2.status == "success"
        assert "done" in result2.content

    def test_timeout_error_message_guides_model_to_solution(self, tmp_path: str) -> None:
        """Test that timeout error message tells model how to fix the issue."""
        mw = ShellMiddleware(workspace_root=str(tmp_path))
        result = mw._run_shell_command("sleep 10", tool_call_id="test", timeout=1)

        # Error message should guide model to use timeout parameter
        assert "timeout parameter" in result.content
        # Error message should show actual timeout used (for model to know to increase)
        assert "1 seconds" in result.content
