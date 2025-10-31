"""Tests for command handlers."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from deepagents_cli.commands import execute_bash_command, handle_command


@pytest.fixture
def mock_agent():
    """Create a mock agent with checkpointer."""
    agent = Mock()
    agent.checkpointer = Mock()
    return agent


@pytest.fixture
def mock_token_tracker():
    """Create a mock token tracker."""
    tracker = Mock()
    tracker.reset = Mock()
    tracker.display_session = Mock()
    return tracker


class TestHandleCommand:
    """Tests for handle_command function."""

    def test_quit_command(self, mock_agent, mock_token_tracker):
        """Test that /quit returns 'exit'."""
        result = handle_command("/quit", mock_agent, mock_token_tracker)
        assert result == "exit"

    def test_exit_command(self, mock_agent, mock_token_tracker):
        """Test that /exit returns 'exit'."""
        result = handle_command("/exit", mock_agent, mock_token_tracker)
        assert result == "exit"

    def test_q_command(self, mock_agent, mock_token_tracker):
        """Test that /q returns 'exit'."""
        result = handle_command("/q", mock_agent, mock_token_tracker)
        assert result == "exit"

    def test_quit_command_case_insensitive(self, mock_agent, mock_token_tracker):
        """Test that quit commands work regardless of case."""
        assert handle_command("/QUIT", mock_agent, mock_token_tracker) == "exit"
        assert handle_command("/Exit", mock_agent, mock_token_tracker) == "exit"

    def test_clear_command(self, mock_agent, mock_token_tracker):
        """Test that /clear resets state and returns True."""
        from langgraph.checkpoint.memory import InMemorySaver

        result = handle_command("/clear", mock_agent, mock_token_tracker)

        assert result is True
        # Verify checkpointer was reset
        assert isinstance(mock_agent.checkpointer, InMemorySaver)
        # Verify token tracker was reset
        mock_token_tracker.reset.assert_called_once()

    def test_help_command(self, mock_agent, mock_token_tracker):
        """Test that /help returns True."""
        with patch("deepagents_cli.commands.show_interactive_help"):
            result = handle_command("/help", mock_agent, mock_token_tracker)
            assert result is True

    def test_tokens_command(self, mock_agent, mock_token_tracker):
        """Test that /tokens displays session and returns True."""
        result = handle_command("/tokens", mock_agent, mock_token_tracker)

        assert result is True
        mock_token_tracker.display_session.assert_called_once()

    def test_unknown_command(self, mock_agent, mock_token_tracker):
        """Test that unknown command returns True."""
        result = handle_command("/unknown", mock_agent, mock_token_tracker)
        assert result is True

    def test_command_with_leading_slash(self, mock_agent, mock_token_tracker):
        """Test commands work with leading slash."""
        result = handle_command("/quit", mock_agent, mock_token_tracker)
        assert result == "exit"

    def test_command_without_leading_slash(self, mock_agent, mock_token_tracker):
        """Test commands work without leading slash."""
        result = handle_command("quit", mock_agent, mock_token_tracker)
        assert result == "exit"

    def test_command_with_whitespace(self, mock_agent, mock_token_tracker):
        """Test commands work with surrounding whitespace."""
        result = handle_command("  /quit  ", mock_agent, mock_token_tracker)
        assert result == "exit"


class TestExecuteBashCommand:
    """Tests for execute_bash_command function."""

    def test_execute_simple_command(self):
        """Test executing a simple bash command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["echo", "hello"], returncode=0, stdout="hello\n", stderr=""
            )

            result = execute_bash_command("!echo hello")
            assert result is True
            mock_run.assert_called_once()

    def test_execute_empty_command(self):
        """Test that empty command is handled."""
        result = execute_bash_command("!")
        assert result is True

    def test_execute_command_with_stderr(self):
        """Test command that produces stderr."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["ls", "nonexistent"], returncode=1, stdout="", stderr="ls: cannot access"
            )

            result = execute_bash_command("!ls nonexistent")
            assert result is True

    def test_execute_command_timeout(self):
        """Test command timeout handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=30)

            result = execute_bash_command("!sleep 100")
            assert result is True

    def test_execute_command_exception(self):
        """Test command execution exception handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Test error")

            result = execute_bash_command("!invalid")
            assert result is True

    def test_execute_command_strips_exclamation(self):
        """Test that leading ! is stripped from command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["pwd"], returncode=0, stdout="/home/user\n", stderr=""
            )

            execute_bash_command("!pwd")

            # Check that the command was called with shell=True and the right command
            call_args = mock_run.call_args
            assert call_args[0][0] == "pwd"  # Command without !
            assert call_args[1]["shell"] is True

    def test_execute_command_nonzero_exit_code(self):
        """Test command with non-zero exit code."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["false"], returncode=1, stdout="", stderr=""
            )

            result = execute_bash_command("!false")
            assert result is True

    def test_execute_command_with_whitespace(self):
        """Test command with leading/trailing whitespace."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["echo", "test"], returncode=0, stdout="test\n", stderr=""
            )

            result = execute_bash_command("  !echo test  ")
            assert result is True
