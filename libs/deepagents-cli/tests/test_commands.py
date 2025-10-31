"""Tests for command handlers."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from deepagents_cli.commands import execute_bash_command, handle_command


@pytest.fixture
def mock_agent() -> Mock:
    """Create a mock agent with checkpointer."""
    agent = Mock()
    agent.checkpointer = Mock()
    return agent


@pytest.fixture
def mock_token_tracker() -> Mock:
    """Create a mock token tracker."""
    tracker = Mock()
    tracker.reset = Mock()
    tracker.display_session = Mock()
    return tracker


class TestHandleCommand:
    """Tests for handle_command function."""

    @pytest.mark.parametrize("command", ["/quit", "/exit", "/q", "/QUIT", "/Exit", "/Q"])
    def test_exit_commands(self, command: str, mock_agent: Mock, mock_token_tracker: Mock) -> None:
        """Test that all exit command variants return 'exit'."""
        result = handle_command(command, mock_agent, mock_token_tracker)
        assert result == "exit"

    def test_clear_command(self, mock_agent: Mock, mock_token_tracker: Mock) -> None:
        """Test that /clear resets state and returns True."""
        from langgraph.checkpoint.memory import InMemorySaver

        result = handle_command("/clear", mock_agent, mock_token_tracker)

        assert result is True
        # Verify checkpointer was reset
        assert isinstance(mock_agent.checkpointer, InMemorySaver)
        # Verify token tracker was reset
        mock_token_tracker.reset.assert_called_once()

    def test_help_command(self, mock_agent: Mock, mock_token_tracker: Mock) -> None:
        """Test that /help returns True."""
        with patch("deepagents_cli.commands.show_interactive_help"):
            result = handle_command("/help", mock_agent, mock_token_tracker)
            assert result is True

    def test_tokens_command(self, mock_agent: Mock, mock_token_tracker: Mock) -> None:
        """Test that /tokens displays session and returns True."""
        result = handle_command("/tokens", mock_agent, mock_token_tracker)

        assert result is True
        mock_token_tracker.display_session.assert_called_once()

    def test_unknown_command(self, mock_agent: Mock, mock_token_tracker: Mock) -> None:
        """Test that unknown command returns True."""
        result = handle_command("/unknown", mock_agent, mock_token_tracker)
        assert result is True

    @pytest.mark.parametrize(
        "command",
        ["/quit", "quit", "  /quit  "],
        ids=["with-slash", "without-slash", "with-whitespace"],
    )
    def test_command_formatting(
        self, command: str, mock_agent: Mock, mock_token_tracker: Mock
    ) -> None:
        """Test commands work with/without leading slash and whitespace."""
        result = handle_command(command, mock_agent, mock_token_tracker)
        assert result == "exit"


class TestExecuteBashCommand:
    """Tests for execute_bash_command function."""

    def test_execute_simple_command(self) -> None:
        """Test executing a simple bash command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["echo", "hello"], returncode=0, stdout="hello\n", stderr=""
            )

            result = execute_bash_command("!echo hello")
            assert result is True
            mock_run.assert_called_once()

    def test_execute_empty_command(self) -> None:
        """Test that empty command is handled."""
        result = execute_bash_command("!")
        assert result is True

    def test_execute_command_with_stderr(self) -> None:
        """Test command that produces stderr."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["ls", "nonexistent"], returncode=1, stdout="", stderr="ls: cannot access"
            )

            result = execute_bash_command("!ls nonexistent")
            assert result is True

    def test_execute_command_timeout(self) -> None:
        """Test command timeout handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=30)

            result = execute_bash_command("!sleep 100")
            assert result is True

    def test_execute_command_exception(self) -> None:
        """Test command execution exception handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Test error")

            result = execute_bash_command("!invalid")
            assert result is True

    def test_execute_command_strips_exclamation(self) -> None:
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

    def test_execute_command_nonzero_exit_code(self) -> None:
        """Test command with non-zero exit code."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["false"], returncode=1, stdout="", stderr=""
            )

            result = execute_bash_command("!false")
            assert result is True

    def test_execute_command_with_whitespace(self) -> None:
        """Test command with leading/trailing whitespace."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["echo", "test"], returncode=0, stdout="test\n", stderr=""
            )

            result = execute_bash_command("  !echo test  ")
            assert result is True
