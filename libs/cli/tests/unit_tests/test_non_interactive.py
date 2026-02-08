"""Tests for non-interactive mode HITL decision logic."""

from unittest.mock import patch

import pytest
from rich.console import Console

from deepagents_cli.non_interactive import _make_hitl_decision


@pytest.fixture
def console() -> Console:
    """Console that captures output."""
    return Console(quiet=True)


class TestMakeHitlDecision:
    """Tests for _make_hitl_decision()."""

    def test_non_shell_action_approved(self, console: Console) -> None:
        """Non-shell actions should be auto-approved."""
        result = _make_hitl_decision(
            {"name": "read_file", "args": {"path": "/tmp/test"}}, console
        )
        assert result == {"type": "approve"}

    def test_shell_without_allow_list_approved(self, console: Console) -> None:
        """Shell commands should be approved when no allow-list is configured."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = None
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "rm -rf /"}}, console
            )
            assert result == {"type": "approve"}

    def test_shell_allowed_command_approved(self, console: Console) -> None:
        """Shell commands in the allow-list should be approved."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls", "cat", "grep"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "ls -la"}}, console
            )
            assert result == {"type": "approve"}

    def test_shell_disallowed_command_rejected(self, console: Console) -> None:
        """Shell commands not in the allow-list should be rejected."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls", "cat", "grep"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "rm -rf /"}}, console
            )
            assert result["type"] == "reject"
            assert "rm -rf /" in result["message"]
            assert "not in the allow-list" in result["message"]

    def test_shell_rejected_message_includes_allowed_commands(
        self, console: Console
    ) -> None:
        """Rejection message should list the allowed commands."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls", "cat"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "whoami"}}, console
            )
            assert "ls" in result["message"]
            assert "cat" in result["message"]

    def test_empty_action_name_approved(self, console: Console) -> None:
        """Actions with empty name should be approved (non-shell)."""
        result = _make_hitl_decision({"name": "", "args": {}}, console)
        assert result == {"type": "approve"}

    def test_shell_piped_command_allowed(self, console: Console) -> None:
        """Piped shell commands where all segments are allowed should pass."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls", "grep"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "ls | grep test"}}, console
            )
            assert result == {"type": "approve"}

    def test_shell_piped_command_with_disallowed_segment(
        self, console: Console
    ) -> None:
        """Piped commands with a disallowed segment should be rejected."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "ls | rm file"}}, console
            )
            assert result["type"] == "reject"

    def test_shell_dangerous_pattern_rejected(self, console: Console) -> None:
        """Dangerous patterns rejected even if base command is allowed."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "ls $(whoami)"}}, console
            )
            assert result["type"] == "reject"

    @pytest.mark.parametrize("tool_name", ["bash", "shell", "execute"])
    def test_all_shell_tool_names_recognised(
        self, tool_name: str, console: Console
    ) -> None:
        """All SHELL_TOOL_NAMES variants should be gated by the allow-list."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls"]
            result = _make_hitl_decision(
                {"name": tool_name, "args": {"command": "rm -rf /"}}, console
            )
            assert result["type"] == "reject"
