"""Unit tests for approval widget expandable command display."""

from unittest.mock import MagicMock

from deepagents_cli.widgets.approval import (
    _SHELL_COMMAND_TRUNCATE_LENGTH,
    ApprovalMenu,
)


class TestCheckExpandableCommand:
    """Tests for `ApprovalMenu._check_expandable_command`."""

    def test_shell_command_over_threshold_is_expandable(self) -> None:
        """Test that shell commands longer than threshold are expandable."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu({"name": "shell", "args": {"command": long_command}})
        assert menu._has_expandable_command is True

    def test_shell_command_at_threshold_not_expandable(self) -> None:
        """Test that shell commands at exactly the threshold are not expandable."""
        exact_command = "x" * _SHELL_COMMAND_TRUNCATE_LENGTH
        menu = ApprovalMenu({"name": "shell", "args": {"command": exact_command}})
        assert menu._has_expandable_command is False

    def test_shell_command_under_threshold_not_expandable(self) -> None:
        """Test that short shell commands are not expandable."""
        menu = ApprovalMenu({"name": "shell", "args": {"command": "echo hello"}})
        assert menu._has_expandable_command is False

    def test_execute_tool_is_expandable(self) -> None:
        """Test that execute tool commands can also be expandable."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu({"name": "execute", "args": {"command": long_command}})
        assert menu._has_expandable_command is True

    def test_non_shell_tool_not_expandable(self) -> None:
        """Test that non-shell tools are never expandable."""
        long_content = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 100)
        menu = ApprovalMenu({"name": "write", "args": {"content": long_content}})
        assert menu._has_expandable_command is False

    def test_multiple_requests_not_expandable(self) -> None:
        """Test that batch requests (multiple tools) are not expandable."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu(
            [
                {"name": "shell", "args": {"command": long_command}},
                {"name": "shell", "args": {"command": "echo hello"}},
            ]
        )
        assert menu._has_expandable_command is False

    def test_missing_command_arg_not_expandable(self) -> None:
        """Test that shell requests without command arg are not expandable."""
        menu = ApprovalMenu({"name": "shell", "args": {}})
        assert menu._has_expandable_command is False


class TestGetCommandDisplay:
    """Tests for `ApprovalMenu._get_command_display`."""

    def test_short_command_shows_full(self) -> None:
        """Test that short commands display in full regardless of expanded state."""
        menu = ApprovalMenu({"name": "shell", "args": {"command": "echo hello"}})
        display = menu._get_command_display(expanded=False)
        assert "echo hello" in display
        assert "press 'e' to expand" not in display

    def test_long_command_truncated_when_not_expanded(self) -> None:
        """Test that long commands are truncated with expand hint."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 50)
        menu = ApprovalMenu({"name": "shell", "args": {"command": long_command}})
        display = menu._get_command_display(expanded=False)
        assert "..." in display
        assert "press 'e' to expand" in display
        # Check that the truncated portion is present
        assert "x" * _SHELL_COMMAND_TRUNCATE_LENGTH in display

    def test_long_command_shows_full_when_expanded(self) -> None:
        """Test that long commands display in full when expanded."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 50)
        menu = ApprovalMenu({"name": "shell", "args": {"command": long_command}})
        display = menu._get_command_display(expanded=True)
        assert long_command in display
        assert "press 'e' to expand" not in display
        assert "..." not in display


class TestToggleExpand:
    """Tests for `ApprovalMenu.action_toggle_expand`."""

    def test_toggle_changes_expanded_state(self) -> None:
        """Test that toggling changes the expanded state."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu({"name": "shell", "args": {"command": long_command}})
        # Need to set up command widget for toggle to work
        menu._command_widget = MagicMock()

        assert menu._command_expanded is False
        menu.action_toggle_expand()
        assert menu._command_expanded is True
        menu.action_toggle_expand()
        assert menu._command_expanded is False

    def test_toggle_does_nothing_for_non_expandable(self) -> None:
        """Test that toggling does nothing for non-expandable commands."""
        menu = ApprovalMenu({"name": "shell", "args": {"command": "echo hello"}})
        menu._command_widget = MagicMock()

        assert menu._command_expanded is False
        menu.action_toggle_expand()
        assert menu._command_expanded is False

    def test_toggle_does_nothing_without_widget(self) -> None:
        """Test that toggling does nothing if command widget is not set."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu({"name": "shell", "args": {"command": long_command}})
        # Explicitly ensure no widget
        menu._command_widget = None

        assert menu._command_expanded is False
        menu.action_toggle_expand()
        assert menu._command_expanded is False
