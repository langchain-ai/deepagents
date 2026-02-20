"""Unit tests for approval widget expandable command display."""

from unittest.mock import MagicMock

import pytest
from rich.markup import render

from deepagents_cli.config import get_glyphs
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
        assert get_glyphs().ellipsis in display
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
        assert get_glyphs().ellipsis not in display

    def test_short_command_shows_full_even_when_expanded_true(self) -> None:
        """Test that short commands show in full even when expanded=True."""
        menu = ApprovalMenu({"name": "shell", "args": {"command": "echo hello"}})
        display = menu._get_command_display(expanded=True)
        assert "echo hello" in display
        assert "press 'e' to expand" not in display
        assert get_glyphs().ellipsis not in display

    def test_command_at_boundary_plus_one_is_expandable(self) -> None:
        """Test off-by-one: command at exactly threshold + 1 is expandable."""
        boundary_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 1)
        menu = ApprovalMenu({"name": "shell", "args": {"command": boundary_command}})
        assert menu._has_expandable_command is True
        display = menu._get_command_display(expanded=False)
        assert get_glyphs().ellipsis in display
        assert "press 'e' to expand" in display

    def test_none_command_value_handled(self) -> None:
        """Test that None command value is handled gracefully."""
        menu = ApprovalMenu({"name": "shell", "args": {"command": None}})
        assert menu._has_expandable_command is False
        display = menu._get_command_display(expanded=False)
        assert "None" in display

    def test_integer_command_value_handled(self) -> None:
        """Test that integer command value is converted to string."""
        menu = ApprovalMenu({"name": "shell", "args": {"command": 12345}})
        assert menu._has_expandable_command is False
        display = menu._get_command_display(expanded=False)
        assert "12345" in display

    def test_command_display_escapes_markup_tags(self) -> None:
        """Shell command display should escape literal Rich tag sequences."""
        command = "echo [/dim] [literal]"
        menu = ApprovalMenu({"name": "shell", "args": {"command": command}})
        display = menu._get_command_display(expanded=True)
        rendered = render(display)
        assert command in rendered.plain


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

    def test_toggle_updates_widget_with_correct_content(self) -> None:
        """Test that toggling calls widget.update() with correct display content."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu({"name": "shell", "args": {"command": long_command}})
        menu._command_widget = MagicMock()

        # First toggle: expand
        menu.action_toggle_expand()
        menu._command_widget.update.assert_called_once()
        expanded_call = menu._command_widget.update.call_args[0][0]
        assert long_command in expanded_call
        assert get_glyphs().ellipsis not in expanded_call

        # Second toggle: collapse
        menu._command_widget.reset_mock()
        menu.action_toggle_expand()
        menu._command_widget.update.assert_called_once()
        collapsed_call = menu._command_widget.update.call_args[0][0]
        assert get_glyphs().ellipsis in collapsed_call
        assert "press 'e' to expand" in collapsed_call

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


class TestToolSetConsistency:
    """Tests for tool set consistency between _MINIMAL_TOOLS and SHELL_TOOL_NAMES."""

    def test_bash_tool_is_expandable(self) -> None:
        """Test that bash tool commands can be expandable like shell commands.

        The 'bash' tool is in _MINIMAL_TOOLS, so it should also support
        expandable command display when the command is long.
        """
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu({"name": "bash", "args": {"command": long_command}})
        # bash should be expandable just like shell
        assert menu._has_expandable_command is True

    def test_bash_short_command_not_expandable(self) -> None:
        """Test that short bash commands are not expandable."""
        menu = ApprovalMenu({"name": "bash", "args": {"command": "ls -la"}})
        assert menu._has_expandable_command is False

    def test_execute_tool_is_minimal(self) -> None:
        """Test that execute tool uses minimal display like shell.

        The 'execute' tool is in SHELL_TOOL_NAMES, so it should use minimal display.
        """
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        # execute should use minimal display like shell/bash
        assert menu._is_minimal is True


class TestGetCommandDisplayGuard:
    """Tests for `_get_command_display` safety guard."""

    def test_raises_on_empty_action_requests(self) -> None:
        """Test that _get_command_display raises RuntimeError with empty requests."""
        menu = ApprovalMenu({"name": "shell", "args": {"command": "echo hello"}})
        # Artificially empty the action_requests to test the guard
        menu._action_requests = []
        with pytest.raises(RuntimeError, match="empty action_requests"):
            menu._get_command_display(expanded=False)
