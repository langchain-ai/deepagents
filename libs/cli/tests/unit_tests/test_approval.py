"""Unit tests for approval widget expandable command display."""

from unittest.mock import MagicMock

import pytest

from deepagents_cli.config import get_glyphs
from deepagents_cli.widgets.approval import (
    _SHELL_COMMAND_TRUNCATE_LENGTH,
    _SHELL_COMMAND_TRUNCATE_LINES,
    ApprovalMenu,
)


class TestCheckExpandableCommand:
    """Tests for `ApprovalMenu._check_expandable_command`."""

    def test_shell_command_over_threshold_is_expandable(self) -> None:
        """Test that shell commands longer than threshold are expandable."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu({"name": "execute", "args": {"command": long_command}})
        assert menu._has_expandable_command is True

    def test_shell_command_at_threshold_not_expandable(self) -> None:
        """Test that shell commands at exactly the threshold are not expandable."""
        exact_command = "x" * _SHELL_COMMAND_TRUNCATE_LENGTH
        menu = ApprovalMenu({"name": "execute", "args": {"command": exact_command}})
        assert menu._has_expandable_command is False

    def test_shell_command_under_threshold_not_expandable(self) -> None:
        """Test that short shell commands are not expandable."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
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
                {"name": "execute", "args": {"command": long_command}},
                {"name": "execute", "args": {"command": "echo hello"}},
            ]
        )
        assert menu._has_expandable_command is False

    def test_missing_command_arg_not_expandable(self) -> None:
        """Test that shell requests without command arg are not expandable."""
        menu = ApprovalMenu({"name": "execute", "args": {}})
        assert menu._has_expandable_command is False

    def test_multiline_command_over_line_threshold_is_expandable(self) -> None:
        """Multi-line commands that exceed the line threshold are expandable.

        Each line stays well under the character threshold, so this regresses
        only if the line-count check is missing.
        """
        command = "\n".join(["echo line"] * (_SHELL_COMMAND_TRUNCATE_LINES + 1))
        menu = ApprovalMenu({"name": "execute", "args": {"command": command}})
        assert menu._has_expandable_command is True

    def test_multiline_command_at_line_threshold_not_expandable(self) -> None:
        """Commands at exactly the line threshold are not expandable."""
        command = "\n".join(["echo line"] * _SHELL_COMMAND_TRUNCATE_LINES)
        menu = ApprovalMenu({"name": "execute", "args": {"command": command}})
        assert menu._has_expandable_command is False


class TestGetCommandDisplay:
    """Tests for `ApprovalMenu._get_command_display`."""

    def test_short_command_shows_full(self) -> None:
        """Test that short commands display in full regardless of expanded state."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        display = menu._get_command_display(expanded=False)
        assert "echo hello" in display.plain
        assert "press 'e' to expand" not in display.plain

    def test_long_command_truncated_when_not_expanded(self) -> None:
        """Test that long commands are truncated with expand hint."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 50)
        menu = ApprovalMenu({"name": "execute", "args": {"command": long_command}})
        display = menu._get_command_display(expanded=False)
        assert get_glyphs().ellipsis in display.plain
        assert "press 'e' to expand" in display.plain
        # Check that the truncated portion is present
        assert "x" * _SHELL_COMMAND_TRUNCATE_LENGTH in display.plain

    def test_long_command_shows_full_when_expanded(self) -> None:
        """Test that long commands display in full when expanded."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 50)
        menu = ApprovalMenu({"name": "execute", "args": {"command": long_command}})
        display = menu._get_command_display(expanded=True)
        assert long_command in display.plain
        assert "press 'e' to expand" not in display.plain
        assert get_glyphs().ellipsis not in display.plain

    def test_short_command_shows_full_even_when_expanded_true(self) -> None:
        """Test that short commands show in full even when expanded=True."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        display = menu._get_command_display(expanded=True)
        assert "echo hello" in display.plain
        assert "press 'e' to expand" not in display.plain
        assert get_glyphs().ellipsis not in display.plain

    def test_command_at_boundary_plus_one_is_expandable(self) -> None:
        """Test off-by-one: command at exactly threshold + 1 is expandable."""
        boundary_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 1)
        menu = ApprovalMenu({"name": "execute", "args": {"command": boundary_command}})
        assert menu._has_expandable_command is True
        display = menu._get_command_display(expanded=False)
        assert get_glyphs().ellipsis in display.plain
        assert "press 'e' to expand" in display.plain

    def test_none_command_value_handled(self) -> None:
        """Test that None command value is handled gracefully."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": None}})
        assert menu._has_expandable_command is False
        display = menu._get_command_display(expanded=False)
        assert "None" in display.plain

    def test_integer_command_value_handled(self) -> None:
        """Test that integer command value is converted to string."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": 12345}})
        assert menu._has_expandable_command is False
        display = menu._get_command_display(expanded=False)
        assert "12345" in display.plain

    def test_command_display_escapes_markup_tags(self) -> None:
        """Shell command display should safely render literal bracket sequences."""
        command = "echo [/dim] [literal]"
        menu = ApprovalMenu({"name": "execute", "args": {"command": command}})
        display = menu._get_command_display(expanded=True)
        assert command in display.plain

    def test_command_display_with_hidden_unicode_shows_warning(self) -> None:
        """Hidden Unicode should be surfaced with explicit warning details."""
        command = "echo a\u202eb"
        menu = ApprovalMenu({"name": "execute", "args": {"command": command}})
        display = menu._get_command_display(expanded=True)
        assert "echo ab" in display.plain
        assert "hidden chars detected" in display.plain
        assert "U+202E" in display.plain
        assert "raw:" in display.plain

    def test_multiline_command_truncated_to_max_lines(self) -> None:
        """Multi-line commands collapse to the line cap with an expand hint."""
        lines = [f"echo {i}" for i in range(_SHELL_COMMAND_TRUNCATE_LINES + 3)]
        command = "\n".join(lines)
        menu = ApprovalMenu({"name": "execute", "args": {"command": command}})
        display = menu._get_command_display(expanded=False)
        plain = display.plain
        assert get_glyphs().ellipsis in plain
        assert "press 'e' to expand" in plain
        for kept in lines[:_SHELL_COMMAND_TRUNCATE_LINES]:
            assert kept in plain
        assert lines[_SHELL_COMMAND_TRUNCATE_LINES] not in plain

    def test_multiline_command_shows_full_when_expanded(self) -> None:
        """Expanded multi-line commands show every line."""
        lines = [f"echo {i}" for i in range(_SHELL_COMMAND_TRUNCATE_LINES + 3)]
        command = "\n".join(lines)
        menu = ApprovalMenu({"name": "execute", "args": {"command": command}})
        display = menu._get_command_display(expanded=True)
        for line in lines:
            assert line in display.plain
        assert "press 'e' to expand" not in display.plain


class TestToggleExpand:
    """Tests for `ApprovalMenu.action_toggle_expand`."""

    def test_toggle_changes_expanded_state(self) -> None:
        """Test that toggling changes the expanded state."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu({"name": "execute", "args": {"command": long_command}})
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
        menu = ApprovalMenu({"name": "execute", "args": {"command": long_command}})
        menu._command_widget = MagicMock()

        # First toggle: expand
        menu.action_toggle_expand()
        menu._command_widget.update.assert_called_once()
        expanded_call = menu._command_widget.update.call_args[0][0]
        assert long_command in expanded_call.plain
        assert get_glyphs().ellipsis not in expanded_call.plain

        # Second toggle: collapse
        menu._command_widget.reset_mock()
        menu.action_toggle_expand()
        menu._command_widget.update.assert_called_once()
        collapsed_call = menu._command_widget.update.call_args[0][0]
        assert get_glyphs().ellipsis in collapsed_call.plain
        assert "press 'e' to expand" in collapsed_call.plain

    def test_toggle_does_nothing_for_non_expandable(self) -> None:
        """Test that toggling does nothing for non-expandable commands."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        menu._command_widget = MagicMock()

        assert menu._command_expanded is False
        menu.action_toggle_expand()
        assert menu._command_expanded is False

    def test_toggle_does_nothing_without_widget(self) -> None:
        """Test that toggling does nothing if command widget is not set."""
        long_command = "x" * (_SHELL_COMMAND_TRUNCATE_LENGTH + 10)
        menu = ApprovalMenu({"name": "execute", "args": {"command": long_command}})
        # Explicitly ensure no widget
        menu._command_widget = None

        assert menu._command_expanded is False
        menu.action_toggle_expand()
        assert menu._command_expanded is False


class TestExecuteToolMinimalDisplay:
    """Tests confirming `execute` is treated as the shell-execution tool."""

    def test_execute_tool_is_minimal(self) -> None:
        """The `execute` tool should use minimal display."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        assert menu._is_minimal is True

    def test_non_shell_tool_is_not_minimal(self) -> None:
        """Tools other than `execute` should not use minimal display."""
        menu = ApprovalMenu({"name": "write_file", "args": {"path": "f.py"}})
        assert menu._is_minimal is False


class TestSecurityWarnings:
    """Tests for approval-level Unicode/URL warning collection."""

    def test_collects_hidden_unicode_warning(self) -> None:
        """Hidden Unicode in args should populate security warnings."""
        menu = ApprovalMenu(
            {"name": "execute", "args": {"command": "echo he\u200bllo"}}
        )
        assert menu._security_warnings
        assert any("hidden Unicode" in warning for warning in menu._security_warnings)

    def test_collects_url_warning_for_suspicious_domain(self) -> None:
        """Suspicious URL args should populate security warnings."""
        menu = ApprovalMenu({"name": "fetch_url", "args": {"url": "https://аpple.com"}})
        assert menu._security_warnings
        assert any(
            "URL" in warning or "Domain" in warning
            for warning in menu._security_warnings
        )


class TestGetCommandDisplayGuard:
    """Tests for `_get_command_display` safety guard."""

    def test_raises_on_empty_action_requests(self) -> None:
        """Test that _get_command_display raises RuntimeError with empty requests."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        # Artificially empty the action_requests to test the guard
        menu._action_requests = []
        with pytest.raises(RuntimeError, match="empty action_requests"):
            menu._get_command_display(expanded=False)


class TestOptionOrdering:
    """Tests for the HITL option ordering: approve, auto-approve, reject."""

    @pytest.mark.parametrize(
        ("index", "expected_type"),
        [
            (0, "approve"),
            (1, "auto_approve_all"),
            (2, "reject"),
        ],
    )
    def test_decision_map_index_maps_to_correct_type(
        self, index: int, expected_type: str
    ) -> None:
        """Each selection index must resolve to its corresponding decision type."""
        import asyncio

        loop = asyncio.new_event_loop()
        future: asyncio.Future[dict[str, str]] = loop.create_future()
        menu = ApprovalMenu({"name": "write", "args": {"path": "f.py", "content": ""}})
        menu.set_future(future)
        menu._handle_selection(index)
        assert future.result() == {"type": expected_type}
        assert menu.display is False
        loop.close()

    @pytest.mark.parametrize(
        ("action", "expected_type"),
        [
            ("action_select_approve", "approve"),
            ("action_select_auto", "auto_approve_all"),
            ("action_select_reject", "reject"),
        ],
    )
    def test_quick_actions_submit_without_rendering_selection(
        self, action: str, expected_type: str
    ) -> None:
        """Quick actions must submit the right decision without repainting."""
        import asyncio

        loop = asyncio.new_event_loop()
        future: asyncio.Future[dict[str, str]] = loop.create_future()
        menu = ApprovalMenu({"name": "write", "args": {"path": "f.py", "content": ""}})
        menu.set_future(future)
        menu._selected = 1
        menu._option_widgets = [MagicMock(), MagicMock(), MagicMock()]
        menu._update_options = MagicMock()  # type: ignore[method-assign]
        getattr(menu, action)()
        assert future.result() == {"type": expected_type}
        assert menu._selected == 1
        assert menu.display is False
        menu._update_options.assert_not_called()  # ty: ignore[unresolved-attribute]
        loop.close()

    @pytest.mark.parametrize(
        ("key", "expected_type"),
        [
            ("1", "approve"),
            ("y", "approve"),
            ("2", "auto_approve_all"),
            ("a", "auto_approve_all"),
            ("3", "reject"),
            ("n", "reject"),
        ],
    )
    async def test_key_binding_resolves_correct_decision(
        self, key: str, expected_type: str
    ) -> None:
        """Pressing a quick key must trigger the correct decision via key dispatch."""
        from textual.app import App, ComposeResult

        decision_received: dict[str, str] | None = None

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}}
                )

            def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
                nonlocal decision_received
                decision_received = event.decision

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()

        assert decision_received == {"type": expected_type}
