"""Unit tests for approval widget expandable command display."""

import asyncio
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets.approval import (
    _SHELL_COMMAND_TRUNCATE_LENGTH,
    _SHELL_COMMAND_TRUNCATE_LINES,
    ApprovalMenu,
)

MenuFactory = Callable[..., tuple[ApprovalMenu, "asyncio.Future[dict[str, str]]"]]


@pytest.fixture
def wired_menu() -> Iterator[MenuFactory]:
    """Build an `ApprovalMenu` wired to a future on a throwaway event loop.

    Yields a factory `(*args, **kwargs) -> (menu, future)` that forwards its
    arguments to `ApprovalMenu` and attaches a fresh future. Every loop it opens
    is closed at teardown, so tests can assert on `future.result()` without
    hand-rolling (and remembering to close) an event loop.
    """
    loops: list[asyncio.AbstractEventLoop] = []

    def _make(*args: Any, **kwargs: Any) -> tuple[ApprovalMenu, asyncio.Future]:
        loop = asyncio.new_event_loop()
        loops.append(loop)
        future: asyncio.Future[dict[str, str]] = loop.create_future()
        menu = ApprovalMenu(*args, **kwargs)
        menu.set_future(future)
        return menu, future

    yield _make
    for loop in loops:
        loop.close()


@pytest.mark.parametrize(
    ("configured", "renderer_value", "expected"),
    [(False, True, False), (True, True, True), (True, False, False)],
)
async def test_diff_line_number_preference_gates_approval_preview(
    configured: bool,
    renderer_value: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """App config can hide valid gutters but cannot enable invalid ones."""
    data = {"show_numbers": renderer_value}
    renderer = MagicMock()
    widget_class = MagicMock()
    renderer.get_approval_widget.return_value = (widget_class, data)
    monkeypatch.setattr(
        "deepagents_code.tui.widgets.approval.get_renderer",
        lambda _tool_name: renderer,
    )

    menu = ApprovalMenu(
        {"name": "edit_file", "args": {}},
        show_diff_line_numbers=configured,
    )
    container = MagicMock()
    container.remove_children = AsyncMock()
    container.mount = AsyncMock()
    menu._tool_info_container = container

    await menu._update_tool_info()

    assert data["show_numbers"] is expected
    widget_class.assert_called_once_with(data)


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
    """Tests for approval, mode-change, and reject ordering."""

    def test_auto_fallback_middle_option_switches_to_manual(self) -> None:
        import asyncio

        loop = asyncio.new_event_loop()
        future: asyncio.Future[dict[str, str]] = loop.create_future()
        menu = ApprovalMenu(
            {
                "name": "delete",
                "args": {"file_path": "old.py"},
                "description": "Auto human fallback (consecutive denials: 3).",
            }
        )
        menu.set_future(future)

        menu._handle_selection(1)

        assert menu._is_auto_fallback
        assert future.result() == {"type": "switch_manual"}
        loop.close()

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
        menu._update_options = MagicMock()  # ty: ignore
        getattr(menu, action)()
        assert future.result() == {"type": expected_type}
        assert menu._selected == 1
        assert menu.display is False
        menu._update_options.assert_not_called()  # ty: ignore
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


class TestAutoOptionEligibility:
    """The Auto option is only offered when Auto can actually be enabled."""

    def test_auto_option_hidden_when_not_eligible(self) -> None:
        """With Auto ineligible, only Approve and Reject are offered."""
        menu = ApprovalMenu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        labels = [label for label, _ in menu._build_options()]
        decisions = [decision for _, decision in menu._build_options()]
        assert menu._num_options == 2
        assert decisions == ["approve", "reject"]
        assert all("Auto" not in label for label in labels)

    def test_auto_option_shown_when_eligible(self) -> None:
        """The default (eligible) layout still offers Auto in the middle."""
        menu = ApprovalMenu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=True,
        )
        decisions = [decision for _, decision in menu._build_options()]
        assert menu._num_options == 3
        assert decisions == ["approve", "auto_approve_all", "reject"]

    def test_reject_index_tracks_layout_when_auto_hidden(self) -> None:
        """Reject is the second (last) option when Auto is hidden."""
        menu = ApprovalMenu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        assert menu._reject_index == 1

    def test_reject_via_second_position_when_auto_hidden(
        self, wired_menu: MenuFactory
    ) -> None:
        """The `2` quick key selects Reject once Auto is removed."""
        menu, future = wired_menu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        menu.action_select_position(1)
        assert future.result() == {"type": "reject"}

    def test_select_auto_is_no_op_when_hidden(self, wired_menu: MenuFactory) -> None:
        """Pressing `a` does nothing when Auto is not offered."""
        menu, future = wired_menu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        menu.action_select_auto()
        assert not future.done()
        assert menu.display is True

    def test_third_position_is_no_op_when_auto_hidden(
        self, wired_menu: MenuFactory
    ) -> None:
        """The `3` quick key has no target once Auto is removed."""
        menu, future = wired_menu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        menu.action_select_position(2)
        assert not future.done()

    def test_help_omits_auto_quick_key_when_hidden(self) -> None:
        """The footer advertises `y/n` (not `y/a/n`) when Auto is hidden."""
        menu = ApprovalMenu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        help_text = menu._compose_help_text()
        assert "y/n quick keys" in help_text
        assert "y/a/n" not in help_text

    def test_auto_fallback_shows_switch_even_when_not_eligible(self) -> None:
        """A live Auto fallback keeps its Switch-to-Manual option."""
        menu = ApprovalMenu(
            {
                "name": "delete",
                "args": {"file_path": "old.py"},
                "description": "Auto human fallback (consecutive denials: 3).",
            },
            auto_mode_eligible=False,
        )
        decisions = [decision for _, decision in menu._build_options()]
        assert decisions == ["approve", "switch_manual", "reject"]
        assert menu._num_options == 3
        # Ineligible session but Auto is live, so the footer still advertises `a`.
        assert "y/a/n quick keys" in menu._compose_help_text()

    def test_first_position_approves_when_auto_hidden(
        self, wired_menu: MenuFactory
    ) -> None:
        """The `1` quick key still approves in the 2-option layout."""
        menu, future = wired_menu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        menu.action_select_position(0)
        assert future.result() == {"type": "approve"}

    def test_batch_labels_when_auto_hidden(self) -> None:
        """Batch (count > 1) Approve/Reject labels stay pluralized with Auto hidden."""
        menu = ApprovalMenu(
            [
                {"name": "execute", "args": {"command": "echo one"}},
                {"name": "execute", "args": {"command": "echo two"}},
            ],
            auto_mode_eligible=False,
        )
        labels = [label for label, _ in menu._build_options()]
        decisions = [decision for _, decision in menu._build_options()]
        assert decisions == ["approve", "reject"]
        assert labels == ["Approve all 2 (y)", "Reject all 2 (n)"]

    def test_update_options_renders_contiguous_numbers_when_auto_hidden(self) -> None:
        """The hidden-Auto layout renders `1.`/`2.` rows, never skipping to `3.`.

        The display number is prefixed in `_update_options`, not baked into the
        labels, so a regression to hardcoded numbering would show `3. Reject` in
        a 2-option menu.
        """
        menu = ApprovalMenu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        menu._option_widgets = [MagicMock(), MagicMock()]
        menu._selected = 0
        menu._update_options()
        rendered = [
            widget.update.call_args.args[0]  # ty: ignore
            for widget in menu._option_widgets
        ]
        assert "1. Approve (y)" in rendered[0]
        assert "2. Reject (n)" in rendered[1]
        assert not any("3." in text for text in rendered)
        # The cursor glyph marks the selected row only.
        assert rendered[0].startswith(f"{get_glyphs().cursor} ")
        assert rendered[1].startswith("  ")

    def test_navigation_wraps_within_two_option_layout(self) -> None:
        """Down/up wrap modulo the visible count, never landing on a phantom index."""
        menu = ApprovalMenu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        menu._option_widgets = [MagicMock(), MagicMock()]
        assert menu._selected == 0
        menu.action_move_up()
        assert menu._selected == 1  # wrapped backwards to the last option
        menu.action_move_down()
        assert menu._selected == 0  # wrapped forward to the first option
        menu.action_move_down()
        assert menu._selected == 1

    def test_select_after_wraparound_resolves_without_index_error(
        self, wired_menu: MenuFactory
    ) -> None:
        """Selecting after wraparound in the 2-option layout resolves cleanly.

        Guards against a regression to a hardcoded ``% 3`` modulus, which would
        leave ``_selected == 2`` and raise ``IndexError`` on selection.
        """
        menu, future = wired_menu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        menu._option_widgets = [MagicMock(), MagicMock()]
        menu.action_move_down()  # 0 -> 1 (reject)
        menu.action_move_down()  # 1 -> 0 (wrap to approve)
        menu.action_select()
        assert future.result() == {"type": "approve"}

    @pytest.mark.parametrize(
        ("key", "expected_type"),
        [
            ("1", "approve"),
            ("y", "approve"),
            ("2", "reject"),
            ("n", "reject"),
        ],
    )
    async def test_key_binding_resolves_decision_when_auto_hidden(
        self, key: str, expected_type: str
    ) -> None:
        """With Auto hidden, keys route to the 2-option layout via real dispatch."""
        from textual.app import App, ComposeResult

        decision_received: dict[str, str] | None = None

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}},
                    auto_mode_eligible=False,
                )

            def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
                nonlocal decision_received
                decision_received = event.decision

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()

        assert decision_received == {"type": expected_type}

    @pytest.mark.parametrize("key", ["a", "3"])
    async def test_hidden_auto_keys_are_no_ops_via_dispatch(self, key: str) -> None:
        """`a` and `3` do nothing through real key dispatch when Auto is hidden."""
        from textual.app import App, ComposeResult

        decision_received: dict[str, str] | None = None

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}},
                    auto_mode_eligible=False,
                )

            def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
                nonlocal decision_received
                decision_received = event.decision

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()

        assert decision_received is None


class TestRejectWithReason:
    """Tests for the free-text reject mode (`action_reject_with_reason`)."""

    def test_help_shows_feedback_hint_on_every_option(self) -> None:
        """The Tab hint is unconditional so quick-key users can discover it."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        for selected in range(menu._num_options):
            menu._selected = selected
            assert "Tab reject with feedback" in menu._compose_help_text()

    def test_help_shows_esc_reject_hint(self) -> None:
        """Esc is advertised alongside the other menu-wide hints."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        assert "Esc reject" in menu._compose_help_text()

    def test_help_drops_menu_hints_while_reason_input_active(self) -> None:
        """The reason-input footer replaces the menu hints entirely."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        menu._reason_input_active = True

        help_text = menu._compose_help_text()

        assert "Enter submit" in help_text
        # "Entirely": none of the menu-mode hints survive into input mode.
        assert "Tab reject with feedback" not in help_text
        assert "quick keys" not in help_text
        assert "navigate" not in help_text
        assert "Esc reject" not in help_text

    def test_update_options_refreshes_help(self) -> None:
        """`_update_options` repaints the footer once per call.

        The hint no longer varies by selection, so this pins the refresh itself
        rather than any per-option hint state.
        """
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        menu._option_widgets = [MagicMock() for _ in range(menu._num_options)]
        menu._help_widget = MagicMock()

        menu._selected = 2
        menu._update_options()

        menu._help_widget.update.assert_called_once()
        assert "Tab reject with feedback" in menu._help_widget.update.call_args.args[0]

    def test_move_actions_no_op_while_input_mode_active(self) -> None:
        """Arrow keys should not move the menu while the reason input is open."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        menu._selected = 2
        menu._reason_input_active = True
        menu._update_options = MagicMock()  # ty: ignore

        menu.action_move_up()
        menu.action_move_down()

        assert menu._selected == 2
        menu._update_options.assert_not_called()  # ty: ignore

    def test_moves_cursor_to_reject_from_another_option(self) -> None:
        """Tab from Approve switches to Reject instead of doing nothing."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        reason_input = MagicMock(value="", display=False)
        menu._reason_input = reason_input
        menu._option_widgets = [MagicMock() for _ in range(menu._num_options)]
        menu._help_widget = MagicMock()
        menu._selected = 0

        menu.action_reject_with_reason()

        assert menu._selected == menu._reject_index
        assert menu._reason_input_active is True
        assert reason_input.display is True
        reason_input.focus.assert_called_once()

    def test_tab_repaints_cursor_onto_reject_row(self) -> None:
        """The rows repaint, so the highlight cannot disagree with the decision.

        Without the `_update_options()` refresh the cursor would still be drawn
        on Approve while Enter submits a reject - the exact mismatch that would
        make a user believe they were approving.
        """
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        menu._reason_input = MagicMock(value="", display=False)
        menu._option_widgets = [MagicMock() for _ in range(menu._num_options)]
        menu._help_widget = MagicMock()
        menu._selected = 0

        menu.action_reject_with_reason()

        rendered = [
            widget.update.call_args.args[0]  # ty: ignore
            for widget in menu._option_widgets
        ]
        cursor = f"{get_glyphs().cursor} "
        assert rendered[menu._reject_index].startswith(cursor)
        assert not rendered[0].startswith(cursor)

    def test_activates_input_mode_when_reject_selected(self) -> None:
        """Tab on Reject flips the menu into reason-input mode."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        reason_input = MagicMock(value="existing", display=False)
        menu._reason_input = reason_input
        menu._option_widgets = [MagicMock() for _ in range(menu._num_options)]
        menu._help_widget = MagicMock()
        menu._selected = 2
        menu.action_reject_with_reason()
        assert menu._reason_input_active is True
        assert reason_input.value == ""  # cleared
        assert reason_input.display is True
        reason_input.focus.assert_called_once()

    def test_second_tab_preserves_typed_reason(self) -> None:
        """A repeat Tab must not wipe a reason already being typed.

        Tab reaches this action even while the `Input` holds focus (the menu's
        binding wins over focus traversal), and it is the reflexive "next field"
        key inside a text box - so the guard is on a routine keystroke, not a
        defensive edge case.
        """
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        reason_input = MagicMock(value="wip", display=True)
        menu._reason_input = reason_input
        menu._option_widgets = [MagicMock() for _ in range(menu._num_options)]
        menu._help_widget = MagicMock()
        menu._reason_input_active = True

        menu.action_reject_with_reason()

        assert reason_input.value == "wip"
        assert menu._reason_input_active is True

    def test_quick_keys_cannot_decide_while_reason_input_active(self) -> None:
        """Approve-side keys must not resolve the call being rejected.

        Focus can land on the menu with the reason field still open (a click on
        the menu body), and there the quick keys read as menu commands rather
        than text. Approving there would run the very command the user is typing
        a rejection for.
        """
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        menu._reason_input = MagicMock(value="wip", display=True)
        menu._reason_input_active = True
        decisions: list[dict[str, str]] = []
        menu.post_message = lambda message: decisions.append(  # ty: ignore
            message.decision
        )

        menu.action_select_approve()
        menu.action_select_auto()
        menu.action_select_position(0)

        assert decisions == []

    def test_handle_selection_attaches_reject_message(self) -> None:
        """A non-empty reason is attached to the reject decision."""
        import asyncio

        loop = asyncio.new_event_loop()
        future: asyncio.Future[dict[str, str]] = loop.create_future()
        menu = ApprovalMenu({"name": "write_file", "args": {"path": "f.py"}})
        menu.set_future(future)
        menu._handle_selection(2, reject_message="please dry-run first")
        assert future.result() == {
            "type": "reject",
            "message": "please dry-run first",
        }
        loop.close()

    def test_handle_selection_omits_empty_reject_message(self) -> None:
        """An empty / `None` reason produces a bare reject decision."""
        import asyncio

        loop = asyncio.new_event_loop()
        future: asyncio.Future[dict[str, str]] = loop.create_future()
        menu = ApprovalMenu({"name": "write_file", "args": {"path": "f.py"}})
        menu.set_future(future)
        menu._handle_selection(2, reject_message=None)
        assert future.result() == {"type": "reject"}
        loop.close()

    def test_action_select_reject_cancels_input_mode_first(self) -> None:
        """Reject shortcut closes the input the first time instead of rejecting."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        reason_input = MagicMock(value="abc", display=True)
        menu._reason_input = reason_input
        menu._help_widget = MagicMock()
        menu._selected = 2
        menu._reason_input_active = True
        menu._handle_selection = MagicMock()  # ty: ignore
        menu.action_select_reject()
        # First call: cancels the input, no decision posted.
        menu._handle_selection.assert_not_called()  # ty: ignore
        assert menu._reason_input_active is False
        assert reason_input.display is False
        # Second call: now it actually rejects.
        menu.action_select_reject()
        menu._handle_selection.assert_called_once_with(2)

    @pytest.mark.parametrize(
        ("auto_mode_eligible", "down_presses"),
        [(True, 2), (False, 1)],
        ids=["auto-shown", "auto-hidden"],
    )
    async def test_tab_then_type_then_enter_submits_reason(
        self, *, auto_mode_eligible: bool, down_presses: int
    ) -> None:
        """Tab → type → Enter sends a reason with either option layout."""
        from textual.app import App, ComposeResult

        decision_received: dict[str, str] | None = None

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}},
                    auto_mode_eligible=auto_mode_eligible,
                )

            def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
                nonlocal decision_received
                decision_received = event.decision

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            await pilot.press(*(["down"] * down_presses))
            await pilot.press("tab")
            await pilot.pause()
            for ch in "dry run first":
                await pilot.press(ch if ch != " " else "space")
            await pilot.press("enter")
            await pilot.pause()

        assert decision_received == {
            "type": "reject",
            "message": "dry run first",
        }

    @pytest.mark.parametrize(
        ("auto_mode_eligible", "down_presses", "start_index"),
        [(True, 0, 0), (False, 0, 0), (True, 1, 1)],
        ids=["from-approve", "from-approve-auto-hidden", "from-auto-row"],
    )
    async def test_tab_from_non_reject_row_submits_reason(
        self, *, auto_mode_eligible: bool, down_presses: int, start_index: int
    ) -> None:
        """Tab moves the cursor and submits, from any row and either layout."""
        from textual.app import App, ComposeResult

        decision_received: dict[str, str] | None = None

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}},
                    auto_mode_eligible=auto_mode_eligible,
                )

            def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
                nonlocal decision_received
                decision_received = event.decision

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            menu = pilot.app.query_one(ApprovalMenu)
            if down_presses:
                await pilot.press(*(["down"] * down_presses))
            assert menu._selected == start_index
            assert start_index != menu._reject_index
            await pilot.press("tab")
            await pilot.pause()
            assert menu._selected == menu._reject_index
            for ch in "use a dry run":
                await pilot.press(ch if ch != " " else "space")
            await pilot.press("enter")
            await pilot.pause()

        assert decision_received == {
            "type": "reject",
            "message": "use a dry run",
        }

    async def test_quick_key_cannot_approve_after_click_strands_reason_field(
        self,
    ) -> None:
        """A click onto the menu must not leave quick keys able to approve.

        `on_blur` stops re-trapping focus during reason mode so the `Input` can
        hold it, which leaves a click on the menu body able to strand an open
        reason field. `on_focus` hands focus back, so `y` types instead of
        approving the command being rejected.
        """
        from textual.app import App, ComposeResult

        decisions: list[dict[str, str]] = []

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}}
                )

            def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
                decisions.append(event.decision)

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            menu = pilot.app.query_one(ApprovalMenu)
            await pilot.press("tab")
            await pilot.pause()
            for ch in "wip":
                await pilot.press(ch)
            await pilot.click(ApprovalMenu)
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()

            assert decisions == []
            assert menu._reason_input_active is True
            assert menu._reason_input is not None
            assert menu._reason_input.value == "wipy"

    async def test_enter_submits_reason_when_menu_holds_focus(self) -> None:
        """Enter must submit the typed reason, not a bare reject that drops it.

        The footer reads `Enter submit` while the field is open, so an Enter that
        reaches the menu instead of the `Input` has to honor it.
        """
        from textual.app import App, ComposeResult

        decisions: list[dict[str, str]] = []

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}}
                )

            def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
                decisions.append(event.decision)

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            menu = pilot.app.query_one(ApprovalMenu)
            await pilot.press("tab")
            await pilot.pause()
            for ch in "wip":
                await pilot.press(ch)
            await pilot.pause()
            # Enter arriving at the menu rather than the Input.
            menu.action_select()
            await pilot.pause()

        assert decisions == [{"type": "reject", "message": "wip"}]

    async def test_blank_reason_submits_plain_reject(self) -> None:
        """Pressing Enter without typing yields a bare reject decision."""
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
            await pilot.press("down", "down")
            await pilot.press("tab")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        assert decision_received == {"type": "reject"}

    async def test_escape_during_reason_cancels_without_deciding(self) -> None:
        """Esc from the reason input must close it without posting a decision.

        Verifies the cancel-first behavior end-to-end: typed reason is dropped,
        no `Decided` posts, and a subsequent `n` still produces a plain reject.
        """
        from textual.app import App, ComposeResult

        decisions: list[dict[str, str]] = []

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}}
                )

            def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
                decisions.append(event.decision)

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            await pilot.press("down", "down")
            await pilot.press("tab")
            await pilot.pause()
            for ch in "wip":
                await pilot.press(ch)
            menu = pilot.app.query_one(ApprovalMenu)
            reason_input = menu._reason_input
            assert reason_input is not None
            # Esc from the reason Input — verify the cancel state directly so
            # the test does not depend on which widget surfaces the key event.
            menu.action_select_reject()
            await pilot.pause()
            assert decisions == []
            assert menu._reason_input_active is False
            assert reason_input.display is False
            # Plain reject still works afterwards.
            await pilot.press("n")
            await pilot.pause()

        assert decisions == [{"type": "reject"}]

    async def test_cancel_after_tab_leaves_cursor_on_reject(self) -> None:
        """Cancelling does not restore the row Tab moved away from.

        A stray Tab from Approve therefore leaves a pending Reject, which is the
        fail-safe direction - but it is a deliberate choice, so pin it. Focus
        returns to the menu rather than bouncing back into the closed field.
        """
        from textual.app import App, ComposeResult

        decisions: list[dict[str, str]] = []

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}}
                )

            def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
                decisions.append(event.decision)

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            menu = pilot.app.query_one(ApprovalMenu)
            assert menu._selected == 0
            await pilot.press("tab")
            await pilot.pause()
            menu.action_select_reject()  # Esc routes here via the App binding.
            await pilot.pause()

            assert decisions == []
            assert menu._selected == menu._reject_index
            assert menu._reason_input_active is False
            assert pilot.app.focused is menu

    async def test_on_blur_keeps_focus_on_input_while_active(self) -> None:
        """Focus-trap must skip re-focus while the reason `Input` is active.

        Without this, the menu would steal focus mid-typing and the typed
        reason would be lost.
        """
        from textual.app import App, ComposeResult

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {"name": "execute", "args": {"command": "echo hello"}}
                )

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            await pilot.press("down", "down")
            await pilot.press("tab")
            await pilot.pause()
            menu = pilot.app.query_one(ApprovalMenu)
            assert menu._reason_input_active is True
            assert menu._reason_input is not None
            # Type something so we can confirm the value survives a blur.
            for ch in "ok":
                await pilot.press(ch)
            await pilot.pause()
            # Force-emit a blur event; menu must not steal focus back.
            from textual.events import Blur

            menu.on_blur(Blur())
            await pilot.pause()
            assert pilot.app.focused is menu._reason_input
            assert menu._reason_input.value == "ok"

    def test_on_input_submitted_drops_inactive_event_without_decision(self) -> None:
        """A stray submit after the input was cancelled must not decide.

        Guards against the Esc-then-Enter race: Esc flips `_reason_input_active`
        off; a queued Submitted should be silently dropped (with debug log)
        rather than posting a phantom reject.
        """
        import asyncio
        from types import SimpleNamespace

        loop = asyncio.new_event_loop()
        future: asyncio.Future[dict[str, str]] = loop.create_future()
        menu = ApprovalMenu({"name": "execute", "args": {"command": "echo hello"}})
        menu.set_future(future)
        reason_input = MagicMock(value="late", display=False)
        menu._reason_input = reason_input
        menu._reason_input_active = False
        menu._handle_selection = MagicMock()  # ty: ignore

        event = SimpleNamespace(input=reason_input, value="late", stop=MagicMock())
        menu.on_input_submitted(event)  # ty: ignore

        event.stop.assert_called_once()
        menu._handle_selection.assert_not_called()  # ty: ignore
        assert not future.done()
        loop.close()
