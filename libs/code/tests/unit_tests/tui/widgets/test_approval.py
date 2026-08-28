"""Unit tests for approval widget expandable command display."""

import asyncio
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets.approval import ApprovalMenu

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


class TestCheckExpandableCommand:
    """Tests for `ApprovalMenu._check_expandable_command`."""


class TestGetCommandDisplay:
    """Tests for `ApprovalMenu._get_command_display`."""

    def test_none_command_value_handled(self) -> None:
        """Test that None command value is handled gracefully."""
        menu = ApprovalMenu({"name": "execute", "args": {"command": None}})
        assert menu._has_expandable_command is False
        display = menu._get_command_display(expanded=False)
        assert "None" in display.plain

    def test_command_display_escapes_markup_tags(self) -> None:
        """Shell command display should safely render literal bracket sequences."""
        command = "echo [/dim] [literal]"
        menu = ApprovalMenu({"name": "execute", "args": {"command": command}})
        display = menu._get_command_display(expanded=True)
        assert command in display.plain


class TestToggleExpand:
    """Tests for `ApprovalMenu.action_toggle_expand`."""


class TestExecuteToolMinimalDisplay:
    """Tests confirming `execute` is treated as the shell-execution tool."""

    async def test_mounted_widget_shows_working_directory(self) -> None:
        """The compact shell approval renders its supplemental description."""
        from textual.app import App, ComposeResult
        from textual.content import Content
        from textual.widgets import Static

        class ApprovalTestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {
                        "name": "execute",
                        "args": {"command": "pwd"},
                        "description": (
                            "Execute Command: pwd\n"
                            "Working Directory: /workspace/thread-a"
                        ),
                    }
                )

        async with ApprovalTestApp().run_test() as pilot:
            await pilot.pause()
            menu = pilot.app.query_one(ApprovalMenu)
            description = menu.query_one(".approval-description", Static)
            command = menu.query_one(".approval-command", Static)

            rendered_description = description.render()
            rendered_command = command.render()
            assert isinstance(rendered_description, Content)
            assert isinstance(rendered_command, Content)
            assert rendered_description.plain == (
                "Working Directory: /workspace/thread-a"
            )
            assert rendered_command.plain == "pwd"


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


class TestOptionOrdering:
    """Tests for approval, mode-change, and reject ordering."""


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

    def test_select_auto_is_no_op_when_hidden(self, wired_menu: MenuFactory) -> None:
        """Pressing `a` does nothing when Auto is not offered."""
        menu, future = wired_menu(
            {"name": "execute", "args": {"command": "echo hello"}},
            auto_mode_eligible=False,
        )
        menu.action_select_auto()
        assert not future.done()
        assert menu.display is True

    def test_auto_fallback_shows_switch_even_when_not_eligible(self) -> None:
        """A live Auto fallback keeps its Switch-to-Manual option."""
        menu = ApprovalMenu(
            {
                "name": "delete",
                "args": {"file_path": "old.py"},
                "description": "Auto human fallback: this action needs your review.",
            },
            auto_mode_eligible=False,
        )
        decisions = [decision for _, decision in menu._build_options()]
        assert decisions == ["approve", "switch_manual", "reject"]
        assert menu._num_options == 3
        # Ineligible session but Auto is live, so the footer still advertises `a`.
        assert "y/a/n quick keys" in menu._compose_help_text()


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
