"""Tests for ThreadSelectorScreen."""

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.cells import cell_len
from rich.style import Style
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Input, Select, Static
from textual.widgets._select import SelectCurrent

from deepagents_code.app import DeepAgentsApp, _ThreadHistoryPayload
from deepagents_code.sessions import ThreadInfo
from deepagents_code.tui.widgets.thread_selector import (
    ContainedSelect,
    ContainedSelectOverlay,
    DeleteThreadConfirmScreen,
    ThreadSelectorScreen,
)

MOCK_THREADS: list[ThreadInfo] = [
    {
        "thread_id": "abc12345",
        "agent_name": "my-agent",
        "updated_at": "2025-01-15T10:30:00",
        "message_count": 5,
        "created_at": "2025-01-15T09:00:00",
        "git_branch": "main",
        "cwd": "/home/user/project-a",
        "initial_prompt": "Hello world",
    },
    {
        "thread_id": "def67890",
        "agent_name": "other-agent",
        "updated_at": "2025-01-14T08:00:00",
        "message_count": 12,
        "created_at": "2025-01-14T07:00:00",
        "git_branch": "feature-x",
        "cwd": "/tmp/workspace",
        "initial_prompt": "Fix the bug",
    },
    {
        "thread_id": "ghi11111",
        "agent_name": "my-agent",
        "updated_at": "2025-01-13T15:45:00",
        "message_count": 3,
        "created_at": "2025-01-13T14:00:00",
        "git_branch": None,
        "cwd": None,
        "initial_prompt": None,
    },
]


def _patch_list_threads(threads: list[ThreadInfo] | None = None) -> Any:  # noqa: ANN401
    """Return a patch context manager for `list_threads`.

    Args:
        threads: Thread list to return. Defaults to `MOCK_THREADS`.
    """
    data = threads if threads is not None else MOCK_THREADS
    return patch(
        "deepagents_code.sessions.list_threads",
        new_callable=AsyncMock,
        return_value=data,
    )


def _patch_columns(columns: dict[str, bool] | None = None) -> Any:  # noqa: ANN401
    """Patch thread config loaders for tests."""
    import contextlib

    from deepagents_code.model_config import THREAD_COLUMN_DEFAULTS, ThreadConfig

    cols = columns if columns is not None else THREAD_COLUMN_DEFAULTS

    @contextlib.contextmanager
    def _ctx() -> Any:  # noqa: ANN401
        with (
            patch(
                "deepagents_code.model_config.load_thread_columns",
                return_value=dict(cols),
            ),
            patch(
                "deepagents_code.model_config.load_thread_sort_order",
                return_value="updated_at",
            ),
            patch(
                "deepagents_code.model_config.load_thread_config",
                return_value=ThreadConfig(
                    columns=dict(cols),
                    relative_time=True,
                    sort_order="updated_at",
                    scope="cwd",
                ),
            ),
        ):
            yield

    return _ctx()


def _patch_available_agents(names: list[str] | None = None) -> Any:  # noqa: ANN401
    """Return a patch context manager for `get_available_agent_names`.

    Args:
        names: Configured agent names to return. Defaults to an empty list so
            the agent filter depends only on the (deterministic) thread data.
    """
    return patch(
        "deepagents_code.agent.get_available_agent_names",
        return_value=list(names) if names is not None else [],
    )


@pytest.fixture(autouse=True)
def _isolate_available_agents() -> Any:  # noqa: ANN401
    """Keep the agent filter independent of the machine's `~/.deepagents/`.

    `_load_threads` scans the user's agents directory to populate the filter
    dropdown. Without isolation the option set would vary by developer/CI host,
    so default it to empty; tests that need specific agents patch it locally.
    """
    with _patch_available_agents():
        yield


def _style_scalar_value(value: object) -> int:
    """Return the integer value from a Textual style scalar.

    Args:
        value: Style value that may be a scalar-like object.

    Returns:
        Integer scalar value.
    """
    scalar = getattr(value, "value", None)
    assert isinstance(scalar, float)
    return int(scalar)


class ThreadSelectorTestApp(App):
    """Test app for ThreadSelectorScreen."""

    def __init__(self, current_thread: str | None = "abc12345") -> None:
        super().__init__()
        self.result: str | None = None
        self.dismissed = False
        self._current_thread = current_thread

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def show_selector(self) -> None:
        """Show the thread selector screen."""

        def handle_result(result: str | None) -> None:
            self.result = result
            self.dismissed = True

        # Disable the default cwd filter so tests don't need to populate the
        # `cwd` field on every mock thread fixture.
        screen = ThreadSelectorScreen(
            current_thread=self._current_thread, filter_cwd=None
        )
        self.push_screen(screen, handle_result)


class AppWithEscapeBinding(App):
    """Test app with a conflicting escape binding."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None = None
        self.dismissed = False
        self.interrupt_called = False

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def action_interrupt(self) -> None:
        """Handle escape."""
        if isinstance(self.screen, ModalScreen):
            self.screen.dismiss(None)
            return
        self.interrupt_called = True

    def show_selector(self) -> None:
        """Show the thread selector screen."""

        def handle_result(result: str | None) -> None:
            self.result = result
            self.dismissed = True

        screen = ThreadSelectorScreen(current_thread="abc12345", filter_cwd=None)
        self.push_screen(screen, handle_result)


class TestContainedSelect:
    """Tests for the custom thread filter select."""

    def test_open_before_overlay_mount_noops(self) -> None:
        """Opening before composition should not leave the select expanded."""
        select = ContainedSelect(
            [("All", "all")], value="all", allow_blank=False, compact=True
        )

        select.action_show_overlay()

        assert not select.expanded


class TestThreadSelectorEscapeKey:
    """Tests for ESC key dismissing the modal."""

    async def test_escape_dismisses_modal(self) -> None:
        """Pressing ESC should dismiss the modal with None result."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                await pilot.press("escape")
                await pilot.pause()

                assert app.dismissed is True
                assert app.result is None

    async def test_escape_with_conflicting_app_binding(self) -> None:
        """ESC should dismiss modal even when app has its own escape binding."""
        with _patch_list_threads():
            app = AppWithEscapeBinding()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                await pilot.press("escape")
                await pilot.pause()

                assert app.dismissed is True
                assert app.result is None
                assert app.interrupt_called is False


class TestThreadSelectorKeyboardNavigation:
    """Tests for keyboard navigation in the modal."""

    async def test_down_arrow_moves_selection(self) -> None:
        """Down arrow should move selection down."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                initial_index = screen._selected_index

                await pilot.press("down")
                await pilot.pause()

                assert screen._selected_index == initial_index + 1

    async def test_up_arrow_wraps_from_top(self) -> None:
        """Up arrow at index 0 should wrap to last thread."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                count = len(screen._threads)

                await pilot.press("up")
                await pilot.pause()

                expected = (0 - 1) % count
                assert screen._selected_index == expected

    async def test_enter_selects_thread(self) -> None:
        """Enter should select the current thread and dismiss."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()

                assert app.dismissed is True
                assert app.result == "abc12345"


class TestThreadSelectorCurrentThread:
    """Tests for current thread highlighting and preselection."""

    async def test_current_thread_is_preselected(self) -> None:
        """Opening the selector should pre-select the current thread."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp(current_thread="def67890")
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                # def67890 is at index 1 in MOCK_THREADS
                assert screen._selected_index == 1

    async def test_unknown_current_thread_defaults_to_zero(self) -> None:
        """Unknown current thread should default to index 0."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp(current_thread="nonexistent")
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._selected_index == 0

    async def test_no_current_thread_defaults_to_zero(self) -> None:
        """No current thread should default to index 0."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._selected_index == 0


class TestThreadSelectorEmptyState:
    """Tests for empty thread list."""

    async def test_no_threads_shows_empty_message(self) -> None:
        """Empty thread list should show a message and escape still works."""
        with _patch_list_threads(threads=[]):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._threads) == 0

                # Enter with no threads should be a no-op (not crash)
                await pilot.press("enter")
                await pilot.pause()

                # Escape should still dismiss
                if not app.dismissed:
                    await pilot.press("escape")
                    await pilot.pause()

                assert app.dismissed is True
                assert app.result is None

    async def test_arrow_keys_on_empty_list_do_not_crash(self) -> None:
        """Arrow keys and page keys on empty list should be no-ops."""
        with _patch_list_threads(threads=[]):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._threads) == 0

                for key in ("up", "down", "pageup", "pagedown"):
                    await pilot.press(key)
                    await pilot.pause()

                assert screen._selected_index == 0

                await pilot.press("escape")
                await pilot.pause()
                assert app.dismissed is True


class TestThreadSelectorNavigateAndSelect:
    """Tests for navigating then selecting a specific thread."""

    async def test_navigate_down_and_select(self) -> None:
        """Navigate to second thread and select it."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                await pilot.press("down")
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()

                assert app.dismissed is True
                assert app.result == "def67890"


class TestThreadSelectorTabSort:
    """Tests for sort toggling and focus traversal in the selector."""

    async def test_sort_select_toggles_sort(self) -> None:
        """The sort select should highlight the active header column."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._sort_by_updated is True
                original_columns = dict(screen._columns)
                header = screen.query_one("#thread-header", Horizontal)
                updated_cell = header.query_one(".thread-cell-updated_at", Static)
                created_cell = header.query_one(".thread-cell-created_at", Static)
                sort_select = screen.query_one("#thread-sort-select", Select)
                assert str(updated_cell.content) == "Updated"
                assert updated_cell.has_class("thread-cell-sorted")
                assert not created_cell.has_class("thread-cell-sorted")
                assert sort_select.value == "updated_at"

                sort_select.value = "created_at"
                await pilot.pause()
                assert screen._sort_by_updated is False
                assert screen._columns == original_columns
                created_cell = header.query_one(".thread-cell-created_at", Static)
                updated_cell = header.query_one(".thread-cell-updated_at", Static)
                assert str(created_cell.content) == "Created"
                assert created_cell.has_class("thread-cell-sorted")
                assert not updated_cell.has_class("thread-cell-sorted")
                assert sort_select.value == "created_at"

    async def test_sort_change_persists_preference(self) -> None:
        """Switching the sort dropdown should persist the new preference."""
        mock_save = MagicMock(return_value=True)
        with (
            _patch_list_threads(),
            _patch_columns(),
            patch(
                "deepagents_code.model_config.save_thread_sort_order",
                mock_save,
            ),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                sort_select = screen.query_one("#thread-sort-select", Select)

                sort_select.value = "created_at"
                await pilot.pause()
                await app.workers.wait_for_complete()
                mock_save.assert_any_call("created_at")

                sort_select.value = "updated_at"
                await pilot.pause()
                await app.workers.wait_for_complete()
                mock_save.assert_any_call("updated_at")

    async def test_sort_save_failure_notifies(self) -> None:
        """A failed sort-order save should surface a warning notification."""
        mock_save = MagicMock(return_value=False)
        with (
            _patch_list_threads(),
            _patch_columns(),
            patch(
                "deepagents_code.model_config.save_thread_sort_order",
                mock_save,
            ),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                sort_select = screen.query_one("#thread-sort-select", Select)

                with patch.object(app, "notify") as mock_notify:
                    sort_select.value = "created_at"
                    await pilot.pause()
                    await app.workers.wait_for_complete()
                    await pilot.pause()

                mock_notify.assert_any_call(
                    "Could not save sort preference", severity="warning"
                )

    async def test_reselecting_active_sort_is_noop(self) -> None:
        """Re-applying the active sort order should not persist or re-sort.

        The early return in `_apply_sort_selection` must fire before any
        persistence or rebuild work when the selection is unchanged.
        """
        mock_save = MagicMock(return_value=True)
        with (
            _patch_list_threads(),
            _patch_columns(),
            patch(
                "deepagents_code.model_config.save_thread_sort_order",
                mock_save,
            ),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._sort_by_updated is True

                screen._apply_sort_selection(True)
                await pilot.pause()
                await app.workers.wait_for_complete()

                assert screen._sort_by_updated is True
                mock_save.assert_not_called()

    async def test_confirming_delete_ignores_sort_select_change(self) -> None:
        """Sort dropdown changes are ignored while a delete is being confirmed."""
        mock_save = MagicMock(return_value=True)
        with (
            _patch_list_threads(),
            _patch_columns(),
            patch(
                "deepagents_code.model_config.save_thread_sort_order",
                mock_save,
            ),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._sort_by_updated is True
                sort_select = screen.query_one("#thread-sort-select", Select)
                screen._confirming_delete = True

                sort_select.value = "created_at"
                for _ in range(10):
                    await pilot.pause()

                # The guard short-circuits before mutating sort state.
                assert screen._sort_by_updated is True
                mock_save.assert_not_called()

    async def test_sorted_header_column_is_highlighted(self) -> None:
        """The active sort column should be highlighted without extra text."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test(size=(100, 24)) as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                header = screen.query_one("#thread-header", Horizontal)
                updated_cell = header.query_one(".thread-cell-updated_at", Static)
                created_cell = header.query_one(".thread-cell-created_at", Static)

                assert updated_cell.render_line(0).text.rstrip() == "Updated"
                assert created_cell.render_line(0).text.rstrip() == "Created"
                assert updated_cell.has_class("thread-cell-sorted")
                assert not created_cell.has_class("thread-cell-sorted")

    async def test_tab_moves_focus_into_column_switches(self) -> None:
        """Tab should move focus from the search input into the controls."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                filter_input = screen.query_one("#thread-filter", Input)
                scope_select = screen.query_one("#thread-scope-select", Select)
                sort_select = screen.query_one("#thread-sort-select", Select)
                thread_id_switch = screen.query_one(
                    f"#{ThreadSelectorScreen._switch_id('thread_id')}",
                    Checkbox,
                )
                agent_name_switch = screen.query_one(
                    f"#{ThreadSelectorScreen._switch_id('agent_name')}",
                    Checkbox,
                )
                messages_switch = screen.query_one(
                    f"#{ThreadSelectorScreen._switch_id('messages')}",
                    Checkbox,
                )

                agent_select = screen.query_one("#thread-agent-select", Select)
                assert filter_input.has_focus

                screen.action_focus_next_filter()
                await pilot.pause()
                assert scope_select.has_focus

                screen.action_focus_next_filter()
                await pilot.pause()
                assert sort_select.has_focus

                screen.action_focus_next_filter()
                await pilot.pause()
                assert agent_select.has_focus

                relative_time_switch = screen.query_one(
                    "#thread-relative-time", Checkbox
                )
                screen.action_focus_next_filter()
                await pilot.pause()
                assert relative_time_switch.has_focus

                screen.action_focus_next_filter()
                await pilot.pause()
                assert thread_id_switch.has_focus

                screen.action_focus_next_filter()
                await pilot.pause()
                assert agent_name_switch.has_focus

                screen.action_focus_next_filter()
                await pilot.pause()
                assert messages_switch.has_focus

    async def test_shift_tab_moves_focus_backward_through_controls(self) -> None:
        """Shift+Tab should move focus backward through the controls."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                filter_input = screen.query_one("#thread-filter", Input)
                scope_select = screen.query_one("#thread-scope-select", Select)
                sort_select = screen.query_one("#thread-sort-select", Select)
                agent_select = screen.query_one("#thread-agent-select", Select)
                assert filter_input.has_focus

                await pilot.press("tab")
                await pilot.pause()
                assert scope_select.has_focus

                await pilot.press("tab")
                await pilot.pause()
                assert sort_select.has_focus

                await pilot.press("tab")
                await pilot.pause()
                assert agent_select.has_focus

                await pilot.press("shift+tab")
                await pilot.pause()
                assert sort_select.has_focus

                await pilot.press("shift+tab")
                await pilot.pause()
                assert scope_select.has_focus

                await pilot.press("shift+tab")
                await pilot.pause()
                assert filter_input.has_focus

    def test_select_options_update_before_overlay_mount_is_safe(self) -> None:
        """Agent option refresh can run before the overlay child is mounted."""
        select = ContainedSelect(
            [("Loading...", "__loading__")],
            value="__loading__",
            allow_blank=False,
            id="thread-agent-select",
            classes="thread-agent-select",
        )

        select.set_options([("All agents", "__all__"), ("agent", "agent")])

        assert select.value == "__all__"

    async def test_thread_load_preserves_open_agent_dropdown_focus(self) -> None:
        """Async thread loading should not move focus away from an open dropdown."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_list_threads(
            *_args: object, **_kwargs: object
        ) -> list[ThreadInfo]:
            started.set()
            await release.wait()
            return MOCK_THREADS

        mock_list = AsyncMock(side_effect=delayed_list_threads)
        with (
            patch("deepagents_code.sessions.list_threads", mock_list),
            _patch_columns(),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await asyncio.wait_for(started.wait(), timeout=1)

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("enter")
                await pilot.pause()

                agent_select = screen.query_one("#thread-agent-select", Select)
                assert agent_select.expanded
                assert isinstance(screen.focused, ContainedSelectOverlay)

                with patch.object(screen, "_scroll_selected_into_view") as mock_scroll:
                    release.set()
                    for _ in range(20):
                        await pilot.pause()
                        if screen._disk_load_complete and screen._option_widgets:
                            break

                assert agent_select.expanded
                assert isinstance(screen.focused, ContainedSelectOverlay)
                mock_scroll.assert_not_called()

    async def test_cached_filter_controls_handle_tab_and_typing(self) -> None:
        """Tab traversal and type-to-search should use cached control lookups."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                filter_input = screen.query_one("#thread-filter", Input)
                scope_select = screen.query_one("#thread-scope-select", Select)
                sort_select = screen.query_one("#thread-sort-select", Select)
                agent_select = screen.query_one("#thread-agent-select", Select)
                controls = screen._filter_focus_order()
                event = MagicMock()
                event.character = "f"
                cached_input = MagicMock(spec=Input)
                cached_input.has_focus = False
                screen._filter_input = cached_input

                with (
                    patch.object(
                        screen,
                        "query_one",
                        side_effect=AssertionError("unexpected DOM query"),
                    ),
                    patch.object(screen, "set_timer"),
                ):
                    assert screen._filter_focus_order() == controls
                    assert controls[0] is filter_input
                    assert controls[1] is scope_select
                    assert controls[2] is sort_select
                    assert controls[3] is agent_select

                    screen.on_key(event)

                cached_input.focus.assert_called_once()
                cached_input.insert_text_at_cursor.assert_called_once_with("f")
                event.stop.assert_called_once()

    async def test_switch_toggle_keeps_focus_on_current_control(self) -> None:
        """Toggling a switch should not bounce focus back to the search input."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                relative_switch = screen.query_one("#thread-relative-time", Checkbox)
                filter_input = screen.query_one("#thread-filter", Input)

                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.pause()
                assert relative_switch.has_focus

                relative_switch.toggle()
                await pilot.pause()

                assert relative_switch.has_focus
                assert not filter_input.has_focus

    async def test_typing_letter_from_controls_refocuses_search(self) -> None:
        """Typing a letter on a control should jump back to fuzzy search."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                filter_input = screen.query_one("#thread-filter", Input)
                relative_switch = screen.query_one("#thread-relative-time", Checkbox)

                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.pause()
                assert relative_switch.has_focus

                await pilot.press("f")
                await pilot.pause()

                assert filter_input.has_focus
                assert filter_input.value == "f"
                assert screen._filter_text == "f"
                assert len(screen._filtered_threads) == 1
                assert screen._filtered_threads[0]["thread_id"] == "def67890"

    async def test_typing_multiple_letters_from_controls_appends_search(self) -> None:
        """Typing multiple letters after refocus should append, not replace."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                filter_input = screen.query_one("#thread-filter", Input)
                relative_switch = screen.query_one("#thread-relative-time", Checkbox)

                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.pause()
                assert relative_switch.has_focus

                await pilot.press("f")
                await pilot.pause()
                await pilot.press("i")
                await pilot.pause()

                assert filter_input.has_focus
                assert filter_input.value == "fi"
                assert screen._filter_text == "fi"
                assert len(screen._filtered_threads) == 1
                assert screen._filtered_threads[0]["thread_id"] == "def67890"

    async def test_space_from_controls_does_not_refocus_search(self) -> None:
        """Space on a control should keep switch behavior instead of search focus."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                filter_input = screen.query_one("#thread-filter", Input)
                relative_switch = screen.query_one("#thread-relative-time", Checkbox)

                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.pause()
                assert relative_switch.has_focus
                assert relative_switch.value is True

                await pilot.press("space")
                await pilot.pause()

                assert relative_switch.has_focus
                assert not filter_input.has_focus
                assert filter_input.value == ""
                assert relative_switch.value is False


class TestThreadSelectorDownWrap:
    """Tests for wrapping from bottom to top."""

    async def test_down_arrow_wraps_from_bottom(self) -> None:
        """Down arrow at last index should wrap to first thread."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                count = len(screen._threads)

                # Navigate to the last item
                for _ in range(count - 1):
                    await pilot.press("down")
                    await pilot.pause()
                assert screen._selected_index == count - 1

                # One more down should wrap to 0
                await pilot.press("down")
                await pilot.pause()
                assert screen._selected_index == 0


class TestThreadSelectorPageNavigation:
    """Tests for pageup/pagedown navigation."""

    async def test_pagedown_moves_selection(self) -> None:
        """Pagedown should move selection forward."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                await pilot.press("pagedown")
                await pilot.pause()

                # Should move forward (clamped to last item with 3 threads)
                assert screen._selected_index == len(MOCK_THREADS) - 1

    async def test_pageup_at_top_is_noop(self) -> None:
        """Pageup at index 0 should be a no-op."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._selected_index == 0

                await pilot.press("pageup")
                await pilot.pause()
                assert screen._selected_index == 0


class _ThreadSelectorScopedTestApp(App):
    """Test app that mounts the picker with an explicit `filter_cwd`."""

    def __init__(self, filter_cwd: str | None) -> None:
        super().__init__()
        self._filter_cwd = filter_cwd
        self.result: str | None = None
        self.dismissed = False

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def show_selector(self) -> None:
        """Mount the selector with a caller-supplied cwd filter."""

        def handle_result(result: str | None) -> None:
            self.result = result
            self.dismissed = True

        screen = ThreadSelectorScreen(current_thread=None, filter_cwd=self._filter_cwd)
        self.push_screen(screen, handle_result)


class TestThreadSelectorScopePersistedDefault:
    """Tests that the picker honors the persisted scope preference on open."""

    def test_persisted_all_scope_starts_unfiltered(self) -> None:
        """A persisted scope of "all" should start the picker with no cwd filter."""
        from deepagents_code.model_config import THREAD_COLUMN_DEFAULTS, ThreadConfig

        with patch(
            "deepagents_code.model_config.load_thread_config",
            return_value=ThreadConfig(
                columns=dict(THREAD_COLUMN_DEFAULTS),
                relative_time=True,
                sort_order="updated_at",
                scope="all",
            ),
        ):
            screen = ThreadSelectorScreen(current_thread=None)
        assert screen._filter_cwd is None

    def test_persisted_cwd_scope_starts_filtered(self) -> None:
        """A persisted scope of "cwd" should scope the picker to the cwd."""
        from deepagents_code.model_config import THREAD_COLUMN_DEFAULTS, ThreadConfig

        with (
            patch(
                "deepagents_code.model_config.load_thread_config",
                return_value=ThreadConfig(
                    columns=dict(THREAD_COLUMN_DEFAULTS),
                    relative_time=True,
                    sort_order="updated_at",
                    scope="cwd",
                ),
            ),
            patch(
                "deepagents_code.tui.widgets.thread_selector._safe_cwd_string",
                return_value="/home/user/project-a",
            ),
        ):
            screen = ThreadSelectorScreen(current_thread=None)
        assert screen._filter_cwd == "/home/user/project-a"


class TestThreadSelectorScopeSelect:
    """Tests for the cwd scope `Select` in the Options panel."""

    async def test_enter_opens_scope_select_without_resuming_thread(self) -> None:
        """Enter on the focused scope control should open its dropdown."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                scope_select = screen.query_one("#thread-scope-select", Select)

                await pilot.press("tab")
                await pilot.pause()
                assert scope_select.has_focus

                await pilot.press("enter")
                await pilot.pause()

                assert scope_select.expanded
                assert not app.dismissed

    async def test_tab_keys_move_open_scope_select_highlight(self) -> None:
        """Tab and Shift+Tab should move the dropdown highlight while open."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                filter_input = screen.query_one("#thread-filter", Input)
                scope_select = screen.query_one("#thread-scope-select", Select)
                sort_select = screen.query_one("#thread-sort-select", Select)

                await pilot.press("tab")
                await pilot.press("enter")
                await pilot.pause()
                assert scope_select.expanded
                overlay = scope_select.query_one(ContainedSelectOverlay)
                assert overlay.highlighted == 1

                await pilot.press("shift+tab")
                await pilot.pause()
                assert scope_select.expanded
                assert overlay.highlighted == 0
                assert not filter_input.has_focus
                assert not app.dismissed

                await pilot.press("tab")
                await pilot.pause()
                assert scope_select.expanded
                assert overlay.highlighted == 1
                assert not sort_select.has_focus
                assert not app.dismissed

    async def test_arrow_keys_move_open_scope_select_not_thread_list(self) -> None:
        """Normal dropdown navigation should not move the thread highlight."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                scope_select = screen.query_one("#thread-scope-select", Select)

                screen._selected_index = 1
                await pilot.press("tab")
                await pilot.press("enter")
                await pilot.pause()
                assert scope_select.expanded
                overlay = scope_select.query_one(ContainedSelectOverlay)
                assert overlay.highlighted == 1

                await pilot.press("up")
                await pilot.pause()
                assert overlay.highlighted == 0
                assert screen._selected_index == 1

                await pilot.press("down")
                await pilot.pause()
                assert overlay.highlighted == 1
                assert screen._selected_index == 1

                await pilot.press("pageup")
                await pilot.pause()
                assert overlay.highlighted == 0
                assert screen._selected_index == 1

                await pilot.press("pagedown")
                await pilot.pause()
                assert overlay.highlighted == 1
                assert screen._selected_index == 1
                assert not app.dismissed

    async def test_escape_closes_open_scope_select_without_dismissing(self) -> None:
        """Esc should close the dropdown before it cancels the selector."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                scope_select = screen.query_one("#thread-scope-select", Select)

                await pilot.press("tab")
                await pilot.press("enter")
                await pilot.pause()
                assert scope_select.expanded

                await pilot.press("escape")
                await pilot.pause()

                assert not scope_select.expanded
                assert scope_select.has_focus
                assert not app.dismissed

    async def test_enter_selects_open_scope_select_without_resuming(self) -> None:
        """Enter should choose the highlighted dropdown option while open."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                scope_select = screen.query_one("#thread-scope-select", Select)

                await pilot.press("tab")
                await pilot.press("enter")
                await pilot.pause()
                assert scope_select.expanded

                await pilot.press("enter")
                await pilot.pause()

                assert not scope_select.expanded
                assert scope_select.has_focus
                assert not app.dismissed

    async def test_select_toggle_requeries_with_new_cwd(self) -> None:
        """Switching the scope dropdown reloads threads with the new cwd kwarg."""
        starting_cwd = "/home/user/project-a"
        mock_list = AsyncMock(return_value=MOCK_THREADS)

        with (
            patch("deepagents_code.sessions.list_threads", mock_list),
            _patch_columns(),
            patch(
                "deepagents_code.tui.widgets.thread_selector._safe_cwd_string",
                return_value=starting_cwd,
            ),
        ):
            app = _ThreadSelectorScopedTestApp(filter_cwd=starting_cwd)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._filter_cwd == starting_cwd

                scope_select = screen.query_one("#thread-scope-select", Select)
                # Sanity: initial query carried the cwd filter.
                assert mock_list.await_count >= 1
                initial_kwargs = [c.kwargs for c in mock_list.await_args_list]
                assert any(kw.get("cwd") == starting_cwd for kw in initial_kwargs), (
                    f"expected starting cwd, got {initial_kwargs}"
                )

                mock_list.reset_mock()
                scope_select.value = "all"
                await pilot.pause()
                await pilot.pause()
                assert screen._filter_cwd is None
                toggled_kwargs = [c.kwargs for c in mock_list.await_args_list]
                assert any(kw.get("cwd") is None for kw in toggled_kwargs), (
                    f"expected re-query with cwd=None, got {toggled_kwargs}"
                )

                mock_list.reset_mock()
                scope_select.value = "cwd"
                await pilot.pause()
                await pilot.pause()
                assert screen._filter_cwd == starting_cwd
                back_kwargs = [c.kwargs for c in mock_list.await_args_list]
                assert any(kw.get("cwd") == starting_cwd for kw in back_kwargs), (
                    f"expected re-query with cwd={starting_cwd!r}, got {back_kwargs}"
                )

    async def test_select_same_value_does_not_requery(self) -> None:
        """Setting the dropdown to its current value is a no-op."""
        mock_list = AsyncMock(return_value=MOCK_THREADS)
        with (
            patch("deepagents_code.sessions.list_threads", mock_list),
            _patch_columns(),
        ):
            app = _ThreadSelectorScopedTestApp(filter_cwd=None)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                scope_select = screen.query_one("#thread-scope-select", Select)
                mock_list.reset_mock()
                # Re-setting to "all" (the active value) must not re-query.
                scope_select.value = "all"
                await pilot.pause()
                mock_list.assert_not_awaited()

    async def test_scope_change_persists_preference(self) -> None:
        """Switching the scope dropdown should persist the new preference."""
        starting_cwd = "/home/user/project-a"
        mock_list = AsyncMock(return_value=MOCK_THREADS)
        mock_save = MagicMock(return_value=True)

        with (
            patch("deepagents_code.sessions.list_threads", mock_list),
            _patch_columns(),
            patch(
                "deepagents_code.tui.widgets.thread_selector._safe_cwd_string",
                return_value=starting_cwd,
            ),
            patch(
                "deepagents_code.model_config.save_thread_scope",
                mock_save,
            ),
        ):
            app = _ThreadSelectorScopedTestApp(filter_cwd=starting_cwd)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                scope_select = screen.query_one("#thread-scope-select", Select)

                scope_select.value = "all"
                await pilot.pause()
                await pilot.pause()
                mock_save.assert_any_call("all")

                scope_select.value = "cwd"
                await pilot.pause()
                await pilot.pause()
                mock_save.assert_any_call("cwd")

    async def test_scope_persists_even_when_cwd_unresolvable(self) -> None:
        """Selecting "Current directory" persists "cwd" even if the cwd is gone.

        When `_safe_cwd_string()` returns `None`, the resolved filter stays
        `None` and the reload short-circuits, but the user's explicit "cwd"
        choice must still be persisted. This pins the intentional ordering of
        the persist call ahead of the `new_cwd == self._filter_cwd` early return.
        """
        mock_list = AsyncMock(return_value=MOCK_THREADS)
        mock_save = MagicMock(return_value=True)

        with (
            patch("deepagents_code.sessions.list_threads", mock_list),
            _patch_columns(),
            patch(
                "deepagents_code.tui.widgets.thread_selector._safe_cwd_string",
                return_value=None,
            ),
            patch(
                "deepagents_code.model_config.save_thread_scope",
                mock_save,
            ),
        ):
            # Start unfiltered ("all"); the cwd is unresolvable below.
            app = _ThreadSelectorScopedTestApp(filter_cwd=None)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._filter_cwd is None
                scope_select = screen.query_one("#thread-scope-select", Select)

                scope_select.value = "cwd"
                await pilot.pause()
                await pilot.pause()

                mock_save.assert_any_call("cwd")
                # Filter stays unfiltered (cwd unresolvable), yet the preference
                # was still persisted before the early return fired.
                assert screen._filter_cwd is None

    async def test_scope_save_failure_notifies(self) -> None:
        """A failed scope save should surface a warning notification."""
        starting_cwd = "/home/user/project-a"
        mock_list = AsyncMock(return_value=MOCK_THREADS)
        mock_save = MagicMock(return_value=False)

        with (
            patch("deepagents_code.sessions.list_threads", mock_list),
            _patch_columns(),
            patch(
                "deepagents_code.tui.widgets.thread_selector._safe_cwd_string",
                return_value=starting_cwd,
            ),
            patch(
                "deepagents_code.model_config.save_thread_scope",
                mock_save,
            ),
        ):
            app = _ThreadSelectorScopedTestApp(filter_cwd=starting_cwd)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                scope_select = screen.query_one("#thread-scope-select", Select)

                with patch.object(app, "notify") as mock_notify:
                    scope_select.value = "all"
                    await app.workers.wait_for_complete()
                    await pilot.pause()

                mock_notify.assert_any_call(
                    "Could not save scope preference", severity="warning"
                )


class TestThreadSelectorClickHandling:
    """Tests for mouse click handling."""

    async def test_click_selects_thread(self) -> None:
        """Clicking a thread option should select and dismiss."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                from deepagents_code.tui.widgets.thread_selector import ThreadOption

                assert len(screen._option_widgets) > 1, (
                    "Expected option widgets to be built"
                )
                second = screen._option_widgets[1]
                second.post_message(
                    ThreadOption.Clicked(second.thread_id, second.index)
                )
                await pilot.pause()

                assert app.dismissed is True
                assert app.result == "def67890"


_WEBBROWSER_OPEN = "deepagents_code.tui.widgets._links.webbrowser.open"


class TestThreadSelectorOnClickOpensLink:
    """Tests for `ThreadSelectorScreen.on_click` opening Rich-style hyperlinks."""

    def test_click_on_link_opens_browser(self) -> None:
        """Clicking a Rich link should call `webbrowser.open`."""
        screen = ThreadSelectorScreen(current_thread=None)
        event = MagicMock()
        event.style = Style(link="https://example.com")

        with patch(_WEBBROWSER_OPEN) as mock_open:
            screen.on_click(event)

        mock_open.assert_called_once_with("https://example.com")
        event.stop.assert_called_once()

    def test_click_without_link_is_noop(self) -> None:
        """Clicking on non-link text should not open the browser."""
        screen = ThreadSelectorScreen(current_thread=None)
        event = MagicMock()
        event.style = Style()

        with patch(_WEBBROWSER_OPEN) as mock_open:
            screen.on_click(event)

        mock_open.assert_not_called()
        event.stop.assert_not_called()

    def test_click_with_browser_error_is_graceful(self) -> None:
        """Browser failure should not crash the widget."""
        screen = ThreadSelectorScreen(current_thread=None)
        event = MagicMock()
        event.style = Style(link="https://example.com")

        with patch(_WEBBROWSER_OPEN, side_effect=OSError("no display")):
            screen.on_click(event)  # should not raise

        event.stop.assert_not_called()


class TestThreadSelectorPointer:
    """Tests for `ThreadSelectorScreen` link hover affordance."""

    def test_hover_over_link_uses_pointer(self) -> None:
        """Hovering a Rich-style link sets a pointer cursor."""
        screen = ThreadSelectorScreen(current_thread=None)
        event = MagicMock()
        event.style = Style(link="https://example.com")
        screen.on_mouse_move(event)
        assert screen.styles.pointer == "pointer"

    def test_hover_off_link_uses_default(self) -> None:
        """Hovering non-link text keeps the default cursor."""
        screen = ThreadSelectorScreen(current_thread=None)
        event = MagicMock()
        event.style = Style()
        screen.on_mouse_move(event)
        assert screen.styles.pointer == "default"

    def test_leave_resets_pointer(self) -> None:
        """Leaving the selector resets the cursor to default."""
        screen = ThreadSelectorScreen(current_thread=None)
        link_event = MagicMock()
        link_event.style = Style(link="https://example.com")
        screen.on_mouse_move(link_event)
        screen.on_leave()
        assert screen.styles.pointer == "default"


class TestThreadSelectorBuildTitle:
    """Tests for _build_title with clickable thread ID."""

    def test_no_current_thread(self) -> None:
        """Title without current thread should be plain text."""
        screen = ThreadSelectorScreen(current_thread=None)
        assert screen._build_title() == "Select Thread"

    def test_current_thread_no_url(self) -> None:
        """Title with current thread but no URL should be a plain string."""
        screen = ThreadSelectorScreen(current_thread="abc12345")
        title = screen._build_title()
        assert isinstance(title, str)
        assert "abc12345" in title

    def test_current_thread_with_url(self) -> None:
        """Title with a LangSmith URL should produce Content with a link."""
        from textual.color import Color as TColor
        from textual.content import Content
        from textual.style import Style as TStyle

        screen = ThreadSelectorScreen(current_thread="abc12345")
        title = screen._build_title(
            thread_url="https://smith.langchain.com/p/t/abc12345"
        )
        assert isinstance(title, Content)
        assert "abc12345" in title.plain

        spans = [
            s for s in title._spans if isinstance(s.style, TStyle) and s.style.link
        ]
        assert len(spans) > 0
        style = spans[0].style
        assert isinstance(style, TStyle)
        from deepagents_code.theme import DARK_COLORS

        assert style.foreground == TColor.parse(DARK_COLORS.primary)

    async def test_title_widget_has_id(self) -> None:
        """Title widget should be queryable by ID for URL updates."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp(current_thread="abc12345")
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                title_widget = screen.query_one("#thread-title", Static)
                assert title_widget is not None


class TestFetchThreadUrl:
    """Tests for _fetch_thread_url background worker."""

    async def test_successful_url_updates_title(self) -> None:
        """Background worker should update the title with a clickable link."""
        from textual.content import Content

        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.tui.widgets.thread_selector.build_langsmith_thread_url",
                return_value="https://smith.langchain.com/p/t/abc12345",
            ),
        ):
            app = ThreadSelectorTestApp(current_thread="abc12345")
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                title_widget = screen.query_one("#thread-title", Static)
                content = title_widget.content
                assert isinstance(content, Content)
                assert "abc12345" in content.plain

    async def test_url_resolves_before_thread_load_completes(self) -> None:
        """Header link should appear without waiting for the DB load to finish."""
        from textual.content import Content
        from textual.style import Style as TStyle

        load_gate = asyncio.Event()

        async def _blocking_list_threads(*_args: Any, **_kwargs: Any) -> list[Any]:
            await load_gate.wait()
            return list(MOCK_THREADS)

        with (
            patch(
                "deepagents_code.sessions.list_threads",
                new=_blocking_list_threads,
            ),
            patch(
                "deepagents_code.tui.widgets.thread_selector.build_langsmith_thread_url",
                return_value="https://smith.langchain.com/p/t/abc12345",
            ),
        ):
            app = ThreadSelectorTestApp(current_thread="abc12345")
            async with app.run_test() as pilot:
                app.show_selector()
                # Pump a few cycles; the thread load is still blocked on the gate.
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert not screen._disk_load_complete
                title_widget = screen.query_one("#thread-title", Static)
                content = title_widget.content
                assert isinstance(content, Content)
                assert "abc12345" in content.plain
                # The thread id must be an actual hyperlink, not just present as
                # text: assert a link-bearing span exists.
                link_spans = [
                    s
                    for s in content._spans
                    if isinstance(s.style, TStyle) and s.style.link
                ]
                assert len(link_spans) > 0

                # Release the gated load and let the worker finish so it drains
                # deterministically instead of racing test teardown.
                load_gate.set()
                await pilot.pause()

    async def test_url_resolves_on_mount_with_cached_rows(self) -> None:
        """Cached-rows path should also resolve the header link at mount time.

        The fix unified both branches to resolve the URL on mount; this locks in
        that the cached path (`_has_initial_threads` is `True`) keeps that
        behavior alongside the no-cache path.
        """
        from textual.content import Content
        from textual.style import Style as TStyle

        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.tui.widgets.thread_selector.build_langsmith_thread_url",
                return_value="https://smith.langchain.com/p/t/abc12345",
            ),
        ):
            app = ThreadSelectorTestApp(current_thread="abc12345")
            async with app.run_test() as pilot:
                screen = ThreadSelectorScreen(
                    current_thread="abc12345",
                    initial_threads=list(MOCK_THREADS),
                    filter_cwd=None,
                )
                app.push_screen(screen)
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()

                assert screen._has_initial_threads
                title_widget = screen.query_one("#thread-title", Static)
                content = title_widget.content
                assert isinstance(content, Content)
                link_spans = [
                    s
                    for s in content._spans
                    if isinstance(s.style, TStyle) and s.style.link
                ]
                assert len(link_spans) > 0

    async def test_no_url_worker_when_no_current_thread(self) -> None:
        """With no current thread, the title stays plain and no URL is resolved."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.tui.widgets.thread_selector.build_langsmith_thread_url",
            ) as mock_build_url,
        ):
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                title_widget = screen.query_one("#thread-title", Static)
                # With no current thread the title stays a plain string (no
                # hyperlink), and the URL builder is never invoked.
                assert title_widget.content == "Select Thread"
                mock_build_url.assert_not_called()

    async def test_timeout_leaves_title_unchanged(self) -> None:
        """Timeout during URL resolution should not crash or change the title."""
        import time

        def _blocking(_tid: str) -> str:
            time.sleep(0.1)
            return "https://example.com"

        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.tui.widgets.thread_selector._URL_FETCH_TIMEOUT",
                0.01,
            ),
            patch(
                "deepagents_code.tui.widgets.thread_selector.build_langsmith_thread_url",
                side_effect=_blocking,
            ),
        ):
            app = ThreadSelectorTestApp(current_thread="abc12345")
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                title_widget = screen.query_one("#thread-title", Static)
                assert isinstance(title_widget.content, str)

    async def test_oserror_leaves_title_unchanged(self) -> None:
        """OSError during URL resolution should not crash or change the title."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.tui.widgets.thread_selector.build_langsmith_thread_url",
                side_effect=OSError("network failure"),
            ),
        ):
            app = ThreadSelectorTestApp(current_thread="abc12345")
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                title_widget = screen.query_one("#thread-title", Static)
                assert isinstance(title_widget.content, str)

    async def test_unexpected_exception_leaves_title_unchanged(self) -> None:
        """Unexpected exception should not crash the thread selector."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.tui.widgets.thread_selector.build_langsmith_thread_url",
                side_effect=AttributeError("SDK changed"),
            ),
        ):
            app = ThreadSelectorTestApp(current_thread="abc12345")
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                title_widget = screen.query_one("#thread-title", Static)
                assert isinstance(title_widget.content, str)

    async def test_none_url_leaves_title_unchanged(self) -> None:
        """When build returns None the title should remain a plain string."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.tui.widgets.thread_selector.build_langsmith_thread_url",
                return_value=None,
            ),
        ):
            app = ThreadSelectorTestApp(current_thread="abc12345")
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                title_widget = screen.query_one("#thread-title", Static)
                content = title_widget.content
                assert isinstance(content, str)
                assert "abc12345" in content


class TestThreadSelectorColumnHeader:
    """Tests for the anchored column header."""

    def test_header_contains_default_column_names(self) -> None:
        """Column header labels should contain visible column names."""
        from deepagents_code.tui.widgets.thread_selector import _format_header_label

        assert "Created" in _format_header_label("created_at")
        assert "Msgs" in _format_header_label("messages")
        assert "Updated" in _format_header_label("updated_at")
        assert "Prompt" in _format_header_label("initial_prompt")

    async def test_header_widget_is_mounted(self) -> None:
        """Column header widget should be present in the mounted screen."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                screen.query_one(".thread-list-header", Horizontal)

    async def test_header_stays_outside_scroll(self) -> None:
        """Header should be outside VerticalScroll (anchored, not scrollable)."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                header = screen.query_one(".thread-list-header", Horizontal)
                assert isinstance(header.parent, Vertical)

    async def test_timestamp_columns_share_width_with_rows(self) -> None:
        """Timestamp header cells should use the same width as row cells."""
        from deepagents_code.tui.widgets.thread_selector import (
            _format_column_value,
            _format_header_label,
        )

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                header = screen.query_one("#thread-header", Horizontal)
                row = screen._option_widgets[0]

                for key in ("created_at", "updated_at"):
                    header_cell = header.query_one(f".thread-cell-{key}", Static)
                    row_cell = row.query_one(f".thread-cell-{key}", Static)
                    expected_width = (
                        max(
                            cell_len(_format_header_label(key)),
                            *(
                                cell_len(
                                    _format_column_value(
                                        thread,
                                        key,
                                        relative_time=screen._relative_time,
                                    )
                                )
                                for thread in screen._filtered_threads
                            ),
                        )
                        + 1
                    )
                    assert header_cell.size.width == row_cell.size.width
                    assert (
                        _style_scalar_value(header_cell.styles.width) == expected_width
                    )
                    assert _style_scalar_value(row_cell.styles.width) == expected_width


class TestThreadSelectorPromptOverflow:
    """Tests for prompt-cell overflow handling."""

    async def test_prompt_cell_renders_ellipsis_when_constrained(self) -> None:
        """Prompt cells should use ellipsis instead of hard clipping."""
        columns = {
            "thread_id": False,
            "messages": False,
            "created_at": False,
            "updated_at": True,
            "git_branch": False,
            "cwd": False,
            "initial_prompt": True,
            "agent_name": False,
        }
        thread = ThreadInfo(**MOCK_THREADS[0])
        thread["initial_prompt"] = (
            "This is a very long prompt that should be truncated "
            "with an ellipsis inside the prompt column"
        )
        threads: list[ThreadInfo] = [thread]

        with _patch_list_threads(threads), _patch_columns(columns):
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test(size=(80, 24)) as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                row = screen._option_widgets[0]
                prompt_cell = row.query_one(".thread-cell-initial_prompt", Static)
                rendered = prompt_cell.render_line(0).text.rstrip()

                assert rendered.endswith("…")


class TestThreadSelectorBranchOverflow:
    """Tests for git-branch overflow handling."""

    async def test_branch_cell_renders_ellipsis_when_truncated(self) -> None:
        """Git branch cells should keep the ellipsis visible when clipped."""
        columns = {
            "thread_id": False,
            "messages": False,
            "created_at": False,
            "updated_at": False,
            "git_branch": True,
            "cwd": False,
            "initial_prompt": False,
            "agent_name": False,
        }
        thread = ThreadInfo(**MOCK_THREADS[0])
        thread["git_branch"] = "feature/very-long-branch-name"
        threads: list[ThreadInfo] = [thread]

        with _patch_list_threads(threads), _patch_columns(columns):
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test(size=(80, 24)) as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                row = screen._option_widgets[0]
                branch_cell = row.query_one(".thread-cell-git_branch", Static)
                rendered = branch_cell.render_line(0).text.rstrip()

                assert rendered.endswith("…")


class TestThreadSelectorAutoWidthColumns:
    """Tests for shared widths on auto-sized columns."""

    async def test_agent_name_column_uses_shared_width_capped_at_twelve(self) -> None:
        """Agent column should size to visible content up to the 12-char cap."""
        from deepagents_code.tui.widgets.thread_selector import (
            _format_column_value,
            _format_header_label,
        )

        columns = {
            "thread_id": False,
            "messages": False,
            "created_at": False,
            "updated_at": False,
            "git_branch": False,
            "cwd": False,
            "initial_prompt": False,
            "agent_name": True,
        }

        with _patch_list_threads(), _patch_columns(columns):
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                header = screen.query_one("#thread-header", Horizontal)
                row = screen._option_widgets[0]
                header_cell = header.query_one(".thread-cell-agent_name", Static)
                row_cell = row.query_one(".thread-cell-agent_name", Static)
                expected_width = (
                    max(
                        cell_len(_format_header_label("agent_name")),
                        *(
                            cell_len(
                                _format_column_value(
                                    thread,
                                    "agent_name",
                                    relative_time=screen._relative_time,
                                )
                            )
                            for thread in screen._filtered_threads
                        ),
                    )
                    + 1
                )

                assert header_cell.size.width == row_cell.size.width
                assert _style_scalar_value(header_cell.styles.width) == expected_width
                assert _style_scalar_value(row_cell.styles.width) == expected_width
                assert (
                    _style_scalar_value(header_cell.styles.min_width) == expected_width
                )
                assert _style_scalar_value(row_cell.styles.min_width) == expected_width
                assert expected_width <= 13


class TestThreadSelectorErrorHandling:
    """Tests for error handling when loading threads fails."""

    async def test_list_threads_error_still_dismissable(self) -> None:
        """Database error should not crash; Escape still works."""
        with patch(
            "deepagents_code.sessions.list_threads",
            new_callable=AsyncMock,
            side_effect=OSError("database is locked"),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._threads) == 0

                assert len(screen._option_widgets) == 0
                # A failed load is still a completed load: the flag must flip so
                # the picker never strands on the "Loading threads..." placeholder.
                assert screen._disk_load_complete is True

                await pilot.press("escape")
                await pilot.pause()

                assert app.dismissed is True
                assert app.result is None

    async def test_unexpected_load_error_surfaces_and_completes(self) -> None:
        """A non-OSError/sqlite3 error must surface and not strand the UI."""
        with patch(
            "deepagents_code.sessions.list_threads",
            new_callable=AsyncMock,
            side_effect=ValueError("malformed row"),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                # The catch-all handler marks the load complete and replaces the
                # loading placeholder with the error message instead of leaving a
                # perpetual "Loading threads..." spinner.
                assert screen._disk_load_complete is True
                with pytest.raises(NoMatches):
                    screen.query_one("#thread-loading", Static)

                await pilot.press("escape")
                await pilot.pause()

                assert app.dismissed is True
                assert app.result is None


class TestThreadSelectorLimit:
    """Tests for thread limit via get_thread_limit()."""

    async def test_custom_limit_is_forwarded(self) -> None:
        """get_thread_limit() return value should be forwarded to list_threads."""
        with (
            patch(
                "deepagents_code.sessions.get_thread_limit",
                return_value=5,
            ),
            _patch_list_threads() as mock_lt,
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                mock_lt.assert_awaited_once()
                call_kwargs = mock_lt.await_args.kwargs
                assert call_kwargs["limit"] == 5
                assert call_kwargs["include_message_count"] is False
                assert call_kwargs["sort_by"] in {"updated", "created"}

    async def test_checkpoint_details_are_loaded_for_initial_render(self) -> None:
        """Visible checkpoint fields should be loaded before first non-cached render."""
        threads_without_details: list[ThreadInfo] = [
            {
                "thread_id": "abc12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-15T10:30:00",
            }
        ]

        async def _populate(
            threads: list[ThreadInfo],
            *,
            include_message_count: bool,
            include_initial_prompt: bool,
        ) -> list[ThreadInfo]:
            await asyncio.sleep(0)
            assert include_message_count is True
            assert include_initial_prompt is True
            for thread in threads:
                thread["message_count"] = 9
                thread["initial_prompt"] = "loaded prompt"
            return threads

        with (
            patch(
                "deepagents_code.sessions.list_threads",
                new_callable=AsyncMock,
                return_value=threads_without_details,
            ) as mock_lt,
            _patch_columns(),
            patch(
                "deepagents_code.sessions.populate_thread_checkpoint_details",
                new_callable=AsyncMock,
                side_effect=_populate,
            ) as mock_populate,
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                for _ in range(10):
                    if mock_populate.await_count >= 1:
                        break
                    await pilot.pause(0.05)

                mock_lt.assert_awaited_once_with(
                    limit=20,
                    include_message_count=False,
                    sort_by="updated",
                    cwd=None,
                )
                mock_populate.assert_awaited_once()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._option_widgets) == 1
                assert screen._threads[0]["message_count"] == 9
                assert screen._threads[0]["initial_prompt"] == "loaded prompt"

    async def test_cached_counts_skip_background_population(self) -> None:
        """If cache fills counts before paint, background populate is skipped."""
        threads_without_counts: list[ThreadInfo] = [
            {
                "thread_id": "abc12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-15T10:30:00",
                "initial_prompt": "prompt",
                "latest_checkpoint_id": "cp_1",
            }
        ]

        def _apply_cached(threads: list[ThreadInfo]) -> int:
            threads[0]["message_count"] = 11
            return 1

        with (
            patch(
                "deepagents_code.sessions.list_threads",
                new_callable=AsyncMock,
                return_value=threads_without_counts,
            ),
            patch(
                "deepagents_code.sessions.apply_cached_thread_message_counts",
                side_effect=_apply_cached,
            ) as mock_apply_cached,
            patch(
                "deepagents_code.sessions.populate_thread_checkpoint_details",
                new_callable=AsyncMock,
            ) as mock_populate,
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()
                await pilot.pause(0.1)

                mock_apply_cached.assert_called_once()
                mock_populate.assert_not_awaited()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._threads[0]["message_count"] == 11


class TestThreadSelectorCheckpointDetailErrors:
    """Tests for thread selector checkpoint-detail load error handling."""

    async def test_unexpected_checkpoint_detail_error_logs_warning(self) -> None:
        """Unexpected checkpoint-load errors should be visible at warning level."""
        screen = ThreadSelectorScreen(
            initial_threads=[
                {
                    "thread_id": "abc12345",
                    "agent_name": "my-agent",
                    "updated_at": "2025-01-15T10:30:00",
                }
            ],
            filter_cwd=None,
        )

        with (
            patch(
                "deepagents_code.sessions.populate_thread_checkpoint_details",
                new_callable=AsyncMock,
                side_effect=RuntimeError("unexpected type mismatch"),
            ),
            patch(
                "deepagents_code.tui.widgets.thread_selector.logger.warning"
            ) as mock_warning,
        ):
            await screen._load_checkpoint_details()

        mock_warning.assert_called_once()


class TestThreadSelectorPrefetchedRows:
    """Tests for rendering with prefetched rows from startup cache."""

    async def test_prefetched_rows_render_without_loading_state(self) -> None:
        """Prefetched rows should render immediately, then refresh from SQLite."""
        prefetched: list[ThreadInfo] = [
            {
                "thread_id": "abc12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-15T10:30:00",
                "message_count": 5,
            }
        ]
        refreshed: list[ThreadInfo] = [
            {
                "thread_id": "new12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-16T12:00:00",
                "message_count": 6,
            },
            {
                "thread_id": "abc12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-15T10:30:00",
                "message_count": 5,
            },
        ]
        app = ThreadSelectorTestApp(current_thread="abc12345")

        gate = asyncio.Event()

        async def _list_threads(*_args: object, **_kwargs: object) -> list[ThreadInfo]:
            await gate.wait()
            return refreshed

        with patch(
            "deepagents_code.sessions.list_threads",
            new_callable=AsyncMock,
            side_effect=_list_threads,
        ) as mock_list_threads:
            async with app.run_test() as pilot:
                app.push_screen(
                    ThreadSelectorScreen(
                        current_thread="abc12345",
                        thread_limit=20,
                        initial_threads=prefetched,
                        filter_cwd=None,
                    )
                )
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._option_widgets) == 1
                with pytest.raises(NoMatches):
                    screen.query_one("#thread-loading", Static)

                gate.set()

                for _ in range(10):
                    if mock_list_threads.await_count >= 1 and len(screen._threads) == 2:
                        break
                    await pilot.pause(0.05)

                mock_list_threads.assert_awaited_once()
                assert mock_list_threads.await_args is not None
                kw = mock_list_threads.await_args.kwargs
                assert kw["limit"] == 20
                assert kw["include_message_count"] is False
                assert kw["sort_by"] in {"updated", "created"}
                assert len(screen._threads) == 2
                assert screen._threads[0]["thread_id"] == "new12345"

    async def test_prefetched_prompt_is_preserved_during_refresh(self) -> None:
        """Refreshing prefetched rows should not blank the prompt column first."""
        prefetched: list[ThreadInfo] = [
            {
                "thread_id": "abc12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-15T10:30:00",
                "latest_checkpoint_id": "cp_1",
                "initial_prompt": "cached prompt",
            }
        ]
        refreshed: list[ThreadInfo] = [
            {
                "thread_id": "abc12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-15T10:30:00",
                "latest_checkpoint_id": "cp_1",
            }
        ]

        from deepagents_code import sessions

        sessions._initial_prompt_cache.clear()
        sessions._initial_prompt_cache["abc12345"] = ("cp_1", "cached prompt")
        try:
            with patch(
                "deepagents_code.sessions.list_threads",
                new_callable=AsyncMock,
                return_value=refreshed,
            ):
                app = ThreadSelectorTestApp(current_thread="abc12345")
                async with app.run_test() as pilot:
                    app.push_screen(
                        ThreadSelectorScreen(
                            current_thread="abc12345",
                            thread_limit=20,
                            initial_threads=prefetched,
                            filter_cwd=None,
                        )
                    )
                    await pilot.pause()
                    await pilot.pause(0.1)

                    screen = app.screen
                    assert isinstance(screen, ThreadSelectorScreen)
                    assert screen._threads[0]["initial_prompt"] == "cached prompt"
        finally:
            sessions._initial_prompt_cache.clear()

    async def test_empty_prefetched_snapshot_still_refreshes(self) -> None:
        """An empty cached snapshot should still hydrate from SQLite in background."""
        refreshed: list[ThreadInfo] = [
            {
                "thread_id": "new12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-16T12:00:00",
                "message_count": 6,
            }
        ]
        app = ThreadSelectorTestApp(current_thread="abc12345")
        with patch(
            "deepagents_code.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=refreshed,
        ) as mock_list_threads:
            async with app.run_test() as pilot:
                app.push_screen(
                    ThreadSelectorScreen(
                        current_thread="abc12345",
                        thread_limit=20,
                        initial_threads=[],
                        filter_cwd=None,
                    )
                )
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                with pytest.raises(NoMatches):
                    screen.query_one("#thread-loading", Static)

                for _ in range(10):
                    if mock_list_threads.await_count >= 1 and len(screen._threads) == 1:
                        break
                    await pilot.pause(0.05)

                mock_list_threads.assert_awaited_once()
                assert mock_list_threads.await_args is not None
                kw = mock_list_threads.await_args.kwargs
                assert kw["limit"] == 20
                assert kw["include_message_count"] is False
                assert kw["sort_by"] in {"updated", "created"}
                assert len(screen._threads) == 1
                assert screen._threads[0]["thread_id"] == "new12345"

    async def test_empty_snapshot_shows_loading_until_disk_load_completes(
        self,
    ) -> None:
        """An empty snapshot must not claim "No threads found" while loading."""
        refreshed: list[ThreadInfo] = [
            {
                "thread_id": "new12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-16T12:00:00",
                "message_count": 6,
            }
        ]
        app = ThreadSelectorTestApp(current_thread="abc12345")

        gate = asyncio.Event()

        async def _list_threads(*_args: object, **_kwargs: object) -> list[ThreadInfo]:
            await gate.wait()
            return refreshed

        with patch(
            "deepagents_code.sessions.list_threads",
            new_callable=AsyncMock,
            side_effect=_list_threads,
        ):
            async with app.run_test() as pilot:
                app.push_screen(
                    ThreadSelectorScreen(
                        current_thread="abc12345",
                        thread_limit=20,
                        initial_threads=[],
                        filter_cwd=None,
                    )
                )
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                # While the disk load is in flight, show the loading placeholder
                # rather than "No threads found".
                assert not screen._disk_load_complete
                screen.query_one("#thread-loading", Static)
                assert not screen._option_widgets

                gate.set()

                for _ in range(10):
                    if len(screen._threads) == 1:
                        break
                    await pilot.pause(0.05)

                assert screen._disk_load_complete
                assert len(screen._option_widgets) == 1

    async def test_empty_snapshot_resolves_to_no_threads_found(self) -> None:
        """An empty disk load must flip the placeholder to "No threads found"."""
        app = ThreadSelectorTestApp(current_thread="abc12345")

        gate = asyncio.Event()

        async def _list_threads(*_args: object, **_kwargs: object) -> list[ThreadInfo]:
            await gate.wait()
            return []

        with patch(
            "deepagents_code.sessions.list_threads",
            new_callable=AsyncMock,
            side_effect=_list_threads,
        ):
            async with app.run_test() as pilot:
                app.push_screen(
                    ThreadSelectorScreen(
                        current_thread="abc12345",
                        thread_limit=20,
                        initial_threads=[],
                        filter_cwd=None,
                    )
                )
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert not screen._disk_load_complete
                screen.query_one("#thread-loading", Static)

                gate.set()

                for _ in range(10):
                    if screen._disk_load_complete:
                        break
                    await pilot.pause(0.05)

                # Once the load completes with zero rows, the loading placeholder
                # must resolve to the real empty state, not a perpetual spinner.
                assert screen._disk_load_complete
                assert not screen._option_widgets
                with pytest.raises(NoMatches):
                    screen.query_one("#thread-loading", Static)
                empty = screen.query_one(".thread-empty", Static)
                assert "No threads found" in str(empty.content)

    def test_build_empty_state_reflects_disk_load_flag(self) -> None:
        """`_build_empty_state` chooses its message from `_disk_load_complete`."""
        screen = ThreadSelectorScreen(
            current_thread="abc12345",
            thread_limit=20,
            initial_threads=[],
            filter_cwd=None,
        )

        loading = screen._build_empty_state()
        assert loading.id == "thread-loading"
        assert "Loading threads..." in str(loading.content)

        screen._disk_load_complete = True
        resolved = screen._build_empty_state()
        assert resolved.id is None
        assert "No threads found" in str(resolved.content)


class TestThreadSelectorInitialSortOrder:
    """Tests for initial sort order applied to prefetched rows."""

    async def test_initial_threads_sorted_by_created_at_preference(self) -> None:
        """Prefetched rows should respect the user's sort preference on first render."""
        # Threads ordered by updated_at (default cache order from list_threads)
        prefetched: list[ThreadInfo] = [
            {
                "thread_id": "newer-updated",
                "agent_name": "agent",
                "updated_at": "2025-01-16T12:00:00",
                "created_at": "2025-01-10T08:00:00",
            },
            {
                "thread_id": "older-updated",
                "agent_name": "agent",
                "updated_at": "2025-01-14T08:00:00",
                "created_at": "2025-01-15T10:00:00",
            },
        ]

        import contextlib

        from deepagents_code.model_config import THREAD_COLUMN_DEFAULTS, ThreadConfig

        @contextlib.contextmanager
        def _patch_sort_created() -> Any:  # noqa: ANN401
            with (
                patch(
                    "deepagents_code.model_config.load_thread_columns",
                    return_value=dict(THREAD_COLUMN_DEFAULTS),
                ),
                patch(
                    "deepagents_code.model_config.load_thread_sort_order",
                    return_value="created_at",
                ),
                patch(
                    "deepagents_code.model_config.load_thread_config",
                    return_value=ThreadConfig(
                        columns=dict(THREAD_COLUMN_DEFAULTS),
                        relative_time=True,
                        sort_order="created_at",
                        scope="cwd",
                    ),
                ),
            ):
                yield

        gate = asyncio.Event()

        async def _list_threads(*_a: object, **_kw: object) -> list[ThreadInfo]:
            await gate.wait()
            return prefetched

        with (
            patch(
                "deepagents_code.sessions.list_threads",
                new_callable=AsyncMock,
                side_effect=_list_threads,
            ),
            _patch_sort_created(),
        ):
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test() as pilot:
                app.push_screen(
                    ThreadSelectorScreen(
                        current_thread=None,
                        thread_limit=20,
                        initial_threads=prefetched,
                        filter_cwd=None,
                    )
                )
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                # With sort by created_at, "older-updated" (created 2025-01-15)
                # should come before "newer-updated" (created 2025-01-10)
                assert len(screen._option_widgets) == 2
                assert screen._option_widgets[0].thread_id == "older-updated"
                assert screen._option_widgets[1].thread_id == "newer-updated"

                gate.set()


class TestThreadSelectorSearch:
    """Tests for fuzzy search filtering."""

    async def test_search_filters_threads(self) -> None:
        """Typing in search should filter threads by initial prompt."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._filtered_threads) == 3

                screen._filter_text = "Hello"
                screen._update_filtered_list()
                assert len(screen._filtered_threads) == 1
                assert screen._filtered_threads[0]["thread_id"] == "abc12345"

    def test_empty_search_returns_all(self) -> None:
        """Empty search text should return all threads."""
        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=MOCK_THREADS,
            filter_cwd=None,
        )
        screen._filter_text = ""
        screen._update_filtered_list()
        assert len(screen._filtered_threads) == 3

    def test_search_by_thread_id(self) -> None:
        """Search should match thread IDs."""
        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=MOCK_THREADS,
            filter_cwd=None,
        )
        screen._filter_text = "def678"
        screen._update_filtered_list()
        assert len(screen._filtered_threads) == 1
        assert screen._filtered_threads[0]["thread_id"] == "def67890"

    async def test_typing_in_search_filters_without_crashing(self) -> None:
        """Typing into the live search box should filter by initial prompt."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                for char in "fix":
                    await pilot.press(char)
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._filtered_threads) == 1
                assert screen._filtered_threads[0]["thread_id"] == "def67890"
                assert app.dismissed is False

    def test_equal_match_scores_do_not_crash_sorting(self) -> None:
        """Fuzzy search should handle tied scores without comparing dict rows."""
        threads: list[ThreadInfo] = [
            {
                "thread_id": "thread-a",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "created_at": "2026-03-08T01:00:00+00:00",
                "initial_prompt": "prompt one",
            },
            {
                "thread_id": "thread-b",
                "agent_name": "agent",
                "updated_at": "2026-03-08T03:00:00+00:00",
                "created_at": "2026-03-08T01:30:00+00:00",
                "initial_prompt": "prompt two",
            },
        ]
        screen = ThreadSelectorScreen(
            current_thread=None, initial_threads=threads, filter_cwd=None
        )

        screen._filter_text = "p"
        screen._update_filtered_list()

        assert [thread["thread_id"] for thread in screen._filtered_threads] == [
            "thread-b",
            "thread-a",
        ]

    async def test_filter_and_build_reuses_precomputed_widths(self) -> None:
        """Filter rebuilds should not recompute column widths twice."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                with (
                    patch.object(
                        screen,
                        "_compute_column_widths",
                        wraps=screen._compute_column_widths,
                    ) as mock_widths,
                    patch.object(screen, "_update_help_widgets"),
                ):
                    screen._filter_text = "fix"
                    await screen._filter_and_build()

                assert mock_widths.call_count == 1


class TestThreadSelectorDelete:
    """Tests for ctrl+d delete functionality."""

    async def test_delete_shows_confirmation(self) -> None:
        """Ctrl+D should show a delete confirmation overlay."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._confirming_delete is False

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()

                assert screen._confirming_delete is True

    async def test_delete_confirmation_uses_screen_overlay(self) -> None:
        """Delete confirmation should open as a modal above the selector."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()

                assert isinstance(app.screen, DeleteThreadConfirmScreen)

    async def test_delete_escape_cancels(self) -> None:
        """Escape during delete confirmation should cancel."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert screen._confirming_delete is True
                assert isinstance(app.screen, DeleteThreadConfirmScreen)

                await pilot.press("escape")
                await pilot.pause()
                await pilot.pause()

                assert screen._confirming_delete is False
                assert app.screen is screen
                assert app.dismissed is False

    async def test_delete_keeps_selection_on_next_thread(self) -> None:
        """Deleting a row should move selection to the next visible thread."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.sessions.delete_thread",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                await pilot.press("down")
                await pilot.pause()
                assert screen._selected_index == 1
                assert screen._filtered_threads[1]["thread_id"] == "def67890"

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert isinstance(app.screen, DeleteThreadConfirmScreen)

                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()

                assert app.screen is screen
                assert screen._selected_index == 1
                selected_thread = screen._filtered_threads[screen._selected_index]
                assert selected_thread["thread_id"] == "ghi11111"

    async def test_delete_last_remaining_thread(self) -> None:
        """Deleting the only thread should leave an empty list without errors."""
        single_thread: list[ThreadInfo] = [
            {
                "thread_id": "only-one",
                "agent_name": "agent",
                "updated_at": "2025-01-15T10:30:00",
            }
        ]
        with (
            _patch_list_threads(single_thread),
            patch(
                "deepagents_code.sessions.delete_thread",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._filtered_threads) == 1

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert isinstance(app.screen, DeleteThreadConfirmScreen)

                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()

                assert app.screen is screen
                assert screen._filtered_threads == []
                assert screen._selected_index == 0

    async def test_delete_last_item_in_list_moves_selection_backward(self) -> None:
        """Deleting the bottom thread should move selection to the previous one."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.sessions.delete_thread",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                last_index = len(screen._filtered_threads) - 1

                for _ in range(last_index):
                    await pilot.press("down")
                    await pilot.pause()
                assert screen._selected_index == last_index

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert isinstance(app.screen, DeleteThreadConfirmScreen)

                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()

                assert app.screen is screen
                assert screen._selected_index < last_index
                assert screen._selected_index == len(screen._filtered_threads) - 1

    async def test_delete_failure_keeps_thread_in_list(self) -> None:
        """DB failure during delete should keep the thread visible."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_code.sessions.delete_thread",
                new_callable=AsyncMock,
                side_effect=OSError("disk full"),
            ),
        ):
            app = ThreadSelectorTestApp(current_thread=None)
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                original_count = len(screen._filtered_threads)

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert isinstance(app.screen, DeleteThreadConfirmScreen)

                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()

                assert app.screen is screen
                assert len(screen._filtered_threads) == original_count


class TestThreadSelectorColumnConfig:
    """Tests for column visibility configuration."""

    def test_default_columns(self) -> None:
        """Default column config should match THREAD_COLUMN_DEFAULTS."""
        from deepagents_code.model_config import THREAD_COLUMN_DEFAULTS

        with _patch_columns():
            screen = ThreadSelectorScreen(current_thread=None)
        assert screen._columns == THREAD_COLUMN_DEFAULTS

    async def test_switch_toggles_column_and_persists(self) -> None:
        """Clicking a column switch should hide the column and save the choice."""
        with (
            _patch_list_threads(),
            _patch_columns(),
            patch(
                "deepagents_code.model_config.save_thread_columns",
                return_value=True,
            ) as mock_save,
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                assert screen._columns["initial_prompt"] is True

                prompt_switch = screen.query_one(
                    f"#{screen._switch_id('initial_prompt')}",
                    Checkbox,
                )
                prompt_switch.value = False
                await pilot.pause()

                assert screen._columns["initial_prompt"] is False
                mock_save.assert_called()
                assert mock_save.call_args.args[0]["initial_prompt"] is False

    async def test_enabling_prompt_column_triggers_prompt_load(self) -> None:
        """Turning on the prompt column should fetch missing prompt data."""
        threads_without_prompt: list[ThreadInfo] = [
            {
                "thread_id": "abc12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-15T10:30:00",
                "message_count": 5,
            }
        ]
        columns = {
            "thread_id": True,
            "messages": True,
            "created_at": True,
            "updated_at": True,
            "git_branch": True,
            "cwd": False,
            "initial_prompt": False,
            "agent_name": True,
        }

        async def _populate(
            rows: list[ThreadInfo],
            *,
            include_message_count: bool,
            include_initial_prompt: bool,
        ) -> list[ThreadInfo]:
            await asyncio.sleep(0)
            assert include_message_count is False
            assert include_initial_prompt is True
            rows[0]["initial_prompt"] = "loaded prompt"
            return rows

        with (
            _patch_list_threads(threads_without_prompt),
            _patch_columns(columns),
            patch(
                "deepagents_code.sessions.populate_thread_checkpoint_details",
                new_callable=AsyncMock,
                side_effect=_populate,
            ) as mock_populate,
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert mock_populate.await_count == 0

                prompt_switch = screen.query_one(
                    f"#{screen._switch_id('initial_prompt')}",
                    Checkbox,
                )
                prompt_switch.value = True

                for _ in range(10):
                    if mock_populate.await_count >= 1:
                        break
                    await pilot.pause(0.05)

                mock_populate.assert_awaited_once()
                assert screen._threads[0]["initial_prompt"] == "loaded prompt"


class TestThreadSelectorControlsOverflow:
    """Tests for short-window overflow handling in the options pane."""

    async def test_options_overflow_shows_ellipsis(self) -> None:
        """A short options pane should show an ellipsis when controls are hidden."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test(size=(100, 12)) as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                scroll = screen.query_one("#thread-controls-scroll", VerticalScroll)
                hint = screen.query_one("#thread-controls-overflow", Static)

                assert scroll.max_scroll_y > 0
                assert hint.display is True

    async def test_options_overflow_ellipsis_hides_at_end(self) -> None:
        """The options ellipsis should disappear after scrolling to the end."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test(size=(100, 12)) as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                scroll = screen.query_one("#thread-controls-scroll", VerticalScroll)
                hint = screen.query_one("#thread-controls-overflow", Static)
                scroll.scroll_end(animate=False)
                await pilot.pause()

                assert scroll.scroll_y == scroll.max_scroll_y
                assert hint.display is False


def _app_test_double(app: DeepAgentsApp) -> Any:  # noqa: ANN401
    """Return `app` as dynamic for test-only Textual method patching.

    Textual apps expose real methods at type-check time, but these tests replace
    them with `MagicMock`/`AsyncMock` instances to isolate thread-switching logic.
    Keeping the dynamic escape hatch here avoids broad casts at each call site.
    """
    return app


def _get_widget_text(widget: Static) -> str:
    """Extract text content from a message widget.

    Args:
        widget: A message widget (e.g., `AppMessage`).

    Returns:
        The text content of the widget.
    """
    return str(getattr(widget, "_content", ""))


class TestResumeThread:
    """Tests for DeepAgentsApp._resume_thread."""

    async def test_no_agent_shows_error(self) -> None:
        """_resume_thread with no agent should show an error message."""
        app = DeepAgentsApp()
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda w: mounted.append(w)
        )
        app._agent = None

        await app._resume_thread("thread-123")

        assert len(mounted) == 1
        assert "no active agent" in _get_widget_text(mounted[0])

    async def test_no_session_state_shows_error(self) -> None:
        """_resume_thread with no session state should show an error message."""
        app = DeepAgentsApp()
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda w: mounted.append(w)
        )
        app._agent = MagicMock()
        app._session_state = None

        await app._resume_thread("thread-123")

        assert len(mounted) == 1
        assert "no active session" in _get_widget_text(mounted[0])

    async def test_already_switching_shows_message(self) -> None:
        """_resume_thread should reject concurrent thread switches."""
        app = DeepAgentsApp()
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda w: mounted.append(w)
        )
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "thread-123"
        app._thread_switching = True

        await app._resume_thread("thread-999")

        assert len(mounted) == 1
        assert "already in progress" in _get_widget_text(mounted[0])

    async def test_already_on_thread_shows_message(self) -> None:
        """_resume_thread when already on the thread should show info message."""
        app = DeepAgentsApp()
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda w: mounted.append(w)
        )
        offer_cwd_switch = AsyncMock(return_value="continue")
        _app_test_double(app)._offer_thread_cwd_switch = offer_cwd_switch
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "thread-123"

        await app._resume_thread("thread-123")

        offer_cwd_switch.assert_awaited_once_with(
            "thread-123",
            restart_server=True,
        )
        assert len(mounted) == 1
        assert "Already on thread" in _get_widget_text(mounted[0])

    async def test_already_on_thread_reports_cwd_switch(
        self,
        tmp_path: Path,
    ) -> None:
        """Same-thread resumes should not say unchanged after cwd switches."""
        current = tmp_path / "current"
        target = tmp_path / "target"
        current.mkdir()
        target.mkdir()
        app = DeepAgentsApp(cwd=current)
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda w: mounted.append(w)
        )

        async def offer_cwd_switch(  # noqa: RUF029  # must be async: awaited as _offer_thread_cwd_switch
            thread_id: str,
            *,
            restart_server: bool,
        ) -> str:
            assert thread_id == "thread-123"
            assert restart_server is True
            app._cwd = str(target)
            return "continue"

        _app_test_double(app)._offer_thread_cwd_switch = offer_cwd_switch
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "thread-123"

        await app._resume_thread("thread-123")

        assert len(mounted) == 1
        assert "Switched to thread directory" in _get_widget_text(mounted[0])
        assert "Already on thread" not in _get_widget_text(mounted[0])

    async def test_successful_switch_updates_ids(self) -> None:
        """Successful _resume_thread should update thread IDs and load history."""
        from textual.css.query import NoMatches as _NoMatches

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        app._pending_messages = MagicMock()
        app._queued_widgets = MagicMock()
        _app_test_double(app)._clear_messages = AsyncMock()
        _app_test_double(app)._update_status = MagicMock()
        mock_payload = MagicMock()
        mock_payload.messages = []
        mock_payload.context_tokens = 0
        _app_test_double(app)._fetch_thread_history_data = AsyncMock(
            return_value=mock_payload
        )
        _app_test_double(app)._load_thread_history = AsyncMock()
        _app_test_double(app)._mount_message = AsyncMock()
        _app_test_double(app).query_one = MagicMock(side_effect=_NoMatches())
        offer_cwd_switch = AsyncMock(return_value="continue")
        _app_test_double(app)._offer_thread_cwd_switch = offer_cwd_switch

        await app._resume_thread("new-thread")

        # In-session switches never offer abort — that is launch-time only.
        # Exact-args match fails if `allow_abort=True` ever leaks in here.
        offer_cwd_switch.assert_awaited_once_with(
            "new-thread",
            restart_server=True,
        )
        assert app._lc_thread_id == "new-thread"
        assert app._session_state.thread_id == "new-thread"
        app._pending_messages.clear.assert_called_once()
        app._queued_widgets.clear.assert_called_once()
        _app_test_double(app)._clear_messages.assert_awaited_once()
        assert app._context_tokens == 0
        app._fetch_thread_history_data.assert_awaited_once_with("new-thread")
        app._load_thread_history.assert_awaited_once_with(
            thread_id="new-thread",
            preloaded_payload=mock_payload,
        )

    @staticmethod
    def _switch_app() -> DeepAgentsApp:
        from textual.css.query import NoMatches as _NoMatches

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        app._pending_messages = MagicMock()
        app._queued_widgets = MagicMock()
        _app_test_double(app)._clear_messages = AsyncMock()
        _app_test_double(app)._update_status = MagicMock()
        mock_payload = MagicMock()
        mock_payload.messages = []
        mock_payload.context_tokens = 0
        _app_test_double(app)._fetch_thread_history_data = AsyncMock(
            return_value=mock_payload
        )
        _app_test_double(app)._load_thread_history = AsyncMock()
        _app_test_double(app)._mount_message = AsyncMock()
        _app_test_double(app).query_one = MagicMock(side_effect=_NoMatches())
        return app

    async def test_switch_arms_model_adoption(self) -> None:
        """An in-session thread switch arms session-only model adoption.

        Mirrors launch-time `-r`: `_load_thread_history` (real, here mocked)
        consumes the flag and adopts the switched-to thread's model.
        """
        app = self._switch_app()

        await app._resume_thread("new-thread")

        assert app._should_adopt_resumed_model is True

    async def test_explicit_model_suppresses_switch_adoption(self) -> None:
        """`--model` keeps the session pinned across in-session switches."""
        app = self._switch_app()
        app._model_explicitly_set = True

        await app._resume_thread("new-thread")

        assert app._should_adopt_resumed_model is False

    async def test_successful_switch_records_previous_thread(self) -> None:
        """A successful switch records the outgoing thread as previous_thread_id.

        Lets a follow-up bare `/threads -r` step back to the thread just left
        rather than resolving `previous == current` and reporting "Already on
        thread".
        """
        app = self._switch_app()

        await app._resume_thread("new-thread")

        session_state = app._session_state
        assert session_state is not None
        assert session_state.previous_thread_id == "old-thread"

    async def test_failure_restores_previous_thread_ids(self) -> None:
        """If _clear_messages raises, thread IDs should be restored."""
        from textual.css.query import NoMatches as _NoMatches

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        app._pending_messages = MagicMock()
        app._queued_widgets = MagicMock()
        from deepagents_code.app import _ThreadHistoryPayload

        mock_payload = _ThreadHistoryPayload(
            messages=[], context_tokens=0, model_spec=""
        )
        _app_test_double(app)._fetch_thread_history_data = AsyncMock(
            return_value=mock_payload
        )
        _app_test_double(app)._clear_messages = AsyncMock(
            side_effect=RuntimeError("UI gone")
        )
        _app_test_double(app)._update_status = MagicMock()
        _app_test_double(app)._mount_message = AsyncMock()
        _app_test_double(app).query_one = MagicMock(side_effect=_NoMatches())

        await app._resume_thread("new-thread")

        assert app._lc_thread_id == "old-thread"
        assert app._session_state.thread_id == "old-thread"
        assert any(
            "Failed to switch" in _get_widget_text(call.args[0])
            for call in _app_test_double(app)._mount_message.call_args_list
        )
        _app_test_double(app)._update_status.assert_any_call("")

    async def test_failure_during_load_history_restores_ids(self) -> None:
        """If _load_thread_history raises, thread IDs should be rolled back."""
        from textual.css.query import NoMatches as _NoMatches

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        app._pending_messages = MagicMock()
        app._queued_widgets = MagicMock()
        mock_payload = MagicMock()
        mock_payload.messages = []
        mock_payload.context_tokens = 0
        _app_test_double(app)._fetch_thread_history_data = AsyncMock(
            return_value=mock_payload
        )
        _app_test_double(app)._clear_messages = AsyncMock()
        _app_test_double(app)._update_status = MagicMock()
        _app_test_double(app)._load_thread_history = AsyncMock(
            side_effect=[RuntimeError("checkpoint corrupt"), None]
        )
        _app_test_double(app)._mount_message = AsyncMock()
        _app_test_double(app).query_one = MagicMock(side_effect=_NoMatches())

        await app._resume_thread("new-thread")

        assert app._lc_thread_id == "old-thread"
        assert app._session_state.thread_id == "old-thread"
        assert any(
            "Failed to switch" in _get_widget_text(call.args[0])
            for call in _app_test_double(app)._mount_message.call_args_list
        )

    async def test_prefetch_failure_keeps_current_thread_visible(self) -> None:
        """Failed prefetch should not clear current conversation state."""
        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        fetch_history_mock = AsyncMock(
            side_effect=RuntimeError("checkpoint read failed")
        )
        clear_messages_mock = AsyncMock()
        mount_message_mock = AsyncMock()
        _app_test_double(app)._fetch_thread_history_data = fetch_history_mock
        _app_test_double(app)._clear_messages = clear_messages_mock
        _app_test_double(app)._mount_message = mount_message_mock

        await app._resume_thread("new-thread")

        assert app._session_state.thread_id == "old-thread"
        assert app._lc_thread_id == "old-thread"
        clear_messages_mock.assert_not_awaited()
        assert any(
            "Failed to switch" in _get_widget_text(call.args[0])
            for call in mount_message_mock.call_args_list
        )

    async def test_prefetch_failure_clears_switch_lock_and_restores_input(self) -> None:
        """Prefetch failures should release switch lock and restore input state."""
        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        app._chat_input = MagicMock()
        _app_test_double(app)._mount_message = AsyncMock()

        with patch.object(
            app,
            "_fetch_thread_history_data",
            new_callable=AsyncMock,
            side_effect=RuntimeError("checkpoint read failed"),
        ):
            await app._resume_thread("new-thread")

        assert app._thread_switching is False
        app._chat_input.set_cursor_active.assert_any_call(active=False)
        app._chat_input.set_cursor_active.assert_any_call(active=True)

    async def test_double_failure_surfaces_restore_failure_hint(self) -> None:
        """If rollback restore fails, user-facing error should mention it."""
        from textual.css.query import NoMatches as _NoMatches

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        app._pending_messages = MagicMock()
        app._queued_widgets = MagicMock()
        _app_test_double(app)._fetch_thread_history_data = AsyncMock(return_value=[])
        _app_test_double(app)._clear_messages = AsyncMock()
        _app_test_double(app)._load_thread_history = AsyncMock(
            side_effect=RuntimeError("checkpoint corrupt")
        )
        mount_message_mock = AsyncMock()
        _app_test_double(app)._mount_message = mount_message_mock
        _app_test_double(app).query_one = MagicMock(side_effect=_NoMatches())

        with patch.object(app, "_update_status") as update_status_mock:
            await app._resume_thread("new-thread")

        assert any(
            "Previous thread history could not be restored"
            in _get_widget_text(call.args[0])
            for call in mount_message_mock.call_args_list
        )
        update_status_mock.assert_any_call("")


class TestFetchThreadHistoryData:
    """Tests for DeepAgentsApp._fetch_thread_history_data."""

    async def test_returns_empty_when_agent_missing(self) -> None:
        """No active agent should return an empty history payload."""
        app = DeepAgentsApp()
        app._agent = None

        payload = await app._fetch_thread_history_data("tid-1")

        assert payload.messages == []
        assert payload.context_tokens == 0

    async def test_returns_empty_when_state_missing(self) -> None:
        """Missing checkpoint state should return an empty history payload."""
        app = DeepAgentsApp()
        app._agent = MagicMock()
        app._agent.aget_state = AsyncMock(return_value=None)

        payload = await app._fetch_thread_history_data("tid-1")

        assert payload.messages == []
        assert payload.context_tokens == 0
        app._agent.aget_state.assert_awaited_once_with(
            {"configurable": {"thread_id": "tid-1"}}
        )

    async def test_returns_empty_when_messages_missing(self) -> None:
        """State with no messages should return an empty history payload."""
        app = DeepAgentsApp()
        app._agent = MagicMock()
        state = MagicMock()
        state.values = {}
        app._agent.aget_state = AsyncMock(return_value=state)

        payload = await app._fetch_thread_history_data("tid-1")

        assert payload.messages == []
        assert payload.context_tokens == 0

    async def test_offloads_conversion_to_thread(self) -> None:
        """Message conversion should be offloaded via `asyncio.to_thread`."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp()
        app._agent = MagicMock()
        raw_messages = [object()]
        state = MagicMock()
        state.values = {"messages": raw_messages}
        app._agent.aget_state = AsyncMock(return_value=state)
        converted = [MessageData(type=MessageType.USER, content="hello")]

        with patch(
            "deepagents_code.app.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=converted,
        ) as to_thread_mock:
            payload = await app._fetch_thread_history_data("tid-1")

        assert payload.messages == converted
        to_thread_mock.assert_awaited_once()
        await_args = to_thread_mock.await_args
        assert await_args is not None
        assert await_args.args[1] == raw_messages

    async def test_extracts_nonzero_context_tokens(self) -> None:
        """Persisted _context_tokens should propagate to the payload."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp()
        app._agent = MagicMock()
        raw_messages = [object()]
        state = MagicMock()
        state.values = {"messages": raw_messages, "_context_tokens": 12000}
        app._agent.aget_state = AsyncMock(return_value=state)
        converted = [MessageData(type=MessageType.USER, content="hello")]

        with patch(
            "deepagents_code.app.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=converted,
        ):
            payload = await app._fetch_thread_history_data("tid-1")

        assert payload.context_tokens == 12000

    async def test_extracts_model_spec(self) -> None:
        """Persisted `_model_spec` should propagate to the payload."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp()
        app._agent = MagicMock()
        raw_messages = [object()]
        state = MagicMock()
        state.values = {
            "messages": raw_messages,
            "_model_spec": "anthropic:claude-sonnet-4-5",
        }
        app._agent.aget_state = AsyncMock(return_value=state)
        converted = [MessageData(type=MessageType.USER, content="hello")]

        with patch(
            "deepagents_code.app.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=converted,
        ):
            payload = await app._fetch_thread_history_data("tid-1")

        assert payload.model_spec == "anthropic:claude-sonnet-4-5"

    async def test_missing_model_spec_is_empty(self) -> None:
        """A legacy thread without `_model_spec` yields `model_spec=""`."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp()
        app._agent = MagicMock()
        raw_messages = [object()]
        state = MagicMock()
        state.values = {"messages": raw_messages}
        app._agent.aget_state = AsyncMock(return_value=state)
        converted = [MessageData(type=MessageType.USER, content="hello")]

        with patch(
            "deepagents_code.app.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=converted,
        ):
            payload = await app._fetch_thread_history_data("tid-1")

        assert payload.model_spec == ""

    async def test_none_context_tokens_coerced_to_zero(self) -> None:
        """`_context_tokens: None` in checkpoint should coerce to 0."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp()
        app._agent = MagicMock()
        raw_messages = [object()]
        state = MagicMock()
        state.values = {"messages": raw_messages, "_context_tokens": None}
        app._agent.aget_state = AsyncMock(return_value=state)
        converted = [MessageData(type=MessageType.USER, content="hello")]

        with patch(
            "deepagents_code.app.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=converted,
        ):
            payload = await app._fetch_thread_history_data("tid-1")

        assert payload.context_tokens == 0


class TestLoadThreadHistory:
    """Tests for DeepAgentsApp._load_thread_history."""

    async def test_preloaded_history_skips_fetch_and_schedules_link(self) -> None:
        """Preloaded history should render without state fetch round-trip."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp(thread_id="tid-1")
        app._agent = MagicMock()
        fetch_history_mock = AsyncMock()
        mount_message_mock = AsyncMock()
        schedule_link_mock = MagicMock()
        _app_test_double(app)._fetch_thread_history_data = fetch_history_mock
        _app_test_double(app)._remove_spacer = AsyncMock()
        _app_test_double(app)._mount_message = mount_message_mock
        _app_test_double(app)._schedule_thread_message_link = schedule_link_mock
        _app_test_double(app).set_timer = MagicMock()

        messages_container = MagicMock()
        messages_container.mount = AsyncMock()
        _app_test_double(app).query_one = MagicMock(return_value=messages_container)

        from deepagents_code.app import _ThreadHistoryPayload

        preloaded = _ThreadHistoryPayload(
            messages=[MessageData(type=MessageType.USER, content="hello")],
            context_tokens=0,
            model_spec="",
        )
        await app._load_thread_history(thread_id="tid-1", preloaded_payload=preloaded)

        fetch_history_mock.assert_not_awaited()
        messages_container.mount.assert_awaited_once()
        mount_message_mock.assert_awaited_once()
        schedule_link_mock.assert_called_once()

    async def test_resume_seeds_context_tokens_from_state(self) -> None:
        """Resuming a thread with persisted tokens should seed the local cache."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp(thread_id="tid-1")

        mount_message_mock = AsyncMock()
        schedule_link_mock = MagicMock()
        _app_test_double(app)._remove_spacer = AsyncMock()
        _app_test_double(app)._mount_message = mount_message_mock
        _app_test_double(app)._schedule_thread_message_link = schedule_link_mock
        _app_test_double(app).set_timer = MagicMock()

        messages_container = MagicMock()
        messages_container.mount = AsyncMock()
        _app_test_double(app).query_one = MagicMock(return_value=messages_container)

        from deepagents_code.app import _ThreadHistoryPayload

        preloaded = _ThreadHistoryPayload(
            messages=[MessageData(type=MessageType.USER, content="hello")],
            context_tokens=8500,
            model_spec="",
        )
        await app._load_thread_history(thread_id="tid-1", preloaded_payload=preloaded)

        assert app._context_tokens == 8500

    async def test_zero_context_tokens_does_not_overwrite_cache(self) -> None:
        """Loading a payload with 0 tokens should not reset an existing cache."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp(thread_id="tid-1")
        app._context_tokens = 5000  # pre-existing cache from a previous thread

        mount_message_mock = AsyncMock()
        schedule_link_mock = MagicMock()
        _app_test_double(app)._remove_spacer = AsyncMock()
        _app_test_double(app)._mount_message = mount_message_mock
        _app_test_double(app)._schedule_thread_message_link = schedule_link_mock
        _app_test_double(app).set_timer = MagicMock()

        messages_container = MagicMock()
        messages_container.mount = AsyncMock()
        _app_test_double(app).query_one = MagicMock(return_value=messages_container)

        from deepagents_code.app import _ThreadHistoryPayload

        preloaded = _ThreadHistoryPayload(
            messages=[MessageData(type=MessageType.USER, content="hello")],
            context_tokens=0,
            model_spec="",
        )
        await app._load_thread_history(thread_id="tid-1", preloaded_payload=preloaded)

        assert app._context_tokens == 5000

    async def test_fallback_fetch_path_used_without_preloaded_data(self) -> None:
        """History should be fetched when preloaded data is not provided."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp(thread_id="tid-1")
        app._agent = MagicMock()
        from deepagents_code.app import _ThreadHistoryPayload

        fetched = _ThreadHistoryPayload(
            messages=[MessageData(type=MessageType.USER, content="hello")],
            context_tokens=0,
            model_spec="",
        )
        fetch_history_mock = AsyncMock(return_value=fetched)
        mount_message_mock = AsyncMock()
        schedule_link_mock = MagicMock()
        _app_test_double(app)._fetch_thread_history_data = fetch_history_mock
        _app_test_double(app)._remove_spacer = AsyncMock()
        _app_test_double(app)._mount_message = mount_message_mock
        _app_test_double(app)._schedule_thread_message_link = schedule_link_mock
        _app_test_double(app).set_timer = MagicMock()

        messages_container = MagicMock()
        messages_container.mount = AsyncMock()
        _app_test_double(app).query_one = MagicMock(return_value=messages_container)

        await app._load_thread_history(thread_id="tid-1")

        fetch_history_mock.assert_awaited_once_with("tid-1")
        messages_container.mount.assert_awaited_once()
        mount_message_mock.assert_awaited_once()
        schedule_link_mock.assert_called_once()

    async def test_assistant_render_failure_does_not_abort_history_load(self) -> None:
        """A single assistant render failure should not abort history loading."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType
        from deepagents_code.tui.widgets.messages import AssistantMessage

        app = DeepAgentsApp(thread_id="tid-1")
        app._agent = MagicMock()
        mount_message_mock = AsyncMock()
        schedule_link_mock = MagicMock()
        _app_test_double(app)._remove_spacer = AsyncMock()
        _app_test_double(app)._mount_message = mount_message_mock
        _app_test_double(app)._schedule_thread_message_link = schedule_link_mock
        _app_test_double(app).set_timer = MagicMock()

        messages_container = MagicMock()
        messages_container.mount = AsyncMock()
        _app_test_double(app).query_one = MagicMock(return_value=messages_container)

        from deepagents_code.app import _ThreadHistoryPayload

        preloaded = _ThreadHistoryPayload(
            messages=[
                MessageData(type=MessageType.ASSISTANT, content="ok"),
                MessageData(type=MessageType.ASSISTANT, content="fail"),
            ],
            context_tokens=0,
            model_spec="",
        )

        def _set_content_side_effect(content: str) -> None:
            if content == "fail":
                msg = "markdown update failed"
                raise RuntimeError(msg)

        with patch.object(
            AssistantMessage,
            "set_content",
            new_callable=AsyncMock,
            side_effect=_set_content_side_effect,
        ) as set_content_mock:
            await app._load_thread_history(
                thread_id="tid-1", preloaded_payload=preloaded
            )

        assert set_content_mock.await_count == 2
        mount_message_mock.assert_awaited_once()
        schedule_link_mock.assert_called_once()

    async def test_early_return_without_thread_id_logs_debug(self) -> None:
        """Missing thread ID should early-return with a debug log entry."""
        app = DeepAgentsApp()
        app._lc_thread_id = None
        app._agent = MagicMock()

        with patch("deepagents_code.app.logger.debug") as debug_mock:
            await app._load_thread_history()

        debug_mock.assert_called_once_with(
            "Skipping history load: no thread ID available"
        )

    async def test_early_return_without_agent_logs_debug(self) -> None:
        """No agent and no preloaded payload should early-return with debug log."""
        app = DeepAgentsApp(thread_id="tid-1")
        app._agent = None

        with patch("deepagents_code.app.logger.debug") as debug_mock:
            await app._load_thread_history(thread_id="tid-1")

        debug_mock.assert_called_once_with(
            "Skipping history load for %s: no active agent and no preloaded data",
            "tid-1",
        )


class TestResumeModelAdoption:
    """Tests for adopting a resumed thread's persisted model on load."""

    @staticmethod
    def _make_app() -> DeepAgentsApp:
        app = DeepAgentsApp(thread_id="tid-1")
        _app_test_double(app)._remove_spacer = AsyncMock()
        _app_test_double(app)._mount_message = AsyncMock()
        _app_test_double(app)._schedule_thread_message_link = MagicMock()
        _app_test_double(app).set_timer = MagicMock()
        messages_container = MagicMock()
        messages_container.mount = AsyncMock()
        _app_test_double(app).query_one = MagicMock(return_value=messages_container)
        return app

    @staticmethod
    def _payload(
        model_spec: str,
        *,
        with_messages: bool = True,
        model_params: dict[str, Any] | None = None,
    ) -> _ThreadHistoryPayload:
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        messages = (
            [MessageData(type=MessageType.USER, content="hello")]
            if with_messages
            else []
        )
        return _ThreadHistoryPayload(
            messages=messages,
            context_tokens=0,
            model_spec=model_spec,
            model_params=model_params,
        )

    async def test_adopts_persisted_model_session_only(self) -> None:
        """Armed flag + persisted spec switches the model without persisting it."""
        app = self._make_app()
        switch_mock = AsyncMock()
        _app_test_double(app)._switch_model = switch_mock
        app._should_adopt_resumed_model = True

        await app._load_thread_history(
            thread_id="tid-1",
            preloaded_payload=self._payload("anthropic:claude-sonnet-4-5"),
        )

        switch_mock.assert_awaited_once()
        call = switch_mock.await_args
        assert call is not None
        assert call.args[0] == "anthropic:claude-sonnet-4-5"
        assert call.kwargs["extra_kwargs"] is None
        assert call.kwargs["persist"] is False
        assert call.kwargs["announce_unchanged"] is False
        assert call.kwargs["from_resume"] is True
        # One-shot: the flag is consumed so later loads don't re-adopt.
        assert app._should_adopt_resumed_model is False

    async def test_adopts_persisted_model_params(self) -> None:
        """Resume restores the invocation params saved with the model spec."""
        app = self._make_app()
        switch_mock = AsyncMock()
        _app_test_double(app)._switch_model = switch_mock
        app._should_adopt_resumed_model = True

        await app._load_thread_history(
            thread_id="tid-1",
            preloaded_payload=self._payload(
                "anthropic:claude-sonnet-4-5",
                model_params={"temperature": 0.7, "max_tokens": 1024},
            ),
        )

        switch_mock.assert_awaited_once()
        call = switch_mock.await_args
        assert call is not None
        assert call.args[0] == "anthropic:claude-sonnet-4-5"
        assert call.kwargs["extra_kwargs"] == {
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        assert call.kwargs["persist"] is False
        assert call.kwargs["from_resume"] is True

    async def test_no_adoption_when_flag_unset(self) -> None:
        """Without the armed flag (e.g. in-session switch), model is untouched."""
        app = self._make_app()
        switch_mock = AsyncMock()
        _app_test_double(app)._switch_model = switch_mock
        app._should_adopt_resumed_model = False

        await app._load_thread_history(
            thread_id="tid-1",
            preloaded_payload=self._payload("anthropic:claude-sonnet-4-5"),
        )

        switch_mock.assert_not_awaited()

    async def test_no_adoption_for_legacy_thread_without_spec(self) -> None:
        """Armed flag but no persisted spec (legacy thread) leaves the model alone."""
        app = self._make_app()
        switch_mock = AsyncMock()
        _app_test_double(app)._switch_model = switch_mock
        app._should_adopt_resumed_model = True

        await app._load_thread_history(
            thread_id="tid-1",
            preloaded_payload=self._payload(""),
        )

        switch_mock.assert_not_awaited()
        # The flag is consumed even without a spec, so a later in-session
        # thread switch can't accidentally re-trigger adoption.
        assert app._should_adopt_resumed_model is False

    async def test_consumes_flag_even_when_history_empty(self) -> None:
        """An empty-history resume still adopts and consumes the one-shot flag.

        Adoption runs before the empty-`messages` early return, so the flag
        can't leak into a later in-session `/threads` switch.
        """
        app = self._make_app()
        switch_mock = AsyncMock()
        _app_test_double(app)._switch_model = switch_mock
        app._should_adopt_resumed_model = True

        await app._load_thread_history(
            thread_id="tid-1",
            preloaded_payload=self._payload(
                "anthropic:claude-sonnet-4-5", with_messages=False
            ),
        )

        switch_mock.assert_awaited_once()
        assert app._should_adopt_resumed_model is False


class TestResumeAdoptionFailureMessage:
    """Tests for DeepAgentsApp._mount_resume_adoption_failure."""

    async def test_names_desired_reason_and_fallback(self) -> None:
        """The notice states the desired model, the reason, and the fallback."""
        app = DeepAgentsApp()
        app._model_override = "openai:gpt-5.1"  # the model we fall back to
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda w: mounted.append(w)
        )

        await app._mount_resume_adoption_failure(
            "anthropic:claude-opus-4-8",
            "missing credentials for 'anthropic'",
            hint="Run `/auth` to use it.",
        )

        assert len(mounted) == 1
        text = _get_widget_text(mounted[0])
        assert "anthropic:claude-opus-4-8" in text  # desired
        assert "missing credentials" in text  # reason
        assert "openai:gpt-5.1" in text  # fallback
        assert "Run `/auth` to use it." in text  # hint

    async def test_omits_fallback_when_no_current_model(self) -> None:
        """With no resolvable current model, the fallback clause is dropped."""
        from deepagents_code.config import settings

        app = DeepAgentsApp()
        app._model_override = None
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda w: mounted.append(w)
        )

        with (
            patch.object(settings, "model_provider", ""),
            patch.object(settings, "model_name", ""),
        ):
            await app._mount_resume_adoption_failure(
                "anthropic:claude-opus-4-8", "the model could not be initialized"
            )

        text = _get_widget_text(mounted[0])
        assert "continuing on" not in text
        assert "anthropic:claude-opus-4-8" in text


class TestEffectiveModelSpec:
    """Tests for DeepAgentsApp._effective_model_spec."""

    async def test_prefers_session_override(self) -> None:
        """A `/model` override wins over the startup default in `settings`."""
        from deepagents_code.config import settings

        app = DeepAgentsApp()
        app._model_override = "openai:gpt-5.1"
        with (
            patch.object(settings, "model_provider", "anthropic"),
            patch.object(settings, "model_name", "claude-sonnet-4-5"),
        ):
            assert app._effective_model_spec() == "openai:gpt-5.1"

    async def test_falls_back_to_settings_spec(self) -> None:
        """With no override, the resolved `provider:model` from settings is used."""
        from deepagents_code.config import settings

        app = DeepAgentsApp()
        app._model_override = None
        with (
            patch.object(settings, "model_provider", "anthropic"),
            patch.object(settings, "model_name", "claude-sonnet-4-5"),
        ):
            assert app._effective_model_spec() == "anthropic:claude-sonnet-4-5"

    async def test_none_when_spec_incomplete(self) -> None:
        """No override and a blank model yields `None` (no malformed spec)."""
        from deepagents_code.config import settings

        app = DeepAgentsApp()
        app._model_override = None
        with (
            patch.object(settings, "model_provider", "anthropic"),
            patch.object(settings, "model_name", ""),
        ):
            assert app._effective_model_spec() is None


class TestUpgradeThreadMessageLink:
    """Tests for DeepAgentsApp._upgrade_thread_message_link."""

    async def test_noop_when_link_does_not_resolve(self) -> None:
        """Plain-string result should leave widget content unchanged."""
        app = DeepAgentsApp()
        _app_test_double(app)._build_thread_message = AsyncMock(
            return_value="Resumed thread: tid-1"
        )
        widget = MagicMock()
        widget.parent = object()
        widget._content = "Resumed thread: tid-1"

        await app._upgrade_thread_message_link(
            widget,
            prefix="Resumed thread",
            thread_id="tid-1",
        )

        widget.update.assert_not_called()
        assert widget._content == "Resumed thread: tid-1"

    async def test_noop_when_widget_unmounted(self) -> None:
        """Unmounted widget should not be updated even when link resolves."""
        from textual.content import Content

        app = DeepAgentsApp()
        _app_test_double(app)._build_thread_message = AsyncMock(
            return_value=Content("Resumed thread: tid-1")
        )
        widget = MagicMock()
        widget.parent = None
        widget._content = "Resumed thread: tid-1"

        await app._upgrade_thread_message_link(
            widget,
            prefix="Resumed thread",
            thread_id="tid-1",
        )

        widget.update.assert_not_called()

    async def test_updates_widget_when_link_resolves(self) -> None:
        """Resolved Content should replace widget content."""
        from textual.content import Content

        app = DeepAgentsApp()
        linked = Content("Resumed thread: tid-1")
        _app_test_double(app)._build_thread_message = AsyncMock(return_value=linked)
        widget = MagicMock()
        widget.parent = object()
        widget._content = "Resumed thread: tid-1"

        await app._upgrade_thread_message_link(
            widget,
            prefix="Resumed thread",
            thread_id="tid-1",
        )

        assert widget._content == linked
        widget.update.assert_called_once_with(linked)


class TestBuildThreadMessage:
    """Tests for DeepAgentsApp._build_thread_message."""

    async def test_plain_text_when_tracing_not_configured(self) -> None:
        """Returns plain string when LangSmith URL is not available."""
        app = DeepAgentsApp()
        target = "deepagents_code.config.build_langsmith_thread_url"
        with patch(target, return_value=None):
            result = await app._build_thread_message("Resumed thread", "tid-123")

        assert result == "Resumed thread: tid-123"
        assert isinstance(result, str)

    async def test_hyperlinked_when_tracing_configured(self) -> None:
        """Returns Content with hyperlink when LangSmith URL is available."""
        from textual.content import Content
        from textual.style import Style as TStyle

        app = DeepAgentsApp()
        url = "https://smith.langchain.com/o/org/projects/p/proj/t/tid-123"
        target = "deepagents_code.config.build_langsmith_thread_url"
        with patch(target, return_value=url):
            result = await app._build_thread_message("Resumed thread", "tid-123")

        assert isinstance(result, Content)
        assert "Resumed thread: " in result.plain
        assert "tid-123" in result.plain
        spans = [
            s for s in result._spans if isinstance(s.style, TStyle) and s.style.link
        ]
        assert len(spans) == 1
        style = spans[0].style
        assert isinstance(style, TStyle)
        assert style.link == url

    async def test_fallback_on_timeout(self) -> None:
        """Returns plain string when URL resolution times out."""
        app = DeepAgentsApp()

        async def _raise_timeout(  # noqa: RUF029  # async signature required to match asyncio.wait_for
            coro: Coroutine[Any, Any, Any], *_: Any, **__: Any
        ) -> None:
            coro.close()
            raise TimeoutError

        with patch("deepagents_code.app.asyncio.wait_for", new=_raise_timeout):
            result = await app._build_thread_message("Resumed thread", "t-1")

        assert isinstance(result, str)
        assert result == "Resumed thread: t-1"

    async def test_fallback_on_exception(self) -> None:
        """Returns plain string when URL resolution raises an exception."""
        app = DeepAgentsApp()
        with patch(
            "deepagents_code.config.build_langsmith_thread_url",
            side_effect=OSError("network error"),
        ):
            result = await app._build_thread_message("Resumed thread", "t-1")

        assert isinstance(result, str)
        assert result == "Resumed thread: t-1"


class TestConvertMessagesToData:
    """Tests for DeepAgentsApp._convert_messages_to_data."""

    def _make_human(self, content: str) -> object:
        """Create a HumanMessage."""
        from langchain_core.messages import HumanMessage

        return HumanMessage(content=content)

    def _make_ai(
        self,
        content: str | list[dict[str, str]] = "",
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> object:
        """Create an AIMessage."""
        from langchain_core.messages import AIMessage

        # LangChain accepts `tool_calls` dynamically, but its overloads don't
        # model this simplified test helper shape.
        return AIMessage(
            content=cast("Any", content), tool_calls=cast("Any", tool_calls or [])
        )

    def _make_tool(
        self,
        content: str,
        tool_call_id: str,
        status: str = "success",
    ) -> object:
        """Create a ToolMessage."""
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=content, tool_call_id=tool_call_id, status=status)

    def test_human_message_conversion(self) -> None:
        """HumanMessage should become a USER MessageData."""
        from deepagents_code.tui.widgets.message_store import MessageType

        msgs = [self._make_human("Hello")]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].type == MessageType.USER
        assert result[0].content == "Hello"

    def test_system_prefix_skipped(self) -> None:
        """HumanMessages starting with [SYSTEM] should be skipped."""
        msgs = [
            self._make_human("[SYSTEM] Auto-injected context"),
            self._make_human("Real user message"),
        ]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].content == "Real user message"

    def test_ai_message_text_content(self) -> None:
        """AIMessage with string content should become ASSISTANT MessageData."""
        from deepagents_code.tui.widgets.message_store import MessageType

        msgs = [self._make_ai("Here is the answer.")]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].type == MessageType.ASSISTANT
        assert result[0].content == "Here is the answer."

    def test_ai_message_content_block_list(self) -> None:
        """AIMessage with list-of-blocks content should extract text."""
        from deepagents_code.tui.widgets.message_store import MessageType

        blocks: list[dict[str, str]] = [
            {"type": "text", "text": "Part 1. "},
            {"type": "text", "text": "Part 2."},
        ]
        msgs = [self._make_ai(blocks)]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].type == MessageType.ASSISTANT
        assert result[0].content == "Part 1. Part 2."

    def test_ai_message_empty_text_skipped(self) -> None:
        """AIMessage with empty text should not produce an ASSISTANT entry."""
        msgs = [self._make_ai("   ")]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 0

    def test_tool_call_matching(self) -> None:
        """ToolMessage should be matched to its AIMessage tool call by ID."""
        from deepagents_code.tui.widgets.message_store import MessageType, ToolStatus

        msgs = [
            self._make_ai(
                tool_calls=[
                    {"id": "tc-1", "name": "read_file", "args": {"path": "/a.py"}}
                ]
            ),
            self._make_tool("file contents", tool_call_id="tc-1"),
        ]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].type == MessageType.TOOL
        assert result[0].tool_name == "read_file"
        assert result[0].tool_status == ToolStatus.SUCCESS
        assert result[0].tool_output == "file contents"

    def test_tool_call_error_status(self) -> None:
        """ToolMessage with error status should set ERROR on the tool data."""
        from deepagents_code.tui.widgets.message_store import ToolStatus

        msgs = [
            self._make_ai(
                tool_calls=[{"id": "tc-2", "name": "bash", "args": {"cmd": "fail"}}]
            ),
            self._make_tool("command failed", tool_call_id="tc-2", status="error"),
        ]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert result[0].tool_status == ToolStatus.ERROR
        assert result[0].tool_output == "command failed"

    def test_unmatched_tool_call_rejected(self) -> None:
        """Tool calls with no matching ToolMessage should be REJECTED."""
        from deepagents_code.tui.widgets.message_store import ToolStatus

        msgs = [
            self._make_ai(tool_calls=[{"id": "tc-3", "name": "bash", "args": {}}]),
        ]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].tool_status == ToolStatus.REJECTED

    def test_mixed_message_sequence(self) -> None:
        """Full conversation with mixed message types should convert correctly."""
        from deepagents_code.tui.widgets.message_store import MessageType, ToolStatus

        msgs = [
            self._make_human("What files are here?"),
            self._make_ai(
                "Let me check.",
                tool_calls=[{"id": "tc-a", "name": "list_files", "args": {"dir": "."}}],
            ),
            self._make_tool("file1.py\nfile2.py", tool_call_id="tc-a"),
            self._make_ai("I found 2 files."),
        ]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 4
        assert result[0].type == MessageType.USER
        assert result[1].type == MessageType.ASSISTANT
        assert result[1].content == "Let me check."
        assert result[2].type == MessageType.TOOL
        assert result[2].tool_status == ToolStatus.SUCCESS
        assert result[3].type == MessageType.ASSISTANT
        assert result[3].content == "I found 2 files."

    def test_empty_messages(self) -> None:
        """Empty input should return empty output."""
        result = DeepAgentsApp._convert_messages_to_data([])
        assert result == []

    def test_skill_message_from_additional_kwargs(self) -> None:
        """HumanMessage with __skill in additional_kwargs → SKILL MessageData."""
        from langchain_core.messages import HumanMessage

        from deepagents_code.tui.widgets.message_store import MessageType

        msg = HumanMessage(
            content="I'm invoking the skill `web-research`.\n---\n# Body\n---",
            additional_kwargs={
                "__skill": {
                    "name": "web-research",
                    "description": "Research topics",
                    "source": "user",
                    "args": "find quantum",
                },
            },
        )
        result = DeepAgentsApp._convert_messages_to_data([msg])

        assert len(result) == 1
        assert result[0].type == MessageType.SKILL
        assert result[0].skill_name == "web-research"
        assert result[0].skill_description == "Research topics"
        assert result[0].skill_source == "user"
        assert result[0].skill_args == "find quantum"
        # Full prompt envelope stored as body for expand view
        assert "web-research" in (result[0].skill_body or "")

    def test_skill_without_name_falls_back_to_user(self) -> None:
        """__skill dict missing name should fall back to USER."""
        from langchain_core.messages import HumanMessage

        from deepagents_code.tui.widgets.message_store import MessageType

        msg = HumanMessage(
            content="some text",
            additional_kwargs={"__skill": {"description": "no name"}},
        )
        result = DeepAgentsApp._convert_messages_to_data([msg])

        assert len(result) == 1
        assert result[0].type == MessageType.USER


class TestColumnKeyConsistency:
    """Verify all column dicts stay in sync."""

    def test_all_column_dicts_share_same_keys(self) -> None:
        """All parallel column dicts must have the same key set."""
        from deepagents_code.model_config import THREAD_COLUMN_DEFAULTS
        from deepagents_code.tui.widgets.thread_selector import (
            _COLUMN_LABELS,
            _COLUMN_ORDER,
            _COLUMN_TOGGLE_LABELS,
            _COLUMN_WIDTHS,
        )

        order_keys = set(_COLUMN_ORDER)
        assert order_keys == set(_COLUMN_WIDTHS), (
            f"_COLUMN_WIDTHS keys differ: "
            f"missing={order_keys - set(_COLUMN_WIDTHS)}, "
            f"extra={set(_COLUMN_WIDTHS) - order_keys}"
        )
        assert order_keys == set(_COLUMN_LABELS), (
            f"_COLUMN_LABELS keys differ: "
            f"missing={order_keys - set(_COLUMN_LABELS)}, "
            f"extra={set(_COLUMN_LABELS) - order_keys}"
        )
        assert order_keys == set(_COLUMN_TOGGLE_LABELS), (
            f"_COLUMN_TOGGLE_LABELS keys differ: "
            f"missing={order_keys - set(_COLUMN_TOGGLE_LABELS)}, "
            f"extra={set(_COLUMN_TOGGLE_LABELS) - order_keys}"
        )
        assert order_keys == set(THREAD_COLUMN_DEFAULTS), (
            f"THREAD_COLUMN_DEFAULTS keys differ: "
            f"missing={order_keys - set(THREAD_COLUMN_DEFAULTS)}, "
            f"extra={set(THREAD_COLUMN_DEFAULTS) - order_keys}"
        )


class TestThreadsMatch:
    """Tests for _threads_match short-circuit comparison."""

    @staticmethod
    def _thread(tid: str, cp: str | None = None) -> ThreadInfo:
        t: ThreadInfo = {
            "thread_id": tid,
            "agent_name": "a",
            "updated_at": "x",
        }
        if cp is not None:
            t["latest_checkpoint_id"] = cp
        return t

    def test_identical_lists_match(self) -> None:
        """Identical thread lists should match."""
        a = [self._thread("t1", "cp1"), self._thread("t2", "cp2")]
        b = [self._thread("t1", "cp1"), self._thread("t2", "cp2")]
        assert ThreadSelectorScreen._threads_match(a, b) is True

    def test_different_lengths_do_not_match(self) -> None:
        """Different-length lists should not match."""
        a = [self._thread("t1")]
        b = [self._thread("t1"), self._thread("t2")]
        assert ThreadSelectorScreen._threads_match(a, b) is False

    def test_different_thread_ids_do_not_match(self) -> None:
        """Different thread IDs at same position should not match."""
        a = [self._thread("t1", "cp1")]
        b = [self._thread("t2", "cp1")]
        assert ThreadSelectorScreen._threads_match(a, b) is False

    def test_different_checkpoint_ids_do_not_match(self) -> None:
        """Lists with different checkpoint IDs should not match."""
        a = [self._thread("t1", "cp1")]
        b = [self._thread("t1", "cp2")]
        assert ThreadSelectorScreen._threads_match(a, b) is False

    def test_reordered_threads_do_not_match(self) -> None:
        """Positional comparison means reordered lists fail."""
        a = [self._thread("t1", "cp1"), self._thread("t2", "cp2")]
        b = [self._thread("t2", "cp2"), self._thread("t1", "cp1")]
        assert ThreadSelectorScreen._threads_match(a, b) is False

    def test_empty_lists_match(self) -> None:
        """Two empty lists should match."""
        assert ThreadSelectorScreen._threads_match([], []) is True


class TestThreadSelectorDomSkip:
    """Tests for skipping DOM rebuild when data matches prewarm cache."""

    async def test_matching_refresh_skips_dom_rebuild(self) -> None:
        """When refreshed threads match prefetched, DOM should not be rebuilt."""
        prefetched: list[ThreadInfo] = [
            {
                "thread_id": "abc12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-15T10:30:00",
                "latest_checkpoint_id": "cp_1",
                "message_count": 5,
            }
        ]
        # Same thread and checkpoint
        refreshed: list[ThreadInfo] = [
            {
                "thread_id": "abc12345",
                "agent_name": "my-agent",
                "updated_at": "2025-01-15T10:30:00",
                "latest_checkpoint_id": "cp_1",
            }
        ]
        app = ThreadSelectorTestApp(current_thread="abc12345")

        with patch(
            "deepagents_code.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=refreshed,
        ):
            async with app.run_test() as pilot:
                app.push_screen(
                    ThreadSelectorScreen(
                        current_thread="abc12345",
                        thread_limit=20,
                        initial_threads=prefetched,
                        filter_cwd=None,
                    )
                )
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                initial_widgets = list(screen._option_widgets)
                assert len(initial_widgets) == 1

                # Wait for background refresh
                for _ in range(10):
                    await pilot.pause(0.05)

                # Same widget objects should still be mounted (no rebuild)
                assert screen._option_widgets == initial_widgets


class TestThreadSelectorAgentFilter:
    """Tests for the agent filter dropdown in the Options panel."""

    def test_agent_filtered_threads_no_filter(self) -> None:
        """No agent filter returns all threads."""
        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=MOCK_THREADS,
            filter_cwd=None,
        )
        assert screen._filter_agent is None
        result = screen._agent_filtered_threads()
        assert len(result) == 3

    def test_agent_filtered_threads_with_filter(self) -> None:
        """Agent filter returns only matching threads."""
        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=MOCK_THREADS,
            filter_cwd=None,
        )
        screen._filter_agent = "my-agent"
        result = screen._agent_filtered_threads()
        assert len(result) == 2
        assert all(t["agent_name"] == "my-agent" for t in result)

    def test_agent_filtered_threads_unknown_agent(self) -> None:
        """Threads with no agent_name are matched by the '(unknown)' sentinel."""
        threads: list[ThreadInfo] = [
            {
                "thread_id": "t1",
                "agent_name": None,
                "updated_at": "2025-01-01T00:00:00",
                "created_at": "2025-01-01T00:00:00",
            },
            {
                "thread_id": "t2",
                "agent_name": "my-agent",
                "updated_at": "2025-01-01T00:00:00",
                "created_at": "2025-01-01T00:00:00",
            },
        ]
        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=threads,
            filter_cwd=None,
        )
        screen._filter_agent = "(unknown)"
        result = screen._agent_filtered_threads()
        assert len(result) == 1
        assert result[0]["thread_id"] == "t1"

    def test_update_filtered_list_respects_agent_filter(self) -> None:
        """_update_filtered_list applies agent filter before fuzzy search."""
        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=MOCK_THREADS,
            filter_cwd=None,
        )
        screen._filter_agent = "other-agent"
        screen._filter_text = ""
        screen._update_filtered_list()
        assert len(screen._filtered_threads) == 1
        assert screen._filtered_threads[0]["thread_id"] == "def67890"

    def test_update_filtered_list_agent_and_text_combined(self) -> None:
        """Agent filter and text filter are ANDed together."""
        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=MOCK_THREADS,
            filter_cwd=None,
        )
        screen._filter_agent = "my-agent"
        screen._filter_text = "Hello"
        screen._update_filtered_list()
        assert len(screen._filtered_threads) == 1
        assert screen._filtered_threads[0]["thread_id"] == "abc12345"

    def test_collect_agent_options_all_sentinel_first(self) -> None:
        """collect_agent_options always returns 'All agents' as the first option."""
        from deepagents_code.tui.widgets.thread_selector import _AGENT_VALUE_ALL

        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=MOCK_THREADS,
            filter_cwd=None,
        )
        options = screen._collect_agent_options()
        assert options[0] == ("All agents", _AGENT_VALUE_ALL)

    def test_collect_agent_options_loading_while_pending(self) -> None:
        """While loading with no known agents, the dropdown shows 'Loading...'."""
        from deepagents_code.tui.widgets.thread_selector import (
            _AGENT_LABEL_LOADING,
            _AGENT_VALUE_ALL,
            _AGENT_VALUE_LOADING,
        )

        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=None,
            filter_cwd=None,
        )
        assert screen._disk_load_complete is False
        assert screen._collect_agent_options() == [
            (_AGENT_LABEL_LOADING, _AGENT_VALUE_LOADING)
        ]
        # Once the disk load completes with no threads, fall back to "All agents".
        screen._disk_load_complete = True
        assert screen._collect_agent_options() == [("All agents", _AGENT_VALUE_ALL)]

    async def test_agent_select_label_refreshes_after_empty_load(self) -> None:
        """The loading placeholder is replaced by the final all-agents label."""
        from deepagents_code.tui.widgets.thread_selector import (
            _AGENT_SELECT_ID,
            _AGENT_VALUE_ALL,
            _AGENT_VALUE_LOADING,
        )

        load_started = asyncio.Event()
        load_finished = asyncio.Event()

        async def list_threads_after_signal(**_: object) -> list[ThreadInfo]:
            load_started.set()
            await load_finished.wait()
            return []

        with (
            patch("deepagents_code.sessions.list_threads", list_threads_after_signal),
            _patch_columns(),
            _patch_available_agents([]),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await asyncio.wait_for(load_started.wait(), timeout=1)
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)
                assert agent_select.value == _AGENT_VALUE_LOADING
                assert str(agent_select.query_one(SelectCurrent).label) == "Loading..."

                load_finished.set()
                await pilot.pause()
                await pilot.pause()

                assert agent_select.value == _AGENT_VALUE_ALL
                assert str(agent_select.query_one(SelectCurrent).label) == "All agents"

    def test_collect_agent_options_sorted_unique(self) -> None:
        """collect_agent_options returns sorted unique agent names."""
        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=MOCK_THREADS,
            filter_cwd=None,
        )
        # Inject mixed-case configured names so the order assertion distinguishes
        # the production case-insensitive `str.casefold` key from a plain
        # case-sensitive sort: the latter would place "Zebra" (uppercase) ahead
        # of the lowercase names rather than last.
        screen._available_agent_names = ["Zebra", "apple"]
        options = screen._collect_agent_options()
        # First entry is the sentinel; the rest are unique agents in
        # case-insensitive order with the thread/config overlap de-duplicated.
        labels = [label for label, _ in options[1:]]
        assert labels == ["apple", "my-agent", "other-agent", "Zebra"]
        assert len(set(labels)) == len(labels)  # no duplicates

    def test_collect_agent_options_includes_configured_agents(self) -> None:
        """Configured agents (the /agents list) appear even without threads."""
        screen = ThreadSelectorScreen(
            current_thread=None,
            initial_threads=MOCK_THREADS,
            filter_cwd=None,
        )
        # Simulate the off-thread scan of ~/.deepagents/ surfacing an agent that
        # has not produced any threads yet, plus one that overlaps a thread.
        screen._available_agent_names = ["my-agent", "zeta-agent"]
        labels = [label for label, _ in screen._collect_agent_options()[1:]]
        # Threadless configured agent is selectable...
        assert "zeta-agent" in labels
        # ...alongside thread-derived names, with the overlap de-duplicated.
        assert "my-agent" in labels
        assert "other-agent" in labels
        assert labels.count("my-agent") == 1
        assert labels == sorted(labels, key=str.casefold)

    async def test_load_threads_populates_configured_agents(self) -> None:
        """`/threads` agent filter mirrors `/agents`, not just agents w/ threads."""
        from deepagents_code.tui.widgets.thread_selector import _AGENT_SELECT_ID

        with (
            _patch_list_threads(),
            _patch_columns(),
            _patch_available_agents(["brand-new-agent"]),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._available_agent_names == ["brand-new-agent"]
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)
                values = set(agent_select._legal_values)
                # Configured-but-threadless agent is offered as a filter option.
                assert "brand-new-agent" in values
                # Thread-derived agents remain selectable too.
                assert "my-agent" in values
                assert "other-agent" in values

    async def test_agent_select_widget_present(self) -> None:
        """The agent Select widget is rendered inside the Options panel."""
        from deepagents_code.tui.widgets.thread_selector import _AGENT_SELECT_ID

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)
                assert agent_select is not None

    async def test_agent_select_filters_thread_list(self) -> None:
        """Changing the agent dropdown filters the visible thread list."""
        from deepagents_code.tui.widgets.thread_selector import _AGENT_SELECT_ID

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._filtered_threads) == 3

                # Set agent filter directly and re-run filtering
                screen._filter_agent = "my-agent"
                screen._update_filtered_list()

                assert len(screen._filtered_threads) == 2
                assert all(
                    t["agent_name"] == "my-agent" for t in screen._filtered_threads
                )

    async def test_reload_clears_stale_agent_filter_and_refilters(self) -> None:
        """Reloading without the selected agent should show all remaining rows."""
        from deepagents_code.tui.widgets.thread_selector import (
            _AGENT_SELECT_ID,
            _AGENT_VALUE_ALL,
        )

        reloaded_threads = [MOCK_THREADS[1]]

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)
                agent_select.value = "my-agent"
                screen._filter_agent = "my-agent"
                screen._update_filtered_list()
                assert len(screen._filtered_threads) == 2

                with _patch_list_threads(reloaded_threads):
                    await screen._load_threads()
                await pilot.pause()

                assert screen._filter_agent is None
                assert agent_select.value == _AGENT_VALUE_ALL
                assert screen._filtered_threads == reloaded_threads

    async def test_enter_opens_agent_select_without_resuming_thread(self) -> None:
        """Enter on the focused agent control should open its dropdown."""
        from deepagents_code.tui.widgets.thread_selector import _AGENT_SELECT_ID

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)

                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.pause()
                assert agent_select.has_focus

                await pilot.press("enter")
                await pilot.pause()

                assert agent_select.expanded
                assert not app.dismissed

    async def test_tab_keys_move_open_agent_select_highlight(self) -> None:
        """Tab and Shift+Tab should move the agent dropdown highlight while open."""
        from deepagents_code.tui.widgets.thread_selector import _AGENT_SELECT_ID

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)
                sort_select = screen.query_one("#thread-sort-select", Select)
                scope_select = screen.query_one("#thread-scope-select", Select)

                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("enter")
                await pilot.pause()
                assert agent_select.expanded
                overlay = agent_select.query_one(ContainedSelectOverlay)
                assert overlay.highlighted == 0

                await pilot.press("tab")
                await pilot.pause()
                assert agent_select.expanded
                assert overlay.highlighted == 1
                assert not sort_select.has_focus
                assert not app.dismissed

                await pilot.press("shift+tab")
                await pilot.pause()
                assert agent_select.expanded
                assert overlay.highlighted == 0
                assert not scope_select.has_focus
                assert not app.dismissed

    async def test_escape_closes_open_agent_select_without_dismissing(self) -> None:
        """Esc should close the agent dropdown before it cancels the selector."""
        from deepagents_code.tui.widgets.thread_selector import _AGENT_SELECT_ID

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)

                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("tab")
                await pilot.press("enter")
                await pilot.pause()
                assert agent_select.expanded

                await pilot.press("escape")
                await pilot.pause()

                assert not agent_select.expanded
                assert agent_select.has_focus
                assert not app.dismissed

    async def test_agent_name_load_failure_does_not_strand_picker(self) -> None:
        """An unexpected agent-scan error must not abort the post-load build.

        `_load_available_agent_names` is awaited after `_load_threads`'s own
        try/except, so a raise there would otherwise skip the filter refresh,
        DOM build, and checkpoint enrichment, leaving the picker half-rendered.
        """
        with (
            _patch_list_threads(),
            _patch_columns(),
            patch(
                "deepagents_code.agent.get_available_agent_names",
                side_effect=RuntimeError("agent scan blew up"),
            ),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                screen: object = None
                for _ in range(20):
                    await pilot.pause()
                    screen = app.screen
                    if (
                        isinstance(screen, ThreadSelectorScreen)
                        and screen._disk_load_complete
                        and screen._option_widgets
                    ):
                        break

                assert isinstance(screen, ThreadSelectorScreen)
                # The load completed and the list was built despite the failure.
                assert screen._disk_load_complete
                assert len(screen._filtered_threads) == 3
                assert screen._option_widgets
                # The filter degraded to thread-derived names only (the scan
                # never assigned, so the cache keeps its empty default).
                assert screen._available_agent_names == []

    async def test_reload_preserves_present_agent_filter(self) -> None:
        """Reloading with the selected agent still present keeps the filter."""
        from deepagents_code.tui.widgets.thread_selector import _AGENT_SELECT_ID

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)
                agent_select.value = "my-agent"
                screen._filter_agent = "my-agent"
                screen._update_filtered_list()
                assert len(screen._filtered_threads) == 2

                # Reload with a thread set that still contains my-agent.
                with _patch_list_threads(MOCK_THREADS):
                    await screen._load_threads()
                for _ in range(20):
                    await pilot.pause()
                    if (
                        screen._filter_agent == "my-agent"
                        and len(screen._filtered_threads) == 2
                    ):
                        break

                # Selection and filter survive the reload (no fallback to All).
                assert screen._filter_agent == "my-agent"
                assert agent_select.value == "my-agent"
                assert len(screen._filtered_threads) == 2
                assert all(
                    t["agent_name"] == "my-agent" for t in screen._filtered_threads
                )

    async def test_agent_dropdown_change_filters_via_event(self) -> None:
        """Selecting an agent via the dropdown event filters the list.

        Drives the real `Select.Changed` handler rather than poking
        `_filter_agent`, covering the `_AGENT_VALUE_ALL` translation and the
        async `_filter_and_build` path the handler schedules.
        """
        from deepagents_code.tui.widgets.thread_selector import (
            _AGENT_SELECT_ID,
            _AGENT_VALUE_ALL,
        )

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert screen._filter_agent is None
                assert len(screen._filtered_threads) == 3
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)

                agent_select.value = "my-agent"
                for _ in range(20):
                    await pilot.pause()
                    if (
                        screen._filter_agent == "my-agent"
                        and len(screen._filtered_threads) == 2
                    ):
                        break
                assert screen._filter_agent == "my-agent"
                assert len(screen._filtered_threads) == 2

                # Returning to the sentinel clears the filter.
                agent_select.value = _AGENT_VALUE_ALL
                for _ in range(20):
                    await pilot.pause()
                    if (
                        screen._filter_agent is None
                        and len(screen._filtered_threads) == 3
                    ):
                        break
                assert screen._filter_agent is None
                assert len(screen._filtered_threads) == 3

    async def test_confirming_delete_ignores_agent_select_change(self) -> None:
        """Agent dropdown changes are ignored while a delete is being confirmed."""
        from deepagents_code.tui.widgets.thread_selector import _AGENT_SELECT_ID

        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                agent_select = screen.query_one(f"#{_AGENT_SELECT_ID}", Select)
                screen._confirming_delete = True

                agent_select.value = "my-agent"
                for _ in range(10):
                    await pilot.pause()

                # The guard short-circuits before mutating the filter state.
                assert screen._filter_agent is None
                assert len(screen._filtered_threads) == 3
