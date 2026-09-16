"""Tests for ThreadSelectorScreen."""

import asyncio
from collections.abc import Coroutine
from contextlib import AbstractContextManager
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widget import MountError
from textual.widgets import Checkbox, Input, Select, Static
from textual.widgets._select import SelectCurrent

from deepagents_code.app import DeepAgentsApp, _ThreadHistoryPayload
from deepagents_code.hooks.manager import HooksManager
from deepagents_code.sessions import ThreadInfo
from deepagents_code.tui.widgets.message_store import MessageData
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


class TestThreadSelectorCurrentThread:
    """Tests for current thread highlighting and preselection."""


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


class TestThreadSelectorTabSort:
    """Tests for sort toggling and focus traversal in the selector."""

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


class TestThreadSelectorDownWrap:
    """Tests for wrapping from bottom to top."""


class TestThreadSelectorPageNavigation:
    """Tests for pageup/pagedown navigation."""


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


class TestThreadSelectorClickHandling:
    """Tests for mouse click handling."""


_WEBBROWSER_OPEN = "deepagents_code.tui.widgets._links.webbrowser.open"


class TestThreadSelectorPointer:
    """Tests for `ThreadSelectorScreen` link hover affordance."""


class TestThreadSelectorBuildTitle:
    """Tests for _build_title with clickable thread ID."""


class TestFetchThreadUrl:
    """Tests for _fetch_thread_url background worker."""

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


class TestThreadSelectorColumnHeader:
    """Tests for the anchored column header."""


class TestThreadSelectorPromptOverflow:
    """Tests for prompt-cell overflow handling."""


class TestThreadSelectorBranchOverflow:
    """Tests for git-branch overflow handling."""


class TestThreadSelectorAutoWidthColumns:
    """Tests for shared widths on auto-sized columns."""


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


class TestThreadSelectorCheckpointDetailErrors:
    """Tests for thread selector checkpoint-detail load error handling."""


class TestThreadSelectorPrefetchedRows:
    """Tests for rendering with prefetched rows from startup cache."""

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


class TestThreadSelectorInitialSortOrder:
    """Tests for initial sort order applied to prefetched rows."""


class TestThreadSelectorSearch:
    """Tests for fuzzy search filtering."""


class TestThreadSelectorDelete:
    """Tests for ctrl+d delete functionality."""

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


class TestThreadSelectorColumnConfig:
    """Tests for column visibility configuration."""

    async def test_relative_time_follows_timestamp_columns(self) -> None:
        """Relative timestamps should appear after both timestamp columns."""
        with _patch_list_threads(), _patch_columns():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                toggle_ids = [
                    toggle.id
                    for toggle in screen.query(".thread-column-toggle").results(
                        Checkbox
                    )
                ]

                created_id = screen._switch_id("created_at")
                updated_id = screen._switch_id("updated_at")
                relative_index = toggle_ids.index("thread-relative-time")
                assert toggle_ids[relative_index - 2 : relative_index] == [
                    created_id,
                    updated_id,
                ]

    async def test_relative_time_visibility_tracks_timestamp_columns(self) -> None:
        """Relative timestamps should hide unless a timestamp column is enabled."""
        from deepagents_code.model_config import THREAD_COLUMN_DEFAULTS

        columns = {
            **THREAD_COLUMN_DEFAULTS,
            "created_at": False,
            "updated_at": False,
        }
        with (
            _patch_list_threads(),
            _patch_columns(columns),
            patch(
                "deepagents_code.model_config.save_thread_columns",
                return_value=True,
            ),
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                created_switch = screen.query_one(
                    f"#{screen._switch_id('created_at')}", Checkbox
                )
                updated_switch = screen.query_one(
                    f"#{screen._switch_id('updated_at')}", Checkbox
                )
                relative_switch = screen.query_one("#thread-relative-time", Checkbox)

                assert relative_switch.display is False
                assert relative_switch not in screen._filter_focus_order()

                created_switch.value = True
                await pilot.pause()
                assert relative_switch.display is True
                assert relative_switch in screen._filter_focus_order()
                # Value must be reasserted when un-hidden so the first visible
                # frame renders the persisted check state, not a stale box.
                assert relative_switch.value is True
                assert relative_switch.value == screen._relative_time

                updated_switch.value = True
                created_switch.value = False
                await pilot.pause()
                assert relative_switch.display is True

                updated_switch.value = False
                await pilot.pause()
                assert relative_switch.display is False
                assert relative_switch not in screen._filter_focus_order()

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


def _app_test_double(app: DeepAgentsApp) -> Any:  # noqa: ANN401
    """Return `app` as dynamic for test-only Textual method patching.

    Textual apps expose real methods at type-check time, but these tests replace
    them with `MagicMock`/`AsyncMock` instances to isolate thread-switching logic.
    Keeping the dynamic escape hatch here avoids broad casts at each call site.
    """
    return app


def _mock_session_state(thread_id: str) -> MagicMock:
    """Return mocked session state carrying a real, inert hooks coordinator.

    Thread switching runs `SessionEnd` through the coordinator, so a bare
    `MagicMock` would hand back an unawaitable attribute.

    Args:
        thread_id: Thread the mocked session starts on.

    Returns:
        A `MagicMock` session state with `thread_id` and `hooks` populated.
    """
    state = MagicMock()
    state.thread_id = thread_id
    state.hooks = HooksManager.inert()
    return state


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

    async def test_managed_cutoff_blocks_switch_without_mutation(self) -> None:
        """A blocked target leaves the current thread and transcript untouched."""
        app = DeepAgentsApp(thread_id="current-thread")
        app._agent = MagicMock()
        app._session_state = _mock_session_state("current-thread")
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda widget: mounted.append(widget)
        )
        _app_test_double(app)._thread_resume_block = AsyncMock(
            return_value="Thread stale-thread cannot be resumed."
        )
        clear_messages = AsyncMock()
        _app_test_double(app)._clear_messages = clear_messages

        await app._resume_thread("stale-thread")

        assert app._session_state.thread_id == "current-thread"
        assert app._lc_thread_id == "current-thread"
        clear_messages.assert_not_awaited()
        assert "cannot be resumed" in _get_widget_text(mounted[0])

    async def test_already_switching_shows_message(self) -> None:
        """_resume_thread should reject concurrent thread switches."""
        app = DeepAgentsApp()
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda w: mounted.append(w)
        )
        app._agent = MagicMock()
        app._session_state = _mock_session_state("thread-123")
        app._thread_switching = True

        await app._resume_thread("thread-999")

        assert len(mounted) == 1
        assert "already in progress" in _get_widget_text(mounted[0])

    @staticmethod
    def _switch_app() -> DeepAgentsApp:
        from textual.css.query import NoMatches as _NoMatches

        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = _mock_session_state("old-thread")
        # Seed server-backed output so `_mount_previous_thread_hint`'s
        # `had_agent_output` gate passes; cases that need the no-work path
        # reset the store themselves. Load-bearing for every hint assertion
        # in this class, so do not drop it as unused setup.
        app._message_store.append(
            MessageData(type=MessageType.ASSISTANT, content="existing response")
        )
        app._pending_messages = MagicMock()
        app._queued_widgets = MagicMock()
        # A faithful double: the real `_clear_messages` empties the store, and
        # the hint gate is only correct because `_resume_thread` samples the
        # store *before* calling it. A bare `AsyncMock` here would let the
        # sample move below the clear with every test still green.
        _app_test_double(app)._clear_messages = AsyncMock(
            side_effect=lambda *_, **__: app._message_store.clear()
        )
        _app_test_double(app)._update_status = MagicMock()
        mock_payload = MagicMock()
        mock_payload.messages = []
        mock_payload.context_tokens = 0
        _app_test_double(app)._fetch_thread_history_data = AsyncMock(
            return_value=mock_payload
        )
        _app_test_double(app)._load_thread_history = AsyncMock()
        _app_test_double(app)._mount_message = AsyncMock()
        _app_test_double(app)._thread_links_configured = MagicMock(return_value=True)
        _app_test_double(app).query_one = MagicMock(side_effect=_NoMatches())
        return app

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

    async def test_switch_survives_goal_review_restore_failure(self) -> None:
        """A goal review that cannot be restored must not undo the switch.

        The deferred remount runs inside the block whose handler rolls the
        whole switch back.
        """
        app = self._switch_app()
        mount_message = AsyncMock()
        _app_test_double(app)._mount_message = mount_message
        _app_test_double(app)._remount_pending_goal_rubric_review = AsyncMock(
            side_effect=RuntimeError("rubric restore exploded")
        )

        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=True),
            ),
            patch(
                "deepagents_code.sessions.get_thread_agent",
                AsyncMock(return_value="agent"),
            ),
            patch.object(app, "_schedule_thread_message_link"),
        ):
            await app._resume_thread("new-thread")

        contents = [
            _get_widget_text(call.args[0]) for call in mount_message.call_args_list
        ]
        assert app._session_state is not None
        assert app._session_state.thread_id == "new-thread"
        assert not any("Failed to switch" in text for text in contents)

    async def test_switch_survives_previous_thread_hint_mount_failure(self) -> None:
        """A hint that cannot be mounted must not fail the switch.

        The mount is the half of the hint that is not covered by the
        resumability guard, and a raise here reaches the same rollback handler.
        """
        app = self._switch_app()
        mounted: list[str] = []
        mount_raised = False

        def mount(widget: Static) -> None:
            nonlocal mount_raised
            text = _get_widget_text(widget)
            if text.startswith("Previous thread:"):
                mount_raised = True
                msg = "container is detached"
                raise MountError(msg)
            mounted.append(text)

        _app_test_double(app)._mount_message = AsyncMock(side_effect=mount)

        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=True),
            ),
            patch(
                "deepagents_code.sessions.get_thread_agent",
                AsyncMock(return_value="agent"),
            ),
            patch.object(app, "_schedule_thread_message_link"),
        ):
            await app._resume_thread("new-thread")

        assert app._session_state is not None
        assert app._session_state.thread_id == "new-thread"
        assert not any("Failed to switch" in text for text in mounted)
        # Anchor: without this the test passes when the hint is suppressed
        # before mounting, so the MountError branch is never exercised.
        assert mount_raised

    async def test_successful_switch_rearms_already_on_thread_toast(self) -> None:
        """A real switch clears suppression so the next no-op toasts again.

        Without the reset, re-selecting A after an A -> B -> A round trip would
        be swallowed by the stale suppression entry left by the first no-op.
        Mirrors the same-model counterpart in `test_model_switch.py`.
        """
        app = self._switch_app()
        notify_mock = MagicMock()
        _app_test_double(app).notify = notify_mock
        _app_test_double(app)._offer_thread_cwd_switch = AsyncMock(
            return_value="continue"
        )

        # Hold the clock still so a second toast is attributable to the reset
        # rather than to the toast lifetime quietly expiring.
        with patch("deepagents_code.app._monotonic", return_value=100.0):
            # No-op records the suppression entry.
            await app._resume_thread("old-thread")
            # Real switches away and back must clear it.
            await app._resume_thread("new-thread")
            await app._resume_thread("old-thread")
            # Identical message, same instant on the clock: only the reset can
            # let this through.
            await app._resume_thread("old-thread")

        unchanged_toasts = [
            call.args[0]
            for call in notify_mock.call_args_list
            if call.args[0].startswith("Already on thread")
        ]
        assert unchanged_toasts == [
            "Already on thread: old-thread",
            "Already on thread: old-thread",
        ]

    async def test_failure_after_switch_restores_previous_thread_pointer(self) -> None:
        """A raise after the back-pointer is set must not leave previous == current.

        `previous_thread_id` is recorded once the switch is materially complete,
        but `_run_session_start_hook` runs after that and can raise. Without an
        explicit restore, rollback would put the session back on the outgoing
        thread while the back-pointer still named it, making a later bare
        `/threads -r` a no-op with nowhere to step back to.
        """
        app = self._switch_app()
        session_state = app._session_state
        assert session_state is not None
        session_state.previous_thread_id = "grandparent-thread"
        _app_test_double(app)._offer_thread_cwd_switch = AsyncMock(
            return_value="continue"
        )
        # Raise on the post-switch call; succeed on the rollback call so the
        # rollback path itself completes.
        _app_test_double(app)._run_session_start_hook = AsyncMock(
            side_effect=[RuntimeError("hook exploded"), True]
        )

        await app._resume_thread("new-thread")

        assert session_state.thread_id == "old-thread"
        assert session_state.previous_thread_id == "grandparent-thread"

    async def test_failure_restores_previous_thread_ids(self) -> None:
        """If _clear_messages raises, thread IDs should be restored."""
        from textual.css.query import NoMatches as _NoMatches

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = _mock_session_state("old-thread")
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
        app._session_state = _mock_session_state("old-thread")
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
        app._session_state = _mock_session_state("old-thread")
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


class TestFetchThreadHistoryData:
    """Tests for DeepAgentsApp._fetch_thread_history_data."""

    @staticmethod
    def _skip_conversion(converted: list[MessageData]) -> AbstractContextManager[Any]:
        """Stub message preparation so metadata assertions run on their own.

        Patches the prepare function rather than `asyncio.to_thread`: `app.py`
        offloads hook transcript recording through `to_thread` too, so a
        module-wide stub would silently disable unrelated work and hand back a
        false pass.
        """

        def prepare(
            _messages: list[Any], *, show_reasoning: bool = False
        ) -> tuple[list[MessageData], tuple[()]]:
            assert not show_reasoning
            return converted, ()

        return patch.object(
            DeepAgentsApp,
            "_prepare_thread_history_messages",
            staticmethod(prepare),
        )

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

        with self._skip_conversion(converted):
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

        with self._skip_conversion(converted):
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

        with self._skip_conversion(converted):
            payload = await app._fetch_thread_history_data("tid-1")

        assert payload.model_spec == ""

    async def test_extracts_cache_endpoint_identity(self) -> None:
        """Persisted cache endpoint should propagate to the restore payload."""
        from deepagents_code.tui.widgets.message_store import MessageData, MessageType

        app = DeepAgentsApp()
        app._agent = MagicMock()
        raw_messages = [object()]
        state = MagicMock()
        state.values = {
            "messages": raw_messages,
            "_last_model_request_at": "2026-08-11T12:30:00+00:00",
            "_last_cache_endpoint": "https://api.anthropic.com/v1",
        }
        app._agent.aget_state = AsyncMock(return_value=state)
        converted = [MessageData(type=MessageType.USER, content="hello")]

        with self._skip_conversion(converted):
            payload = await app._fetch_thread_history_data("tid-1")

        assert payload.cache_state is not None
        assert (
            payload.cache_state["_last_cache_endpoint"]
            == "https://api.anthropic.com/v1"
        )

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

        with self._skip_conversion(converted):
            payload = await app._fetch_thread_history_data("tid-1")

        assert payload.context_tokens == 0


class TestLoadThreadHistory:
    """Tests for DeepAgentsApp._load_thread_history."""

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

    async def test_resume_seeds_cost_for_empty_thread_history(self) -> None:
        """Checkpoint metadata restores before the empty-transcript return."""
        from deepagents_code.app import _ThreadHistoryPayload

        app = DeepAgentsApp(thread_id="tid-1")
        app._set_session_cost(9.0)
        app._add_provisional_cost(0.5)
        app._thread_stats.record_request(
            "old-model",
            100,
            10,
            cost_usd=9.0,
        )
        preloaded = _ThreadHistoryPayload(
            messages=[],
            context_tokens=8500,
            model_spec="",
            session_cost_usd=1.25,
        )

        await app._load_thread_history(
            thread_id="tid-1",
            preloaded_payload=preloaded,
        )

        assert app._context_tokens == 8500
        assert app._session_cost_usd == pytest.approx(1.25)
        assert app._thread_restored_cost_usd == pytest.approx(1.25)
        assert app._displayed_cost_usd == pytest.approx(1.25)
        assert app._thread_stats.request_count == 0

    async def test_resume_restores_cache_endpoint_identity(self) -> None:
        """Resumed threads retain the endpoint used for their cached prefix."""
        from deepagents_code.app import _ThreadHistoryPayload

        app = DeepAgentsApp(thread_id="tid-1")
        preloaded = _ThreadHistoryPayload(
            messages=[],
            context_tokens=8500,
            model_spec="anthropic:claude-sonnet-4-6",
            cache_state={
                "_last_model_request_at": "2026-08-11T12:30:00+00:00",
                "_last_cache_model_spec": "anthropic:claude-sonnet-4-6",
                "_last_cache_endpoint": "https://api.anthropic.com/v1",
                "_model_spec": "anthropic:claude-sonnet-4-6",
                "_model_params": None,
            },
        )

        await app._load_thread_history(
            thread_id="tid-1",
            preloaded_payload=preloaded,
        )

        assert app._last_cache_endpoint == "https://api.anthropic.com/v1"

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
        from deepagents_code.config import runtime_state

        app = DeepAgentsApp()
        app._model_override = None
        mounted: list[Static] = []
        _app_test_double(app)._mount_message = AsyncMock(
            side_effect=lambda w: mounted.append(w)
        )

        with (
            patch.object(runtime_state, "model_provider", ""),
            patch.object(runtime_state, "model_name", ""),
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
        """A `/model` override wins over process-wide runtime model state."""
        from deepagents_code.config import runtime_state

        app = DeepAgentsApp()
        app._model_override = "openai:gpt-5.1"
        with (
            patch.object(runtime_state, "model_provider", "anthropic"),
            patch.object(runtime_state, "model_name", "claude-sonnet-4-5"),
        ):
            assert app._effective_model_spec() == "openai:gpt-5.1"

    async def test_falls_back_to_settings_spec(self) -> None:
        """With no override, the resolved runtime `provider:model` is used."""
        from deepagents_code.config import runtime_state

        app = DeepAgentsApp()
        app._model_override = None
        with (
            patch.object(runtime_state, "model_provider", "anthropic"),
            patch.object(runtime_state, "model_name", "claude-sonnet-4-5"),
        ):
            assert app._effective_model_spec() == "anthropic:claude-sonnet-4-5"

    async def test_none_when_spec_incomplete(self) -> None:
        """No override and a blank model yields `None` (no malformed spec)."""
        from deepagents_code.config import runtime_state

        app = DeepAgentsApp()
        app._model_override = None
        with (
            patch.object(runtime_state, "model_provider", "anthropic"),
            patch.object(runtime_state, "model_name", ""),
        ):
            assert app._effective_model_spec() is None


class TestThreadLinksConfigured:
    """Tests for DeepAgentsApp._thread_links_configured."""

    def test_false_without_langsmith_project(self) -> None:
        """Tracing without a project cannot produce thread links."""
        app = DeepAgentsApp()
        with patch(
            "deepagents_code.config.get_langsmith_project_name", return_value=None
        ):
            assert app._thread_links_configured() is False

    def test_true_with_langsmith_project(self) -> None:
        """An active LangSmith project enables thread links."""
        app = DeepAgentsApp()
        with patch(
            "deepagents_code.config.get_langsmith_project_name", return_value="project"
        ):
            assert app._thread_links_configured() is True


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

    async def test_linked_content_matches_plain_app_message_styling(self) -> None:
        """Linked notes carry the same styling `AppMessage` gives plain strings.

        The expected style is read off a real `AppMessage` rather than hardcoded,
        so that if `AppMessage` stops styling plain strings `dim italic` this
        fails instead of silently locking in the divergence it exists to catch.
        `AppMessage` records the style as an unresolved spec string, so compare
        parsed styles rather than span representations.
        """
        from textual.content import Content
        from textual.style import Style as TStyle

        from deepagents_code.tui.widgets.messages import AppMessage

        plain_spans = AppMessage("Previous thread: tid-123").render()._spans
        assert len(plain_spans) == 1
        plain_style = plain_spans[0].style
        expected = (
            TStyle.parse(plain_style) if isinstance(plain_style, str) else plain_style
        )

        app = DeepAgentsApp()
        url = "https://smith.langchain.com/o/org/projects/p/proj/t/tid-123"
        target = "deepagents_code.config.build_langsmith_thread_url"
        with patch(target, return_value=url):
            result = await app._build_thread_message(
                "Previous thread", "tid-123", suffix=" (Resume with /threads -r)"
            )

        assert isinstance(result, Content)
        # Sum the spans to prove every character is styled, not just that the
        # spans present are correct: an unstyled gap is the original regression.
        covered = 0
        for span in result._spans:
            assert isinstance(span.style, TStyle)
            assert span.style.dim == expected.dim
            assert span.style.italic == expected.italic
            covered += span.end - span.start
        assert covered == len(result.plain)

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

    def test_system_prefix_skipped(self) -> None:
        """HumanMessages starting with [SYSTEM] should be skipped."""
        msgs = [
            self._make_human("[SYSTEM] Auto-injected context"),
            self._make_human("Real user message"),
        ]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].content == "Real user message"

    def test_known_internal_sources_are_skipped_without_prefix(self) -> None:
        from langchain_core.messages import HumanMessage

        messages = [
            HumanMessage(
                content=f"hidden {source}",
                additional_kwargs={"lc_source": source},
            )
            for source in (
                "goal_state",
                "goal_control",
                "rubric_grader",
                "summarization",
                "local_context",
            )
        ]
        messages.append(HumanMessage(content="real user message"))

        result = DeepAgentsApp._convert_messages_to_data(messages)

        assert len(result) == 1
        assert result[0].content == "real user message"

    def test_unknown_source_remains_visible(self) -> None:
        from langchain_core.messages import HumanMessage

        result = DeepAgentsApp._convert_messages_to_data(
            [
                HumanMessage(
                    content="connector user message",
                    additional_kwargs={"lc_source": "slack"},
                )
            ]
        )

        assert len(result) == 1
        assert result[0].content == "connector user message"

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

    def test_reloaded_ask_user_row_keeps_its_questions(self) -> None:
        """A reloaded `ask_user` row needs its questions to render answers.

        `_format_ask_user_output` takes its answer count from the structured
        questions in `tool_args` — recovered from the preceding
        `AIMessage.tool_calls[].args`, never from the `ToolMessage`. Without them
        the row degrades to generic formatting, so this pins the args plumbing the
        whole feature rests on.
        """
        from deepagents_code._ask_user_types import ASK_USER_FAILED_SUMMARY
        from deepagents_code.tui.widgets.message_store import ToolStatus
        from deepagents_code.tui.widgets.messages import ToolCallMessage

        args = {"questions": [{"question": "Deploy?"}]}
        msgs = [
            self._make_ai(
                tool_calls=[{"id": "tc-1", "name": "ask_user", "args": args}]
            ),
            self._make_tool(
                "Q: Deploy?\nA: (error: ask_user interaction failed)",
                tool_call_id="tc-1",
                status="error",
            ),
        ]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].tool_args == args
        assert result[0].tool_status == ToolStatus.ERROR

        widget = result[0].to_widget()
        assert isinstance(widget, ToolCallMessage)
        widget._restore_deferred_state()
        formatted = widget._format_ask_user_output(str(widget._output), is_preview=True)
        assert formatted.content.plain == ASK_USER_FAILED_SUMMARY

    def test_checkpoint_edit_restores_diff(self) -> None:
        """Checkpointed edit arguments rebuild the diff omitted from graph state."""
        from deepagents_code.tui.widgets.message_store import MessageType

        tool, diff = DeepAgentsApp._convert_messages_to_data(
            [
                self._make_ai(
                    tool_calls=[
                        {
                            "id": "tc-edit",
                            "name": "edit_file",
                            "args": {
                                "file_path": "a.py",
                                "old_string": "old\n",
                                "new_string": "new\n",
                            },
                        }
                    ]
                ),
                self._make_tool("Updated file", tool_call_id="tc-edit"),
            ]
        )

        assert tool.tool_diff_superseded is True
        assert diff.type == MessageType.DIFF
        assert diff.diff_file_path == "a.py"
        assert "-old" in diff.content
        assert "+new" in diff.content
        widget = diff.to_widget()
        assert all(getattr(row, "selection_prefix", 2) == 2 for row in widget.compose())

    def test_ai_message_reasoning_blocks_follow_preference(self) -> None:
        from deepagents_code.tui.widgets.message_store import MessageType

        messages = [
            self._make_ai(
                [
                    {"type": "text", "text": "Before "},
                    {"type": "reasoning", "reasoning": "Thinking"},
                    {"type": "text", "text": "after"},
                ]
            )
        ]

        hidden = DeepAgentsApp._convert_messages_to_data(messages)
        visible = DeepAgentsApp._convert_messages_to_data(messages, show_reasoning=True)

        assert [(message.type, message.content) for message in hidden] == [
            (MessageType.ASSISTANT, "Before after")
        ]
        assert [(message.type, message.content) for message in visible] == [
            (MessageType.ASSISTANT, "Before "),
            (MessageType.REASONING, "Thinking"),
            (MessageType.ASSISTANT, "after"),
        ]


class TestColumnKeyConsistency:
    """Verify all column dicts stay in sync."""


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

    def test_different_checkpoint_ids_do_not_match(self) -> None:
        """Lists with different checkpoint IDs should not match."""
        a = [self._thread("t1", "cp1")]
        b = [self._thread("t1", "cp2")]
        assert ThreadSelectorScreen._threads_match(a, b) is False


class TestThreadSelectorAgentFilter:
    """Tests for the agent filter dropdown in the Options panel."""

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
