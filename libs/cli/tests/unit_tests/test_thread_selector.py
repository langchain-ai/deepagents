"""Tests for ThreadSelectorScreen."""

from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.style import Style
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.sessions import ThreadInfo
from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

MOCK_THREADS: list[ThreadInfo] = [
    {
        "thread_id": "abc12345",
        "agent_name": "my-agent",
        "updated_at": "2025-01-15T10:30:00",
        "message_count": 5,
    },
    {
        "thread_id": "def67890",
        "agent_name": "other-agent",
        "updated_at": "2025-01-14T08:00:00",
        "message_count": 12,
    },
    {
        "thread_id": "ghi11111",
        "agent_name": "my-agent",
        "updated_at": "2025-01-13T15:45:00",
        "message_count": 3,
    },
]


def _patch_list_threads(threads: list[ThreadInfo] | None = None) -> Any:  # noqa: ANN401
    """Return a patch context manager for `list_threads`.

    Args:
        threads: Thread list to return. Defaults to `MOCK_THREADS`.
    """
    data = threads if threads is not None else MOCK_THREADS
    return patch(
        "deepagents_cli.sessions.list_threads",
        new_callable=AsyncMock,
        return_value=data,
    )


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

        screen = ThreadSelectorScreen(current_thread=self._current_thread)
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

        screen = ThreadSelectorScreen(current_thread="abc12345")
        self.push_screen(screen, handle_result)


class TestThreadSelectorEscapeKey:
    """Tests for ESC key dismissing the modal."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_j_k_navigation(self) -> None:
        """j/k keys should navigate like down/up arrows."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                await pilot.press("j")
                await pilot.pause()
                assert screen._selected_index == 1

                await pilot.press("k")
                await pilot.pause()
                assert screen._selected_index == 0

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_arrow_keys_on_empty_list_do_not_crash(self) -> None:
        """Arrow keys, j/k, and page keys on empty list should be no-ops."""
        with _patch_list_threads(threads=[]):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                assert len(screen._threads) == 0

                # All navigation keys should be safe on an empty list
                for key in ("up", "down", "j", "k", "pageup", "pagedown"):
                    await pilot.press(key)
                    await pilot.pause()

                assert screen._selected_index == 0

                await pilot.press("escape")
                await pilot.pause()
                assert app.dismissed is True


class TestThreadSelectorNavigateAndSelect:
    """Tests for navigating then selecting a specific thread."""

    @pytest.mark.asyncio
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


class TestThreadSelectorTabNavigation:
    """Tests for tab/shift+tab navigation."""

    @pytest.mark.asyncio
    async def test_tab_moves_down(self) -> None:
        """Tab should move selection down."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                await pilot.press("tab")
                await pilot.pause()
                assert screen._selected_index == 1

    @pytest.mark.asyncio
    async def test_shift_tab_moves_up(self) -> None:
        """Shift+tab should move selection up."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                # Move down first, then shift+tab back
                await pilot.press("tab")
                await pilot.pause()
                assert screen._selected_index == 1

                await pilot.press("shift+tab")
                await pilot.pause()
                assert screen._selected_index == 0


class TestThreadSelectorDownWrap:
    """Tests for wrapping from bottom to top."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


class TestThreadSelectorClickHandling:
    """Tests for mouse click handling."""

    @pytest.mark.asyncio
    async def test_click_selects_thread(self) -> None:
        """Clicking a thread option should select and dismiss."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                # Post a Clicked message from the second option widget.
                # (pilot.click(type) always hits the first match, so we
                # exercise the handler directly for an exact-widget test.)
                from deepagents_cli.widgets.thread_selector import ThreadOption

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


_WEBBROWSER_OPEN = "deepagents_cli.widgets._links.webbrowser.open"


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


class TestThreadSelectorFormatLabel:
    """Tests for _format_option_label static method."""

    def test_selected_shows_cursor(self) -> None:
        """Selected option should include a cursor glyph."""
        label = ThreadSelectorScreen._format_option_label(
            MOCK_THREADS[0], selected=True, current=False
        )
        # Should not start with spaces (cursor glyph present)
        assert not label.startswith("  ")

    def test_unselected_has_no_cursor(self) -> None:
        """Unselected option should start with spaces instead of cursor."""
        label = ThreadSelectorScreen._format_option_label(
            MOCK_THREADS[0], selected=False, current=False
        )
        assert label.startswith("  ")

    def test_current_shows_suffix(self) -> None:
        """Current thread should show (current) suffix."""
        label = ThreadSelectorScreen._format_option_label(
            MOCK_THREADS[0], selected=False, current=True
        )
        assert "(current)" in label

    def test_not_current_no_suffix(self) -> None:
        """Non-current thread should not show (current) suffix."""
        label = ThreadSelectorScreen._format_option_label(
            MOCK_THREADS[0], selected=False, current=False
        )
        assert "(current)" not in label

    def test_missing_agent_name_shows_unknown(self) -> None:
        """Thread with no agent_name should show 'unknown'."""
        thread = ThreadInfo(thread_id="test123", agent_name=None, updated_at=None)
        label = ThreadSelectorScreen._format_option_label(
            thread, selected=False, current=False
        )
        assert "unknown" in label

    def test_includes_message_count(self) -> None:
        """Label should include message count."""
        label = ThreadSelectorScreen._format_option_label(
            MOCK_THREADS[0], selected=False, current=False
        )
        assert "5" in label

    def test_columns_align_with_header(self) -> None:
        """Option labels should align with the column header."""
        header = ThreadSelectorScreen._format_header()
        label = ThreadSelectorScreen._format_option_label(
            MOCK_THREADS[0], selected=False, current=False
        )
        # "Thread" column starts at the same offset as the thread ID
        assert header.index("Thread") == label.index("abc12345")

    def test_long_values_are_truncated(self) -> None:
        """Thread ID and agent name exceeding column width are truncated."""
        thread = ThreadInfo(
            thread_id="abcdef1234567890",
            agent_name="very-long-agent-name-here",
            updated_at=None,
            message_count=0,
        )
        label = ThreadSelectorScreen._format_option_label(
            thread, selected=False, current=False
        )
        # Thread ID column is 10 chars, agent column is 14 chars
        assert "abcdef1234567890" not in label
        assert "abcdef1234" in label
        assert "very-long-agent-name-here" not in label
        assert "very-long-agen" in label


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
        """Title with a LangSmith URL should produce a Rich Text with a link."""
        from rich.text import Text

        screen = ThreadSelectorScreen(current_thread="abc12345")
        title = screen._build_title(
            thread_url="https://smith.langchain.com/p/t/abc12345"
        )
        assert isinstance(title, Text)
        assert "abc12345" in title.plain

        # Verify the thread ID span carries a cyan + link style
        spans = [s for s in title._spans if s.style and "link" in str(s.style)]
        assert len(spans) > 0
        assert "cyan" in str(spans[0].style)

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_successful_url_updates_title(self) -> None:
        """Background worker should update the title with a clickable link."""
        from rich.text import Text

        with (
            _patch_list_threads(),
            patch(
                "deepagents_cli.widgets.thread_selector.build_langsmith_thread_url",
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
                content = title_widget._Static__content
                assert isinstance(content, Text)
                assert "abc12345" in content.plain

    @pytest.mark.asyncio
    async def test_timeout_leaves_title_unchanged(self) -> None:
        """Timeout during URL resolution should not crash or change the title."""
        import time

        def _blocking(_tid: str) -> str:
            time.sleep(3)
            return "https://example.com"

        with (
            _patch_list_threads(),
            patch(
                "deepagents_cli.widgets.thread_selector.build_langsmith_thread_url",
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
                assert isinstance(title_widget._Static__content, str)

    @pytest.mark.asyncio
    async def test_oserror_leaves_title_unchanged(self) -> None:
        """OSError during URL resolution should not crash or change the title."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_cli.widgets.thread_selector.build_langsmith_thread_url",
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
                assert isinstance(title_widget._Static__content, str)

    @pytest.mark.asyncio
    async def test_unexpected_exception_leaves_title_unchanged(self) -> None:
        """Unexpected exception should not crash the thread selector."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_cli.widgets.thread_selector.build_langsmith_thread_url",
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
                assert isinstance(title_widget._Static__content, str)

    @pytest.mark.asyncio
    async def test_none_url_leaves_title_unchanged(self) -> None:
        """When build returns None the title should remain a plain string."""
        with (
            _patch_list_threads(),
            patch(
                "deepagents_cli.widgets.thread_selector.build_langsmith_thread_url",
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
                content = title_widget._Static__content
                assert isinstance(content, str)
                assert "abc12345" in content


class TestThreadSelectorColumnHeader:
    """Tests for the anchored column header."""

    def test_header_contains_column_names(self) -> None:
        """Column header string should contain all column names."""
        header = ThreadSelectorScreen._format_header()
        assert "Thread" in header
        assert "Agent" in header
        assert "Msgs" in header
        assert "Updated" in header

    @pytest.mark.asyncio
    async def test_header_widget_is_mounted(self) -> None:
        """Column header widget should be present in the mounted screen."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)
                screen.query_one(".thread-list-header", Static)

    @pytest.mark.asyncio
    async def test_header_stays_outside_scroll(self) -> None:
        """Header should be outside VerticalScroll (anchored, not scrollable)."""
        with _patch_list_threads():
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, ThreadSelectorScreen)

                header = screen.query_one(".thread-list-header", Static)
                # Header's parent should be the Vertical, not VerticalScroll
                assert isinstance(header.parent, Vertical)


class TestThreadSelectorErrorHandling:
    """Tests for error handling when loading threads fails."""

    @pytest.mark.asyncio
    async def test_list_threads_error_still_dismissable(self) -> None:
        """Database error should not crash; Escape still works."""
        with patch(
            "deepagents_cli.sessions.list_threads",
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

                # No option widgets should have been created
                assert len(screen._option_widgets) == 0

                # Escape should still dismiss
                await pilot.press("escape")
                await pilot.pause()

                assert app.dismissed is True
                assert app.result is None


class TestThreadSelectorLimit:
    """Tests for thread limit via get_thread_limit()."""

    @pytest.mark.asyncio
    async def test_custom_limit_is_forwarded(self) -> None:
        """get_thread_limit() return value should be forwarded to list_threads."""
        with (
            patch(
                "deepagents_cli.sessions.get_thread_limit",
                return_value=5,
            ),
            _patch_list_threads() as mock_lt,
        ):
            app = ThreadSelectorTestApp()
            async with app.run_test() as pilot:
                app.show_selector()
                await pilot.pause()

                mock_lt.assert_awaited_once_with(limit=5, include_message_count=True)


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

    @pytest.mark.asyncio
    async def test_no_agent_shows_error(self) -> None:
        """_resume_thread with no agent should show an error message."""
        app = DeepAgentsApp()
        mounted: list[Static] = []
        app._mount_message = AsyncMock(side_effect=lambda w: mounted.append(w))  # type: ignore[assignment]
        app._agent = None

        await app._resume_thread("thread-123")

        assert len(mounted) == 1
        assert "no active agent" in _get_widget_text(mounted[0])

    @pytest.mark.asyncio
    async def test_no_session_state_shows_error(self) -> None:
        """_resume_thread with no session state should show an error message."""
        app = DeepAgentsApp()
        mounted: list[Static] = []
        app._mount_message = AsyncMock(side_effect=lambda w: mounted.append(w))  # type: ignore[assignment]
        app._agent = MagicMock()
        app._session_state = None

        await app._resume_thread("thread-123")

        assert len(mounted) == 1
        assert "no active session" in _get_widget_text(mounted[0])

    @pytest.mark.asyncio
    async def test_already_on_thread_shows_message(self) -> None:
        """_resume_thread when already on the thread should show info message."""
        app = DeepAgentsApp()
        mounted: list[Static] = []
        app._mount_message = AsyncMock(side_effect=lambda w: mounted.append(w))  # type: ignore[assignment]
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "thread-123"

        await app._resume_thread("thread-123")

        assert len(mounted) == 1
        assert "Already on thread" in _get_widget_text(mounted[0])

    @pytest.mark.asyncio
    async def test_successful_switch_updates_ids(self) -> None:
        """Successful _resume_thread should update thread IDs and load history."""
        from textual.css.query import NoMatches as _NoMatches

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        app._pending_messages = MagicMock()
        app._queued_widgets = MagicMock()
        app._clear_messages = AsyncMock()  # type: ignore[assignment]
        app._token_tracker = MagicMock()
        app._update_status = MagicMock()  # type: ignore[assignment]
        app._load_thread_history = AsyncMock()  # type: ignore[assignment]
        app._mount_message = AsyncMock()  # type: ignore[assignment]
        app.query_one = MagicMock(side_effect=_NoMatches())  # type: ignore[assignment]

        await app._resume_thread("new-thread")

        assert app._lc_thread_id == "new-thread"
        assert app._session_state.thread_id == "new-thread"
        app._pending_messages.clear.assert_called_once()
        app._queued_widgets.clear.assert_called_once()
        app._clear_messages.assert_awaited_once()
        app._token_tracker.reset.assert_called_once()
        app._load_thread_history.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failure_restores_previous_thread_ids(self) -> None:
        """If _clear_messages raises, thread IDs should be restored."""
        from textual.css.query import NoMatches as _NoMatches

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        app._pending_messages = MagicMock()
        app._queued_widgets = MagicMock()
        app._clear_messages = AsyncMock(side_effect=RuntimeError("UI gone"))  # type: ignore[assignment]
        app._mount_message = AsyncMock()  # type: ignore[assignment]
        app.query_one = MagicMock(side_effect=_NoMatches())  # type: ignore[assignment]

        await app._resume_thread("new-thread")

        # Thread IDs should be restored to previous values
        assert app._lc_thread_id == "old-thread"
        assert app._session_state.thread_id == "old-thread"
        # Should show error message
        assert any(
            "Failed to switch" in _get_widget_text(call.args[0])
            for call in app._mount_message.call_args_list  # type: ignore[union-attr]
        )

    @pytest.mark.asyncio
    async def test_failure_during_load_history_restores_ids(self) -> None:
        """If _load_thread_history raises, thread IDs should be rolled back."""
        from textual.css.query import NoMatches as _NoMatches

        app = DeepAgentsApp(thread_id="old-thread")
        app._agent = MagicMock()
        app._session_state = MagicMock()
        app._session_state.thread_id = "old-thread"
        app._pending_messages = MagicMock()
        app._queued_widgets = MagicMock()
        app._clear_messages = AsyncMock()  # type: ignore[assignment]
        app._token_tracker = MagicMock()
        app._update_status = MagicMock()  # type: ignore[assignment]
        # First call (in try block) fails; second call (in rollback) succeeds
        app._load_thread_history = AsyncMock(  # type: ignore[assignment]
            side_effect=[RuntimeError("checkpoint corrupt"), None]
        )
        app._mount_message = AsyncMock()  # type: ignore[assignment]
        app.query_one = MagicMock(side_effect=_NoMatches())  # type: ignore[assignment]

        await app._resume_thread("new-thread")

        assert app._lc_thread_id == "old-thread"
        assert app._session_state.thread_id == "old-thread"
        assert any(
            "Failed to switch" in _get_widget_text(call.args[0])
            for call in app._mount_message.call_args_list  # type: ignore[union-attr]
        )


class TestBuildThreadMessage:
    """Tests for DeepAgentsApp._build_thread_message."""

    @pytest.mark.asyncio
    async def test_plain_text_when_tracing_not_configured(self) -> None:
        """Returns plain string when LangSmith URL is not available."""
        app = DeepAgentsApp()
        with patch("deepagents_cli.app.build_langsmith_thread_url", return_value=None):
            result = await app._build_thread_message("Resumed thread", "tid-123")

        assert result == "Resumed thread: tid-123"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_hyperlinked_when_tracing_configured(self) -> None:
        """Returns Rich Text with hyperlink when LangSmith URL is available."""
        from rich.text import Text

        app = DeepAgentsApp()
        url = "https://smith.langchain.com/o/org/projects/p/proj/t/tid-123"
        with patch("deepagents_cli.app.build_langsmith_thread_url", return_value=url):
            result = await app._build_thread_message("Resumed thread", "tid-123")

        assert isinstance(result, Text)
        assert "Resumed thread: " in result.plain
        assert "tid-123" in result.plain
        # Verify the thread ID span has the link style
        spans = [s for s in result._spans if s.style and "link" in str(s.style)]
        assert len(spans) == 1
        assert url in str(spans[0].style)

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self) -> None:
        """Returns plain string when URL resolution times out."""
        app = DeepAgentsApp()
        with patch(
            "deepagents_cli.app.asyncio.wait_for",
            side_effect=TimeoutError,
        ):
            result = await app._build_thread_message("Resumed thread", "t-1")

        assert isinstance(result, str)
        assert result == "Resumed thread: t-1"

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self) -> None:
        """Returns plain string when URL resolution raises an exception."""
        app = DeepAgentsApp()
        with patch(
            "deepagents_cli.app.build_langsmith_thread_url",
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

        return AIMessage(content=content, tool_calls=tool_calls or [])  # type: ignore[no-matching-overload]

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
        from deepagents_cli.widgets.message_store import MessageType

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
        from deepagents_cli.widgets.message_store import MessageType

        msgs = [self._make_ai("Here is the answer.")]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].type == MessageType.ASSISTANT
        assert result[0].content == "Here is the answer."

    def test_ai_message_content_block_list(self) -> None:
        """AIMessage with list-of-blocks content should extract text."""
        from deepagents_cli.widgets.message_store import MessageType

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
        from deepagents_cli.widgets.message_store import MessageType, ToolStatus

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
        from deepagents_cli.widgets.message_store import ToolStatus

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
        from deepagents_cli.widgets.message_store import ToolStatus

        msgs = [
            self._make_ai(tool_calls=[{"id": "tc-3", "name": "bash", "args": {}}]),
        ]
        result = DeepAgentsApp._convert_messages_to_data(msgs)

        assert len(result) == 1
        assert result[0].tool_status == ToolStatus.REJECTED

    def test_mixed_message_sequence(self) -> None:
        """Full conversation with mixed message types should convert correctly."""
        from deepagents_cli.widgets.message_store import MessageType, ToolStatus

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
