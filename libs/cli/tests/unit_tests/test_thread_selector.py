"""Tests for ThreadSelectorScreen."""

from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.screen import ModalScreen

from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

MOCK_THREADS = [
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


def _patch_list_threads(threads: list[dict] | None = None) -> Any:  # noqa: ANN401
    """Return a patch context manager for list_threads.

    Args:
        threads: Thread list to return. Defaults to MOCK_THREADS.
    """
    data = threads if threads is not None else MOCK_THREADS
    return patch(
        "deepagents_cli.widgets.thread_selector.list_threads",
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
                assert screen._selected_index > 0

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

                # Click the second option
                if len(screen._option_widgets) > 1:
                    await pilot.click(type(screen._option_widgets[1]))
                    await pilot.pause()

                    assert app.dismissed is True
                    assert app.result is not None


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
        thread = {"thread_id": "test123", "updated_at": None}
        label = ThreadSelectorScreen._format_option_label(
            thread, selected=False, current=False
        )
        assert "unknown" in label

    def test_includes_message_count(self) -> None:
        """Label should include message count."""
        label = ThreadSelectorScreen._format_option_label(
            MOCK_THREADS[0], selected=False, current=False
        )
        assert "5 msgs" in label


class TestThreadSelectorErrorHandling:
    """Tests for error handling when loading threads fails."""

    @pytest.mark.asyncio
    async def test_list_threads_error_still_dismissable(self) -> None:
        """Database error should not crash; Escape still works."""
        with patch(
            "deepagents_cli.widgets.thread_selector.list_threads",
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
