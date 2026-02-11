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
