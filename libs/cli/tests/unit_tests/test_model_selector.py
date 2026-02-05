"""Tests for ModelSelectorScreen."""

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container

from deepagents_cli.widgets.model_selector import ModelSelectorScreen


class ModelSelectorTestApp(App):
    """Test app for ModelSelectorScreen."""

    def __init__(self) -> None:
        super().__init__()
        self.result: tuple[str, str] | None = None
        self.dismissed = False

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def show_selector(self) -> None:
        """Show the model selector screen."""

        def handle_result(result: tuple[str, str] | None) -> None:
            self.result = result
            self.dismissed = True

        screen = ModelSelectorScreen(
            current_model="claude-sonnet-4-5",
            current_provider="anthropic",
        )
        self.push_screen(screen, handle_result)


class TestModelSelectorEscapeKey:
    """Tests for ESC key dismissing the modal."""

    @pytest.mark.asyncio
    async def test_escape_dismisses_modal(self) -> None:
        """Pressing ESC should dismiss the modal with None result."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            # Press ESC - this should dismiss the modal
            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is None

    @pytest.mark.asyncio
    async def test_escape_works_when_input_focused(self) -> None:
        """ESC should work even when the filter input is focused."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            # Type something to ensure input is focused
            await pilot.press("c", "l", "a", "u", "d", "e")
            await pilot.pause()

            # Press ESC - should still dismiss
            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is None


class TestModelSelectorKeyboardNavigation:
    """Tests for keyboard navigation in the modal."""

    @pytest.mark.asyncio
    async def test_down_arrow_moves_selection(self) -> None:
        """Down arrow should move selection down."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)
            initial_index = screen._selected_index

            await pilot.press("down")
            await pilot.pause()

            assert screen._selected_index == initial_index + 1

    @pytest.mark.asyncio
    async def test_up_arrow_moves_selection(self) -> None:
        """Up arrow should move selection up (wrapping to end)."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)
            assert screen._selected_index == 0

            await pilot.press("up")
            await pilot.pause()

            # Should wrap to last item
            expected = len(screen._filtered_models) - 1
            assert screen._selected_index == expected

    @pytest.mark.asyncio
    async def test_enter_selects_model(self) -> None:
        """Enter should select the current model and dismiss."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is not None
            assert isinstance(app.result, tuple)
            assert len(app.result) == 2


class TestModelSelectorFiltering:
    """Tests for search filtering."""

    @pytest.mark.asyncio
    async def test_typing_filters_models(self) -> None:
        """Typing in the filter input should filter models."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Type a filter
            await pilot.press("c", "l", "a", "u", "d", "e")
            await pilot.pause()

            assert screen._filter_text == "claude"

    @pytest.mark.asyncio
    async def test_custom_model_spec_entry(self) -> None:
        """User can enter a custom provider:model spec."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            # Type a custom model spec
            for char in "custom:my-model":
                await pilot.press(char)
            await pilot.pause()

            # Press enter to select
            await pilot.press("enter")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result == ("custom:my-model", "custom")
