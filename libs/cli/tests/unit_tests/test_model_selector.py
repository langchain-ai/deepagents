"""Tests for ModelSelectorScreen."""

from typing import ClassVar

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.screen import ModalScreen

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


class AppWithEscapeBinding(App):
    """Test app that has a conflicting escape binding like DeepAgentsApp.

    This reproduces the real-world scenario where the app binds escape
    to action_interrupt, which would intercept escape before the modal.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.result: tuple[str, str] | None = None
        self.dismissed = False
        self.interrupt_called = False

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def action_interrupt(self) -> None:
        """Handle escape - dismiss modal if present, otherwise mark as called."""
        if isinstance(self.screen, ModalScreen):
            self.screen.dismiss(None)
            return
        self.interrupt_called = True

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

    @pytest.mark.asyncio
    async def test_escape_with_conflicting_app_binding(self) -> None:
        """ESC should dismiss modal even when app has its own escape binding.

        This test reproduces the bug where DeepAgentsApp's escape binding
        for action_interrupt would intercept escape before the modal could
        handle it, causing the modal to not close.
        """
        app = AppWithEscapeBinding()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            # Press ESC - this should dismiss the modal, not call action_interrupt
            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is None
            # The interrupt action should NOT have been called because modal was open
            assert app.interrupt_called is False


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
        """Up arrow should move selection up (wrapping to end if at 0)."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)
            initial_index = screen._selected_index
            count = len(screen._filtered_models)

            await pilot.press("up")
            await pilot.pause()

            # Should move up by one, wrapping if at 0
            expected = (initial_index - 1) % count
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

    @pytest.mark.asyncio
    async def test_enter_selects_highlighted_model_not_filter_text(self) -> None:
        """Enter selects highlighted model, not raw filter text."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Type a partial spec with colon that matches existing models
            for char in "anthropic:claude":
                await pilot.press(char)
            await pilot.pause()

            # Should have filtered results
            assert len(screen._filtered_models) > 0

            # Press enter - should select the highlighted model, not raw text
            await pilot.press("enter")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is not None
            # Result should be a full model spec from the list, not "anthropic:claude"
            model_spec, provider = app.result
            assert model_spec != "anthropic:claude"
            assert provider == "anthropic"


class TestModelSelectorCurrentModelPreselection:
    """Tests for pre-selecting the current model when opening the selector."""

    @pytest.mark.asyncio
    async def test_current_model_is_preselected(self) -> None:
        """Opening the selector should pre-select the current model, not first."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # The test app sets current model to "anthropic:claude-sonnet-4-5"
            # Find its index in the filtered models
            current_spec = "anthropic:claude-sonnet-4-5"
            expected_index = None
            for i, (model_spec, _) in enumerate(screen._filtered_models):
                if model_spec == current_spec:
                    expected_index = i
                    break

            assert expected_index is not None, f"{current_spec} not found in models"
            assert screen._selected_index == expected_index, (
                f"Expected current model at index {expected_index} to be selected, "
                f"but index {screen._selected_index} was selected instead"
            )

    @pytest.mark.asyncio
    async def test_clearing_filter_reselects_current_model(self) -> None:
        """Clearing the filter should re-select the current model."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Find the current model's index
            current_spec = "anthropic:claude-sonnet-4-5"
            current_index = None
            for i, (model_spec, _) in enumerate(screen._filtered_models):
                if model_spec == current_spec:
                    current_index = i
                    break
            assert current_index is not None

            # Type something that filters to no/few results
            await pilot.press("x", "y", "z")
            await pilot.pause()

            # Now clear the filter by backspacing
            await pilot.press("backspace", "backspace", "backspace")
            await pilot.pause()

            # Selection should be back to the current model
            assert screen._selected_index == current_index, (
                f"After clearing filter, expected index {current_index} "
                f"but got {screen._selected_index}"
            )
