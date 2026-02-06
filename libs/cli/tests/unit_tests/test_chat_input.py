"""Unit tests for ChatInput widget and completion popup."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container

from deepagents_cli.widgets.chat_input import (
    ChatInput,
    CompletionOption,
    CompletionPopup,
)


class TestCompletionOption:
    """Test CompletionOption widget."""

    def test_clicked_message_contains_index(self) -> None:
        """Clicked message should contain the option index."""
        message = CompletionOption.Clicked(index=2)
        assert message.index == 2

    def test_init_stores_attributes(self) -> None:
        """CompletionOption should store label, description, index, and state."""
        option = CompletionOption(
            label="/help",
            description="Show help",
            index=1,
            is_selected=True,
        )
        assert option._label == "/help"
        assert option._description == "Show help"
        assert option._index == 1
        assert option._is_selected is True

    def test_set_selected_updates_state(self) -> None:
        """set_selected should update internal state."""
        option = CompletionOption(
            label="/help",
            description="Show help",
            index=0,
            is_selected=False,
        )
        assert option._is_selected is False

        option.set_selected(selected=True)
        assert option._is_selected is True

        option.set_selected(selected=False)
        assert option._is_selected is False


class TestCompletionPopup:
    """Test CompletionPopup widget."""

    def test_option_clicked_message_contains_index(self) -> None:
        """OptionClicked message should contain the clicked index."""
        message = CompletionPopup.OptionClicked(index=3)
        assert message.index == 3

    def test_init_state(self) -> None:
        """CompletionPopup should initialize with empty options."""
        popup = CompletionPopup()
        assert popup._options == []
        assert popup._selected_index == 0
        assert popup.can_focus is False


class TestCompletionPopupIntegration:
    """Integration tests for CompletionPopup with Textual."""

    @pytest.mark.asyncio
    async def test_update_suggestions_shows_popup(self) -> None:
        """update_suggestions should show the popup when given suggestions."""

        class TestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield CompletionPopup(id="popup")

        app = TestApp()
        async with app.run_test() as pilot:
            popup = app.query_one("#popup", CompletionPopup)

            # Initially hidden
            assert popup.styles.display == "none"

            # Update with suggestions
            popup.update_suggestions(
                [("/help", "Show help"), ("/clear", "Clear chat")],
                selected_index=0,
            )
            await pilot.pause()

            # Should be visible
            assert popup.styles.display == "block"

    @pytest.mark.asyncio
    async def test_update_suggestions_creates_option_widgets(self) -> None:
        """update_suggestions should create CompletionOption widgets."""

        class TestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield CompletionPopup(id="popup")

        app = TestApp()
        async with app.run_test() as pilot:
            popup = app.query_one("#popup", CompletionPopup)

            popup.update_suggestions(
                [("/help", "Show help"), ("/clear", "Clear chat")],
                selected_index=0,
            )
            # Allow async rebuild to complete
            await pilot.pause()

            # Should have created 2 option widgets
            options = popup.query(CompletionOption)
            assert len(options) == 2

    @pytest.mark.asyncio
    async def test_empty_suggestions_hides_popup(self) -> None:
        """Empty suggestions should hide the popup."""

        class TestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield CompletionPopup(id="popup")

        app = TestApp()
        async with app.run_test() as pilot:
            popup = app.query_one("#popup", CompletionPopup)

            # Show popup first
            popup.update_suggestions(
                [("/help", "Show help")],
                selected_index=0,
            )
            await pilot.pause()
            assert popup.styles.display == "block"

            # Hide with empty suggestions
            popup.update_suggestions([], selected_index=0)
            await pilot.pause()

            assert popup.styles.display == "none"


class TestCompletionOptionClick:
    """Test click handling on CompletionOption."""

    @pytest.mark.asyncio
    async def test_click_on_option_posts_message(self) -> None:
        """Clicking on an option should post a Clicked message."""

        class TestApp(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.clicked_indices: list[int] = []

            def compose(self) -> ComposeResult:
                with Container():
                    yield CompletionOption(
                        label="/help",
                        description="Show help",
                        index=0,
                        id="opt0",
                    )
                    yield CompletionOption(
                        label="/clear",
                        description="Clear chat",
                        index=1,
                        id="opt1",
                    )

            def on_completion_option_clicked(
                self, event: CompletionOption.Clicked
            ) -> None:
                self.clicked_indices.append(event.index)

        app = TestApp()
        async with app.run_test() as pilot:
            # Click on first option
            opt0 = app.query_one("#opt0", CompletionOption)
            await pilot.click(opt0)

            assert 0 in app.clicked_indices

            # Click on second option
            opt1 = app.query_one("#opt1", CompletionOption)
            await pilot.click(opt1)

            assert 1 in app.clicked_indices


class TestCompletionPopupClickBubbling:
    """Test that clicks on options bubble up through the popup."""

    @pytest.mark.asyncio
    async def test_popup_receives_option_click_and_posts_message(self) -> None:
        """Popup should receive option clicks and post OptionClicked message."""

        class TestApp(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.option_clicked_indices: list[int] = []

            def compose(self) -> ComposeResult:
                yield CompletionPopup(id="popup")

            def on_completion_popup_option_clicked(
                self, event: CompletionPopup.OptionClicked
            ) -> None:
                self.option_clicked_indices.append(event.index)

        app = TestApp()
        async with app.run_test() as pilot:
            popup = app.query_one("#popup", CompletionPopup)

            # Add suggestions to create option widgets
            popup.update_suggestions(
                [("/help", "Show help"), ("/clear", "Clear chat")],
                selected_index=0,
            )
            await pilot.pause()

            # Click on the first option
            options = popup.query(CompletionOption)
            await pilot.click(options[0])

            assert 0 in app.option_clicked_indices

            # Click on second option
            await pilot.click(options[1])
            assert 1 in app.option_clicked_indices
