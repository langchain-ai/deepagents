"""Unit tests for ChatInput widget and completion popup."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static

from deepagents_cli.widgets.autocomplete import SLASH_COMMANDS
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


class _ChatInputTestApp(App[None]):
    """Minimal app that hosts a ChatInput for testing."""

    def compose(self) -> ComposeResult:
        yield ChatInput(id="chat-input")


def _prompt_text(prompt: Static) -> str:
    """Read the current text content of a Static widget."""
    return str(prompt._Static__content)  # type: ignore[attr-defined]  # accessing internal content store


class TestPromptIndicator:
    """Test that the prompt indicator reflects the current input mode."""

    @pytest.mark.asyncio
    async def test_prompt_shows_bang_in_bash_mode(self) -> None:
        """Setting mode to 'bash' should change prompt to '!' and apply bash styling."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            prompt = chat_input.query_one("#prompt", Static)

            assert _prompt_text(prompt) == ">"
            assert not chat_input.has_class("mode-bash")

            chat_input.mode = "bash"
            await pilot.pause()
            assert _prompt_text(prompt) == "!"
            assert chat_input.has_class("mode-bash")

    @pytest.mark.asyncio
    async def test_prompt_shows_slash_in_command_mode(self) -> None:
        """Setting mode to 'command' should change prompt and styling."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            prompt = chat_input.query_one("#prompt", Static)

            chat_input.mode = "command"
            await pilot.pause()
            assert _prompt_text(prompt) == "/"
            assert chat_input.has_class("mode-command")

    @pytest.mark.asyncio
    async def test_prompt_reverts_to_default_on_normal_mode(self) -> None:
        """Resetting mode to 'normal' should revert indicator and classes."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            prompt = chat_input.query_one("#prompt", Static)

            chat_input.mode = "bash"
            await pilot.pause()
            assert _prompt_text(prompt) == "!"
            assert chat_input.has_class("mode-bash")

            chat_input.mode = "normal"
            await pilot.pause()
            assert _prompt_text(prompt) == ">"
            assert not chat_input.has_class("mode-bash")
            assert not chat_input.has_class("mode-command")

    @pytest.mark.asyncio
    async def test_mode_change_posts_message(self) -> None:
        """Setting mode should post a ModeChanged message."""
        messages: list[ChatInput.ModeChanged] = []

        class RecordingApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ChatInput()

            def on_chat_input_mode_changed(self, event: ChatInput.ModeChanged) -> None:
                messages.append(event)

        app = RecordingApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)

            chat_input.mode = "bash"
            await pilot.pause()
            assert any(m.mode == "bash" for m in messages)


class TestHistoryNavigationFlag:
    """Test that _navigating_history resets when history is exhausted."""

    @pytest.mark.asyncio
    async def test_down_arrow_at_bottom_resets_navigating_flag(self) -> None:
        """Pressing down with no history should not leave _navigating_history stuck."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            assert not text_area._navigating_history

            await pilot.press("down")
            await pilot.pause()

            assert not text_area._navigating_history

    @pytest.mark.asyncio
    async def test_autocomplete_works_after_down_arrow(self) -> None:
        """Typing '/' after pressing down should still trigger completions."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            # Press down at the bottom of empty history
            await pilot.press("down")
            await pilot.pause()

            # Now type '/' — completions should appear
            text_area.insert("/")
            await pilot.pause()

            assert chat_input._completion_manager is not None
            controller = chat_input._completion_manager._active
            assert controller is not None


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


class TestDismissCompletion:
    """Test ChatInput.dismiss_completion edge cases."""

    @pytest.mark.asyncio
    async def test_dismiss_returns_false_when_no_suggestions(self) -> None:
        """dismiss_completion returns False when nothing is shown."""
        app = _ChatInputTestApp()
        async with app.run_test():
            chat = app.query_one("#chat-input", ChatInput)
            assert chat.dismiss_completion() is False

    @pytest.mark.asyncio
    async def test_dismiss_clears_popup_and_state(self) -> None:
        """dismiss_completion hides popup and resets all state."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-input", ChatInput)
            popup = chat.query_one(CompletionPopup)

            # Trigger slash completion
            assert chat._text_area is not None
            chat._text_area.text = "/"
            await pilot.pause()

            # Completion should be active
            assert chat._current_suggestions
            assert popup.styles.display == "block"

            # Dismiss
            result = chat.dismiss_completion()
            assert result is True

            # All state should be cleaned up
            assert chat._current_suggestions == []
            assert popup.styles.display == "none"
            assert chat._text_area._completion_active is False

    @pytest.mark.asyncio
    async def test_dismiss_is_idempotent(self) -> None:
        """Calling dismiss_completion twice is safe."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-input", ChatInput)

            assert chat._text_area is not None
            chat._text_area.text = "/"
            await pilot.pause()
            assert chat._current_suggestions

            assert chat.dismiss_completion() is True
            # Second call is a no-op
            assert chat.dismiss_completion() is False

    @pytest.mark.asyncio
    async def test_completion_reappears_after_dismiss(self) -> None:
        """Typing / after dismiss_completion re-opens the menu."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-input", ChatInput)
            popup = chat.query_one(CompletionPopup)

            assert chat._text_area is not None

            # Show → dismiss
            chat._text_area.text = "/"
            await pilot.pause()
            assert chat._current_suggestions
            chat.dismiss_completion()

            # Clear input and retype /
            chat._text_area.text = ""
            await pilot.pause()
            chat._text_area.text = "/"
            await pilot.pause()

            # Menu should reappear with all commands
            assert len(chat._current_suggestions) == len(SLASH_COMMANDS)
            assert popup.styles.display == "block"

    @pytest.mark.asyncio
    async def test_popup_hide_cancels_pending_rebuild(self) -> None:
        """Hiding the popup clears pending suggestions so a stale rebuild is a no-op."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            popup = app.query_one(CompletionPopup)

            # Schedule a rebuild then immediately hide
            popup.update_suggestions([("/help", "Show help")], selected_index=0)
            popup.hide()

            # Let the queued _rebuild_options run
            await pilot.pause()

            # Popup should remain hidden with no option widgets
            assert popup.styles.display == "none"
            assert popup.query(CompletionOption) is not None  # query exists
            assert len(popup.query(CompletionOption)) == 0
