"""Unit tests for ChatInput widget and completion popup."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static

from deepagents_cli.command_registry import SLASH_COMMANDS
from deepagents_cli.input import MediaTracker
from deepagents_cli.widgets import chat_input as chat_input_module
from deepagents_cli.widgets.autocomplete import MAX_SUGGESTIONS
from deepagents_cli.widgets.chat_input import (
    ChatInput,
    ChatTextArea,
    CompletionOption,
    CompletionPopup,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from textual.pilot import Pilot


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


class _RecordingApp(App[None]):
    """App that records ChatInput.Submitted events for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[ChatInput.Submitted] = []

    def compose(self) -> ComposeResult:
        yield ChatInput(id="chat-input")

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        self.submitted.append(event)


class _ImagePasteApp(App[None]):
    """App that wires a shared tracker into ChatInput for paste tests."""

    def __init__(self) -> None:
        super().__init__()
        self.tracker = MediaTracker()

    def compose(self) -> ComposeResult:
        yield ChatInput(id="chat-input", image_tracker=self.tracker)


class _ImagePasteRecordingApp(App[None]):
    """App that records submitted values while using image tracker wiring."""

    def __init__(self) -> None:
        super().__init__()
        self.tracker = MediaTracker()
        self.submitted: list[ChatInput.Submitted] = []

    def compose(self) -> ComposeResult:
        yield ChatInput(id="chat-input", image_tracker=self.tracker)

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        self.submitted.append(event)


async def _pause_for_strip(pilot: Pilot[None]) -> None:
    """Wait two frames so the prefix-strip text-change event propagates."""
    await pilot.pause()
    await pilot.pause()


def _prompt_text(prompt: Static) -> str:
    """Read the current text content of a Static widget."""
    return str(prompt._Static__content)  # type: ignore[attr-defined]  # accessing internal content store


def _render_text_area_line(text_area: ChatTextArea, y: int = 0) -> str:
    """Render a text-area line and trim widget padding for assertions."""
    return text_area.render_line(y).text.rstrip()


class TestPromptIndicator:
    """Test that the prompt indicator reflects the current input mode."""

    async def test_prompt_shows_bang_in_shell_mode(self) -> None:
        """Mode 'shell' should change prompt to '!' and apply styling."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            prompt = chat_input.query_one("#prompt", Static)

            assert _prompt_text(prompt) == ">"
            assert not chat_input.has_class("mode-shell")

            chat_input.mode = "shell"
            await pilot.pause()
            assert _prompt_text(prompt) == "$"
            assert chat_input.has_class("mode-shell")

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

    async def test_prompt_reverts_to_default_on_normal_mode(self) -> None:
        """Resetting mode to 'normal' should revert indicator and classes."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            prompt = chat_input.query_one("#prompt", Static)

            chat_input.mode = "shell"
            await pilot.pause()
            assert _prompt_text(prompt) == "$"
            assert chat_input.has_class("mode-shell")

            chat_input.mode = "normal"
            await pilot.pause()
            assert _prompt_text(prompt) == ">"
            assert not chat_input.has_class("mode-shell")
            assert not chat_input.has_class("mode-command")

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

            chat_input.mode = "shell"
            await pilot.pause()
            assert any(m.mode == "shell" for m in messages)


class TestHistoryNavigationFlag:
    """Test that _skip_history_change_events resets when history is exhausted."""

    async def test_down_arrow_at_bottom_resets_navigating_flag(self) -> None:
        """Pressing down with no history should not leave the skip counter stuck."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            assert text_area._skip_history_change_events == 0

            await pilot.press("down")
            await pilot.pause()

            assert text_area._skip_history_change_events == 0

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

            # Now type '/' — the prefix is stripped but completions appear
            # via the virtual prefix path.
            text_area.insert("/")
            await _pause_for_strip(pilot)

            assert chat_input.mode == "command"
            assert chat_input._completion_manager is not None
            controller = chat_input._completion_manager._active
            assert controller is not None

    async def test_counter_resets_after_successful_recall(self) -> None:
        """Counter should return to 0 after a history entry is recalled."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            # Seed history with an entry
            chat_input._history._entries.append("previous entry")

            # Recall via up arrow (cursor starts at (0,0) on empty input)
            await pilot.press("up")
            await pilot.pause()

            assert text_area.text == "previous entry"
            assert text_area._skip_history_change_events == 0

    async def test_autocomplete_works_after_history_recall(self) -> None:
        """Typing '/' after recalling history should trigger completions."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            # Seed and recall a history entry
            chat_input._history._entries.append("previous entry")
            await pilot.press("up")
            await pilot.pause()
            assert text_area.text == "previous entry"

            # Clear and type '/' — autocomplete should activate
            text_area.clear_text()
            await pilot.pause()
            text_area.insert("/")
            await _pause_for_strip(pilot)

            assert chat_input.mode == "command"
            assert chat_input._completion_manager is not None
            controller = chat_input._completion_manager._active
            assert controller is not None

    async def test_multiple_rapid_recalls_drain_counter(self) -> None:
        """Multiple set_text_from_history calls should each reserve a skip."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            # Call set_text_from_history twice without letting events process
            text_area.set_text_from_history("first")
            text_area.set_text_from_history("second")
            assert text_area._skip_history_change_events == 2

            # Let both Changed events fire and drain the counter
            await pilot.pause()
            await pilot.pause()
            assert text_area._skip_history_change_events == 0
            assert text_area.text == "second"

    async def test_clear_text_suppresses_own_changed_event(self) -> None:
        """clear_text increments the counter so its Changed event is skipped."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            # Recall a history entry, then immediately clear
            chat_input._history._entries.append("recalled")
            await pilot.press("up")
            await pilot.pause()
            assert text_area.text == "recalled"

            text_area.clear_text()
            # Counter should be 1 (for the clear's own Changed event)
            assert text_area._skip_history_change_events == 1
            await pilot.pause()
            assert text_area._skip_history_change_events == 0

    async def test_negative_counter_resets_with_warning(self) -> None:
        """Defensive check: negative counter is logged and reset to 0."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            # Force counter negative (simulates a bug elsewhere)
            text_area._skip_history_change_events = -1
            text_area.insert("x")
            await pilot.pause()

            assert text_area._skip_history_change_events == 0


class TestHistoryBoundaryNavigation:
    """Test that history navigation only triggers at input boundaries."""

    async def test_up_arrow_only_triggers_at_cursor_start(self) -> None:
        """Up arrow should only navigate history when cursor is at (0, 0)."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("previous entry")

            # Type some text — cursor ends up at the end
            chat._text_area.insert("hello")
            await pilot.pause()
            assert chat._text_area.cursor_location == (0, 5)

            # Up arrow should NOT trigger history (cursor not at start)
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "hello"

    async def test_up_arrow_triggers_at_cursor_zero(self) -> None:
        """Up arrow should navigate history when cursor is at (0, 0)."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("say hello world")

            # Type text then move cursor to start
            chat._text_area.insert("hello")
            await pilot.pause()
            chat._text_area.move_cursor((0, 0))
            await pilot.pause()

            # Up arrow should trigger history (cursor at start)
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "say hello world"

    async def test_down_arrow_navigates_from_start_when_in_history(self) -> None:
        """Down arrow at start navigates history when `_in_history` is True."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.extend(["first", "second"])

            # Navigate into history first (cursor at start on empty)
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "second"

            # Move cursor to start — still a boundary
            chat._text_area.move_cursor((0, 0))
            await pilot.pause()

            # _in_history is True so down at start (boundary) still navigates
            await pilot.press("down")
            await pilot.pause()
            assert chat._text_area.text == ""

    async def test_down_arrow_does_not_trigger_at_non_end(self) -> None:
        """Down arrow should not navigate history when cursor is not at end."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("previous entry")

            # Type text — cursor ends up at the end
            chat._text_area.insert("hello world")
            await pilot.pause()

            # Move cursor to middle (not at end)
            chat._text_area.move_cursor((0, 5))
            await pilot.pause()

            # Down arrow should NOT trigger history
            await pilot.press("down")
            await pilot.pause()
            assert chat._text_area.text == "hello world"

    async def test_down_arrow_at_end_triggers_history(self) -> None:
        """Down arrow at end of text should navigate history forward."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.extend(["first", "second"])

            # Navigate up twice into history
            await pilot.press("up")
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "first"

            # Cursor should be at end after set_text_from_history
            # Down arrow at end should navigate forward
            await pilot.press("down")
            await pilot.pause()
            assert chat._text_area.text == "second"

    async def test_up_at_middle_of_multiline_does_not_trigger(self) -> None:
        """Up arrow on a middle line should not navigate history."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("previous entry")

            # Insert multiline text and place cursor on line 1
            chat._text_area.text = "line one\nline two\nline three"
            await pilot.pause()
            chat._text_area.move_cursor((1, 3))
            await pilot.pause()

            # Up arrow should move cursor, not navigate history
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "line one\nline two\nline three"

    async def test_in_history_allows_up_from_end(self) -> None:
        """When browsing history, up arrow at end should also navigate."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.extend(["first", "second"])

            # Navigate into history
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "second"
            assert chat._text_area._in_history is True

            # Cursor is at end after set_text_from_history; up should
            # still navigate because _in_history is True and at boundary
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "first"

    async def test_in_history_resets_after_submission(self) -> None:
        """Submitting should clear the _in_history flag."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("recalled entry")

            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area._in_history is True

            await pilot.press("enter")
            await pilot.pause()
            assert chat._text_area._in_history is False

    async def test_in_history_resets_after_navigating_past_end(self) -> None:
        """Pressing down past history end should set `_in_history` to False."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("only entry")

            # Navigate up into history
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "only entry"
            assert chat._text_area._in_history is True

            # Navigate down past the end — returns to original (empty) input
            await pilot.press("down")
            await pilot.pause()
            assert chat._text_area.text == ""
            assert chat._text_area._in_history is False


class TestCompletionPopupClickBubbling:
    """Test that clicks on options bubble up through the popup."""

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

    async def test_dismiss_returns_false_when_no_suggestions(self) -> None:
        """dismiss_completion returns False when nothing is shown."""
        app = _ChatInputTestApp()
        async with app.run_test():
            chat = app.query_one("#chat-input", ChatInput)
            assert chat.dismiss_completion() is False

    async def test_dismiss_clears_popup_and_state(self) -> None:
        """dismiss_completion hides popup and resets all state."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-input", ChatInput)
            popup = chat.query_one(CompletionPopup)

            # Trigger slash completion — the "/" prefix is stripped from the
            # text area but completions appear via virtual prefix synthesis.
            assert chat._text_area is not None
            chat._text_area.text = "/"
            await _pause_for_strip(pilot)

            # Completion should be active
            assert chat.mode == "command"
            assert chat._current_suggestions
            assert popup.styles.display == "block"

            # Dismiss
            result = chat.dismiss_completion()
            assert result is True

            # All state should be cleaned up
            assert chat._current_suggestions == []
            assert popup.styles.display == "none"
            assert chat._text_area._completion_active is False

    async def test_dismiss_is_idempotent(self) -> None:
        """Calling dismiss_completion twice is safe."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-input", ChatInput)

            assert chat._text_area is not None
            chat._text_area.text = "/"
            await _pause_for_strip(pilot)
            assert chat._current_suggestions

            assert chat.dismiss_completion() is True
            # Second call is a no-op
            assert chat.dismiss_completion() is False

    async def test_completion_reappears_after_dismiss(self) -> None:
        """Typing / after dismiss_completion re-opens the menu."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one("#chat-input", ChatInput)
            popup = chat.query_one(CompletionPopup)

            assert chat._text_area is not None

            # Show → dismiss
            chat._text_area.text = "/"
            await _pause_for_strip(pilot)
            assert chat._current_suggestions
            chat.dismiss_completion()

            # Clear input — mode persists (backspace-on-empty exits)
            chat._text_area.text = ""
            await pilot.pause()
            assert chat.mode == "command"

            # Exit mode via backspace on empty
            await pilot.press("backspace")
            await pilot.pause()
            assert chat.mode == "normal"

            # Retype / — prefix stripped, mode becomes command, completions appear
            chat._text_area.text = "/"
            await _pause_for_strip(pilot)

            # Menu should reappear with all commands
            assert len(chat._current_suggestions) == min(
                len(SLASH_COMMANDS), MAX_SUGGESTIONS
            )
            assert popup.styles.display == "block"

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


class TestModePrefixStripping:
    """Test that mode-trigger characters are stripped from text input."""

    async def test_typing_bang_strips_prefix_and_sets_shell_mode(self) -> None:
        """Setting text to `'!ls'` should strip to `'ls'` and enter shell mode."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "!ls"
            await _pause_for_strip(pilot)

            assert chat.mode == "shell"
            assert chat._text_area.text == "ls"

    async def test_typing_slash_strips_prefix_and_sets_command_mode(self) -> None:
        """Setting text to `'/'` should strip to `''` and enter command mode."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "/"
            await _pause_for_strip(pilot)

            assert chat.mode == "command"
            assert chat._text_area.text == ""

    async def test_mode_stays_on_empty_text(self) -> None:
        """Clearing text after entering shell mode should stay in mode."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter shell mode
            chat._text_area.text = "!ls"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"

            # Clear text — mode should persist (backspace on empty exits)
            chat._text_area.text = ""
            await pilot.pause()
            assert chat.mode == "shell"

    async def test_backspace_on_empty_exits_mode(self) -> None:
        """Backspace on empty input in shell mode should reset to normal."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter shell mode
            chat._text_area.text = "!ls"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"

            # Clear text — still in shell mode
            chat._text_area.text = ""
            await pilot.pause()
            assert chat.mode == "shell"

            # Backspace on empty — exits mode
            await pilot.press("backspace")
            await pilot.pause()
            assert chat.mode == "normal"

    async def test_backspace_on_single_char_stays_in_mode(self) -> None:
        """Deleting last char in command mode should stay in mode, not exit."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter command mode and type a character
            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"

            chat._text_area.insert("h")
            await pilot.pause()
            assert chat._text_area.text == "h"

            # Backspace deletes 'h' — should stay in command mode
            await pilot.press("backspace")
            await pilot.pause()
            assert chat._text_area.text == ""
            assert chat.mode == "command"

            # Second backspace on empty — exits mode
            await pilot.press("backspace")
            await pilot.pause()
            assert chat.mode == "normal"

    async def test_backspace_at_cursor_zero_with_text_exits_mode(self) -> None:
        """Backspace at cursor position 0 with text after cursor exits mode."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter command mode and type some text
            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"

            chat._text_area.insert("help")
            await pilot.pause()
            assert chat._text_area.text == "help"

            # Move cursor to position 0 (beginning of field)
            chat._text_area.move_cursor((0, 0))
            await pilot.pause()

            # Backspace at position 0 with text after cursor — should exit mode
            await pilot.press("backspace")
            await pilot.pause()
            assert chat.mode == "normal"

    async def test_backspace_exit_mode_dismisses_completion(self) -> None:
        """Exiting mode via backspace-on-empty should hide the completion popup."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            popup = chat.query_one(CompletionPopup)
            assert chat._text_area is not None

            # Enter command mode — completions appear
            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"
            assert chat._current_suggestions

            # Backspace on empty — exits mode and hides popup
            await pilot.press("backspace")
            await pilot.pause()
            assert chat.mode == "normal"
            assert chat._current_suggestions == []
            assert popup.styles.display == "none"

    async def test_slash_completion_works_after_strip(self) -> None:
        """Entering command mode and typing `'h'` should trigger completions."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Type "/" to enter command mode
            chat._text_area.text = "/"
            await _pause_for_strip(pilot)
            assert chat.mode == "command"

            # Now type "h" — the virtual prefix makes the controller see "/h"
            chat._text_area.text = "h"
            await pilot.pause()

    async def test_submission_prepends_shell_prefix(self) -> None:
        """Submitting in shell mode should prepend `'!'` to the value."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter shell mode
            chat._text_area.text = "!ls"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"
            assert chat._text_area.text == "ls"

            # Submit
            await pilot.press("enter")
            await pilot.pause()

            # Should have received "!ls"
            assert len(app.submitted) == 1
            assert app.submitted[0].value == "!ls"
            assert app.submitted[0].mode == "shell"

    async def test_submission_prepends_command_prefix(self) -> None:
        """Submitting in command mode should prepend `'/'` to the value."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter command mode — "/" is stripped, then type command text.
            # Use insert() rather than .text= so cursor stays at end, as
            # it would in real typing.
            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"

            # Dismiss completion so Enter takes the direct submission path
            chat.dismiss_completion()

            chat._text_area.insert("help")
            await pilot.pause()

            # Submit — text is "help", mode is "command"
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "/help"
            assert app.submitted[0].mode == "command"

    async def test_mode_resets_after_submission(self) -> None:
        """Mode should reset to normal after submitting."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter shell mode and submit
            chat._text_area.text = "!ls"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"

            await pilot.press("enter")
            await pilot.pause()

            assert chat.mode == "normal"
            assert chat._text_area.text == ""

    async def test_mode_sticky_during_typing(self) -> None:
        """Mode should persist while typing in shell/command mode."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter shell mode
            chat._text_area.text = "!echo hello"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"
            assert chat._text_area.text == "echo hello"

            # Continue typing — mode stays shell
            chat._text_area.text = "echo hello world"
            await pilot.pause()
            assert chat.mode == "shell"

    async def test_shell_mode_does_not_trigger_completions(self) -> None:
        """Typing in shell mode should not trigger completions."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "!echo"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"
            assert chat._current_suggestions == []

    async def test_submission_does_not_double_prefix(self) -> None:
        """If text already starts with prefix, submission should not add another."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Manually set mode and text that already has prefix
            chat.mode = "shell"
            chat._stripping_prefix = True  # prevent mode re-detection
            chat._text_area.text = "!already-prefixed"
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "!already-prefixed"


class TestExitModePreservesText:
    """Exiting shell/command mode should preserve typed text."""

    async def test_exit_shell_mode_keeps_text(self) -> None:
        """Pressing Escape in shell mode should switch to normal but keep text."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter shell mode with some text
            chat._text_area.text = "!ls -la"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"
            assert chat._text_area.text == "ls -la"

            # Exit mode — text should be preserved
            assert chat.exit_mode() is True
            assert chat.mode == "normal"
            assert chat._text_area.text == "ls -la"

    async def test_exit_command_mode_keeps_text(self) -> None:
        """Pressing Escape in command mode should switch to normal but keep text."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"

            chat.dismiss_completion()
            chat._text_area.insert("help")
            await pilot.pause()
            assert chat._text_area.text == "help"

            assert chat.exit_mode() is True
            assert chat.mode == "normal"
            assert chat._text_area.text == "help"


class TestHistoryRecallModeReset:
    """Regression: history recall must not inherit a stale shell/command mode."""

    async def test_history_non_prefixed_entry_resets_shell_mode(self) -> None:
        """Recalling a normal-mode entry while in shell mode should reset to normal."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Seed history with a normal-mode entry
            chat._history._entries.append("echo hello")

            # Enter shell mode, then clear text so the history query is
            # empty (matches all entries) — we're testing mode reset, not
            # substring filtering.
            chat._text_area.text = "!ls"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"
            chat._text_area.text = ""
            await pilot.pause()

            # Press up to recall the non-prefixed history entry through
            # the ChatInput handler (which normalizes mode).
            await pilot.press("up")
            await pilot.pause()

            # Mode must have reset to normal
            assert chat.mode == "normal"
            assert chat._text_area.text == "echo hello"

            # Submitting should NOT prepend "!"
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "echo hello"
            assert app.submitted[0].mode == "normal"

    async def test_history_prefixed_entry_keeps_mode(self) -> None:
        """Recalling a shell-prefixed entry should re-enter shell mode."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Seed history with a shell-mode entry
            chat._history._entries.append("!ls")

            # Press up to recall the prefixed entry
            await pilot.press("up")
            await _pause_for_strip(pilot)

            assert chat.mode == "shell"
            assert chat._text_area.text == "ls"

            # Submit — should prepend "!"
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "!ls"
            assert app.submitted[0].mode == "shell"

    async def test_history_non_prefixed_entry_resets_command_mode(self) -> None:
        """Recalling a normal entry while in command mode should reset to normal."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Seed history with a normal-mode entry
            chat._history._entries.append("hello world")

            # Enter command mode
            chat._text_area.text = "/"
            await _pause_for_strip(pilot)
            assert chat.mode == "command"

            # Dismiss completion so up arrow goes to history, not completion nav
            chat.dismiss_completion()

            # Recall the non-prefixed entry
            await pilot.press("up")
            await pilot.pause()

            assert chat.mode == "normal"

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "hello world"
            assert app.submitted[0].mode == "normal"


class TestSlashCompletionCursorMapping:
    """Regression: virtual-to-real index translation for slash replacement."""

    async def test_tab_completion_mid_token_preserves_suffix(self) -> None:
        """Applying slash completion mid-token should keep text after cursor."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter command mode through typed input so cursor is at end.
            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("he")
            await pilot.pause()
            assert chat.mode == "command"
            assert chat._text_area.text == "he"
            await pilot.press("left")
            await pilot.pause()

            # Apply selected slash completion via keyboard path.
            await pilot.press("tab")
            await _pause_for_strip(pilot)

            assert chat._text_area.text == "help e"

    async def test_click_completion_mid_token_preserves_suffix(self) -> None:
        """Click-selecting slash completion mid-token should keep suffix text."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("he")
            await pilot.pause()
            await pilot.press("left")
            await pilot.pause()

            chat.on_completion_popup_option_clicked(
                CompletionPopup.OptionClicked(index=0)
            )
            await _pause_for_strip(pilot)

            assert chat._text_area.text == "help e"

    async def test_click_completion_at_end_updates_hint_without_extra_frame(
        self,
    ) -> None:
        """Click-selecting a command should render final text immediately."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("re")
            await pilot.pause()

            chat.on_completion_popup_option_clicked(
                CompletionPopup.OptionClicked(index=0)
            )

            assert chat._text_area.text == "remember "
            assert chat._text_area.argument_hint == "[context]"
            assert _render_text_area_line(chat._text_area) == "remember [context]"

    async def test_tab_completion_at_end_replaces_whole_token(self) -> None:
        """Tab-completing at end should replace all typed command text."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter command mode through typed input so cursor is at end.
            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("he")
            await pilot.pause()
            assert chat.mode == "command"
            assert chat._text_area.text == "he"

            await pilot.press("tab")
            await _pause_for_strip(pilot)

            assert chat._text_area.text == "help "

    async def test_normal_mode_replace_is_unaffected(self) -> None:
        """In normal mode (no prefix), coordinates pass through unchanged."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "hello @wor"
            await pilot.pause()
            assert chat.mode == "normal"

            # Replace @wor (positions 6..10) with @world
            chat.replace_completion_range(6, 10, "@world")
            await pilot.pause()

            assert chat._text_area.text == "hello @world "


class TestHistorySlashPrefixRecall:
    """Test that recalling a slash-prefixed history entry enters command mode."""

    async def test_history_slash_prefixed_entry_enters_command_mode(self) -> None:
        """Recalling a `/help` history entry should enter command mode."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("/help")

            await pilot.press("up")
            await _pause_for_strip(pilot)

            assert chat.mode == "command"
            assert chat._text_area.text == "help"

            chat.dismiss_completion()
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "/help"
            assert app.submitted[0].mode == "command"


class TestCompletionIndexToTextIndex:
    """Edge-case tests for _completion_index_to_text_index clamping."""

    async def test_negative_mapped_index_clamps_to_zero(self) -> None:
        """A completion index below the prefix length should clamp to 0."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter command mode so prefix_len == 1
            chat._text_area.text = "/"
            await _pause_for_strip(pilot)
            assert chat.mode == "command"

            # index=0 in completion space -> 0 - 1 = -1 -> clamped to 0
            assert chat._completion_index_to_text_index(0) == 0

    async def test_overflow_index_clamps_to_text_length(self) -> None:
        """A completion index beyond text length should clamp to len(text)."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "/he"
            await _pause_for_strip(pilot)
            # text is now "he" (len 2), prefix_len is 1
            # index=100 -> 100 - 1 = 99 -> clamped to 2
            assert chat._completion_index_to_text_index(100) == 2

    async def test_normal_mode_passes_through(self) -> None:
        """In normal mode (prefix_len=0), index maps 1:1."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "hello"
            await pilot.pause()
            assert chat._completion_index_to_text_index(3) == 3


class TestHistoryRecallSuppressesCompletions:
    """Test that history navigation does not trigger completions."""

    async def test_history_recall_does_not_trigger_completions(self) -> None:
        """Recalling a history entry with '@' should not open file completions."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("tell me about @package.json")

            await pilot.press("up")
            await pilot.pause()

            assert chat._text_area.text == "tell me about @package.json"
            assert chat._current_suggestions == []


class TestDroppedImagePaste:
    """Tests for drag/drop image-path handling via paste events."""

    async def test_forward_delete_removes_placeholder(self, tmp_path) -> None:
        """Forward-delete should remove `[image N]` as a single token."""
        img_path = tmp_path / "fwddelete.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="magenta")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(str(img_path))
            await pilot.pause()
            assert chat._text_area.text == "[image 1] "

            # Move cursor to start and press forward-delete
            chat._text_area.move_cursor((0, 0))
            await pilot.pause()
            await pilot.press("delete")
            await pilot.pause()

            # Forward-delete removes the placeholder token but not the
            # trailing space (unlike backspace which catches it).
            assert "[image" not in chat._text_area.text
            assert app.tracker.get_images() == []
            assert app.tracker.next_image_id == 1

    async def test_backspace_removes_full_image_placeholder(self, tmp_path) -> None:
        """Backspace should remove `[image N]` as a single token."""
        img_path = tmp_path / "backspace.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="cyan")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(str(img_path))
            await pilot.pause()
            assert chat._text_area.text == "[image 1] "

            await pilot.press("backspace")
            await pilot.pause()

            assert chat._text_area.text == ""
            assert app.tracker.get_images() == []
            assert app.tracker.next_image_id == 1

    async def test_readding_after_delete_restarts_image_counter(self, tmp_path) -> None:
        """Re-adding after deleting all placeholders should restart at `[image 1]`."""
        img_path = tmp_path / "readd.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="red")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(str(img_path))
            await pilot.pause()
            assert chat._text_area.text == "[image 1] "

            await pilot.press("backspace")
            await pilot.pause()
            assert app.tracker.next_image_id == 1

            chat.handle_external_paste(str(img_path))
            await pilot.pause()
            assert chat._text_area.text == "[image 1] "
            assert len(app.tracker.get_images()) == 1
            assert app.tracker.next_image_id == 2

    async def test_handle_external_paste_attaches_dropped_image(self, tmp_path) -> None:
        """External paste routing should attach dropped images."""
        img_path = tmp_path / "external.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="blue")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            assert chat.handle_external_paste(str(img_path))
            await pilot.pause()

            assert chat._text_area.text.strip() == "[image 1]"
            assert len(app.tracker.get_images()) == 1

    async def test_handle_external_paste_attaches_unquoted_path_with_spaces(
        self, tmp_path
    ) -> None:
        """External paste should attach raw absolute paths that include spaces."""
        img_path = tmp_path / "Screenshot 1.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="orange")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            assert chat.handle_external_paste(str(img_path))
            await pilot.pause()

            assert chat._text_area.text.strip() == "[image 1]"
            assert len(app.tracker.get_images()) == 1

    async def test_handle_external_paste_inserts_plain_text(self) -> None:
        """External paste should insert text when payload is not a file path."""
        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            assert chat.handle_external_paste("hello world")
            await pilot.pause()

            assert chat._text_area.text == "hello world"
            assert app.tracker.get_images() == []

    async def test_paste_image_path_attaches_image_and_inserts_placeholder(
        self, tmp_path
    ) -> None:
        """Pasting a dropped image path should attach and insert `[image N]`."""
        img_path = tmp_path / "drop.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="blue")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await chat._text_area._on_paste(events.Paste(str(img_path)))
            await pilot.pause()

            assert chat._text_area.text.strip() == "[image 1]"
            assert len(app.tracker.get_images()) == 1

    async def test_paste_non_image_path_keeps_original_text(self, tmp_path) -> None:
        """Non-image dropped paths should keep the default path paste behavior."""
        file_path = tmp_path / "notes.txt"
        file_path.write_text("hello")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await chat._text_area._on_paste(events.Paste(str(file_path)))
            await pilot.pause()

            assert chat._text_area.text.endswith(str(file_path).lstrip("/"))
            assert app.tracker.get_images() == []

    async def test_inline_quoted_path_payload_rewrites_to_placeholder(
        self, tmp_path
    ) -> None:
        """Quoted dropped path text should rewrite inline to `[image N]`."""
        img_path = tmp_path / "vscode-drop.png"
        from PIL import Image

        image = Image.new("RGB", (3, 3), color="teal")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Simulate terminals that drop paths as plain quoted text.
            chat._text_area.text = f"'{img_path}'"
            await pilot.pause()

            assert chat._text_area.text == "[image 1] "
            assert len(app.tracker.get_images()) == 1

    async def test_key_burst_quoted_path_rewrites_without_showing_raw_path(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fast quoted-path key bursts should flush as `[image N]` placeholders."""
        # This test exercises burst parsing behavior, not scheduler precision.
        # CI workers can exceed the default 30ms inter-key gap, which would
        # flush mid-sequence and make the test flaky.
        monkeypatch.setattr(chat_input_module, "_PASTE_BURST_CHAR_GAP_SECONDS", 1.0)
        monkeypatch.setattr(chat_input_module, "_PASTE_BURST_FLUSH_DELAY_SECONDS", 0.25)

        img_path = tmp_path / "vscode-burst.png"
        from PIL import Image

        image = Image.new("RGB", (3, 3), color="navy")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            payload = f"'{img_path}'"
            for char in payload:
                await chat._text_area._on_key(events.Key(char, char))

            # Burst text is buffered and should not be inserted verbatim.
            assert chat._text_area.text == ""

            await pilot.pause(0.35)

            assert chat._text_area.text == "[image 1] "
            assert len(app.tracker.get_images()) == 1

    async def test_submit_absolute_path_without_paste_event_attaches_image(
        self, tmp_path
    ) -> None:
        """Submission should still attach when terminal inserts path as plain text."""
        img_path = tmp_path / "dragged.png"
        from PIL import Image

        image = Image.new("RGB", (3, 3), color="green")
        image.save(img_path, format="PNG")

        app = _ImagePasteRecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Simulate terminals that insert dropped paths as regular text.
            chat._text_area.text = str(img_path)
            await pilot.pause()

            assert chat.mode == "normal"
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "[image 1]"
            assert app.submitted[0].mode == "normal"
            assert len(app.tracker.get_images()) == 1

    async def test_submit_absolute_path_with_spaces_stays_normal_mode(
        self, tmp_path
    ) -> None:
        """Absolute paths with spaces should not trigger slash-command mode."""
        img_path = tmp_path / "Screenshot 1.png"
        from PIL import Image

        image = Image.new("RGB", (3, 3), color="green")
        image.save(img_path, format="PNG")

        app = _ImagePasteRecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Simulate terminals that insert dropped paths as regular text.
            chat._text_area.text = str(img_path)
            await pilot.pause()

            assert chat.mode == "normal"
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "[image 1]"
            assert app.submitted[0].mode == "normal"
            assert len(app.tracker.get_images()) == 1

    async def test_submit_absolute_path_with_spaces_and_trailing_text(
        self, tmp_path
    ) -> None:
        """Path-with-spaces plus prompt text should stay normal and attach image."""
        img_path = tmp_path / "Screenshot 1.png"
        from PIL import Image

        image = Image.new("RGB", (3, 3), color="green")
        image.save(img_path, format="PNG")

        app = _ImagePasteRecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = f"{img_path} what's in this"
            await pilot.pause()

            assert chat.mode == "normal"
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "[image 1] what's in this"
            assert app.submitted[0].mode == "normal"
            assert len(app.tracker.get_images()) == 1

    async def test_submit_leading_path_with_trailing_text_attaches_image(
        self, tmp_path
    ) -> None:
        """Leading pasted path should attach while preserving trailing prompt text."""
        img_path = tmp_path / "leading-path.png"
        from PIL import Image

        image = Image.new("RGB", (3, 3), color="green")
        image.save(img_path, format="PNG")

        app = _ImagePasteRecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = f"'{img_path}' what's in this image?"
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "[image 1] what's in this image?"
            assert app.submitted[0].mode == "normal"
            assert len(app.tracker.get_images()) == 1

    async def test_submit_falls_back_to_leading_image_when_full_path_non_image(
        self, tmp_path
    ) -> None:
        """Leading image token should win over full non-image payload resolution."""
        img_path = tmp_path / "fallback.png"
        from PIL import Image

        image = Image.new("RGB", (3, 3), color="green")
        image.save(img_path, format="PNG")

        payload_path = tmp_path / "fallback.png analyze"
        payload_path.write_text("not an image")

        app = _ImagePasteRecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = str(payload_path)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "[image 1] analyze"
            assert app.submitted[0].mode == "normal"
            assert len(app.tracker.get_images()) == 1

    async def test_submit_leading_path_handles_unicode_space_variants(
        self, tmp_path
    ) -> None:
        """Submitted leading path should recover Unicode-space filename variants."""
        from PIL import Image

        img_path = tmp_path / "Screenshot 2026-02-26 at 2.02.42\u202fAM.png"
        image = Image.new("RGB", (3, 3), color="green")
        image.save(img_path, format="PNG")

        pasted_with_ascii_space = str(img_path).replace("\u202f", " ")

        app = _ImagePasteRecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = f"'{pasted_with_ascii_space}' analyze this"
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "[image 1] analyze this"
            assert app.submitted[0].mode == "normal"
            assert len(app.tracker.get_images()) == 1

    async def test_sync_resumes_after_submit_skip(self, tmp_path) -> None:
        """Image tracker sync should resume after the post-submit skip event."""
        img_path = tmp_path / "sync_resume.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="yellow")
        image.save(img_path, format="PNG")

        app = _ImagePasteRecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Paste an image and submit
            chat.handle_external_paste(str(img_path))
            await pilot.pause()
            assert chat._text_area.text == "[image 1] "

            await pilot.press("enter")
            await pilot.pause()

            # After submit, the skip counter fires for the clear_text event.
            # Typing new text should now sync normally (tracker is cleared).
            chat._text_area.insert("hello")
            await pilot.pause()

            # The tracker should have synced and cleared images since
            # the new text has no placeholders.
            assert app.tracker.get_images() == []
            assert app.tracker.next_image_id == 1

    async def test_submit_recovers_if_command_mode_already_stripped_path(
        self, tmp_path
    ) -> None:
        """If slash mode stripped a dropped path, submission should recover it."""
        img_path = tmp_path / "recover.png"
        from PIL import Image

        image = Image.new("RGB", (2, 2), color="purple")
        image.save(img_path, format="PNG")

        app = _ImagePasteRecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Simulate previously stripped leading slash.
            chat.mode = "command"
            chat._text_area.text = str(img_path).lstrip("/")
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "[image 1]"
            assert app.submitted[0].mode == "normal"
            assert len(app.tracker.get_images()) == 1


def _make_mp4_bytes() -> bytes:
    """Return minimal valid MP4 ftyp box bytes."""
    return (
        b"\x00\x00\x00\x14"  # box size (20 bytes)
        b"ftyp"  # box type
        b"mp42"  # major brand
        b"\x00\x00\x00\x00"  # minor version
        b"mp42"  # compatible brand
    )


class TestDroppedVideoPaste:
    """Tests for drag/drop video-path handling via paste events."""

    async def test_paste_video_attaches_and_inserts_placeholder(
        self, tmp_path: Path
    ) -> None:
        """Dropping a valid .mp4 should insert `[video 1]` placeholder."""
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(_make_mp4_bytes())

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            assert chat.handle_external_paste(str(video_path))
            await pilot.pause()

            assert "[video 1]" in chat._text_area.text
            assert len(app.tracker.get_videos()) == 1

    async def test_backspace_removes_video_placeholder(self, tmp_path: Path) -> None:
        """Backspace should remove `[video N]` as a single token."""
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(_make_mp4_bytes())

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(str(video_path))
            await pilot.pause()
            assert "[video 1]" in chat._text_area.text

            await pilot.press("backspace")
            await pilot.pause()

            assert "[video" not in chat._text_area.text
            assert app.tracker.get_videos() == []
            assert app.tracker.next_video_id == 1

    async def test_forward_delete_removes_video_placeholder(
        self, tmp_path: Path
    ) -> None:
        """Forward-delete should remove `[video N]` as a single token."""
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(_make_mp4_bytes())

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(str(video_path))
            await pilot.pause()
            assert "[video 1]" in chat._text_area.text

            chat._text_area.move_cursor((0, 0))
            await pilot.pause()
            await pilot.press("delete")
            await pilot.pause()

            assert "[video" not in chat._text_area.text
            assert app.tracker.get_videos() == []

    async def test_mixed_image_and_video_drop(self, tmp_path: Path) -> None:
        """Dropping an image and video should produce both placeholder types."""
        from PIL import Image

        img_path = tmp_path / "photo.png"
        image = Image.new("RGB", (4, 4), color="red")
        image.save(img_path, format="PNG")

        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(_make_mp4_bytes())

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            payload = f"{img_path}\n{video_path}"
            chat.handle_external_paste(payload)
            await pilot.pause()

            text = chat._text_area.text
            assert "[image 1]" in text
            assert "[video 1]" in text
            assert len(app.tracker.get_images()) == 1
            assert len(app.tracker.get_videos()) == 1


class TestBackslashEnterNewline:
    """Test that backslash followed quickly by enter inserts a newline.

    Some terminals (e.g. VSCode built-in) send a literal backslash followed
    by enter when the user presses shift+enter.  The widget detects this
    pair and collapses it into a newline.
    """

    async def test_backslash_then_enter_inserts_newline(self) -> None:
        """Rapid backslash + enter should produce a newline, not submit."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.insert("hello")
            await pilot.pause()

            await pilot.press("backslash")
            await pilot.press("enter")
            await pilot.pause()

            assert "\n" in ta.text
            assert "\\" not in ta.text
            assert len(app.submitted) == 0

    async def test_backslash_alone_inserts_normally(self) -> None:
        """A lone backslash should be inserted immediately as normal text."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            await pilot.press("backslash")
            await pilot.pause()

            assert ta.text == "\\"

    async def test_backslash_then_letter_inserts_both(self) -> None:
        """Backslash followed by a letter should insert both characters."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            await pilot.press("backslash")
            await pilot.press("a")
            await pilot.pause()

            assert ta.text == "\\a"

    async def test_backslash_enter_on_empty_prompt_does_not_submit(self) -> None:
        """Backslash + enter on empty prompt should not submit."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            await pilot.press("backslash")
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 0
            assert "\\" not in ta.text
            assert ta.text == "\n"

    async def test_backslash_then_slow_enter_submits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backslash + enter beyond the timing gap should submit normally."""
        # Set gap to 0 so any real delay exceeds it.
        monkeypatch.setattr(chat_input_module, "_BACKSLASH_ENTER_GAP_SECONDS", 0.0)

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.insert("hello")
            await pilot.pause()

            await pilot.press("backslash")
            await asyncio.sleep(0.05)
            await pilot.press("enter")
            await pilot.pause()

            # Should have submitted (backslash included in text)
            assert len(app.submitted) == 1


class TestVSCodeSpaceWorkaround:
    """VS Code 1.110 sends space as CSI u (character=None, is_printable=False).

    Our workaround in _on_key detects this and manually inserts a space.
    See https://github.com/Textualize/textual/issues/6408.
    """

    async def test_space_with_none_character_inserts_space(self) -> None:
        """A space key event with character=None should still insert a space."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.insert("hello")
            await pilot.pause()

            # Simulate VS Code 1.110 CSI u space: key='space', character=None
            await ta._on_key(events.Key("space", None))
            await pilot.pause()

            assert ta.text == "hello "

    async def test_normal_space_still_works(self) -> None:
        """A normal space key event (character=' ') should still work."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.insert("hello")
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause()

            assert ta.text == "hello "


class TestCtrlUDeleteToLineStart:
    """Test that ctrl+u deletes from cursor to start of line (readline convention)."""

    async def test_ctrl_u_deletes_to_line_start(self) -> None:
        """ctrl+u with cursor mid-line should delete text before the cursor."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.insert("hello world")
            await pilot.pause()
            # Cursor at end after insert — move to col 5
            ta.move_cursor((0, 5))
            await pilot.pause()

            await pilot.press("ctrl+u")
            await pilot.pause()

            assert ta.text == " world"
            assert ta.cursor_location == (0, 0)

    async def test_ctrl_u_at_end_of_line_clears_line(self) -> None:
        """ctrl+u at end of single line should clear it entirely."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.insert("hello world")
            await pilot.pause()

            await pilot.press("ctrl+u")
            await pilot.pause()

            assert ta.text == ""
            assert ta.cursor_location == (0, 0)

    async def test_ctrl_u_on_empty_input_is_noop(self) -> None:
        """ctrl+u on already empty input should leave text empty."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            await pilot.press("ctrl+u")
            await pilot.pause()

            assert ta.text == ""
            assert ta.cursor_location == (0, 0)

    async def test_ctrl_u_at_start_of_line_is_noop(self) -> None:
        """ctrl+u at column 0 should not delete anything."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.text = "hello world"
            await pilot.pause()
            ta.move_cursor((0, 0))
            await pilot.pause()

            await pilot.press("ctrl+u")
            await pilot.pause()

            assert ta.text == "hello world"
            assert ta.cursor_location == (0, 0)

    async def test_ctrl_u_multiline_only_affects_current_line(self) -> None:
        """ctrl+u in a multiline buffer should only delete on the cursor's line."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.text = "line one\nline two\nline three"
            await pilot.pause()
            # Place cursor at col 4 on line 1
            ta.move_cursor((1, 4))
            await pilot.pause()

            await pilot.press("ctrl+u")
            await pilot.pause()

            assert ta.text == "line one\n two\nline three"
            assert ta.cursor_location == (1, 0)


class _TextAreaTypingApp(App[None]):
    """Minimal app that captures ChatTextArea.Typing and ChatInput.Typing events."""

    def __init__(self) -> None:
        super().__init__()
        self.text_area_typing_count = 0
        self.chat_input_typing_count = 0

    def compose(self) -> ComposeResult:
        yield ChatInput(id="chat-input")

    def on_chat_text_area_typing(
        self,
        event: ChatTextArea.Typing,  # noqa: ARG002
    ) -> None:
        self.text_area_typing_count += 1

    def on_chat_input_typing(
        self,
        event: ChatInput.Typing,  # noqa: ARG002
    ) -> None:
        self.chat_input_typing_count += 1


class TestChatTextAreaTypingEmission:
    """ChatTextArea should emit Typing on printable keys and backspace."""

    async def test_printable_key_emits_typing(self) -> None:
        """Pressing a printable character should emit ChatTextArea.Typing."""
        app = _TextAreaTypingApp()
        async with app.run_test() as pilot:
            text_area = app.query_one(ChatTextArea)
            text_area.focus()
            await pilot.pause()

            before = app.text_area_typing_count
            await pilot.press("a")
            await pilot.pause()

            assert app.text_area_typing_count > before

    async def test_backspace_emits_typing(self) -> None:
        """Pressing backspace should emit ChatTextArea.Typing."""
        app = _TextAreaTypingApp()
        async with app.run_test() as pilot:
            text_area = app.query_one(ChatTextArea)
            text_area.focus()
            await pilot.press("h")
            await pilot.pause()

            before = app.text_area_typing_count
            await pilot.press("backspace")
            await pilot.pause()

            assert app.text_area_typing_count > before

    async def test_enter_does_not_emit_typing(self) -> None:
        """Pressing enter should NOT emit ChatTextArea.Typing."""
        app = _TextAreaTypingApp()
        async with app.run_test() as pilot:
            text_area = app.query_one(ChatTextArea)
            text_area.focus()
            await pilot.pause()
            initial = app.text_area_typing_count
            await pilot.press("enter")
            await pilot.pause()

            assert app.text_area_typing_count == initial


class TestChatInputTypingBubble:
    """ChatInput.Typing should bubble from ChatTextArea.Typing."""

    async def test_typing_bubbles_to_chat_input(self) -> None:
        """ChatInput.Typing count should track ChatTextArea.Typing."""
        app = _TextAreaTypingApp()
        async with app.run_test() as pilot:
            text_area = app.query_one(ChatTextArea)
            text_area.focus()
            await pilot.press("x")
            await pilot.press("y")
            await pilot.pause()

            assert app.chat_input_typing_count == 2


class TestArgumentHints:
    """Test inline argument-hint ghost text for slash commands."""

    def test_rebuild_argument_hints_populates_lookup(self) -> None:
        """Commands with hints produce a name → hint mapping."""
        from deepagents_cli.command_registry import CommandEntry

        commands = [
            CommandEntry("/remember", "Update memory", "", "[context]"),
            CommandEntry("/help", "Show help", "", ""),
            CommandEntry("/skill-creator", "Create skills", "", "[task]"),
        ]
        chat = ChatInput()
        chat._rebuild_argument_hints(commands)
        assert chat._argument_hints == {
            "remember": "[context]",
            "skill-creator": "[task]",
        }

    def test_rebuild_argument_hints_excludes_empty(self) -> None:
        """Commands without hints are excluded from the lookup."""
        from deepagents_cli.command_registry import CommandEntry

        commands = [
            CommandEntry("/help", "Show help", "", ""),
            CommandEntry("/quit", "Exit", "", ""),
        ]
        chat = ChatInput()
        chat._rebuild_argument_hints(commands)
        assert chat._argument_hints == {}

    async def test_hint_shown_after_command_and_space(self) -> None:
        """Ghost text appears when text is a known command + trailing space."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Enter command mode and type "remember "
            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("remember ")
            await pilot.pause()

            assert chat._text_area.argument_hint == "[context]"
            assert _render_text_area_line(chat._text_area) == "remember [context]"

    async def test_hint_cleared_when_args_typed(self) -> None:
        """Ghost text disappears once the user starts typing arguments."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("remember ")
            await pilot.pause()
            assert chat._text_area.argument_hint == "[context]"

            chat._text_area.insert("x")
            await pilot.pause()
            assert chat._text_area.argument_hint == ""
            assert _render_text_area_line(chat._text_area) == "remember x"

    async def test_hint_stays_at_end_when_cursor_moves(self) -> None:
        """Moving the cursor should not move the rendered argument hint."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("remember ")
            await pilot.pause()

            assert _render_text_area_line(chat._text_area) == "remember [context]"

            for _ in "remember ":
                await pilot.press("left")
            await pilot.pause()

            assert chat._text_area.cursor_location == (0, 0)
            assert chat._text_area.argument_hint == "[context]"
            assert _render_text_area_line(chat._text_area) == "remember [context]"

    async def test_hint_clears_when_extra_space_is_inserted(self) -> None:
        """Typing another space should leave the exact placeholder state."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("remember ")
            await pilot.pause()

            assert chat._text_area.argument_hint == "[context]"

            await pilot.press("left")
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()

            assert chat._text_area.text == "remember  "
            assert chat._text_area.argument_hint == ""
            assert _render_text_area_line(chat._text_area) == "remember"

    async def test_hint_cleared_when_command_mode_exits_via_submit(self) -> None:
        """Submitting a command clears ghost text when mode resets to normal."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("remember ")
            await pilot.pause()
            assert chat._text_area.argument_hint == "[context]"

            await pilot.press("enter")
            await pilot.pause()

            assert chat.mode == "normal"
            assert chat._text_area.argument_hint == ""
            assert len(app.submitted) == 1
            assert app.submitted[0].value == "/remember"

    async def test_hint_cleared_when_backspace_exits_command_mode(self) -> None:
        """Backspace mode exit clears ghost text without needing a text edit."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("remember ")
            await pilot.pause()
            assert chat._text_area.argument_hint == "[context]"

            chat._text_area.move_cursor((0, 0))
            await pilot.pause()
            await pilot.press("backspace")
            await pilot.pause()

            assert chat.mode == "normal"
            assert chat._text_area.text == "remember "
            assert chat._text_area.argument_hint == ""
            assert _render_text_area_line(chat._text_area) == "remember"

    async def test_hint_not_shown_in_normal_mode(self) -> None:
        """Ghost text does not appear when not in command mode."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("remember ")
            await pilot.pause()

            assert chat.mode == "normal"
            assert chat._text_area.argument_hint == ""

    async def test_hint_not_shown_for_unknown_command(self) -> None:
        """Ghost text does not appear for commands without argument hints."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("help ")
            await pilot.pause()

            assert chat._text_area.argument_hint == ""

    async def test_pre_key_dismiss_hides_popup_on_space(self) -> None:
        """Popup is hidden before TextArea processes the space character."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            popup = chat.query_one(CompletionPopup)
            assert chat._text_area is not None

            # Trigger command mode with active suggestions
            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            chat._text_area.insert("rem")
            await pilot.pause()
            assert chat._current_suggestions
            assert popup.styles.display == "block"

            # Type space — popup should dismiss
            await pilot.press("space")
            await pilot.pause()
            assert popup.styles.display == "none"


class TestScrollCursorVisibleDesync:
    """scroll_cursor_visible should not crash on cursor/document desync."""

    async def test_returns_zero_offset_on_value_error(self) -> None:
        """When super() raises ValueError, return Offset(0, 0)."""
        from unittest.mock import patch

        from textual.geometry import Offset
        from textual.widgets import TextArea

        app = _TextAreaTypingApp()
        async with app.run_test() as pilot:
            text_area = app.query_one(ChatTextArea)
            text_area.focus()
            await pilot.pause()

            with patch.object(
                TextArea,
                "scroll_cursor_visible",
                side_effect=ValueError("line index out of bounds"),
            ):
                result = text_area.scroll_cursor_visible()

            assert result == Offset(0, 0)
