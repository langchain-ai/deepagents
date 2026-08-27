"""Unit tests for ChatInput widget and completion popup."""

from __future__ import annotations

import asyncio
import html
from typing import TYPE_CHECKING

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import Container
from textual.widgets import Static
from textual.widgets.text_area import Selection

from deepagents_code import _textual_patches as _textual_patches, theme
from deepagents_code.command_registry import get_slash_commands
from deepagents_code.input import MediaTracker
from deepagents_code.media_utils import ImageData, create_multimodal_content
from deepagents_code.tui.widgets import (
    _paste_textarea as paste_textarea_module,
    chat_input as chat_input_module,
)
from deepagents_code.tui.widgets.autocomplete import MAX_SUGGESTIONS
from deepagents_code.tui.widgets.chat_input import (
    _CHAT_INPUT_AUTO_MAX_HEIGHT,
    _CHAT_INPUT_MANUAL_MAX_HEIGHT,
    _CHAT_INPUT_RESERVED_SCREEN_ROWS,
    _COMPLETION_POPUP_MAX_HEIGHT,
    ChatInput,
    ChatInputBox,
    ChatInputResizeHandle,
    ChatTextArea,
    CompletionOption,
    CompletionPopup,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from pathlib import Path

    from textual.pilot import Pilot


class TestCompletionOption:
    """Test CompletionOption widget."""

    def test_clicked_message_contains_index(self) -> None:
        """Clicked message should contain the option index."""
        message = CompletionOption.Clicked(index=2)
        assert message.index == 2


class TestCompletionPopup:
    """Test CompletionPopup widget."""

    def test_option_clicked_message_contains_index(self) -> None:
        """OptionClicked message should contain the clicked index."""
        message = CompletionPopup.OptionClicked(index=3)
        assert message.index == 3


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


class _ChatInputResizeTestApp(App[None]):
    """App that positions the chat input at the bottom for drag tests."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #spacer {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="spacer")
        yield ChatInput(id="chat-input")


class _DispatchError(RuntimeError):
    """Stand-in for a burst dispatch that raises (media decode, notify, ...)."""


class _RecordingApp(App[None]):
    """App that records ChatInput.Submitted events for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[ChatInput.Submitted] = []

    def compose(self) -> ComposeResult:
        yield ChatInput(id="chat-input")

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        self.submitted.append(event)


def _capture_notifications(
    monkeypatch: pytest.MonkeyPatch, app: App[None]
) -> list[tuple[str, dict[str, object]]]:
    """Patch ``app.notify`` and return a list recording each call.

    Each entry is ``(message, kwargs)`` so tests can assert both the toast
    text and the notification options (e.g. ``markup``, ``timeout``).
    """
    calls: list[tuple[str, dict[str, object]]] = []

    def _record(message: str, *_args: object, **kwargs: object) -> None:
        calls.append((str(message), kwargs))

    monkeypatch.setattr(app, "notify", _record)
    return calls


async def _noop() -> None:
    pass


class _RefreshController:
    def __init__(self) -> None:
        self.cwd_values: list[Path] = []
        self.force_values: list[bool] = []

    def set_cwd(self, cwd: Path) -> None:
        self.cwd_values.append(cwd)

    def warm_cache(self, *, force: bool = False) -> Coroutine[object, object, None]:
        self.force_values.append(force)
        return _noop()


_RESIZE_SCREEN_HEIGHT = 24
"""Terminal height the resize tests run at, so expectations can be derived."""

_EXPANDED_HEIGHT = min(
    _CHAT_INPUT_MANUAL_MAX_HEIGHT,
    _RESIZE_SCREEN_HEIGHT - _CHAT_INPUT_RESERVED_SCREEN_ROWS,
)
"""Composer height a fully expanded drag reaches at `_RESIZE_SCREEN_HEIGHT`."""


class TestChatInputResize:
    """Tests for resizing the chat input from its top border."""

    async def test_double_click_from_a_partial_manual_height_collapses(self) -> None:
        """A part-way manual height collapses rather than expanding.

        Deliberate: the gesture keys off whether a manual height is what pins
        the composer, not off whether it reached the maximum, so one
        double-click undoes any drag the user can see.
        """
        app = _ChatInputResizeTestApp()
        async with app.run_test(size=(80, _RESIZE_SCREEN_HEIGHT)) as pilot:
            box = app.query_one(ChatInputBox)
            handle = app.query_one(ChatInputResizeHandle)
            text_area = app.query_one(ChatTextArea)
            text_area.insert("one\ntwo\nthree")
            box.set_manual_height(5)
            await pilot.pause()
            assert text_area.size.height == 5

            await pilot.double_click(handle, offset=(5, 0))
            await pilot.pause()

            assert box._requested_height is None
            assert text_area.size.height == 3
            # Genuinely automatic again, not a height that happens to equal the
            # draft: auto growth is back in charge and a further toggle expands.
            assert text_area._settled_content_height() == _CHAT_INPUT_AUTO_MAX_HEIGHT
            await pilot.double_click(handle, offset=(5, 0))
            await pilot.pause()
            assert box._requested_height == _EXPANDED_HEIGHT

    async def test_double_click_expands_a_manual_height_hidden_by_the_draft(
        self,
    ) -> None:
        """A drag floored by the draft expands instead of collapsing.

        Dragging a tall draft smaller stores a request the content floor then
        refuses to render, so the composer never moves. Collapsing that request
        would not move it either, leaving the user with two dead gestures in a
        row -- so this case expands.
        """
        app = _ChatInputResizeTestApp()
        async with app.run_test(size=(80, _RESIZE_SCREEN_HEIGHT)) as pilot:
            box = app.query_one(ChatInputBox)
            handle = app.query_one(ChatInputResizeHandle)
            text_area = app.query_one(ChatTextArea)
            text_area.insert("one\ntwo\nthree\nfour\nfive\nsix")
            await pilot.pause()
            assert text_area.size.height == 6

            # Drag downward to shrink; the floor keeps all six rows visible.
            await pilot.mouse_down(handle, offset=(5, 0))
            await pilot.hover(handle, offset=(5, 3))
            await pilot.mouse_up(handle, offset=(5, 3))
            await pilot.pause()
            assert box._requested_height == 3
            assert text_area.size.height == 6

            await pilot.double_click(handle, offset=(5, 0))
            await pilot.pause()

            assert box._requested_height == _EXPANDED_HEIGHT
            assert text_area.size.height == _EXPANDED_HEIGHT

    async def test_double_click_expands_after_a_one_row_drag_jitter(self) -> None:
        """A single row of travel during a press still leaves "expand" next.

        `on_mouse_move` only suppresses sub-cell jitter, so drifting a whole row
        and back within one press does establish a manual height. It renders no
        differently from automatic sizing, so the double-click that follows must
        still expand.
        """
        app = _ChatInputResizeTestApp()
        async with app.run_test(size=(80, _RESIZE_SCREEN_HEIGHT)) as pilot:
            box = app.query_one(ChatInputBox)
            handle = app.query_one(ChatInputResizeHandle)
            text_area = app.query_one(ChatTextArea)
            await pilot.pause()

            await pilot.mouse_down(handle, offset=(5, 0))
            await pilot.hover(handle, offset=(5, 1))
            await pilot.hover(handle, offset=(5, 0))
            await pilot.mouse_up(handle, offset=(5, 0))
            await pilot.pause()
            assert box._requested_height == 1
            assert text_area.size.height == 1

            await pilot.double_click(handle, offset=(5, 0))
            await pilot.pause()

            assert box._requested_height == _EXPANDED_HEIGHT
            assert text_area.size.height == _EXPANDED_HEIGHT

    async def test_unmount_mid_drag_releases_mouse_capture(self) -> None:
        """Tearing the input down during a drag does not strand the capture.

        A leaked capture would leave the whole app deaf to mouse input.
        """
        app = _ChatInputResizeTestApp()
        async with app.run_test(size=(80, _RESIZE_SCREEN_HEIGHT)) as pilot:
            chat_input = app.query_one(ChatInput)
            handle = app.query_one(ChatInputResizeHandle)
            await pilot.pause()

            await pilot.mouse_down(handle, offset=(5, 0))
            assert app.mouse_captured is handle

            await chat_input.remove()
            await pilot.pause()

            assert app.mouse_captured is None


class TestChatInputScrollbar:
    """Regression tests for the chat input's vertical scrollbar behavior.

    `ChatTextArea` is `height: auto; max-height: 8; overflow-y: auto`. The base
    `TextArea` grows its `virtual_size` height the moment a row is inserted, a
    frame before this auto-height widget's container reflows to match. Left to
    the base `_refresh_scrollbars`, that one-frame mismatch makes a short draft
    look like it overflows and flashes the vertical scrollbar on, then off. The
    `ChatTextArea._refresh_scrollbars` override corrects the comparison height
    so the bar appears only on genuine overflow.
    """


class TestChatTextAreaKeybindings:
    """Regression tests for terminal key aliases in the chat input."""


class TestDiscardText:
    """Tests for the undoable draft clear behind esc+esc and the `[ X ]` button."""


class TestInputActionButtons:
    """Tests for the `[ X ]` clear and `[ COPY ]` buttons in the chat input."""

    async def test_buttons_render_labels(self) -> None:
        """The action button labels render as text, not Rich markup tags."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            # Buttons only appear once a draft exists.
            text_area.insert("draft")
            await pilot.pause()
            rendered = html.unescape(app.export_screenshot()).replace("\xa0", " ")

        assert "[ X ]" in rendered
        assert "[ COPY ]" in rendered


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
    return str(prompt._Static__content)  # ty: ignore  # accessing internal content store


def _render_text_area_line(text_area: ChatTextArea, y: int = 0) -> str:
    """Render a text-area line and trim widget padding for assertions."""
    return text_area.render_line(y).text.rstrip()


class TestPromptIndicator:
    """Test that the prompt indicator reflects the current input mode."""

    async def test_incognito_shell_to_shell_clears_incognito_styling(self) -> None:
        """Transitioning out of incognito must clear the incognito styling.

        Regression guard: a future change forgetting to drop the incognito
        title or CSS class would leave stale styling on the input.
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)

            input_box = chat_input.query_one("#input-box")
            chat_input.mode = "shell_incognito"
            await pilot.pause()
            assert input_box.border_title == "incognito"
            assert chat_input.has_class("mode-shell-incognito")

            chat_input.mode = "shell"
            await pilot.pause()
            assert input_box.border_title is None
            assert not chat_input.has_class("mode-shell-incognito")
            assert chat_input.has_class("mode-shell")


class TestShellSyntaxHighlighting:
    """Shell command modes should render native shell styles in the chat input."""

    async def test_highlight_failure_never_shows_stale_text(self) -> None:
        """A failed highlight must fall back to the document, not a stale draft.

        The rendered text has to match the buffer that Enter would submit. If
        the cache marker were committed before `highlight()` ran, a failure
        would leave the marker on the new text and the cached lines on the old,
        so every later call would take the cache-hit path and render the
        previous draft indefinitely.
        """
        from unittest.mock import patch

        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = app.query_one(ChatTextArea)
            text_area.text = "echo first"
            chat_input.mode = "shell"
            await pilot.pause()
            assert text_area.get_line(0).plain == "echo first"

            text_area.text = "echo second"
            with patch.object(
                chat_input_module,
                "highlight",
                side_effect=RuntimeError("lexer exploded"),
            ):
                line = text_area.get_line(0)

            assert line.plain == "echo second"
            # Degradation persists rather than re-raising every frame, and the
            # text stays correct once the patch is lifted.
            assert text_area.get_line(0).plain == "echo second"

    async def test_cursor_line_keeps_shell_highlight_colors(self) -> None:
        """Rendered strip on the cursor line should keep token colors.

        Regression test: `TextArea._render_line` stylizes the whole cursor line
        with `cursor_line_style`, which carries the widget text color. Without
        the foreground strip in `ChatTextArea._render_line`, that paints over
        the syntax spans and every rendered token collapses to one color. The
        other tests in this class only assert on `get_line()`, which runs
        before the cursor-line style is applied.
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = app.query_one(ChatTextArea)
            text_area.text = 'FOO="bar" echo "$FOO"'
            chat_input.mode = "shell"
            await pilot.pause()

            # Put the cursor on the line being rendered, but at end-of-line.
            # The block cursor inverts the cell it sits on, which contributes a
            # second color on its own - enough to satisfy the assertion below
            # even with the token colors flattened. Parking it past the last
            # character puts it on trailing padding, which the `.strip()`
            # filter drops, so only real token colors are counted.
            text_area.move_cursor((0, len(text_area.text)))
            strip = text_area.render_line(0)
            colors = {
                segment.style.color.triplet
                for segment in strip
                if segment.text.strip() and segment.style and segment.style.color
            }
            # Distinct syntax colors must survive to the rendered strip, not
            # flatten to the single cursor-line text color.
            assert len(colors) > 1


class TestModeSwitchNoJitter:
    """Regression tests: mode glyph and completion popup update atomically.

    Switching modes (e.g. `/` → `!` or `!` → `/`) must change the prompt glyph
    and completion popup visibility in the same frame. A deferred ordering that
    closes the popup one frame before the glyph changes (or vice versa) creates
    visible jitter.
    """

    async def test_slash_to_bang_updates_glyph_and_popup_same_frame(self) -> None:
        """Switching from command mode to shell mode atomically hides popup."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            prompt = chat.query_one("#prompt", Static)
            popup = chat.query_one(CompletionPopup)
            assert chat._text_area is not None

            # Enter command mode — popup visible, glyph is "/"
            await pilot.press("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"
            assert _prompt_text(prompt) == "/"
            assert popup.styles.display == "block"

            # Switch to shell mode — popup hidden AND glyph is "$" after one pause
            await pilot.press("!")
            await pilot.pause()
            assert chat.mode == "shell"
            assert _prompt_text(prompt) == "$"
            assert popup.styles.display == "none"

    async def test_bang_to_slash_updates_glyph_and_popup_same_frame(self) -> None:
        """Switching from shell mode to command mode atomically shows popup."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            prompt = chat.query_one("#prompt", Static)
            popup = chat.query_one(CompletionPopup)
            assert chat._text_area is not None

            # Enter shell mode first — popup hidden, glyph is "$"
            await pilot.press("!")
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"
            assert _prompt_text(prompt) == "$"
            assert popup.styles.display == "none"

            # Switch to command mode — popup visible AND glyph is "/" after one pause
            await pilot.press("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"
            assert _prompt_text(prompt) == "/"
            assert popup.styles.display == "block"


class TestHistoryNavigationFlag:
    """Test that _skip_history_change_events resets when history is exhausted."""


class TestSetValueAtEnd:
    """Tests for programmatically setting input text at the end cursor position."""


class TestInsertAtCursor:
    """Tests for undoable prompt insertion without submission."""


class TestRefocusClickSuppression:
    """Clicks that re-focus the terminal window should not move the cursor."""

    async def test_refocus_click_does_not_move_cursor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A click within the refocus window only restores focus."""
        # Widen the window so the test never depends on how fast the event loop
        # delivers the click after the refocus stamp (avoids wall-clock flake).
        monkeypatch.setattr(
            chat_input_module, "_REFOCUS_CLICK_SUPPRESS_WINDOW_SECONDS", 60.0
        )
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            text_area = chat._text_area
            assert text_area is not None

            text_area.insert("hello world")
            text_area.move_cursor((0, 0))
            await pilot.pause()
            assert text_area.cursor_location == (0, 0)

            chat._notify_app_blur()
            chat._notify_app_focus()
            await pilot.click(ChatTextArea, offset=(6, 0))
            await pilot.pause()

            assert text_area.cursor_location == (0, 0)


class TestCursorHiddenWhileUnfocused:
    """The chat input must never blink a cursor it cannot type into.

    Textual's `TextArea._draw_cursor` ignores `has_focus` while blinking is on,
    and mouse-down sets `_selecting`, so the matching mouse-up always restarts
    the blink through `_end_mouse_selection`. Clicking the chat input while a
    focus-trapping widget (e.g. the `edit_file` approval menu, which re-focuses
    itself on blur) owns the keyboard therefore left a blinking cursor in a
    field that could neither receive keystrokes nor be typed into.
    """

    async def test_click_while_approval_traps_focus_shows_no_cursor(self) -> None:
        """Clicking the chat input under an `edit_file` approval draws no cursor."""
        from deepagents_code.tui.widgets.approval import ApprovalMenu

        class _ApprovalApp(App[None]):
            def compose(self) -> ComposeResult:
                yield ApprovalMenu(
                    {
                        "name": "edit_file",
                        "args": {
                            "file_path": "main.py",
                            "old_string": "a",
                            "new_string": "b",
                        },
                    }
                )
                yield ChatInput(id="chat-input")

        app = _ApprovalApp()
        async with app.run_test(size=(80, 24)) as pilot:
            chat = app.query_one(ChatInput)
            text_area = chat._text_area
            assert text_area is not None
            menu = app.query_one(ApprovalMenu)
            menu.focus()
            await pilot.pause()
            assert text_area._draw_cursor is False

            # Relies on `pilot.click` pausing before each event, so the menu's
            # deferred `on_blur` refocus lands before the mouse-up restarts the
            # blink; the phantom cursor is settled by the time `click` returns.
            await pilot.click(ChatTextArea)
            await pilot.pause()

            # Pin the precondition. Without this, a `pilot` that stopped
            # interleaving would leave the input focused, `_watch_has_focus`
            # would hide the cursor on the later refocus, and the assertion
            # below would pass for a reason unrelated to the fix.
            assert text_area.has_focus is False

            assert app.focused is menu
            assert text_area._draw_cursor is False
            # `_draw_cursor` alone is a 50/50 read against a regression that
            # leaves the timer running, since it is False for half of every
            # blink cycle. The parked timer is the deterministic signal.
            assert text_area.blink_timer._active.is_set() is False


class TestHistoryBoundaryNavigation:
    """Test that history navigation only triggers at input boundaries."""

    async def test_up_at_end_of_single_line_snaps_cursor_first(self) -> None:
        """Up at end of single-line typed input snaps cursor to start, no history."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            # Entry must contain "hello" — substring-filtered history.
            chat._history._entries.append("say hello world")

            chat._text_area.insert("hello")
            await pilot.pause()
            assert chat._text_area.cursor_location == (0, 5)

            # First up moves the cursor to (0, 0) — there is no row above.
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "hello"
            assert chat._text_area.cursor_location == (0, 0)

            # Second up has no further cursor movement available, so history.
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "say hello world"


class TestCompletionPopupClickBubbling:
    """Test that clicks on options bubble up through the popup."""


class TestDismissCompletion:
    """Test ChatInput.dismiss_completion edge cases."""

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
                len(get_slash_commands()), MAX_SUGGESTIONS
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

    async def test_handle_mode_prefix_keystroke_switches_without_text_change(
        self,
    ) -> None:
        """A typed mode selector is consumed without inserting the character.

        Regression guard for the `!`-flash: `handle_mode_prefix_keystroke`
        consumes the keystroke and flips the mode directly when needed, so the
        trigger is never inserted (and thus never flashes for a frame before
        stripping).
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            assert chat.handle_mode_prefix_keystroke("!") is True
            await pilot.pause()
            assert chat.mode == "shell"
            assert chat._text_area.text == ""

            # Second bang promotes to incognito, still without inserted text.
            assert chat.handle_mode_prefix_keystroke("!") is True
            await pilot.pause()
            assert chat.mode == "shell_incognito"
            assert chat._text_area.text == ""

            # A third bang in incognito is literal body text — not consumed.
            assert chat.handle_mode_prefix_keystroke("!") is False
            # Non-trigger characters are never consumed.
            assert chat.handle_mode_prefix_keystroke("a") is False

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

    async def test_backspace_on_empty_incognito_exits_to_normal(self) -> None:
        """Backspace cancels incognito mode instead of demoting to shell mode."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await pilot.press("!")
            await pilot.press("!")
            await _pause_for_strip(pilot)
            assert chat.mode == "shell_incognito"
            assert chat._text_area.text == ""

            await pilot.press("backspace")
            await pilot.pause()
            assert chat.mode == "normal"
            assert chat._text_area.text == ""

    async def test_backspace_at_cursor_zero_with_text_stays_in_mode(self) -> None:
        """Backspace only exits a mode prompt when the input is empty."""
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

            # Backspace at position 0 with text after cursor is a text-editing
            # no-op; it should not cancel the active mode.
            await pilot.press("backspace")
            await pilot.pause()
            assert chat.mode == "command"
            assert chat._text_area.text == "help"

    async def test_third_bang_stays_in_incognito_shell_mode(self) -> None:
        """Typing `!`+`!`+`!` must not demote `shell_incognito` back to `shell`.

        Regression guard for the privacy-sensitive parser path: a stray third
        bang should be treated as command-body content, not as a mode change
        out of incognito.
        """
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await pilot.press("!")
            await _pause_for_strip(pilot)
            await pilot.press("!")
            await _pause_for_strip(pilot)
            assert chat.mode == "shell_incognito"

            await pilot.press("!")
            await _pause_for_strip(pilot)
            assert chat.mode == "shell_incognito"
            assert chat._text_area.text == "!"

            chat._text_area.insert("ls")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].mode == "shell_incognito"
            assert app.submitted[0].value == "!!!ls"

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


class TestExitModePreservesText:
    """Exiting shell/command mode should preserve typed text."""

    async def test_exit_empty_shell_mode_does_not_restore_prefix(self) -> None:
        """Escape cancels shell mode; it does not turn `!` back into text."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await pilot.press("!")
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"
            assert chat._text_area.text == ""

            assert chat.exit_mode() is True
            assert chat.mode == "normal"
            assert chat._text_area.text == ""

    async def test_exit_empty_incognito_mode_does_not_restore_prefix(self) -> None:
        """Escape cancels incognito mode; it does not turn `!!` back into text."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await pilot.press("!")
            await pilot.press("!")
            await _pause_for_strip(pilot)
            assert chat.mode == "shell_incognito"
            assert chat._text_area.text == ""

            assert chat.exit_mode() is True
            assert chat.mode == "normal"
            assert chat._text_area.text == ""

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


class TestSlashCompletionCursorMapping:
    """Regression: virtual-to-real index translation for slash replacement."""


class TestHistorySlashPrefixRecall:
    """Test that recalling a slash-prefixed history entry enters command mode."""


class TestCompletionIndexToTextIndex:
    """Edge-case tests for _completion_index_to_text_index clamping."""


class TestHistoryRecallSuppressesCompletions:
    """Test that history navigation does not trigger completions."""


class TestDroppedImagePaste:
    """Tests for drag/drop image-path handling via paste events."""

    async def test_backspace_from_line_below_image_keeps_placeholder(
        self, tmp_path
    ) -> None:
        """Backspace on the line below `[image N]` rejoins lines, keeps token.

        Two images dropped on separate lines render as `[image 1]`, a newline,
        and then `[image 2]`, with no trailing space after the first token. The
        newline sits immediately after the first
        token's closing bracket. Backspacing from the start of the second line
        must remove only the line break, not delete `[image 1]` atomically with
        it (the regression this fix addresses for the media code path).
        """
        from PIL import Image

        img1 = tmp_path / "one.png"
        img2 = tmp_path / "two.png"
        Image.new("RGB", (4, 4), color="cyan").save(img1, format="PNG")
        Image.new("RGB", (4, 4), color="magenta").save(img2, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(f"{img1}\n{img2}")
            await pilot.pause()
            assert chat._text_area.text == "[image 1]\n[image 2]"

            chat._text_area.move_cursor((1, 0))
            await pilot.pause()

            await pilot.press("backspace")
            await pilot.pause()

            # The line break is removed and both placeholders survive rather
            # than `[image 1]` being deleted atomically with the newline.
            assert chat._text_area.text == "[image 1][image 2]"
            assert len(app.tracker.get_images()) == 2

    async def test_typed_image_placeholder_is_not_atomic(self) -> None:
        """Manually typed `[image N]` (no attachment) edits char-by-char.

        Regression test: placeholder-shaped text the user typed must not be
        treated as an atomic media token, so backspace removes a single
        character instead of deleting the whole `[image 2]`.
        """
        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "[image 2]"
            await pilot.pause()
            assert app.tracker.get_images() == []

            chat._text_area.move_cursor((0, len("[image 2]")))
            await pilot.pause()
            await pilot.press("backspace")
            await pilot.pause()

            assert chat._text_area.text == "[image 2"

    async def test_typed_placeholder_not_atomic_alongside_real_image(
        self, tmp_path
    ) -> None:
        """A typed look-alike is char-editable while a real one stays atomic."""
        img_path = tmp_path / "real.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="green")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(str(img_path))
            await pilot.pause()
            assert chat._text_area.text == "[image 1] "

            # Append a manually typed placeholder-shaped token that is not
            # backed by any attachment.
            chat._text_area.text = "[image 1] [image 2]"
            await pilot.pause()
            assert len(app.tracker.get_images()) == 1

            chat._text_area.move_cursor((0, len("[image 1] [image 2]")))
            await pilot.pause()
            await pilot.press("backspace")
            await pilot.pause()

            # Only one character of the typed token is removed; the real
            # `[image 1]` placeholder is untouched and still tracked.
            assert chat._text_area.text == "[image 1] [image 2"
            assert len(app.tracker.get_images()) == 1

    async def test_real_image_placeholder_still_atomic_with_typed_lookalike(
        self, tmp_path
    ) -> None:
        """The real `[image 1]` deletes atomically even beside a typed token."""
        img_path = tmp_path / "atomic.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="purple")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(str(img_path))
            await pilot.pause()
            chat._text_area.text = "[image 2] [image 1]"
            await pilot.pause()
            assert len(app.tracker.get_images()) == 1

            # Cursor just after the real trailing `[image 1]` token.
            chat._text_area.move_cursor((0, len("[image 2] [image 1]")))
            await pilot.pause()
            await pilot.press("backspace")
            await pilot.pause()

            # The whole real placeholder is removed atomically, leaving the
            # typed look-alike intact.
            assert chat._text_area.text == "[image 2] "

    async def test_submit_remaps_span_onto_stripped_value(self, tmp_path) -> None:
        """`_submit_value` re-maps placeholder spans onto the final submitted text.

        Regression: spans captured against the raw draft go stale when submit
        strips leading whitespace (and expands pastes), so the adapter would
        strip the wrong token from the model-facing message. The span must
        follow the transform.
        """
        img_path = tmp_path / "submit.png"
        from PIL import Image

        Image.new("RGB", (4, 4), color="navy").save(img_path, format="PNG")

        app = _ImagePasteRecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(str(img_path))
            await pilot.pause()

            # Leading whitespace shifts every offset when submit strips it.
            chat._text_area.text = "  look [image 1]"
            await pilot.pause()
            img = app.tracker.get_images()[0]
            assert img.placeholder_span == (7, 16)

            chat._submit_value(chat._text_area.text.strip())
            await pilot.pause()

            assert app.submitted[-1].value == "look [image 1]"
            # The span now indexes the submitted value, not the raw draft.
            assert img.placeholder_span == (5, 14)
            content = create_multimodal_content(
                app.submitted[-1].value, app.tracker.get_images()
            )
            assert content[0]["text"] == "look"

    async def test_key_burst_absolute_path_preserves_leading_slash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rapid absolute path recovers the slash consumed by command mode."""
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 1.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_BURST_FLUSH_DELAY_SECONDS", 0.25
        )

        img_path = tmp_path / "absolute-burst.png"
        from PIL import Image

        Image.new("RGB", (3, 3), color="navy").save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            for char in str(img_path):
                await chat._text_area._on_key(events.Key(char, char))

            assert chat.mode == "normal"
            assert chat._text_area.text == ""

            await pilot.pause(0.35)

            assert chat._text_area.text == "[image 1] "
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

    async def test_typed_video_placeholder_is_not_atomic(self) -> None:
        """Manually typed `[video N]` (no attachment) edits char-by-char.

        Mirrors the image look-alike guard for the video code path, which shares
        `_bound_media_placeholders` but was otherwise only tested for bound
        tokens.
        """
        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "[video 2]"
            await pilot.pause()
            assert app.tracker.get_videos() == []

            chat._text_area.move_cursor((0, len("[video 2]")))
            await pilot.pause()
            await pilot.press("backspace")
            await pilot.pause()

            assert chat._text_area.text == "[video 2"


class TestPathPayloadDetectionGating:
    """Single-keystroke edits should skip the blocking path-detection helpers.

    `_is_dropped_path_payload` and `_apply_inline_dropped_path_replacement`
    reach `Path.exists()` / `Path.is_file()` via
    `deepagents_code.input.parse_pasted_path_payload`, which are synchronous
    stat syscalls on the event-loop thread. They are only meaningful when a
    text change inserts more than one character (drag-drop / bracketed paste);
    on normal typing they cost real wall-clock time for no possible match.
    """


class TestBackslashEnterNewline:
    """Test that backslash followed quickly by enter inserts a newline.

    Some terminals (e.g. VSCode built-in) send a literal backslash followed
    by enter when the user presses shift+enter.  The widget detects this
    pair and collapses it into a newline.
    """

    async def test_backslash_then_enter_inserts_newline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rapid backslash + enter should produce a newline, not submit."""
        # Widen the gap so wall-clock timing between pilot.press calls on slow
        # CI runners cannot push the enter past the 150ms default and trip the
        # submit path.
        monkeypatch.setattr(paste_textarea_module, "_BACKSLASH_ENTER_GAP_SECONDS", 60.0)

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


class TestVSCodeSpaceWorkaround:
    """VS Code 1.110 sends space as CSI u (character=None, is_printable=False).

    Our workaround in _on_key detects this and manually inserts a space.
    See https://github.com/Textualize/textual/issues/6408.
    """


class TestLockKeysDoNotType:
    """Lock keys must never insert text.

    Under the kitty keyboard protocol with associated-text reporting (iTerm2,
    VS Code's xterm.js, etc.), pressing Caps Lock arrives as
    Key(key='caps_lock', character='A'), which would otherwise make TextArea
    insert a stray letter.
    """

    @pytest.mark.parametrize(
        "lock_key",
        [
            "caps_lock",
            "num_lock",
            "scroll_lock",
            # Modifier-prefixed variants: the lock bit can arrive alongside
            # other modifier bits, so the key string is suffixed.
            "ctrl+caps_lock",
            "alt+ctrl+hyper+meta+super+caps_lock",
        ],
    )
    async def test_lock_key_with_associated_text_inserts_nothing(
        self, lock_key: str
    ) -> None:
        """A lock-key event carrying associated text should insert nothing."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.insert("hello")
            await pilot.pause()

            # iTerm2/kitty protocol reports the would-be text as `character`.
            await ta._on_key(events.Key(lock_key, "A"))
            await pilot.pause()

            assert ta.text == "hello"


class TestCtrlUDeleteToLineStart:
    """Test that ctrl+u deletes from cursor to start of line (readline convention)."""


class TestModifiedBackspaceDeleteWordLeft:
    """Test modified Backspace aliases for word deletion."""

    @pytest.mark.parametrize("key", ["ctrl+backspace", "alt+backspace"])
    async def test_modified_backspace_deletes_paste_placeholder_atomically(
        self, key: str
    ) -> None:
        """Modified Backspace should not corrupt a collapsed-paste token."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste("p" * 900)
            await pilot.pause()
            assert chat._text_area.text == "[Pasted text #1]"

            await pilot.press(key)
            await pilot.pause()

            assert chat._text_area.text == ""
            assert 1 in chat._pasted_contents

    @pytest.mark.parametrize("key", ["ctrl+backspace", "alt+backspace"])
    async def test_modified_backspace_after_tab_deletes_placeholder_atomically(
        self, key: str
    ) -> None:
        """Modified Backspace preserves token integrity after a tab."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste("p" * 900)
            chat._text_area.insert("\t")
            await pilot.pause()
            assert chat._text_area.text == "[Pasted text #1]\t"

            await pilot.press(key)
            await pilot.pause()

            assert chat._text_area.text == ""
            assert 1 in chat._pasted_contents


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


class TestChatInputTypingBubble:
    """ChatInput.Typing should bubble from ChatTextArea.Typing."""


class TestArgumentHints:
    """Test inline argument-hint ghost text for slash commands."""

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


class TestSetCursorStyle:
    """`ChatInput.set_cursor_style` updates the rendered cursor component."""

    async def test_switches_between_underline_and_block(self) -> None:
        """Underline adds its cursor class and block restores Textual's default."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.set_cursor_style(style="underline")
            await pilot.pause()

            assert chat._text_area.has_class("cursor-underline")
            underline = chat._text_area.get_component_rich_style("text-area--cursor")
            assert underline.underline is True

            chat.set_cursor_style(style="block")
            await pilot.pause()

            assert not chat._text_area.has_class("cursor-underline")
            block = chat._text_area.get_component_rich_style("text-area--cursor")
            assert block.underline is not True


class TestSetCursorBlink:
    """`ChatInput.set_cursor_blink` toggles cursor blink without changing focus."""

    async def test_toggles_reactive(self) -> None:
        """Pause flips `cursor_blink` to False; resume flips it back to True."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None
            assert chat._text_area.cursor_blink is True

            chat.set_cursor_blink(blink=False)
            await pilot.pause()
            assert chat._text_area.cursor_blink is False

            chat.set_cursor_blink(blink=True)
            await pilot.pause()
            assert chat._text_area.cursor_blink is True


class TestPasteBurstEnterSuppression:
    """Multi-line pastes replayed as key events must not submit mid-stream.

    Terminals without bracketed paste deliver a paste as rapid `Char`/`Enter`
    key events. A short run of fast keystrokes arms a suppression window so the
    embedded `enter` events insert newlines instead of submitting.
    """

    async def test_rapid_burst_with_newline_does_not_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fast keystroke run stays visible and enter inserts a newline."""
        # Widen the burst gap so wall-clock delays between pilot.press calls on
        # slow CI runners still register as a single rapid burst.
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_ENTER_SUPPRESS_WINDOW_SECONDS", 60.0
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            for char in "hello":
                await pilot.press(char)
            assert ta.text == "hello"
            assert ta._paste_burst_buffer == ""

            await pilot.press("enter")
            await pilot.press("w")
            await pilot.pause(0.15)

            assert len(app.submitted) == 0
            assert "\n" in ta.text


class TestPasteBurstPromotion:
    """Promotion of a visible rapid run into the hidden paste buffer.

    Rapid typing stays in the document until something confirms a paste: an
    embedded newline, a dropped-path shape, or a length no human reaches at
    burst speed. These tests drive the real `_on_key` path, since the chat
    input's key handling interleaves several branches ahead of the burst
    helpers.
    """

    @pytest.mark.parametrize("payload", ["hello world", '"hello world"'])
    async def test_ordinary_rapid_typing_is_never_promoted(
        self, monkeypatch: pytest.MonkeyPatch, payload: str
    ) -> None:
        """A short rapid run, including quoted text, stays fully visible."""
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            for char in payload:
                await pilot.press(char)

            # Asserted before any pause: the flush timer would restore the text
            # and hide the very regression this covers. Typing must be visible
            # *while* typing, not once it stops.
            assert ta.text == payload
            assert ta._paste_burst_buffer == ""

            await pilot.pause(0.15)

            assert ta.text == payload
            assert ta._paste_burst_buffer == ""

    async def test_promotion_falls_back_when_selection_is_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A selection at Enter blocks promotion; the newline is inserted plainly.

        Promoting would delete the user's selected range rather than the run.
        """
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_ENTER_SUPPRESS_WINDOW_SECONDS", 60.0
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            for char in "abc":
                await pilot.press(char)
            ta.selection = Selection((0, 0), (0, 3))

            await ta._on_key(events.Key("enter", None))
            await pilot.pause()

            assert ta._paste_burst_buffer == ""
            assert "abc" in ta.text
            assert len(app.submitted) == 0

    async def test_vscode_space_workaround_keeps_run_in_sync(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CSI-u space is tracked, so a burst containing one still promotes.

        VS Code sends space as a key with no character; the workaround inserts
        it directly and returns before the burst helpers, so it must feed the
        tracker itself or the run text diverges from the document.
        """
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_ENTER_SUPPRESS_WINDOW_SECONDS", 60.0
        )
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_BURST_FLUSH_DELAY_SECONDS", 0.25
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            for char in "ab":
                await ta._on_key(events.Key(char, char))
            await ta._on_key(events.Key("space", None))
            for char in "cd":
                await ta._on_key(events.Key(char, char))

            assert ta.text == "ab cd"
            assert ta._paste_burst_run_text == "ab cd"

            await ta._on_key(events.Key("enter", None))
            await pilot.pause()

            # Promotion succeeded, so the run moved into the buffer rather than
            # failing verification and falling back to a plain newline.
            assert ta.text == ""
            assert ta._paste_burst_buffer == "ab cd\n"
            assert len(app.submitted) == 0

    async def test_rapid_typing_in_command_mode_stays_visible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fast typing after a `/` is not mistaken for a dropped path.

        The slash-recovery hook prepends `/` before asking whether the payload
        looks like a path, so a guard phrased as a question about the `/`-prefixed
        candidate is vacuously true for any text. If that is the only guard, every
        rapid run in command mode is hidden and the input silently leaves command
        mode mid-command.
        """
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_BURST_FLUSH_DELAY_SECONDS", 0.25
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            for char in "/git":
                await pilot.press(char)
            for char in "add":
                await pilot.press(char)
            await pilot.pause(0.35)

            assert ta.text == "gitadd"
            assert chat.mode == "command"
            assert ta._paste_burst_buffer == ""

    async def test_rapid_slash_command_with_path_argument_is_not_promoted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A command whose argument is a path keeps its command semantics.

        The payload contains a separator, so a separator-only test would treat
        `read src/main.py` as the tail of an absolute path — injecting a `/` and
        dropping out of command mode. Whitespace before the separator rules it out.
        """
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_BURST_FLUSH_DELAY_SECONDS", 0.25
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            # Driven through `_on_key` rather than `pilot.press` so slash-command
            # completion cannot rewrite the text out from under the assertion.
            for char in "/read src/main.py":
                key = "space" if char == " " else char
                await ta._on_key(events.Key(key, char))
            await pilot.pause(0.35)

            # Still a command, and nothing was hidden or slash-prefixed. (The
            # space itself is swallowed by the open completion popup, so the exact
            # text is not asserted here.)
            assert chat.mode == "command"
            assert ta._paste_burst_buffer == ""
            assert not ta.text.startswith("/")
            assert ta.text.endswith("src/main.py")

    async def test_rapid_absolute_path_that_does_not_exist_keeps_its_slash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recovered slash survives the insert that follows a failed parse.

        Only an existing path takes the dropped-path branch. Everything else
        falls through to a plain insert at offset 0, which trips mode-prefix
        detection a second time — stripping the recovered slash again and losing
        the character for good.
        """
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_BURST_FLUSH_DELAY_SECONDS", 0.25
        )
        missing = tmp_path / "no-such-file.txt"

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            for char in str(missing):
                await ta._on_key(events.Key(char, char))
            await pilot.pause(0.35)

            assert ta.text == str(missing)
            assert chat.mode == "normal"
            assert chat.value == str(missing)


class TestPasteCollapseHelpers:
    """Unit tests for the paste_collapse module helpers."""


class TestPasteCollapseIntegration:
    """Integration tests for paste collapsing in ChatInput."""

    async def test_backspace_removes_only_targeted_paste_placeholder(self) -> None:
        """Backspace deletes only the placeholder at the cursor, not others."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste("A" * 900)
            await pilot.pause()
            chat.handle_external_paste("B" * 900)
            await pilot.pause()
            assert "[Pasted text #1]" in chat._text_area.text
            assert "[Pasted text #2]" in chat._text_area.text

            await pilot.press("backspace")
            await pilot.pause()

            # Exact equality (not a substring check): a non-atomic delete that
            # removed a single char would leave "[Pasted text #2" and still
            # satisfy a `"[Pasted text #2]" not in text` assertion.
            assert chat._text_area.text == "[Pasted text #1]"

    async def test_typed_paste_placeholder_is_not_atomic(self) -> None:
        """A paste placeholder with no backing content edits char-by-char.

        Regression test for the bound-token guard: `[Pasted text #99]` that the
        user typed (or a stale token whose id is absent from `_pasted_contents`)
        must not delete atomically, so backspace removes a single character.
        Without the `id not in pasted_ids` arm this whole token would vanish.
        """
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None
            assert chat._pasted_contents == {}

            chat._text_area.text = "[Pasted text #99]"
            chat._text_area.move_cursor((0, len("[Pasted text #99]")))
            await pilot.pause()
            await pilot.press("backspace")
            await pilot.pause()

            assert chat._text_area.text == "[Pasted text #99"

    async def test_backspace_removes_multiline_paste_placeholder(self) -> None:
        """Backspace atomically deletes the `+M lines` placeholder variant."""
        multi_line = "\n".join(f"line {i}" for i in range(5))
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(multi_line)
            await pilot.pause()
            # The multi-line form carries a "+M lines" suffix, so its span is
            # longer than the bare "[Pasted text #N]" token.
            assert chat._text_area.text == "[Pasted text #1 +4 lines]"

            await pilot.press("backspace")
            await pilot.pause()

            assert chat._text_area.text == ""
            assert 1 in chat._pasted_contents

    async def test_backspace_from_line_below_placeholder_keeps_it(self) -> None:
        """Backspace on a line below a placeholder rejoins lines, keeps token.

        Regression: a newline immediately after a `[Pasted text #N]` placeholder
        was treated as an auto-inserted trailing separator, so backspacing from
        the start of the next line deleted the whole placeholder instead of just
        removing the line break. The cursor should land at the end of the
        placeholder line with the placeholder intact.
        """
        big_text = "p" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()
            assert chat._text_area.text == "[Pasted text #1]"

            chat._text_area.insert("\n")
            await pilot.pause()
            assert chat._text_area.cursor_location == (1, 0)

            await pilot.press("backspace")
            await pilot.pause()

            assert chat._text_area.text == "[Pasted text #1]"
            assert chat._text_area.cursor_location == (0, len("[Pasted text #1]"))
            assert 1 in chat._pasted_contents

    async def test_orphan_cleanup_skips_empty_text(self) -> None:
        """Setting text to empty must not trigger orphan cleanup."""
        big_text = "G" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()

            chat._text_area.text = ""
            await pilot.pause()

            assert 1 in chat._pasted_contents


class TestPromptSearchPanel:
    """Inline prompt history search (first Ctrl+R tier)."""

    def _seed_history(self, chat: ChatInput, prompts: list[str]) -> None:
        for prompt in prompts:
            chat._history.add(prompt)

    async def test_ctrl_r_opens_inline_panel_above_input(self, tmp_path) -> None:
        from deepagents_code.tui.widgets.prompt_search import (
            PromptSearchInput,
            PromptSearchPanel,
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            chat._history.history_file = tmp_path / "history.jsonl"
            self._seed_history(chat, ["first prompt", "second prompt"])
            await pilot.pause()
            assert chat._text_area is not None
            chat._text_area.insert("second")
            await pilot.pause()

            tier = chat.open_prompt_search()
            await pilot.pause()
            await pilot.pause()

            assert tier == "inline"
            panel = app.query_one(PromptSearchPanel)
            assert panel.styles.display == "block"
            assert chat._prompt_search_active is True
            assert app.query_one(PromptSearchInput).value == "second"
            assert chat._prompt_search_filtered == ["second prompt"]
            # Seeding the filter does not consume or change the draft.
            assert chat._text_area.text == "second"

    async def test_typing_filters_results(self, tmp_path) -> None:
        from deepagents_code.tui.widgets.prompt_search import PromptSearchPanel

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            chat._history.history_file = tmp_path / "history.jsonl"
            self._seed_history(chat, ["fix the bug", "add feature", "fix tests"])
            await pilot.pause()

            chat.open_prompt_search()
            await pilot.pause()
            await pilot.pause()

            for char in "fix":
                await pilot.press(char)
            await pilot.pause()
            await pilot.pause()

            # newest-first order, both "fix" prompts retained
            assert chat._prompt_search_filtered == ["fix tests", "fix the bug"]
            panel = app.query_one(PromptSearchPanel)
            assert panel.styles.display == "block"

    async def test_escape_restores_draft(self, tmp_path) -> None:
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            chat._history.history_file = tmp_path / "history.jsonl"
            self._seed_history(chat, ["some prompt"])
            await pilot.pause()
            assert chat._text_area is not None
            chat._text_area.insert("my draft")
            await pilot.pause()

            chat.open_prompt_search()
            await pilot.pause()
            await pilot.pause()
            # Type a query (does not touch the draft)
            await pilot.press("x")
            await pilot.pause()
            assert chat._text_area.text == "my draft"

            await pilot.press("escape")
            await pilot.pause()

            assert chat._prompt_search_active is False
            assert chat._text_area.text == "my draft"

    async def test_escape_preserves_concurrently_updated_draft(self, tmp_path) -> None:
        """Cancel should not replace a draft changed outside prompt search."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            chat._history.history_file = tmp_path / "history.jsonl"
            self._seed_history(chat, ["some prompt"])
            await pilot.pause()
            assert chat._text_area is not None
            chat._text_area.insert("original draft")
            chat.open_prompt_search()
            await pilot.pause()

            chat.value = "external editor result"
            await pilot.press("escape")
            await pilot.pause()

            assert chat._prompt_search_active is False
            assert chat.value == "external editor result"

    async def test_backspace_on_empty_query_cancels(self, tmp_path) -> None:
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            chat._history.history_file = tmp_path / "history.jsonl"
            self._seed_history(chat, ["hello"])
            await pilot.pause()
            assert chat._text_area is not None

            chat.open_prompt_search()
            await pilot.pause()
            await pilot.pause()
            await pilot.press("backspace")
            await pilot.pause()

            assert chat._prompt_search_active is False
            assert chat._text_area.text == ""

    async def test_selection_past_first_page_stays_mounted_and_visible(
        self, tmp_path
    ) -> None:
        """Regression: rows beyond the first page must exist to be shown.

        The panel windows the DOM around the selection, so navigating past the
        first page with arrows must keep the selected row mounted and scrolled
        into view rather than pointing at a row that was never rendered.
        """
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            chat._history.history_file = tmp_path / "history.jsonl"
            self._seed_history(chat, [f"prompt {i}" for i in range(30)])
            await pilot.pause()

            chat.open_prompt_search()
            await pilot.pause()
            await pilot.pause()
            panel = chat._prompt_search
            assert panel is not None

            # Arrow past the first page of 5 rows.
            for _ in range(12):
                await pilot.press("down")
            await pilot.pause()
            await pilot.pause()

            assert chat._prompt_search_index == 12
            mounted = {option.index for option in panel._options}
            assert chat._prompt_search_index in mounted
            selected = [
                o for o in panel._options if o.index == chat._prompt_search_index
            ]
            assert selected[0].is_selected

            # Enter inserts the exact windowed prompt.
            await pilot.press("enter")
            await pilot.pause()
            assert chat._text_area is not None
            assert chat._text_area.text == "prompt 17"  # newest-first: 29-12

    async def test_empty_state_reopens_cleanly_after_each_hide(self, tmp_path) -> None:
        """Empty -> hide cycles must not accumulate orphaned empty rows."""
        from textual.widgets import Static

        from deepagents_code.tui.widgets.prompt_search import PromptSearchOption

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            chat._history.history_file = tmp_path / "history.jsonl"
            self._seed_history(chat, ["hello world"])
            await pilot.pause()

            for _ in range(3):
                chat.open_prompt_search()
                await pilot.pause()
                await pilot.pause()
                await pilot.press("z")
                await pilot.pause()
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

            panel = chat._prompt_search
            assert panel is not None
            results = panel.query_one("#prompt-search-results")
            empty_rows = [
                child
                for child in results.children
                if isinstance(child, Static)
                and not isinstance(child, PromptSearchOption)
            ]
            # Exactly one: `<= 1` also passes when the row vanishes entirely,
            # which is the other half of the bug this locks in.
            assert len(empty_rows) == 1

    async def test_unreadable_history_is_not_reported_as_empty(
        self, tmp_path: Path
    ) -> None:
        """An unreadable history file must not claim the user has no prompts."""
        from textual.widgets import Static

        from deepagents_code.tui.widgets.prompt_search import PromptSearchOption

        # A directory in place of the file is an OSError on read, the same path
        # permissions and encoding failures take.
        history_file = tmp_path / "history.jsonl"
        history_file.mkdir()
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            chat._history.history_file = history_file
            # Nothing cached in memory, so the read failure is the only reason
            # the list comes back empty.
            chat._history._entries = []
            await pilot.pause()

            chat.open_prompt_search()
            await pilot.pause()
            await pilot.pause()

            panel = chat._prompt_search
            assert panel is not None
            results = panel.query_one("#prompt-search-results")
            messages = [
                str(child.render())
                for child in results.children
                if isinstance(child, Static)
                and not isinstance(child, PromptSearchOption)
            ]
            assert messages
            assert "Could not read prompt history" in messages[0]
            assert "No prompts yet" not in messages[0]
