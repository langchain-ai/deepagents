"""Unit tests for ChatInput widget and completion popup."""

from __future__ import annotations

import asyncio
import html
from typing import TYPE_CHECKING

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static
from textual.widgets.text_area import Selection

from deepagents_code import _textual_patches as _textual_patches
from deepagents_code.command_registry import SLASH_COMMANDS
from deepagents_code.input import MediaTracker
from deepagents_code.media_utils import ImageData
from deepagents_code.widgets import chat_input as chat_input_module
from deepagents_code.widgets.autocomplete import MAX_SUGGESTIONS
from deepagents_code.widgets.chat_input import (
    ChatInput,
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


class TestChatInputFileCacheRefresh:
    def test_refresh_file_cache_uses_exclusive_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat = ChatInput()
        controller = _RefreshController()
        worker_calls: list[dict[str, object]] = []

        def fake_run_worker(
            work: Coroutine[object, object, None], **kwargs: object
        ) -> None:
            work.close()
            worker_calls.append(kwargs)

        monkeypatch.setattr(chat, "_file_controller", controller, raising=False)
        monkeypatch.setattr(chat, "run_worker", fake_run_worker)

        chat._refresh_file_cache()

        assert controller.force_values == [True]
        assert worker_calls == [
            {
                "exclusive": True,
                "group": chat_input_module._FILE_CACHE_WORKER_GROUP,
                "exit_on_error": False,
            }
        ]

    def test_set_cwd_uses_file_cache_worker_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat = ChatInput()
        controller = _RefreshController()
        worker_calls: list[dict[str, object]] = []

        def fake_run_worker(
            work: Coroutine[object, object, None], **kwargs: object
        ) -> None:
            work.close()
            worker_calls.append(kwargs)

        monkeypatch.setattr(chat, "_file_controller", controller, raising=False)
        monkeypatch.setattr(chat, "run_worker", fake_run_worker)

        chat.set_cwd(tmp_path)

        assert controller.cwd_values == [tmp_path]
        assert controller.force_values == [False]
        assert worker_calls == [
            {
                "exclusive": False,
                "group": chat_input_module._FILE_CACHE_WORKER_GROUP,
                "exit_on_error": False,
            }
        ]

    def test_refresh_file_cache_noop_without_controller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refresh is a no-op before the file controller is wired up."""
        chat = ChatInput()
        worker_calls: list[object] = []
        monkeypatch.setattr(
            chat,
            "run_worker",
            lambda *args, **kwargs: worker_calls.append((args, kwargs)),
        )

        assert getattr(chat, "_file_controller", None) is None
        chat._refresh_file_cache()

        assert worker_calls == []

    async def test_on_mount_schedules_file_cache_refresh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`on_mount` schedules the periodic `@` file-cache refresh.

        Without this the cache would only warm on mount and cwd switches, so
        files created or deleted mid-session would never surface.
        """
        interval_calls: list[tuple[float, object]] = []

        def fake_set_interval(
            _self: ChatInput, interval: float, callback: object, **_kwargs: object
        ) -> None:
            interval_calls.append((interval, callback))

        monkeypatch.setattr(ChatInput, "set_interval", fake_set_interval)

        app = _ChatInputTestApp()
        async with app.run_test():
            chat = app.query_one(ChatInput)
            assert (
                chat_input_module._FILE_CACHE_REFRESH_INTERVAL_SECONDS,
                chat._refresh_file_cache,
            ) in interval_calls


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

    async def test_newline_into_short_draft_never_flashes_scrollbar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A newline below `max-height` must never show the vertical scrollbar.

        Records the scrollbar decision on every refresh triggered by the insert
        (not just the settled state) so the one-frame flash is caught. Fails
        against the unpatched base behavior, which shows the bar mid-reflow.
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            text_area = chat.input_widget
            assert text_area is not None
            text_area.focus()
            await pilot.pause()

            decisions: list[bool] = []
            original = ChatTextArea._refresh_scrollbars

            def _record(self: ChatTextArea) -> None:
                original(self)
                decisions.append(self.show_vertical_scrollbar)

            monkeypatch.setattr(ChatTextArea, "_refresh_scrollbars", _record)
            await pilot.press("shift+enter")
            for _ in range(4):
                await pilot.pause()

            assert text_area.text == "\n"
            assert text_area.max_scroll_y == 0
            assert decisions, "expected a scrollbar refresh during the insert"
            assert not any(decisions), (
                f"vertical scrollbar flashed during newline insert: {decisions}"
            )
            assert text_area.show_vertical_scrollbar is False

    async def test_overflowing_draft_keeps_visible_scrollbar(self) -> None:
        """A draft taller than `max-height` keeps a real, scrollable bar."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            text_area = chat.input_widget
            assert text_area is not None
            text_area.focus()
            await pilot.pause()

            for _ in range(15):
                await pilot.press("shift+enter")
            for _ in range(3):
                await pilot.pause()

            assert text_area.max_scroll_y > 0
            assert text_area.show_vertical_scrollbar is True
            assert text_area.scrollbar_size_vertical > 0
            # The cursor stays in view at the bottom of the overflowing draft.
            rel_y = text_area.cursor_location[0] - text_area.scroll_offset.y
            assert 0 <= rel_y < text_area.size.height

    async def test_settled_content_height_resolves_max_height(self) -> None:
        """The flash-suppression bound resolves to `max-height` in content rows.

        Guards the override silently disabling itself: if `max-height` stops
        resolving to a fixed cell count, `_settled_content_height` returns
        `None` and the flash returns.
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            text_area = app.query_one(ChatInput).input_widget
            assert text_area is not None
            await pilot.pause()
            # max-height: 8 with no border/padding -> 8 content rows.
            assert text_area._settled_content_height() == 8


class TestChatTextAreaKeybindings:
    """Regression tests for terminal key aliases in the chat input."""

    def test_newline_bindings_do_not_shadow_enter_alias(self) -> None:
        """`ctrl+m` is carriage return in terminals, so it must remain plain Enter."""
        newline_keys = {
            key.strip()
            for binding in ChatTextArea.BINDINGS
            if binding.action == "insert_newline"
            for key in binding.key.split(",")
        }

        assert "ctrl+m" not in newline_keys
        assert "ctrl+m" not in ChatTextArea._NEWLINE_KEYS

    def test_modified_backspace_deletes_word_left(self) -> None:
        """Modified Backspace aliases should delete the previous word."""
        word_delete_keys = {
            key.strip()
            for binding in ChatTextArea.BINDINGS
            if binding.action == "delete_word_left"
            for key in binding.key.split(",")
        }

        assert "ctrl+backspace" in word_delete_keys
        assert "alt+backspace" in word_delete_keys


class TestDiscardText:
    """Tests for the undoable draft clear behind esc+esc and the `[ X ]` button."""

    async def test_discard_text_clears_and_reports_cleared(self) -> None:
        """`discard_text` empties the draft and returns True when text existed."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            text_area.insert("a draft I changed my mind about")
            await pilot.pause()

            assert chat_input.discard_text() is True
            await pilot.pause()
            assert chat_input.value == ""

    async def test_discard_text_no_op_when_empty(self) -> None:
        """`discard_text` returns False and leaves the media-skip counter alone."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            await pilot.pause()
            before = chat_input._skip_media_sync_events
            assert chat_input.discard_text() is False
            # An empty no-op must not bump the skip counter: a stray increment
            # would later swallow a legitimate media sync, desyncing placeholders.
            assert chat_input._skip_media_sync_events == before

    async def test_discard_text_is_undoable(self) -> None:
        """The cleared draft is restorable via the TextArea undo (ctrl+z)."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            text_area.insert("restore me")
            await pilot.pause()

            assert chat_input.discard_text() is True
            await pilot.pause()
            assert chat_input.value == ""

            text_area.undo()
            await pilot.pause()
            assert chat_input.value == "restore me"

    async def test_discard_text_preserves_media_for_undo(self) -> None:
        """Undoing a cleared media draft keeps placeholder media attached."""
        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            placeholder = app.tracker.add_image(
                ImageData(base64_data="abc", format="png", placeholder="")
            )
            text_area.insert(placeholder)
            await pilot.pause()

            assert len(app.tracker.get_images()) == 1
            assert chat_input.discard_text() is True
            await pilot.pause()
            assert chat_input.value == ""
            assert len(app.tracker.get_images()) == 1

            text_area.undo()
            await pilot.pause()
            assert chat_input.value == placeholder
            assert len(app.tracker.get_images()) == 1


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

    async def test_buttons_render_on_input_border(self) -> None:
        """Buttons sit on the box's top border line, above full-width text.

        They render on the border row (not a content row), so the text area
        keeps the full width and the draft is never overlapped. The top-right
        corner stays visible, a first-row text click still reaches the text
        area, and a button click hits the button.
        """
        app = _ChatInputTestApp()
        async with app.run_test(size=(60, 24)) as pilot:
            await pilot.pause()
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            # A long single line would wrap to the text area's full width.
            text_area.insert("Z" * 200)
            await pilot.pause()

            box = chat_input.query_one("#input-box")
            clear = chat_input.query_one("#clear-button", Static)
            copy = chat_input.query_one("#copy-button", Static)

            # Text area spans the full width inside the border.
            assert text_area.region.right == box.content_region.right

            # Buttons render on the top border row, above the first text row.
            assert clear.region.y == box.region.y
            assert copy.region.y == box.region.y
            assert text_area.region.y > box.region.y

            # The top-right corner stays visible (buttons stop short of the edge).
            assert copy.region.right < box.region.right

            # No overlap: a first-row click reaches the text area, and a click on
            # a button hits the button.
            left_widget, _ = app.screen.get_widget_at(
                text_area.region.x + 1, text_area.region.y
            )
            assert left_widget is text_area
            button_widget, _ = app.screen.get_widget_at(
                copy.region.x + 1, copy.region.y
            )
            assert button_widget is copy

    async def test_buttons_hidden_until_draft_entered(self) -> None:
        """The buttons appear only while the draft has non-whitespace content."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            actions = chat_input.query_one("#input-actions")

            # Empty draft: nothing to clear or copy, so the buttons stay hidden.
            assert actions.display is False

            # Whitespace-only input has nothing worth acting on: still hidden.
            text_area.insert("  \n\n  ")
            await pilot.pause()
            assert actions.display is False

            # Real content reveals them.
            text_area.insert("draft")
            await pilot.pause()
            assert actions.display is True

            # Clearing the draft hides them again.
            chat_input.discard_text()
            await pilot.pause()
            assert actions.display is False

    async def test_history_navigation_hides_buttons_in_same_frame(self) -> None:
        """Emptying the draft via history/clear hides the buttons synchronously."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            actions = chat_input.query_one("#input-actions")

            # Recalling content shows the buttons in the same frame (no pause).
            text_area.set_text_from_history("recalled", cursor_at_end=True)
            assert actions.display is True

            # Tabbing forward to an empty draft hides them in the same frame,
            # before the suppressed Changed event would otherwise process.
            text_area.set_text_from_history("", cursor_at_end=True)
            assert actions.display is False

            # clear_text empties the draft and hides them synchronously too.
            text_area.set_text_from_history("recalled", cursor_at_end=True)
            assert actions.display is True
            text_area.clear_text()
            assert actions.display is False

    async def test_copy_button_double_click_does_not_select_label(self) -> None:
        """Double-clicking `[ COPY ]` should not trigger Textual word selection."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            text_area.insert("draft")  # buttons only render with a draft
            await pilot.pause()

            await pilot.double_click("#copy-button", offset=(3, 0))
            await pilot.pause()

            assert app.screen.get_selected_text() is None

    async def test_clear_button_clears_input(self) -> None:
        """Clicking `[ X ]` empties the draft."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            text_area.insert("clear me")
            await pilot.pause()

            await pilot.click("#clear-button")
            await pilot.pause()
            assert chat_input.value == ""

    async def test_clear_button_is_undoable(self) -> None:
        """A draft cleared via `[ X ]` can be restored with undo."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            text_area.insert("undo me")
            await pilot.pause()

            await pilot.click("#clear-button")
            await pilot.pause()
            assert chat_input.value == ""

            text_area.undo()
            await pilot.pause()
            assert chat_input.value == "undo me"

    async def test_clear_button_exits_command_mode(self) -> None:
        """Clicking `[ X ]` should not leave a stale slash-command mode active."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None

            text_area.insert("/")
            await _pause_for_strip(pilot)
            assert chat_input.mode == "command"
            assert chat_input._current_suggestions

            text_area.insert("help")
            await pilot.pause()
            await pilot.click("#clear-button")
            await pilot.pause()

            assert chat_input.mode == "normal"
            assert chat_input.value == ""
            assert chat_input._current_suggestions == []

            text_area.insert("hello")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "hello"
            assert app.submitted[0].mode == "normal"

    async def test_copy_button_copies_input(self, monkeypatch) -> None:
        """Clicking `[ COPY ]` sends the draft to the clipboard helper."""
        import deepagents_code.clipboard as clipboard_module

        copied: list[str] = []

        def fake_copy(_app: object, text: str) -> tuple[bool, str | None]:
            copied.append(text)
            return True, None

        monkeypatch.setattr(clipboard_module, "copy_text_to_clipboard", fake_copy)

        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            text_area.insert("copy me")
            await pilot.pause()

            await pilot.click("#copy-button")
            await pilot.pause()

        assert copied == ["copy me"]

    async def test_copy_button_failure_warns(self, monkeypatch) -> None:
        """A failed `[ COPY ]` surfaces a warning toast instead of failing silently."""
        import deepagents_code.clipboard as clipboard_module

        def fake_copy(_app: object, _text: str) -> tuple[bool, str | None]:
            return False, "boom"

        monkeypatch.setattr(clipboard_module, "copy_text_to_clipboard", fake_copy)

        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            text_area.insert("copy me")
            await pilot.pause()

            notifications: list[tuple[str, object]] = []
            monkeypatch.setattr(
                app,
                "notify",
                lambda message, **kwargs: notifications.append(
                    (message, kwargs.get("severity"))
                ),
            )

            await pilot.click("#copy-button")
            await pilot.pause()

        assert notifications == [("Failed to copy input: boom", "warning")]

    async def test_clear_button_refocuses_input(self) -> None:
        """Clicking `[ X ]` returns focus to the text area so typing can continue."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            text_area.insert("clear me")
            await pilot.pause()

            await pilot.click("#clear-button")
            await pilot.pause()
            assert text_area.has_focus

    async def test_copy_button_refocuses_input(self, monkeypatch) -> None:
        """`[ COPY ]` returns focus to the input (not the non-focusable button)."""
        import deepagents_code.clipboard as clipboard_module

        monkeypatch.setattr(
            clipboard_module,
            "copy_text_to_clipboard",
            lambda _app, _text: (True, None),
        )

        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input.input_widget
            assert text_area is not None
            text_area.insert("copy me")  # buttons only render with a draft
            await pilot.pause()

            await pilot.click("#copy-button")
            await pilot.pause()
            assert text_area.has_focus


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

    async def test_prompt_shows_shell_style_in_incognito_shell_mode(self) -> None:
        """Incognito shell mode sets the `$` prompt, border title, and class."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            prompt = chat_input.query_one("#prompt", Static)

            chat_input.mode = "shell_incognito"
            await pilot.pause()

            input_box = chat_input.query_one("#input-box")
            assert _prompt_text(prompt) == "$"
            assert input_box.border_title == "incognito"
            assert chat_input.has_class("mode-shell-incognito")

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
            assert chat_input.border_title is None
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
            text_area.set_text_from_history("first", cursor_at_end=False)
            text_area.set_text_from_history("second", cursor_at_end=False)
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


class TestSetValueAtEnd:
    """Tests for programmatically setting input text at the end cursor position."""

    async def test_places_cursor_at_end(self) -> None:
        """set_value_at_end loads text and lands the cursor after the last char."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            chat_input.set_value_at_end("ls -la")
            await pilot.pause()

            assert text_area.text == "ls -la"
            assert text_area.cursor_location == (0, len("ls -la"))

    async def test_multiline_places_cursor_at_end(self) -> None:
        """set_value_at_end handles multi-line text by targeting the last line."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInput)
            text_area = chat_input._text_area
            assert text_area is not None

            chat_input.set_value_at_end("first\nsecond")
            await pilot.pause()

            assert text_area.text == "first\nsecond"
            assert text_area.cursor_location == (1, len("second"))


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

    async def test_click_while_focused_moves_cursor(self) -> None:
        """A click without a preceding refocus moves the cursor normally."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            text_area = chat._text_area
            assert text_area is not None

            text_area.insert("hello world")
            text_area.move_cursor((0, 0))
            await pilot.pause()
            assert text_area.cursor_location == (0, 0)

            await pilot.click(ChatTextArea, offset=(6, 0))
            await pilot.pause()

            assert text_area.cursor_location != (0, 0)

    def test_consume_refocus_click_requires_blur(self) -> None:
        """Without a preceding blur, focus does not arm click suppression."""
        text_area = ChatTextArea()
        text_area._notify_app_focus()
        assert text_area._consume_refocus_click() is False

    def test_consume_refocus_click_fires_once(self) -> None:
        """Only the first click after a refocus is suppressed."""
        text_area = ChatTextArea()
        text_area._notify_app_blur()
        text_area._notify_app_focus()
        assert text_area._consume_refocus_click() is True
        assert text_area._consume_refocus_click() is False

    def test_consume_refocus_click_expires_after_window(self) -> None:
        """A click landing after the window elapses moves the cursor normally."""
        text_area = ChatTextArea()
        text_area._notify_app_blur()
        text_area._notify_app_focus()
        # Backdate the refocus stamp past the window so the gap check fails.
        text_area._refocus_time = (
            chat_input_module.time.monotonic()
            - chat_input_module._REFOCUS_CLICK_SUPPRESS_WINDOW_SECONDS
            - 0.01
        )
        assert text_area._consume_refocus_click() is False

    def test_consume_refocus_click_rearms_each_cycle(self) -> None:
        """Suppression re-arms on every blur/focus cycle, not just the first."""
        text_area = ChatTextArea()
        text_area._notify_app_blur()
        text_area._notify_app_focus()
        assert text_area._consume_refocus_click() is True
        # A second cycle must arm suppression again.
        text_area._notify_app_blur()
        text_area._notify_app_focus()
        assert text_area._consume_refocus_click() is True


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

    async def test_up_at_cursor_zero_navigates_history(self) -> None:
        """Up at (0, 0) goes straight to history."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("say hello world")

            chat._text_area.insert("hello")
            await pilot.pause()
            chat._text_area.move_cursor((0, 0))
            await pilot.pause()

            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "say hello world"

    async def test_down_at_non_end_moves_cursor_not_history(self) -> None:
        """Down with a row below moves the cursor, not history."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("previous entry")

            chat._text_area.text = "line one\nline two"
            chat._text_area.move_cursor((0, 3))
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()
            assert chat._text_area.text == "line one\nline two"
            cursor_row, _ = chat._text_area.cursor_location
            assert cursor_row == 1

    async def test_up_in_middle_of_multiline_moves_cursor(self) -> None:
        """Up from a middle row moves the cursor, not history."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("previous entry")

            chat._text_area.text = "line one\nline two\nline three"
            chat._text_area.move_cursor((1, 3))
            await pilot.pause()

            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "line one\nline two\nline three"
            cursor_row, _ = chat._text_area.cursor_location
            assert cursor_row == 0

    async def test_up_load_places_cursor_at_top(self) -> None:
        """A history entry loaded via up has cursor at (0, 0).

        This is what enables continuous up-navigation: the next up press
        immediately triggers another history previous without snapping the
        cursor first.
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("line one\nline two")

            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "line one\nline two"
            assert chat._text_area.cursor_location == (0, 0)

    async def test_continuous_up_navigates_through_history(self) -> None:
        """Repeated up presses walk back through history without manual cursor moves."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.extend(
                ["oldest", "middle entry\nwith two lines", "newest"]
            )

            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "newest"

            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "middle entry\nwith two lines"

            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "oldest"

    async def test_continuous_down_navigates_forward_through_history(self) -> None:
        """Repeated down presses walk forward through history.

        After down-navigation, cursor lands at the end of the loaded entry,
        so the next down press triggers another history next.
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.extend(["oldest", "middle", "newest"])

            # Walk up to the oldest entry.
            for _ in range(3):
                await pilot.press("up")
                await pilot.pause()
            assert chat._text_area.text == "oldest"
            assert chat._text_area.cursor_location == (0, 0)

            # Switching direction requires one snap-to-end press first.
            await pilot.press("down")
            await pilot.pause()
            assert chat._text_area.text == "oldest"
            assert chat._text_area.cursor_location == (0, len("oldest"))

            # Subsequent down presses navigate forward continuously.
            await pilot.press("down")
            await pilot.pause()
            assert chat._text_area.text == "middle"

            await pilot.press("down")
            await pilot.pause()
            assert chat._text_area.text == "newest"

    async def test_down_past_newest_restores_typed_input(self) -> None:
        """Down past the newest history entry restores the user's typed input."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("only entry")

            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "only entry"

            # Cursor at (0, 0) after up-load; need to move to end first.
            chat._text_area.move_cursor((0, len("only entry")))
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()
            assert chat._text_area.text == ""

    async def test_typed_newlines_up_from_end_walks_rows(self) -> None:
        """Up from the end of multi-row typed input walks the cursor up rows."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "abc\ndef\nghi"
            chat._text_area.move_cursor((2, len("ghi")))
            await pilot.pause()

            for expected_row in (1, 0):
                await pilot.press("up")
                await pilot.pause()
                assert chat._text_area.text == "abc\ndef\nghi"
                cursor_row, _ = chat._text_area.cursor_location
                assert cursor_row == expected_row

            # Cursor is now at (0, 3); next up snaps to (0, 0).
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "abc\ndef\nghi"
            assert chat._text_area.cursor_location == (0, 0)

    async def test_soft_wrapped_single_row_navigates_visual_lines(self) -> None:
        """Up/down on a soft-wrapped single doc row walks visual lines.

        A row-based history trigger (`row == 0`) would incorrectly fire on the
        last visual line of a wrapped doc row. The cursor-cannot-move check
        avoids that: visual lines below the top of the wrapped row still have
        a "row above" in the wrapped document, so cursor movement wins.
        """
        app = _ChatInputTestApp()
        # Constrain width so a long single-line entry wraps to several
        # visual lines but stays on doc row 0.
        async with app.run_test(size=(20, 24)) as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("history entry")

            long_line = "word " * 30  # ~150 chars, well past wrap width
            chat._text_area.text = long_line.strip()
            chat._text_area.move_cursor((0, len(chat._text_area.text)))
            await pilot.pause()

            # Cursor on the last visual line of doc row 0 — up should walk
            # back through visual lines, not fire history.
            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == long_line.strip()
            # Cursor should still be on doc row 0 but at a smaller column
            # corresponding to the previous visual line.
            row, col = chat._text_area.cursor_location
            assert row == 0
            assert col < len(chat._text_area.text)

    async def test_shift_up_at_top_extends_selection_not_history(self) -> None:
        """`shift+up` at (0, 0) should not fire history navigation.

        The action_cursor_up guard requires `not select`, so shift+up must
        fall through to TextArea's selection-extending behavior even when
        the cursor literally cannot move further up.
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("older entry")

            chat._text_area.text = "hello"
            chat._text_area.move_cursor((0, 3))
            await pilot.pause()

            # shift+up at row 0 should not replace text with history.
            await pilot.press("shift+up")
            await pilot.pause()
            assert chat._text_area.text == "hello"

    async def test_up_with_unmatched_query_is_noop(self) -> None:
        """Up at (0,0) with typed text that matches no history entry is a no-op.

        `HistoryManager.get_previous` filters by substring; when typed text
        doesn't appear in any entry, the load is skipped. The text area
        should stay unchanged (and the bell on the handler is allowed to
        ring as a boundary signal).
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("totally different entry")

            chat._text_area.insert("abc")
            chat._text_area.move_cursor((0, 0))
            await pilot.pause()

            await pilot.press("up")
            await pilot.pause()
            assert chat._text_area.text == "abc"
            assert chat._text_area.cursor_location == (0, 0)


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

    async def test_redundant_typed_slash_keystroke_stays_command_mode(self) -> None:
        """A redundant `/` at the command prompt is consumed as a mode selector."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await pilot.press("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"
            assert chat._text_area.text == ""

            await pilot.press("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"
            assert chat._text_area.text == ""

    async def test_typed_bang_keystroke_skips_strip_round_trip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pressing `!` enters shell mode without an insert-then-strip round trip."""
        strip_calls: list[int] = []
        original = ChatInput._strip_mode_prefix

        def _spy(self: ChatInput, length: int = 1) -> None:
            strip_calls.append(length)
            original(self, length)

        monkeypatch.setattr(ChatInput, "_strip_mode_prefix", _spy)

        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await pilot.press("!")
            await _pause_for_strip(pilot)

            assert chat.mode == "shell"
            assert chat._text_area.text == ""
            assert strip_calls == []

    async def test_typed_slash_keystroke_enters_command_mode_with_completions(
        self,
    ) -> None:
        """Pressing `/` enters command mode and activates completions, no flash."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await pilot.press("/")
            await _pause_for_strip(pilot)

            assert chat.mode == "command"
            assert chat._text_area.text == ""
            assert chat._completion_manager is not None
            assert chat._completion_manager._active is not None

    async def test_typed_bang_not_at_start_is_literal(self) -> None:
        """A `!` typed mid-text is body content, not a mode switch."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("ab")
            await pilot.pause()
            await pilot.press("!")
            await _pause_for_strip(pilot)

            assert chat.mode == "normal"
            assert chat._text_area.text == "ab!"

    async def test_typed_trigger_at_cursor_zero_with_text_switches_mode(self) -> None:
        """A trigger typed at start of existing text switches mode, keeps text.

        Exercises the `cursor_location == (0, 0)` arm of the `_on_key` guard
        with a non-empty input: the keystroke is consumed (no inserted `!`) and
        the body text is preserved, matching the legacy insert-then-strip path.
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("abc")
            await pilot.pause()
            chat._text_area.move_cursor((0, 0))
            await pilot.pause()

            await pilot.press("!")
            await _pause_for_strip(pilot)

            assert chat.mode == "shell"
            assert chat._text_area.text == "abc"

    async def test_typed_trigger_with_selection_is_not_intercepted(self) -> None:
        """A trigger typed over a selection replaces it instead of switching.

        A backward selection puts the cursor at `(0, 0)` while leaving the
        selection non-empty, so only the `selection.is_empty` arm of the
        `_on_key` guard keeps the keystroke from being intercepted. Removing
        that arm would swallow the `/` and strand the selected text in the
        input, which this test catches.
        """
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("ab")
            await pilot.pause()
            # Anchor at end, cursor at start: cursor_location is (0, 0) but the
            # selection is non-empty.
            chat._text_area.selection = Selection((0, 2), (0, 0))
            await pilot.pause()
            assert chat._text_area.cursor_location == (0, 0)
            assert not chat._text_area.selection.is_empty

            await pilot.press("/")
            await _pause_for_strip(pilot)

            # TextArea replaced the selected "ab" with "/", which the change
            # handler then detected and stripped into command mode. The key
            # point: the selected text did not survive as literal input.
            assert chat._text_area.text == ""
            assert chat.mode == "command"

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

    async def test_submission_prepends_incognito_shell_prefix(self) -> None:
        """Submitting in incognito shell mode should preserve the `'!!'` prefix."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "!!pwd"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell_incognito"
            assert chat._text_area.text == "pwd"

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "!!pwd"
            assert app.submitted[0].mode == "shell_incognito"

    async def test_typing_second_bang_enters_incognito_shell_mode(self) -> None:
        """Typing `!!pwd` as separate keypresses should submit incognito shell."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await pilot.press("!")
            await _pause_for_strip(pilot)
            assert chat.mode == "shell"
            assert chat._text_area.text == ""

            await pilot.press("!")
            await _pause_for_strip(pilot)
            assert chat.mode == "shell_incognito"
            assert chat._text_area.text == ""

            chat._text_area.insert("pwd")
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "!!pwd"
            assert app.submitted[0].mode == "shell_incognito"

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

    async def test_pasted_three_bangs_routes_to_incognito(self) -> None:
        """Pasting `!!!ls` must enter `shell_incognito` with body `!ls`."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.text = "!!!ls"
            await _pause_for_strip(pilot)
            assert chat.mode == "shell_incognito"
            assert chat._text_area.text == "!ls"

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].mode == "shell_incognito"
            assert app.submitted[0].value == "!!!ls"

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

    async def test_stale_enter_single_slash_match_submits_completion(self) -> None:
        """Fast Enter on an unambiguous slash prefix should submit the match."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.mode = "command"
            chat._text_area.text = "mod"
            chat._text_area.move_cursor((0, 3))
            chat._text_area.set_completion_active(active=False)

            await chat._text_area._on_key(events.Key("enter", None))
            await pilot.pause()

            assert [event.value for event in app.submitted] == ["/model"]
            assert app.submitted[0].mode == "command"

    async def test_stale_enter_multiple_slash_matches_shows_popup(self) -> None:
        """Fast Enter on an ambiguous slash prefix should show choices."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.mode = "command"
            chat._text_area.text = "re"
            chat._text_area.move_cursor((0, 2))
            chat._text_area.set_completion_active(active=False)

            await chat._text_area._on_key(events.Key("enter", None))
            await pilot.pause()

            labels = [label for label, _ in chat._current_suggestions]
            assert not app.submitted
            assert "/reload" in labels
            assert "/remember" in labels
            assert chat._text_area._completion_active is True

    async def test_stale_enter_multiple_slash_matches_next_enter_selects(
        self,
    ) -> None:
        """Popup shown by stale Enter should remain keyboard-operable."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.mode = "command"
            chat._text_area.text = "re"
            chat._text_area.move_cursor((0, 2))
            chat._text_area.set_completion_active(active=False)

            await chat._text_area._on_key(events.Key("enter", None))
            await pilot.pause()
            assert not app.submitted
            assert chat._current_suggestions
            selected_label = chat._current_suggestions[chat._current_selected_index][0]

            await chat.on_key(events.Key("enter", None))
            await pilot.pause()

            assert [event.value for event in app.submitted] == [selected_label]
            assert app.submitted[0].mode == "command"

    async def test_stale_enter_submits_exact_restart_command(self) -> None:
        """Exact restart command should submit without requiring autocomplete."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.mode = "command"
            chat._text_area.text = "restart"
            chat._text_area.move_cursor((0, 7))
            chat._text_area.set_completion_active(active=False)

            await chat._text_area._on_key(events.Key("enter", None))
            await pilot.pause()

            assert [event.value for event in app.submitted] == ["/restart"]
            assert app.submitted[0].mode == "command"

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

    async def test_history_incognito_shell_entry_enters_incognito_mode(self) -> None:
        """Recalling a `!!` history entry should enter incognito shell mode."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._history._entries.append("!!pwd")

            await pilot.press("up")
            await _pause_for_strip(pilot)

            assert chat.mode == "shell_incognito"
            assert chat._text_area.text == "pwd"

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "!!pwd"
            assert app.submitted[0].mode == "shell_incognito"


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

    async def test_paste_image_path_skips_literal_placeholder_in_draft(
        self, tmp_path
    ) -> None:
        """Attaching media does not bind a literal placeholder already in text."""
        img_path = tmp_path / "drop.png"
        from PIL import Image

        image = Image.new("RGB", (4, 4), color="blue")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None
            chat._text_area.text = "restore [image 1] "
            chat._text_area.move_cursor_to_end()

            await chat._text_area._on_paste(events.Paste(str(img_path)))
            await pilot.pause()

            assert chat._text_area.text == "restore [image 1] [image 2] "
            assert [img.placeholder for img in app.tracker.get_images()] == [
                "[image 2]"
            ]

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


class TestPathPayloadDetectionGating:
    """Single-keystroke edits should skip the blocking path-detection helpers.

    `_is_dropped_path_payload` and `_apply_inline_dropped_path_replacement`
    reach `Path.exists()` / `Path.is_file()` via
    `deepagents_code.input.parse_pasted_path_payload`, which are synchronous
    stat syscalls on the event-loop thread. They are only meaningful when a
    text change inserts more than one character (drag-drop / bracketed paste);
    on normal typing they cost real wall-clock time for no possible match.
    """

    async def test_typing_does_not_invoke_path_detection(self) -> None:
        """Char-by-char keypresses must not run path-detection helpers."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            detect_calls = 0
            replace_calls = 0
            original_detect = chat._is_dropped_path_payload
            original_replace = chat._apply_inline_dropped_path_replacement

            def counting_detect(text: str) -> bool:
                nonlocal detect_calls
                detect_calls += 1
                return original_detect(text)

            def counting_replace(text: str) -> bool:
                nonlocal replace_calls
                replace_calls += 1
                return original_replace(text)

            chat._is_dropped_path_payload = counting_detect  # ty: ignore
            chat._apply_inline_dropped_path_replacement = counting_replace  # ty: ignore

            for char in "hello":
                await pilot.press(char)
            await pilot.pause()

            assert detect_calls == 0
            assert replace_calls == 0

    async def test_bulk_text_change_invokes_path_detection(
        self, tmp_path: Path
    ) -> None:
        """Multi-char Changed events (drag-drop / paste) must still detect paths."""
        target = tmp_path / "dropped.txt"
        target.write_text("payload")

        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            detect_calls = 0
            original_detect = chat._is_dropped_path_payload

            def counting_detect(text: str) -> bool:
                nonlocal detect_calls
                detect_calls += 1
                return original_detect(text)

            chat._is_dropped_path_payload = counting_detect  # ty: ignore

            ta.text = str(target)
            await pilot.pause()

            assert detect_calls >= 1

    async def test_replacement_edit_with_small_length_delta_detects_path(
        self, tmp_path: Path
    ) -> None:
        """Replacing selected text with a similar-length path should attach it."""
        img_path = tmp_path / "similar-length.png"
        from PIL import Image

        image = Image.new("RGB", (3, 3), color="orange")
        image.save(img_path, format="PNG")

        app = _ImagePasteApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.text = "x" * len(str(img_path))
            await pilot.pause()

            ta.text = str(img_path)
            await pilot.pause()

            assert ta.text == "[image 1] "
            assert chat.mode == "normal"
            assert len(app.tracker.get_images()) == 1


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
        monkeypatch.setattr(chat_input_module, "_BACKSLASH_ENTER_GAP_SECONDS", 60.0)

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

    @pytest.mark.parametrize(
        "newline_keys",
        [
            pytest.param(["shift+enter"], id="modifier_enter"),
            pytest.param(["ctrl+j"], id="ctrl_j"),
            pytest.param(["backslash", "enter"], id="vscode_backslash_fallback"),
        ],
    )
    async def test_newline_past_max_height_scrolls_cursor_into_view(
        self, monkeypatch: pytest.MonkeyPatch, newline_keys: list[str]
    ) -> None:
        """Every newline-insertion path keeps the cursor in view past max-height.

        All three paths (binding, `_NEWLINE_KEYS` branch, and the
        backslash+enter fallback for terminals that emulate shift+enter)
        must route through `action_insert_newline`, where the
        `call_after_refresh(scroll_cursor_visible)` keeps the cursor visible.
        """
        # Widen the backslash+enter gap so the fallback test isn't racy on CI.
        monkeypatch.setattr(chat_input_module, "_BACKSLASH_ENTER_GAP_SECONDS", 60.0)

        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            # Build a doc tall enough to overflow the widget's max height,
            # then move the cursor to the last row.
            ta.text = "\n".join(f"row {i}" for i in range(15))
            ta.move_cursor((14, len("row 14")))
            await pilot.pause()
            assert ta.scroll_offset.y > 0

            for key in newline_keys:
                await pilot.press(key)
            await pilot.pause()

            cursor_row = ta.cursor_location[0]
            assert cursor_row == 15
            rel_y = cursor_row - ta.scroll_offset.y
            assert 0 <= rel_y < ta.size.height, (
                f"cursor row {cursor_row} not in viewport "
                f"[{ta.scroll_offset.y}, {ta.scroll_offset.y + ta.size.height})"
            )

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

    async def test_backslash_enter_on_empty_prompt_does_not_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backslash + enter on empty prompt should not submit."""
        monkeypatch.setattr(chat_input_module, "_BACKSLASH_ENTER_GAP_SECONDS", 60.0)

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


class TestModifiedBackspaceDeleteWordLeft:
    """Test modified Backspace aliases for word deletion."""

    @pytest.mark.parametrize("key", ["ctrl+backspace", "alt+backspace"])
    async def test_modified_backspace_deletes_previous_word(self, key: str) -> None:
        """Modified Backspace should delete the word before the cursor."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.insert("hello world")
            await pilot.pause()

            await pilot.press(key)
            await pilot.pause()

            assert ta.text == "hello "
            assert ta.cursor_location == (0, 6)


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
        from deepagents_code.command_registry import CommandEntry

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
        from deepagents_code.command_registry import CommandEntry

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
        """Backspace mode exit clears stale ghost text without a text edit."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await _pause_for_strip(pilot)
            assert chat.mode == "command"
            assert chat._text_area.text == ""

            chat._text_area.argument_hint = "[context]"
            await pilot.press("backspace")
            await pilot.pause()

            assert chat.mode == "normal"
            assert chat._text_area.text == ""
            assert chat._text_area.argument_hint == ""
            assert _render_text_area_line(chat._text_area) == ""

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

    async def test_preserves_widget_focus(self) -> None:
        """Pausing must not blur the widget."""
        app = _ChatInputTestApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None
            chat._text_area.focus()
            await pilot.pause()

            chat.set_cursor_blink(blink=False)
            await pilot.pause()

            assert chat._text_area.has_focus is True


class TestPasteBurstEnterSuppression:
    """Multi-line pastes replayed as key events must not submit mid-stream.

    Terminals without bracketed paste deliver a paste as rapid `Char`/`Enter`
    key events. A short run of fast keystrokes arms a suppression window so the
    embedded `enter` events insert newlines instead of submitting.
    """

    async def test_rapid_burst_with_newline_does_not_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fast keystroke run followed by enter inserts a newline."""
        # Widen the burst gap so wall-clock delays between pilot.press calls on
        # slow CI runners still register as a single rapid burst.
        monkeypatch.setattr(chat_input_module, "_PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            chat_input_module, "_PASTE_ENTER_SUPPRESS_WINDOW_SECONDS", 60.0
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            for char in "hello":
                await pilot.press(char)
            await pilot.press("enter")
            await pilot.press("w")
            await pilot.pause()

            assert len(app.submitted) == 0
            assert "\n" in ta.text

    async def test_slow_typing_then_enter_submits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deliberate typing (no burst) keeps enter as submit."""
        # Force every inter-key gap to exceed the burst threshold.
        monkeypatch.setattr(chat_input_module, "_PASTE_BURST_CHAR_GAP_SECONDS", 0.0)

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            for char in "hello":
                await pilot.press(char)
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "hello"

    async def test_single_line_burst_then_manual_enter_submits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-line paste followed by manual enter still submits."""
        monkeypatch.setattr(chat_input_module, "_PASTE_BURST_CHAR_GAP_SECONDS", 0.03)
        monkeypatch.setattr(
            chat_input_module, "_PASTE_ENTER_SUPPRESS_WINDOW_SECONDS", 0.12
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.text = "abc"
            now = chat_input_module.time.monotonic()
            ta._paste_burst_last_key_time = (
                now - chat_input_module._PASTE_BURST_CHAR_GAP_SECONDS - 0.01
            )
            ta._paste_burst_window_until = (
                now + chat_input_module._PASTE_ENTER_SUPPRESS_WINDOW_SECONDS
            )

            await ta._on_key(events.Key("enter", None))
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == "abc"
            assert "\n" not in ta.text

    async def test_suppressed_enter_rearms_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A suppressed enter extends the window so trailing lines stay grouped."""
        monkeypatch.setattr(chat_input_module, "_PASTE_BURST_CHAR_GAP_SECONDS", 0.03)
        monkeypatch.setattr(
            chat_input_module, "_PASTE_ENTER_SUPPRESS_WINDOW_SECONDS", 0.12
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.text = "abc"
            now = chat_input_module.time.monotonic()
            # Fresh keystroke within the char gap and an open (but nearly
            # closed) window: this enter belongs to a replayed paste.
            ta._paste_burst_last_key_time = now
            original_until = now + 0.01
            ta._paste_burst_window_until = original_until

            await ta._on_key(events.Key("enter", None))
            await pilot.pause()

            assert len(app.submitted) == 0
            assert "\n" in ta.text
            assert ta._paste_burst_window_until is not None
            assert ta._paste_burst_window_until > original_until

    async def test_blank_line_paste_keeps_consecutive_enter_grouped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A delayed second enter in a blank-line paste does not submit."""
        monkeypatch.setattr(chat_input_module, "_PASTE_BURST_CHAR_GAP_SECONDS", 0.03)
        monkeypatch.setattr(
            chat_input_module, "_PASTE_ENTER_SUPPRESS_WINDOW_SECONDS", 60.0
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            ta.text = "abc"
            ta.move_cursor((0, len(ta.text)))
            now = chat_input_module.time.monotonic()
            ta._paste_burst_last_key_time = now
            ta._paste_burst_window_until = now + 60.0

            await ta._on_key(events.Key("enter", None))
            await pilot.pause()
            assert ta.text == "abc\n"

            ta._paste_burst_last_key_time = (
                chat_input_module.time.monotonic()
                - chat_input_module._PASTE_BURST_CHAR_GAP_SECONDS
                - 0.01
            )
            await ta._on_key(events.Key("enter", None))
            await pilot.pause()

            assert len(app.submitted) == 0
            assert ta.text == "abc\n\n"

    async def test_slash_command_enter_still_submits_during_burst(self) -> None:
        """Slash-command context keeps enter dispatching even after a burst."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            ta = chat._text_area
            assert ta is not None

            for char in "/help":
                await pilot.press(char)
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1


class TestPasteCollapseHelpers:
    """Unit tests for the paste_collapse module helpers."""

    def test_should_collapse_short_text(self) -> None:
        """Short single-line text should not be collapsed."""
        from deepagents_code.paste_collapse import should_collapse_paste

        assert should_collapse_paste("hello") is False

    def test_should_collapse_long_text(self) -> None:
        """Text exceeding the character threshold should be collapsed."""
        from deepagents_code.paste_collapse import should_collapse_paste

        assert should_collapse_paste("x" * 801) is True

    def test_should_not_collapse_at_char_boundary(self) -> None:
        """Text exactly at the character threshold should not be collapsed."""
        from deepagents_code.paste_collapse import should_collapse_paste

        assert should_collapse_paste("x" * 800) is False

    def test_should_collapse_multi_line(self) -> None:
        """Text with more lines than the threshold should be collapsed."""
        from deepagents_code.paste_collapse import should_collapse_paste

        assert should_collapse_paste("line1\nline2\nline3\nline4") is True

    def test_should_not_collapse_two_lines(self) -> None:
        """Exactly two newlines (three lines) should not be collapsed by line count."""
        from deepagents_code.paste_collapse import should_collapse_paste

        assert should_collapse_paste("line1\nline2\nline3") is False

    def test_format_paste_ref_single_line(self) -> None:
        """Single-line paste ref has no line count suffix."""
        from deepagents_code.paste_collapse import format_paste_ref

        assert format_paste_ref(1, 0) == "[Pasted text #1]"

    def test_format_paste_ref_multi_line(self) -> None:
        """Multi-line paste ref includes the line count."""
        from deepagents_code.paste_collapse import format_paste_ref

        assert format_paste_ref(3, 5) == "[Pasted text #3 +5 lines]"

    def test_expand_paste_refs(self) -> None:
        """expand_paste_refs replaces placeholders with stored content."""
        from deepagents_code.paste_collapse import PastedContent, expand_paste_refs

        contents = {
            1: PastedContent(content="FIRST\nSECOND"),
            2: PastedContent(content="third"),
        }
        text = "before [Pasted text #1 +1 lines] after [Pasted text #2] end"
        expanded = expand_paste_refs(text, contents)
        assert expanded == "before FIRST\nSECOND after third end"

    def test_expand_paste_refs_out_of_order_and_repeated(self) -> None:
        """Placeholders expand by id regardless of order, and repeats reuse content."""
        from deepagents_code.paste_collapse import PastedContent, expand_paste_refs

        contents = {
            1: PastedContent(content="ONE"),
            2: PastedContent(content="TWO"),
        }
        text = "[Pasted text #2] [Pasted text #1] [Pasted text #1]"
        assert expand_paste_refs(text, contents) == "TWO ONE ONE"

    def test_expand_paste_refs_content_with_backslashes(self) -> None:
        """Stored content is inserted literally, not as a regex replacement."""
        from deepagents_code.paste_collapse import PastedContent, expand_paste_refs

        contents = {1: PastedContent(content=r"\1 \g<0> back\slash")}
        expanded = expand_paste_refs("[Pasted text #1]", contents)
        assert expanded == r"\1 \g<0> back\slash"

    def test_expand_paste_refs_unknown_id_left_as_is(self) -> None:
        """Placeholders with unknown IDs are left unchanged."""
        from deepagents_code.paste_collapse import expand_paste_refs

        text = "[Pasted text #99 +5 lines]"
        assert expand_paste_refs(text, {}) == text

    def test_expand_paste_refs_empty_contents(self) -> None:
        """Expanding with no stored contents returns text unchanged."""
        from deepagents_code.paste_collapse import expand_paste_refs

        text = "hello [Pasted text #1] world"
        assert expand_paste_refs(text, {}) == text


class TestPasteCollapseIntegration:
    """Integration tests for paste collapsing in ChatInput."""

    async def test_large_paste_inserts_placeholder(self) -> None:
        """Pasting text > 800 chars inserts a placeholder, not the full text."""
        big_text = "x" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()

            assert "[Pasted text #1]" in chat._text_area.text
            assert "x" * 900 not in chat._text_area.text
            assert 1 in chat._pasted_contents
            assert chat._pasted_contents[1].content == big_text

    async def test_small_paste_inserts_directly(self) -> None:
        """Pasting text under the threshold is inserted as-is."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste("short text")
            await pilot.pause()

            assert chat._text_area.text == "short text"
            assert len(chat._pasted_contents) == 0

    async def test_multi_line_paste_inserts_placeholder(self) -> None:
        """Pasting text with > 2 newlines inserts a placeholder."""
        multi_line = "\n".join(f"line {i}" for i in range(5))
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(multi_line)
            await pilot.pause()

            assert "[Pasted text #1 +4 lines]" in chat._text_area.text
            assert "line 0" not in chat._text_area.text

    async def test_submit_expands_placeholder(self) -> None:
        """Submitting text with a placeholder sends the full expanded content."""
        big_text = "A" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == big_text

    async def test_submit_clears_paste_contents(self) -> None:
        """Paste contents are cleared after submission."""
        big_text = "B" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()
            assert len(chat._pasted_contents) == 1

            await pilot.press("enter")
            await pilot.pause()

            assert len(chat._pasted_contents) == 0

    async def test_multiple_pastes_get_incrementing_ids(self) -> None:
        """Consecutive large pastes get incrementing placeholder IDs."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste("X" * 900)
            await pilot.pause()
            chat.handle_external_paste("Y" * 900)
            await pilot.pause()

            assert "[Pasted text #1]" in chat._text_area.text
            assert "[Pasted text #2]" in chat._text_area.text
            assert chat._pasted_contents[1].content == "X" * 900
            assert chat._pasted_contents[2].content == "Y" * 900

    async def test_multiple_pastes_submit_expands_all(self) -> None:
        """Submitting with multiple placeholders expands all of them."""
        text_a = "A" * 900
        text_b = "B" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(text_a)
            await pilot.pause()
            chat.handle_external_paste(text_b)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == text_a + text_b

    async def test_text_around_placeholder_preserved_on_submit(self) -> None:
        """Text typed around a placeholder is preserved on submission."""
        big_text = "C" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            for char in "fix this: ":
                await pilot.press(char)
            await pilot.pause()

            chat.handle_external_paste(big_text)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == f"fix this: {big_text}"

    async def test_paste_content_survives_undo_restored_placeholder(self) -> None:
        """A restored placeholder still expands after an undoable edit."""
        big_text = "D" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()
            placeholder = chat._text_area.text
            assert 1 in chat._pasted_contents

            chat._text_area.text = "all gone"
            await pilot.pause()

            chat._text_area.text = f"fix: {placeholder}"
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == f"fix: {big_text}"

    @pytest.mark.parametrize(
        ("pasted", "mode", "visible"),
        [
            ("/help " + ("x" * 900), "command", "help " + ("x" * 900)),
            ("!echo " + ("x" * 900), "shell", "echo " + ("x" * 900)),
            ("!!echo " + ("x" * 900), "shell_incognito", "echo " + ("x" * 900)),
        ],
    )
    async def test_prefixed_large_paste_renders_mode_instead_of_collapsing(
        self, pasted: str, mode: str, visible: str
    ) -> None:
        """Mode-prefixed large pastes stay visible so the prompt shows mode."""
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(pasted)
            await pilot.pause()

            assert chat.mode == mode
            assert chat._text_area.text == visible
            assert chat._pasted_contents == {}

            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == pasted
            assert app.submitted[0].mode == mode

    async def test_identical_second_paste_expands_placeholder(self) -> None:
        """Pasting identical content again expands the placeholder to full text."""
        text = "S" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(text)
            await pilot.pause()
            assert chat._text_area.text == "[Pasted text #1]"

            chat.handle_external_paste(text)
            await pilot.pause()

            # Second identical paste expands the placeholder inline. The stored
            # copy stays available because this edit can be undone.
            assert chat._text_area.text == text
            assert chat._pasted_contents[1].content == text

            await pilot.press("enter")
            await pilot.pause()
            assert app.submitted[0].value == text

    async def test_repeat_paste_expansion_keeps_content_for_undo(self) -> None:
        """Undo-restored placeholders still expand after repeat-paste expansion."""
        text = "U" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(text)
            await pilot.pause()
            placeholder = chat._text_area.text

            chat.handle_external_paste(text)
            await pilot.pause()
            assert chat._text_area.text == text

            chat._text_area.text = placeholder
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.submitted) == 1
            assert app.submitted[0].value == text

    async def test_identical_second_paste_expands_preserving_surrounding_text(
        self,
    ) -> None:
        """Expanding a repeated paste keeps text typed around the placeholder."""
        text = "S" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            for char in "fix: ":
                await pilot.press(char)
            chat.handle_external_paste(text)
            await pilot.pause()
            assert chat._text_area.text == "fix: [Pasted text #1]"

            chat.handle_external_paste(text)
            await pilot.pause()

            assert chat._text_area.text == f"fix: {text}"
            assert chat._pasted_contents[1].content == text

    async def test_repeat_paste_skips_stale_deleted_placeholder_ids(self) -> None:
        """Repeat expansion targets a visible placeholder, not stale paste data."""
        text = "R" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(text)
            await pilot.pause()
            assert chat._text_area.text == "[Pasted text #1]"

            await pilot.press("backspace")
            await pilot.pause()
            assert chat._text_area.text == ""
            assert chat._pasted_contents[1].content == text

            chat.handle_external_paste(text)
            await pilot.pause()
            assert chat._text_area.text == "[Pasted text #2]"

            chat.handle_external_paste(text)
            await pilot.pause()

            assert chat._text_area.text == text
            assert chat._pasted_contents[1].content == text
            assert chat._pasted_contents[2].content == text

    async def test_bracketed_paste_event_collapses(self) -> None:
        """A real Paste event over the threshold collapses to a placeholder.

        Exercises the production path (`_on_paste` -> `PastedText` message ->
        `on_chat_text_area_pasted_text`) rather than `handle_external_paste`.
        """
        big_text = "z" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            await chat._text_area._on_paste(events.Paste(big_text))
            await pilot.pause()

            assert "[Pasted text #1]" in chat._text_area.text
            assert big_text not in chat._text_area.text
            assert chat._pasted_contents[1].content == big_text

    async def test_paste_burst_flush_collapses_large_payload(self) -> None:
        """A large buffered paste burst collapses to a placeholder on flush."""
        big_text = "q" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area._paste_burst_buffer = big_text
            await chat._text_area._flush_paste_burst()
            await pilot.pause()

            assert "[Pasted text #1]" in chat._text_area.text
            assert big_text not in chat._text_area.text
            assert chat._pasted_contents[1].content == big_text

    async def test_backspace_removes_full_paste_placeholder(self) -> None:
        """Backspace deletes a [Pasted text #N] placeholder as a single token."""
        big_text = "p" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()
            assert "[Pasted text #1]" in chat._text_area.text

            await pilot.press("backspace")
            await pilot.pause()

            assert chat._text_area.text == ""
            # Backing content is preserved so an undo can restore the token;
            # it is cleared only at submit.
            assert 1 in chat._pasted_contents

    async def test_forward_delete_removes_full_paste_placeholder(self) -> None:
        """Forward delete removes a paste placeholder as a single token."""
        big_text = "p" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()
            chat._text_area.move_cursor((0, 0))

            await pilot.press("delete")
            await pilot.pause()

            assert chat._text_area.text == ""
            # Like backspace, forward delete leaves the backing content for undo;
            # it is cleared only at submit.
            assert 1 in chat._pasted_contents

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

    async def test_identical_multiline_repaste_expands_placeholder(self) -> None:
        """Repeat-pasting multi-line content expands its `+M lines` placeholder."""
        multi_line = "\n".join(f"line {i}" for i in range(5))
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(multi_line)
            await pilot.pause()
            assert chat._text_area.text == "[Pasted text #1 +4 lines]"

            chat.handle_external_paste(multi_line)
            await pilot.pause()

            assert chat._text_area.text == multi_line
            assert chat._pasted_contents[1].content == multi_line

    async def test_copy_button_expands_placeholders(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The copy button copies expanded text, not the placeholder."""
        big_text = "E" * 900
        copied: list[str] = []

        def capture_copy(_app_arg: object, text: str, **_kwargs: object) -> None:
            copied.append(text)

        # _copy_via_button imports copy_text_with_feedback at call time, so
        # patching the module attribute is picked up without a manual restore.
        monkeypatch.setattr(
            "deepagents_code.clipboard.copy_text_with_feedback", capture_copy
        )

        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()

            chat._copy_via_button()
            await pilot.pause()

        assert len(copied) == 1
        assert copied[0] == big_text

    async def test_paste_content_survives_undoable_clear(self) -> None:
        """Undoable clear (discard_text) must not delete paste contents.

        After clearing and undoing, the restored placeholder must still
        have its backing content so submission expands correctly.
        """
        big_text = "F" * 900
        app = _RecordingApp()
        async with app.run_test() as pilot:
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat.handle_external_paste(big_text)
            await pilot.pause()
            assert 1 in chat._pasted_contents

            # Undoable clear empties the text area
            chat.discard_text()
            await pilot.pause()

            # Paste contents must survive so undo can restore them
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
