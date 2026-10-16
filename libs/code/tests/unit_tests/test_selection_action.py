"""Tests for asking about selected conversation text."""

from pathlib import Path

import pytest
from textual.containers import Container
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Input, Markdown, Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.widgets.messages import (
    AppMessage,
    AssistantMessage,
    UserMessage,
)
from deepagents_code.tui.widgets.selection_action import (
    SelectionAction,
    is_transcript_selection,
)


async def test_selection_action_preserves_draft_and_focus(tmp_path: Path) -> None:
    """A real drag and click inserts the selection without sending the draft."""
    (tmp_path / "config.toml").write_text("[ui]\nask_about_selection = true\n")
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        area.load_text("Explain: ")
        area.move_cursor((0, 9))
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        await pilot.mouse_down(message, offset=(0, 0))
        await pilot.hover(message, offset=(8, 0))
        await pilot.mouse_up(message, offset=(8, 0))
        await pilot.pause()
        action = app.query_one(SelectionAction)
        selected = action.selected_text
        assert selected.strip()
        await pilot.click(action)
        await pilot.pause()
        assert area.text == "Explain: \n\n> " + selected + "\n\n"
        assert area.cursor_location == (4, 0)
        assert area.has_focus
        # The terminal cursor must follow the chat input after it grows. The
        # headless driver settles it even without
        # `ChatInput._settle_cursor_after_insert`, so this pins the outcome but
        # does not guard that helper; a real terminal is the check for it.
        assert app.cursor_position == area.cursor_screen_offset
        assert not app.query(SelectionAction)
        assert not app.screen.selections


async def test_selection_action_dismisses_without_selection() -> None:
    """Clicking elsewhere removes the shortcut instead of retaining stale text."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        await _drag_select(pilot, message)
        assert app.query(SelectionAction)
        await pilot.click(message, offset=(2, 0))
        await pilot.pause()
        assert not app.query(SelectionAction)


@pytest.mark.parametrize("key", ["a", "enter", "escape", "left"])
async def test_selection_action_dismissed_by_keypress(key: str) -> None:
    """Typing in the focused composer removes the shortcut."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        await _drag_select(pilot, message)
        assert app.query(SelectionAction)
        await pilot.press(key)
        await pilot.pause()
        assert not app.query(SelectionAction)


@pytest.mark.parametrize(
    ("config", "enabled"),
    [
        ("", True),
        ("[ui]\nask_about_selection = true\n", True),
        ("[ui]\nask_about_selection = false\n", False),
        ('[ui]\nask_about_selection = "invalid"\n', True),
    ],
)
async def test_selection_action_config(
    tmp_path: Path, config: str, *, enabled: bool
) -> None:
    """The action defaults on and honors an explicit config opt-out."""
    (tmp_path / "config.toml").write_text(config)
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        await pilot.mouse_down(message, offset=(0, 0))
        await pilot.hover(message, offset=(8, 0))
        await pilot.mouse_up(message, offset=(8, 0))
        await pilot.pause()
        assert app.screen.get_selected_text()
        assert bool(app.query(SelectionAction)) is enabled


@pytest.mark.parametrize(
    "pending",
    [
        "_pending_approval_widget",
        "_pending_ask_user_widget",
        "_pending_goal_review_widget",
    ],
)
async def test_selection_action_preserves_pending_prompt(
    monkeypatch: pytest.MonkeyPatch, pending: str
) -> None:
    """A pending prompt hides the action and keeps focus and the draft."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        area.load_text("existing draft")
        prompt = Input()
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message, prompt)
        monkeypatch.setattr(app, pending, prompt)
        prompt.focus()
        await pilot.pause()
        await _drag_select(pilot, message)
        assert app.screen.get_selected_text()
        assert not app.query(SelectionAction)
        assert prompt.has_focus
        await pilot.press("y")
        await pilot.pause()
        assert prompt.value == "y"
        assert area.text == "existing draft"


async def test_selection_action_warns_when_prompt_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt that appears after the action shows leaves the draft unchanged."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        area.load_text("existing draft")
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        await _drag_select(pilot, message)
        action = app.query_one(SelectionAction)
        monkeypatch.setattr(app, "_pending_approval_widget", Input())
        notices: list[str] = []
        monkeypatch.setattr(app, "notify", lambda msg, **_: notices.append(msg))
        await pilot.click(action)
        await pilot.pause()
        assert area.text == "existing draft"
        assert notices == ["Answer the pending prompt before adding the selection"]


async def test_selection_action_dismissed_by_scroll_and_resize() -> None:
    """The shortcut leaves when the transcript or terminal moves under it."""
    app = DeepAgentsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        messages = app.query_one("#messages", Container)
        message = UserMessage("selected text")
        await messages.mount(message)
        await messages.mount_all(UserMessage(f"filler {i}") for i in range(40))
        await pilot.pause()
        app.query_one("#chat").scroll_home(animate=False)
        await pilot.pause()
        await _drag_select(pilot, message)
        assert app.query(SelectionAction)
        app.query_one("#chat").scroll_down(animate=False)
        await pilot.pause()
        assert not app.query(SelectionAction)

        await _drag_select(pilot, message)
        assert app.query(SelectionAction)
        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert not app.query(SelectionAction)


async def test_selection_action_warns_when_insert_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A composer that rejects the insert reports it and keeps the selection."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        await _drag_select(pilot, message)
        monkeypatch.setattr(app._chat_input, "insert_at_cursor", lambda _text: False)
        notices: list[str] = []
        monkeypatch.setattr(app, "notify", lambda msg, **_: notices.append(msg))
        app.post_message(SelectionAction.Ask("selected text"))
        await pilot.pause()
        assert notices == ["Could not add the selection: the composer is unavailable"]
        assert app.screen.selections


async def test_multiline_quote_keeps_cursor_visible() -> None:
    """A long quote preserves line breaks and reveals the following blank line."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        text = "\n".join(["quoted line"] * 30)
        action = SelectionAction(text)
        action.styles.offset = (0, 0)
        await app.screen.mount(action)
        await pilot.pause()
        await pilot.click(action)
        await pilot.pause()
        assert area.text == "\n".join(["> quoted line"] * 30) + "\n\n"
        assert area.cursor_location == (31, 0)
        assert area.has_focus
        # Like the cursor check above, headless runs pass this without the
        # settle helper, so it pins the outcome rather than guarding the fix.
        assert area.scroll_y > 0


async def _drag_select(pilot: Pilot[None], widget: Widget) -> None:
    await pilot.mouse_down(widget, offset=(0, 0))
    await pilot.hover(widget, offset=(8, 0))
    await pilot.mouse_up(widget, offset=(8, 0))
    await pilot.pause()


@pytest.mark.parametrize(
    ("widget", "shown"),
    [
        (Static("tool output text", id="output-full"), True),
        (Static("tool args text", classes="tool-args"), True),
        (Static("Waiting for typing to finish..."), False),
        (AppMessage("Checking for updates..."), False),
    ],
)
async def test_selection_action_only_for_transcript(
    widget: Static, *, shown: bool
) -> None:
    """Message and tool content get the action; conversation chrome does not."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await app.query_one("#messages", Container).mount(widget)
        await pilot.pause()
        await _drag_select(pilot, widget)
        assert app.screen.get_selected_text()
        assert bool(app.query(SelectionAction)) is shown


async def test_selection_action_for_assistant_markdown() -> None:
    """Rendered assistant Markdown blocks count as transcript text."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        message = AssistantMessage("selected assistant text")
        await app.query_one("#messages", Container).mount(message)
        await message.write_initial_content()
        await pilot.pause()
        block = message.query_one(Markdown).children[0]
        await _drag_select(pilot, block)
        assert app.screen.get_selected_text()
        assert app.query(SelectionAction)


async def test_selection_spanning_chrome_is_not_transcript() -> None:
    """One chrome widget in a mixed selection rejects the whole selection."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        message = UserMessage("selected text")
        status = Static("Running...", classes="tool-status")
        await app.query_one("#messages", Container).mount(message, status)
        await pilot.pause()
        assert is_transcript_selection([message])
        assert not is_transcript_selection([message, status])
        assert not is_transcript_selection([app.query_one("#welcome-banner")])


async def test_messages_outside_the_transcript_are_not_transcript() -> None:
    """A message widget outside `#messages`, such as in a modal, is rejected."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        inside = UserMessage("in the transcript")
        outside = UserMessage("in a side panel")
        await app.query_one("#messages", Container).mount(inside)
        await app.screen.mount(outside)
        await pilot.pause()
        assert is_transcript_selection([inside])
        assert not is_transcript_selection([outside])


async def test_selection_action_draws_above_autocomplete() -> None:
    """The action's layer exists on the screen and stacks above autocomplete."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        await _drag_select(pilot, message)
        layers = app.screen.layers
        layer = app.query_one(SelectionAction).styles.layer
        assert layers.index(layer) > layers.index("autocomplete")
