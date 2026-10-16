"""Tests for asking about selected conversation text."""

from pathlib import Path

import pytest
from textual.containers import Container
from textual.pilot import Pilot
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Input, Markdown, Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.widgets.messages import (
    AppMessage,
    AssistantMessage,
    DiffMessage,
    UserMessage,
)
from deepagents_code.tui.widgets.selection_action import (
    SELECTION_ACTION_WIDTH,
    SelectionAction,
    is_transcript_selection,
)


async def _drag_select(pilot: Pilot[None], widget: Widget) -> None:
    await pilot.mouse_down(widget, offset=(0, 0))
    await pilot.hover(widget, offset=(8, 0))
    await pilot.mouse_up(widget, offset=(8, 0))
    await pilot.pause()


async def test_selection_action_preserves_draft_and_focus() -> None:
    """A real drag and click inserts the selection without sending the draft."""
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
        await _drag_select(pilot, message)
        action = app.query_one(SelectionAction)
        selected = action.selected_text
        assert selected.strip()
        await pilot.click(action)
        await pilot.pause()
        assert area.text == "Explain: \n\n> " + selected + "\n\n"
        assert area.cursor_location == (4, 0)
        assert area.has_focus
        # The terminal cursor must follow the chat input after it grows, which
        # `ChatInput._settle_cursor_after_insert` handles.
        assert app.cursor_position == area.cursor_screen_offset
        assert not app.query(SelectionAction)
        assert not app.screen.selections


async def test_selection_action_dismisses_without_selection() -> None:
    """Clicking elsewhere removes the action instead of retaining stale text."""
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
    """Any key, including ones the focused chat input consumes, removes it."""
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
        await _drag_select(pilot, message)
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


@pytest.mark.parametrize(
    ("draft", "cursor", "expected"),
    [
        ("please fix this", (0, 3), "please fix this\n\n> q\n\n"),
        ("please fix this", (0, 0), "please fix this\n\n> q\n\n"),
        ("> a\n\n", (0, 0), "> a\n\n> q\n\n"),
        ("note\n", (1, 0), "note\n\n> q\n\n"),
    ],
)
async def test_selection_action_appends_after_draft(
    draft: str, cursor: tuple[int, int], expected: str
) -> None:
    """The quote goes after the draft with one blank line, wherever the cursor is."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        area.load_text(draft)
        area.move_cursor(cursor)
        app.post_message(SelectionAction.Ask("q"))
        await pilot.pause()
        assert area.text == expected
        assert area.cursor_location == area.document.end


async def test_selection_action_closes_prompt_search() -> None:
    """An open Ctrl+R search closes so the quote lands in the visible draft."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        area.load_text("draft")
        assert app._chat_input.open_prompt_search() == "inline"
        await pilot.pause()
        app.post_message(SelectionAction.Ask("q"))
        await pilot.pause()
        assert not app._chat_input._prompt_search_active
        assert area.text == "draft\n\n> q\n\n"
        assert area.has_focus


@pytest.mark.parametrize("mode", ["shell", "shell_incognito", "command"])
async def test_selection_action_leaves_empty_special_mode(mode: str) -> None:
    """An empty shell or command draft switches to normal before the quote."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        app._chat_input.mode = mode
        app.post_message(SelectionAction.Ask("selected text"))
        await pilot.pause()
        assert app._chat_input.mode == "normal"
        assert area.text == "> selected text\n\n"


@pytest.mark.parametrize("mode", ["shell", "command"])
async def test_selection_action_keeps_special_mode_draft(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A typed shell or command draft is never turned into a quote."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        app._chat_input.mode = mode
        area.load_text("ls")
        notices: list[str] = []
        monkeypatch.setattr(app, "notify", lambda msg, **_: notices.append(msg))
        app.post_message(SelectionAction.Ask("selected text"))
        await pilot.pause()
        assert app._chat_input.mode == mode
        assert area.text == "ls"
        assert notices == [
            "Clear the shell or command input before adding the selection"
        ]


async def test_selection_action_follows_scroll_until_out_of_view() -> None:
    """The action moves with its selection and leaves once it scrolls away."""
    app = DeepAgentsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        messages = app.query_one("#messages", Container)
        chat = app.query_one("#chat")
        message = UserMessage("selected text")
        await messages.mount_all(UserMessage(f"filler {i}") for i in range(20))
        await messages.mount(message)
        await messages.mount_all(UserMessage(f"filler {i}") for i in range(20))
        await pilot.pause()
        chat.scroll_to_center(message, animate=False)
        await pilot.pause()
        await _drag_select(pilot, message)
        action = app.query_one(SelectionAction)
        gap = message.region.y - action.region.y
        chat.scroll_relative(y=2, animate=False)
        await pilot.pause()
        assert app.query_one(SelectionAction) is action
        assert message.region.y - action.region.y == gap
        chat.scroll_end(animate=False)
        await pilot.pause()
        assert not app.query(SelectionAction)


async def test_selection_action_dismissed_by_resize() -> None:
    """The action leaves when the terminal resizes under it."""
    app = DeepAgentsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()

        await _drag_select(pilot, message)
        assert app.query(SelectionAction)
        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert not app.query(SelectionAction)


async def test_selection_action_warns_when_insert_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chat input that rejects the insert reports it."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        monkeypatch.setattr(app._chat_input, "insert_at_end", lambda _text: False)
        notices: list[str] = []
        monkeypatch.setattr(app, "notify", lambda msg, **_: notices.append(msg))
        app.post_message(SelectionAction.Ask("selected text"))
        await pilot.pause()
        assert notices == [
            "Could not add the selection to the chat input. Copy and paste it instead."
        ]


async def test_selection_action_warns_when_modal_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A modal that opens before the click lands leaves the draft unchanged."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        area.load_text("existing draft")
        await app.push_screen(ModalScreen())
        notices: list[str] = []
        monkeypatch.setattr(app, "notify", lambda msg, **_: notices.append(msg))
        app.post_message(SelectionAction.Ask("selected text"))
        await pilot.pause()
        assert area.text == "existing draft"
        assert notices == ["Close the open dialog before adding the selection"]


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
        # Unlike the terminal cursor check in
        # `test_selection_action_preserves_draft_and_focus`, headless runs pass
        # this without the settle helper, so it pins the outcome only.
        assert area.scroll_y > 0


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
        assert not is_transcript_selection([])
        assert not is_transcript_selection([message, status])
        assert not is_transcript_selection([app.query_one("#welcome-banner")])


async def test_messages_outside_the_transcript_are_not_transcript() -> None:
    """A message widget mounted outside `#messages` is rejected."""
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


async def test_selection_action_stays_on_screen_at_right_edge() -> None:
    """A selection ending at the right edge keeps the whole label visible."""
    app = DeepAgentsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        message = UserMessage("x" * 70)
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        await pilot.mouse_down(message, offset=(0, 0))
        await pilot.hover(message, offset=(message.size.width - 1, 0))
        await pilot.mouse_up(message, offset=(message.size.width - 1, 0))
        await pilot.pause()
        action = app.query_one(SelectionAction)
        assert action.region.right <= app.screen.size.width
        assert action.region.width == SELECTION_ACTION_WIDTH


async def test_whitespace_selection_shows_no_action() -> None:
    """Selecting only blank space offers nothing to ask about."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        message = UserMessage("        ")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        await _drag_select(pilot, message)
        assert app.screen.get_selected_text()
        assert not app.query(SelectionAction)


async def test_diff_body_is_transcript_but_header_is_not() -> None:
    """Diff lines count as transcript text; the diff header is chrome."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        diff = DiffMessage("@@ -1 +1 @@\n-old\n+new\n", "file.py")
        await app.query_one("#messages", Container).mount(diff)
        await pilot.pause()
        header = diff.query_one(".diff-header")
        body = [w for w in diff.query(Static) if not w.has_class("diff-header")]
        assert body
        assert is_transcript_selection(body)
        assert not is_transcript_selection([header])


async def test_selection_action_skips_a_replaced_screen() -> None:
    """A screen switch before the deferred show leaves no action anywhere."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message)
        await pilot.pause()
        base = app.screen
        await pilot.mouse_down(message, offset=(0, 0))
        await pilot.hover(message, offset=(8, 0))
        await app.push_screen(Screen())
        await app._show_selection_action(base, 0, 0)
        await pilot.pause()
        assert not base.query(SelectionAction)
        assert not app.screen.query(SelectionAction)


async def test_selection_action_returns_focus_to_composer() -> None:
    """Clicking the action moves focus back to the chat input."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        other = Input()
        message = UserMessage("selected text")
        await app.query_one("#messages", Container).mount(message, other)
        other.focus()
        await pilot.pause()
        await _drag_select(pilot, message)
        await pilot.click(app.query_one(SelectionAction))
        await pilot.pause()
        assert area.has_focus
