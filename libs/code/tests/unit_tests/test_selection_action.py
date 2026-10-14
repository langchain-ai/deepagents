"""Tests for asking about selected conversation text."""

from pathlib import Path

import pytest
from textual.containers import Container
from textual.widgets import Input, Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.widgets.selection_action import SelectionAction


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
        message = Static("selected text", id="selection-source")
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
        assert not app.query(SelectionAction)
        assert not app.screen.selections


async def test_selection_action_dismisses_without_selection(tmp_path: Path) -> None:
    """Clicking elsewhere removes the shortcut instead of retaining stale text."""
    (tmp_path / "config.toml").write_text("[ui]\nask_about_selection = true\n")
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await app.screen.mount(SelectionAction("old selection"))
        app.screen.clear_selection()
        await app._show_selection_action(app.screen, 0, 0)
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
        message = Static("selected text")
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
    """Clicking the action leaves a pending prompt focused and the draft unchanged."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        assert app._chat_input is not None
        area = app._chat_input._text_area
        assert area is not None
        area.load_text("existing draft")
        prompt = Input()
        message = Static("selected text")
        await app.query_one("#messages", Container).mount(message, prompt)
        monkeypatch.setattr(app, pending, prompt)
        prompt.focus()
        await pilot.pause()
        await pilot.mouse_down(message, offset=(0, 0))
        await pilot.hover(message, offset=(8, 0))
        await pilot.mouse_up(message, offset=(8, 0))
        await pilot.pause()
        await pilot.click(app.query_one(SelectionAction))
        await pilot.pause()
        assert prompt.has_focus
        assert area.text == "existing draft"
        await pilot.press("y")
        await pilot.pause()
        assert prompt.value == "y"
        assert area.text == "existing draft"


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
        assert area.scroll_y > 0
