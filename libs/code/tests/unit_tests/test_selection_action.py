"""Tests for asking about selected conversation text."""

from textual.containers import Container
from textual.widgets import Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.widgets.selection_action import SelectionAction


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
        assert area.text == "Explain: " + selected + "\n\n"
        assert area.has_focus
        assert not app.query(SelectionAction)
        assert not app.screen.selections


async def test_selection_action_dismisses_without_selection() -> None:
    """Clicking elsewhere removes the shortcut instead of retaining stale text."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await app.screen.mount(SelectionAction("old selection"))
        app.screen.clear_selection()
        await app._show_selection_action(app.screen, 0, 0)
        await pilot.pause()
        assert not app.query(SelectionAction)
