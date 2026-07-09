"""Tests for the persistent `TodoPanel` widget.

Each test mounts the panel in a minimal App, feeds it authoritative todo state,
and asserts on observable state / rendered content. Uses the Textual
`run_test()` pilot harness. Validation is also exercised directly against the
pure `_validate` helper, since it must fail closed on untrusted payloads.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.tui.widgets.todo_panel import TodoPanel


class PanelApp(App):
    """Minimal app that mounts a TodoPanel for testing."""

    def compose(self) -> ComposeResult:
        yield TodoPanel(id="panel")


def _render(widget: Static) -> str:
    content = widget.render()
    plain = getattr(content, "plain", None)
    return plain if isinstance(plain, str) else str(content)


class TestLifecycle:
    async def test_hidden_until_first_todos(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", TodoPanel)
            assert not panel.has_class("-visible")

    async def test_visible_and_rendered_after_todos_set(self) -> None:
        async with PanelApp().run_test(size=(120, 24)) as pilot:
            panel = pilot.app.query_one("#panel", TodoPanel)
            panel.set_todos(
                [
                    {"content": "Explore the code", "status": "completed"},
                    {"content": "Implement the panel", "status": "in_progress"},
                    {"content": "Write tests", "status": "pending"},
                ]
            )
            await pilot.pause()
            assert panel.has_class("-visible")
            assert panel._counts() == (1, 1, 1)  # active, pending, completed
            body = _render(pilot.app.query_one("#todo-panel-body", Static))
            assert "Implement the panel" in body
            assert "1 active" in body
            assert "1 pending" in body
            assert "1 done" in body

    async def test_empty_todos_hide_panel(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", TodoPanel)
            panel.set_todos([{"content": "x", "status": "pending"}])
            await pilot.pause()
            assert panel.has_class("-visible")
            panel.set_todos([])
            await pilot.pause()
            assert not panel.has_class("-visible")

    async def test_reset_clears_and_hides(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", TodoPanel)
            panel.set_todos([{"content": "x", "status": "in_progress"}])
            await pilot.pause()
            assert panel.has_class("-visible")
            panel.reset()
            await pilot.pause()
            assert not panel.has_class("-visible")
            assert panel._todos == []


class TestValidation:
    def test_rejects_non_list(self) -> None:
        assert TodoPanel._validate("not-a-list") == []
        assert TodoPanel._validate({"todos": []}) == []

    def test_rejects_unknown_status(self) -> None:
        assert TodoPanel._validate([{"content": "x", "status": "bogus"}]) == []

    def test_rejects_non_dict_item(self) -> None:
        assert (
            TodoPanel._validate([{"content": "x", "status": "pending"}, "nope"]) == []
        )

    def test_rejects_missing_or_blank_content(self) -> None:
        assert TodoPanel._validate([{"status": "pending"}]) == []
        assert TodoPanel._validate([{"content": "   ", "status": "pending"}]) == []

    def test_accepts_well_formed_todos(self) -> None:
        records = TodoPanel._validate(
            [
                {"content": "First", "status": "completed"},
                {"content": "Second", "status": "pending"},
            ]
        )
        assert [(r.content, r.status) for r in records] == [
            ("First", "completed"),
            ("Second", "pending"),
        ]

    def test_sanitizes_control_characters(self) -> None:
        records = TodoPanel._validate(
            [{"content": "clean\x00\x1btext", "status": "pending"}]
        )
        assert len(records) == 1
        assert "\x00" not in records[0].content
        assert "\x1b" not in records[0].content
