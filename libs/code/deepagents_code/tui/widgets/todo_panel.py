"""Persistent todo checklist docked above the chat input.

The agent's `write_todos` tool writes the authoritative todo list into graph
state. This widget renders that list in a compact, always-visible panel just
above the input (like Claude Code and other CLIs) so the current plan stays on
screen instead of scrolling away inside the transcript.

Trust note: todo `content` strings are LLM-authored and therefore untrusted.
Every rendered string is routed through `sanitize_control_chars` (stripping
control/escape/bidi characters) and rendered only via `Content.styled` /
`markup=False` `Static` updates, so embedded markup and terminal escapes cannot
influence rendering or panel state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from textual.containers import Vertical
from textual.content import Content
from textual.css.query import NoMatches, TooManyMatches
from textual.widgets import Static

from deepagents_code.config import get_glyphs
from deepagents_code.theme import get_theme_colors
from deepagents_code.unicode_security import sanitize_control_chars

if TYPE_CHECKING:
    from textual.app import ComposeResult

logger = logging.getLogger(__name__)

TodoStatus = Literal["pending", "in_progress", "completed"]

_VALID_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "completed"})
_MAX_CONTENT_CHARS = 200


@dataclass
class _TodoRecord:
    """One validated, display-ready todo item."""

    content: str
    """Sanitized todo text."""

    status: TodoStatus
    """Lifecycle state; one of pending, in_progress, completed."""


class TodoPanel(Vertical):
    """Docked checklist showing the agent's current todos above the input.

    Hidden until the first non-empty todo list arrives. Fed by authoritative
    graph state (the `todos` channel), not by tool-call arguments, so it always
    reflects the committed plan and survives message compaction.
    """

    can_focus = False
    can_focus_children = False

    DEFAULT_CSS = """
    TodoPanel {
        height: auto;
        background: $surface;
        border-top: solid $primary;
        display: none;
        padding: 0 2;
    }

    TodoPanel.-visible {
        display: block;
    }

    TodoPanel #todo-panel-body {
        width: 1fr;
        height: auto;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize an empty, hidden panel."""
        super().__init__(**kwargs)
        self._todos: list[_TodoRecord] = []
        self._last_render: str | None = None

    def compose(self) -> ComposeResult:  # noqa: PLR6301 — Textual widget method
        """Yield the single body Static that renders the checklist."""
        yield Static("", id="todo-panel-body", markup=False)

    def set_todos(self, todos: object) -> None:
        """Replace the displayed todos from authoritative graph state.

        Args:
            todos: Raw `todos` value from the graph state update. Anything that
                is not a list of well-formed `{content, status}` dicts is
                ignored defensively, since the payload is model-authored.
        """
        records = self._validate(todos)
        self._todos = records
        if not records:
            self._hide()
            return
        self._show()
        self._refresh()

    def reset(self) -> None:
        """Clear all todos and hide the panel (e.g. on `/clear`)."""
        self._todos = []
        self._last_render = None
        self._hide()

    @staticmethod
    def _validate(todos: object) -> list[_TodoRecord]:
        """Coerce an untrusted `todos` payload into display-ready records.

        Returns:
            Validated records, or an empty list when the payload is malformed.
        """
        if not isinstance(todos, list):
            return []
        records: list[_TodoRecord] = []
        for item in todos:
            if not isinstance(item, dict):
                return []
            content = item.get("content")
            status = item.get("status")
            if not isinstance(content, str) or status not in _VALID_STATUSES:
                return []
            text = sanitize_control_chars(
                content, keep_newlines=False, max_length=_MAX_CONTENT_CHARS
            ).strip()
            if not text:
                return []
            # `status` is a plain object narrowed only by the runtime
            # `_VALID_STATUSES` membership check above; cast to the Literal.
            records.append(_TodoRecord(content=text, status=cast("TodoStatus", status)))
        return records

    def _counts(self) -> tuple[int, int, int]:
        """Return `(active, pending, completed)` counts for the current todos.

        Returns:
            A `(active, pending, completed)` tuple over the current todos.
        """
        active = sum(1 for t in self._todos if t.status == "in_progress")
        completed = sum(1 for t in self._todos if t.status == "completed")
        pending = len(self._todos) - active - completed
        return active, pending, completed

    def _show(self) -> None:
        """Make the panel visible (idempotent)."""
        self.add_class("-visible")

    def _hide(self) -> None:
        """Hide the panel (idempotent)."""
        self.remove_class("-visible")

    def _header(self) -> Content:
        """Build the header line: title plus active/pending/done counts.

        Returns:
            Styled `Content` for the header row.
        """
        colors = get_theme_colors(self)
        active, pending, completed = self._counts()
        parts: list[Content] = [Content.styled("Todos", "bold")]
        stats: list[Content] = []
        if active:
            stats.append(Content.styled(f"{active} active", colors.warning))
        if pending:
            stats.append(Content.styled(f"{pending} pending", "dim"))
        if completed:
            stats.append(Content.styled(f"{completed} done", colors.success))
        if stats:
            parts.extend(
                (
                    Content.styled("   ", "dim"),
                    Content.styled(" | ", "dim").join(stats),
                )
            )
        return Content.assemble(*parts)

    def _line(self, todo: _TodoRecord) -> Content:
        """Render a single todo row with a status glyph.

        Returns:
            Styled `Content` for one checklist row.
        """
        colors = get_theme_colors(self)
        glyphs = get_glyphs()
        if todo.status == "completed":
            prefix = Content.styled(f"  {glyphs.checkmark} ", colors.success)
            return Content.assemble(prefix, Content.styled(todo.content, "dim"))
        if todo.status == "in_progress":
            prefix = Content.styled(f"  {glyphs.circle_filled} ", colors.warning)
            return Content.assemble(prefix, Content(todo.content))
        prefix = Content.styled(f"  {glyphs.circle_empty} ", "dim")
        return Content.assemble(prefix, Content(todo.content))

    def _refresh(self) -> None:
        """Re-render the checklist, skipping unchanged content to avoid flicker."""
        lines: list[Content] = [self._header()]
        lines.extend(self._line(todo) for todo in self._todos)
        content = Content("\n").join(lines)
        if self._last_render == content.plain:
            return
        try:
            self.query_one("#todo-panel-body", Static).update(content)
        except (NoMatches, TooManyMatches):  # not mounted yet
            return
        self._last_render = content.plain
