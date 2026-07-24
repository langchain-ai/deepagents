"""Persistent inline display for the current goal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.content import Content
from textual.widgets import Static

if TYPE_CHECKING:
    from deepagents_code.resume_state import GoalStatus


class GoalStatusPanel(Static):
    """Keep the current goal and lifecycle state visible above the input."""

    def __init__(self, *, id: str | None = None) -> None:  # noqa: A002
        """Initialize an empty hidden goal panel."""
        super().__init__("", id=id, classes="goal-status-panel")
        self.display = False

    def set_goal(
        self,
        objective: str | None,
        status: GoalStatus | None,
        note: str | None,
    ) -> None:
        """Render the current goal or hide the panel when no goal exists.

        Args:
            objective: Persisted goal objective, if set.
            status: Current lifecycle state.
            note: Blocker or completion note associated with the state.
        """
        if not objective:
            self.update("")
            self.display = False
            return

        current = status or "active"
        label = "completed" if current == "complete" else current
        content = Content.from_markup(
            "[bold]Goal · $status[/bold]\n$objective",
            status=label,
            objective=objective,
        )
        if note and current in {"blocked", "complete"}:
            content += Content.from_markup("\n[dim]$note[/dim]", note=note)
        self.update(content)
        self.display = True
