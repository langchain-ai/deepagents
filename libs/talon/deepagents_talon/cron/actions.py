"""Durable tool intents for objective runs with interrupted destination writes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@dataclass
class ActionRun:
    """Bind tool receipts and successful destination reads to one objective check."""

    path: Path
    job: str
    run: str
    current: Callable[[], bool] | None = field(default=None, kw_only=True)
    reads: set[str] = field(default_factory=set)
    active: set[str] = field(default_factory=set)
    applied: set[tuple[str, str]] = field(default_factory=set)

    @staticmethod
    def arguments(args: dict[str, object]) -> str:
        """Bound historical arguments and normalize key order across retries."""
        return json.dumps(args, default=str, sort_keys=True)[:4000]

    def unresolved(self) -> list[dict[str, str]]:
        """Return interrupted calls, including the arguments needed for destination lookup."""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT run, call_id, tool, arguments FROM objective_actions WHERE job=?",
                (self.job,),
            )
            return [dict(row) for row in rows if row["call_id"] not in self.active]

    @contextmanager
    def dispatch(self, call_id: str, name: str, args: dict[str, object]) -> Iterator[None]:
        """Commit intent before dispatch. Leave an uncertain call durable on interruption."""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT INTO objective_actions VALUES(?,?,?,?,?)",
                (self.job, self.run, call_id, name, self.arguments(args)),
            )
        self.active.add(call_id)
        try:
            yield
        finally:
            self.active.discard(call_id)

    def confirm(self, call_id: str, tool_name: str) -> None:
        """Drop an intent after its tool returned successfully."""
        with closing(sqlite3.connect(self.path)) as db, db:
            note = f"Tool receipt: {tool_name} ({call_id}), status success."
            db.execute(
                "UPDATE checks SET work=substr(work || ?,-4000) WHERE run=?",
                ("\n" + note, self.run),
            )
            db.execute(
                "DELETE FROM objective_actions WHERE job=? AND run=? AND call_id=?",
                (self.job, self.run, call_id),
            )


CURRENT_ACTIONS: ContextVar[ActionRun | None] = ContextVar("talon_actions", default=None)


@tool
def reconcile_job_action(
    call_id: str, read_call_id: str, outcome: Literal["applied", "absent"], reference: str
) -> str:
    """Record a conclusive destination read for an interrupted action before further writes.

    Use applied for a matching destination effect and absent only for authoritative absence.
    Supply the successful read tool call ID and the destination evidence reference.
    """
    run = CURRENT_ACTIONS.get()
    if run is None or read_call_id not in run.reads or not reference.strip():
        return "A successful destination read and its evidence reference are required."
    if call_id in run.active:
        return "Action is still executing."
    with closing(sqlite3.connect(run.path)) as db, db:
        row = db.execute(
            "DELETE FROM objective_actions WHERE job=? AND call_id=? RETURNING tool, arguments",
            (run.job, call_id),
        ).fetchone()
        if row is None:
            return "No matching unresolved action."
        note = f"Reconciled {row[0]} ({call_id}): {outcome} at {reference[:1000]}"
        db.execute(
            "UPDATE checks SET work=substr(work || ?,-4000) WHERE run=?", ("\n" + note, run.run)
        )
    if outcome == "applied":
        run.applied.add((row[0], row[1]))
    return f"Recorded: {note}."


def successful(result: object) -> bool:
    """Whether a tool returned an unambiguous success receipt."""
    return (
        isinstance(result, ToolMessage)
        and result.status == "success"
        and not (isinstance(result.content, str) and result.content.startswith("Error:"))
    )
