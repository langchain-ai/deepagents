"""Durable, owner-scoped history of scheduled job runs."""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from deepagents_talon.cron.jobs import CronJob, CronOrigin


_SAVE_WORK: ContextVar[Callable[[str], None] | None] = ContextVar("job_save_work", default=None)


@tool
def record_job_work(note: str) -> str:
    """Checkpoint verified work, evidence references, and the next step during a job run."""
    save = _SAVE_WORK.get()
    if save is None:
        return "Available only during a scheduled job run."
    save(note)
    return "Work saved to job history."


class ObjectiveReport(BaseModel):
    """One check's private work summary and optional owner-facing message."""

    status: Literal["checked", "partial", "blocked", "unverified", "complete"]
    summary: str = Field(min_length=1, max_length=4000)
    notice: str = Field(default="", max_length=2000)

    blocker_key: str = Field(default="", max_length=200)

    next_wake_at: datetime | None = None

    @field_validator("next_wake_at", mode="before")
    @classmethod
    def _parse_wake(cls, value: object) -> datetime | None:
        """Preserve findings even if the proposed wake is unusable."""
        if isinstance(value, str):
            try:
                wake = datetime.fromisoformat(value)
                if wake.tzinfo is not None:
                    return wake.astimezone(UTC)
            except ValueError, OverflowError:
                pass
        return None

    @classmethod
    def from_reply(cls, text: str) -> ObjectiveReport:
        """Keep an unstructured result as an unverified note for later inspection."""
        payload = text.strip()
        if payload.startswith(("```json\n", "```\n")) and payload.endswith("```"):
            payload = payload.split("\n", 1)[1][:-3].strip()
        try:
            return cls.model_validate_json(payload)
        except ValidationError:
            return cls(
                status="unverified", summary=text.strip()[:4000] or "No work summary returned."
            )


class JobJournal:
    """Store bounded check summaries independently of the scheduler's job list."""

    def __init__(self, directory: Path) -> None:
        """Open the assistant's local journal, creating it on first use."""
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = directory / "job-history.sqlite3"
        self.path.touch(mode=0o600, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS checks ("
                "run TEXT PRIMARY KEY, job TEXT NOT NULL, name TEXT NOT NULL, "
                "channel TEXT NOT NULL, "
                "chat TEXT NOT NULL, owner TEXT NOT NULL, started TEXT DEFAULT CURRENT_TIMESTAMP, "
                "status TEXT NOT NULL, summary TEXT NOT NULL, "
                "work TEXT NOT NULL DEFAULT '', read_tools TEXT NOT NULL DEFAULT '')"
            )
            db.execute("CREATE INDEX IF NOT EXISTS job_checks ON checks(channel, chat, owner, job)")

            db.execute(
                "CREATE TABLE IF NOT EXISTS blockers (job TEXT PRIMARY KEY, "
                "key TEXT NOT NULL, notice TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0)"
            )

    def blocker_key(self, job: str) -> str:
        """Return the current blocker identity for the next isolated check."""
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT key FROM blockers WHERE job=?", (job,)).fetchone()
            return row[0] if row else ""

    def notice(self, job: CronJob, report: ObjectiveReport) -> str:
        """Keep one notice pending until delivered, or until progress clears the blocker."""
        with closing(sqlite3.connect(self.path)) as db, db:
            if report.status != "blocked":
                if report.status in {"checked", "partial", "complete"}:
                    db.execute("DELETE FROM blockers WHERE job=?", (job.id,))
                return ""
            notice = (
                report.notice.strip()
                or f"Job {job.name} is blocked. Ask for its saved check results."
            )
            key = (report.blocker_key.strip() or report.notice.strip() or report.summary).casefold()
            db.execute(
                "INSERT INTO blockers(job,key,notice) VALUES(?,?,?) ON CONFLICT(job) "
                "DO UPDATE SET key=excluded.key,notice=excluded.notice,delivered=0 "
                "WHERE blockers.key != excluded.key",
                (job.id, key, notice),
            )
            row = db.execute(
                "SELECT notice FROM blockers WHERE job=? AND delivered=0", (job.id,)
            ).fetchone()
            return row[0] if row else ""

    def mark_delivered(self, job: str, notice: str) -> None:
        """Acknowledge a blocker only after the channel accepted its message."""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE blockers SET delivered=1 WHERE job=? AND notice=?", (job, notice))

    def start(self, job: CronJob, run: str) -> None:
        """Record the start before doing work, including when the host subsequently exits."""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT INTO checks(run,job,name,channel,chat,owner,status,summary) "
                "VALUES(?,?,?,?,?,?,'started','No outcome recorded yet.')",
                (
                    run,
                    job.id,
                    job.name,
                    job.origin.channel or "",
                    job.origin.conversation_id,
                    job.origin.sender_id or "",
                ),
            )

    @contextmanager
    def capture_work(self, run: str) -> Iterator[None]:
        """Bind checkpoints to the executing check without exposing a writable job ID."""

        def save(note: str) -> None:
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute(
                    "UPDATE checks SET work=substr(work || ?,-4000) "
                    "WHERE run=? AND status='started'",
                    ("\n" + note[:2000], run),
                )

        token = _SAVE_WORK.set(save)
        try:
            yield
        finally:
            _SAVE_WORK.reset(token)

    def finish(self, run: str, status: str, summary: str, *, reads: Sequence[str] = ()) -> None:
        """Persist the last known outcome for this host-generated run ID."""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "UPDATE checks SET status=?, summary=?, read_tools=? WHERE run=?",
                (status, summary[:4000], ", ".join(reads)[:2000], run),
            )

    def history(
        self, job: str | None, origin: CronOrigin, *, offset: int = 0, limit: int = 10
    ) -> list[dict[str, str]]:
        """Return one bounded page, newest first, within the originating owner's thread."""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT job, name, started, status, summary, work, read_tools "
                "FROM checks WHERE channel=? "
                "AND chat=? AND owner=? AND (? IS NULL OR job=?) ORDER BY rowid DESC "
                "LIMIT ? OFFSET ?",
                (
                    origin.channel or "",
                    origin.conversation_id,
                    origin.sender_id or "",
                    job,
                    job,
                    limit,
                    max(0, offset),
                ),
            )
            return [dict(row) for row in rows]
