"""Durable, owner-scoped history of scheduled job runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from deepagents_talon.cron.jobs import CronJob, CronOrigin


_SAVE_WORK: ContextVar[Callable[[str], None] | None] = ContextVar("job_save_work", default=None)
_NOTE_LIMIT = 2000


@tool
def record_job_work(note: str) -> str:
    """Checkpoint verified work, evidence references, and the next step during a job run."""
    save = _SAVE_WORK.get()
    if save is None:
        return "Available only during a scheduled job run."
    save(note)
    if len(note) > _NOTE_LIMIT:
        return "Work saved with its middle truncated. Checkpoint any missing reference separately."
    return "Work saved to job history."


class ObjectiveReport(BaseModel):
    """One check's private work summary and optional owner-facing message."""

    status: Literal["checked", "partial", "blocked", "unverified", "complete"]
    summary: str = Field(min_length=1, max_length=4000)
    notice: str = Field(default="", max_length=2000)

    blocker_key: str = Field(default="", max_length=200)

    next_wake_at: datetime | None = None
    wake_reason: Literal["poll", "owner_timing", "expected_event"] = "poll"
    completion_criteria: list[str] | None = None
    evidence: dict[str, str | None] = Field(default_factory=dict, max_length=64)

    def next_wake(
        self, previous: dict[str, str] | None, now: datetime, deadline: datetime | None
    ) -> datetime | None:
        """Double the observed polling interval after consecutive unchanged checks."""
        if (
            self.status != "checked"
            or self.wake_reason != "poll"
            or self.next_wake_at is None
            or self.next_wake_at <= now
            or not previous
            or previous["status"] != "checked"
        ):
            return self.next_wake_at
        elapsed = now - datetime.fromisoformat(previous["started"]).replace(tzinfo=UTC)
        delay = min(timedelta(hours=4), max(timedelta(minutes=10), elapsed * 2))
        wake = max(now + delay, self.next_wake_at)
        return min(wake, deadline) if deadline else wake

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
        """Recover usable fields from JSON reports and retain unstructured replies as notes."""
        payload = text.strip()
        if payload.startswith(("```json\n", "```\n")) and payload.endswith("```"):
            payload = payload.split("\n", 1)[1][:-3].strip()
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                _normalize_report(data)
            return cls.model_validate(data)
        except ValueError, RecursionError:
            return cls(
                status="unverified", summary=text.strip()[:4000] or "No work summary returned."
            )


def _normalize_report(data: dict[str, object]) -> None:
    """Keep display and scheduling errors from discarding verified milestone references."""
    if data.get("wake_reason") is None:
        data.pop("wake_reason", None)
    if isinstance(summary := data.get("summary"), dict) and summary:
        data["summary"] = json.dumps(summary, ensure_ascii=False)[:4000]
    evidence = data.get("evidence")
    if not isinstance(evidence, dict):
        return
    invalid = [key for key, ref in evidence.items() if ref is not None and not isinstance(ref, str)]
    if invalid:
        for key in invalid:
            del evidence[key]
        if data.get("status") == "complete":
            data["status"] = "partial"
        if isinstance(summary := data.get("summary"), str):
            data["summary"] = f"Supply text references for {', '.join(invalid)}. {summary}"[:4000]


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

            db.execute(
                "CREATE TABLE IF NOT EXISTS evidence (job TEXT, revision TEXT, criterion TEXT, "
                "reference TEXT NOT NULL, PRIMARY KEY(job,revision,criterion))"
            )

            db.execute(
                "CREATE TABLE IF NOT EXISTS criteria (job TEXT, revision TEXT, "
                "items TEXT NOT NULL, PRIMARY KEY(job,revision))"
            )

    def verification(self, job: CronJob) -> tuple[tuple[str, ...] | None, dict[str, str]]:
        """Return standing criteria and evidence, including those learned for older jobs."""
        with closing(sqlite3.connect(self.path)) as db:
            revision = _revision(job)
            criteria = _criteria(db, job, revision)
            return criteria, _saved_evidence(db, job.id, revision, limit=-1 if criteria else 12)

    def finish_check(
        self, job: CronJob, run: str, report: ObjectiveReport, reads: set[str]
    ) -> ObjectiveReport:
        """Commit coverage, milestone evidence and the effective outcome together."""
        revision = _revision(job)
        report = _check_coverage(job, report, reads)
        with closing(sqlite3.connect(self.path)) as db, db:
            if job.completion_criteria is None and report.completion_criteria is not None:
                db.execute(
                    "INSERT OR IGNORE INTO criteria VALUES(?,?,?)",
                    (job.id, revision, json.dumps(report.completion_criteria)),
                )
            criteria = _criteria(db, job, revision)
            _update_evidence(db, job.id, revision, criteria, report)
            if report.status == "complete":
                evidence = _saved_evidence(db, job.id, revision)
                report = _check_completion(report, criteria, evidence)
            db.execute(
                "UPDATE checks SET status=?,summary=?,read_tools=? WHERE run=?",
                (report.status, report.summary[:4000], ", ".join(sorted(reads))[:2000], run),
            )
        return report

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
            self.append_work(run, note)

        token = _SAVE_WORK.set(save)
        try:
            yield
        finally:
            _SAVE_WORK.reset(token)

    def append_work(self, run: str, note: str) -> None:
        """Keep a bounded note, preserving the next step at the end of long checkpoints."""
        if len(note) > _NOTE_LIMIT:
            marker = "\n[truncated]\n"
            head = _NOTE_LIMIT // 2
            note = note[:head] + marker + note[-(_NOTE_LIMIT - head - len(marker)) :]
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "UPDATE checks SET work=substr(work || ?,-4000) WHERE run=? AND status='started'",
                ("\n" + note, run),
            )

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


def _revision(job: CronJob) -> str:
    instructions = [job.prompt, job.completion_criteria, job.required_sources, job.read_sources]
    return hashlib.sha256(json.dumps(instructions).encode()).hexdigest()


def _saved_evidence(
    db: sqlite3.Connection, job: str, revision: str, *, limit: int = -1
) -> dict[str, str]:
    return dict(
        db.execute(
            "SELECT criterion,reference FROM evidence WHERE job=? AND revision=? "
            "ORDER BY rowid DESC LIMIT ?",
            (job, revision, limit),
        )
    )


def _criteria(db: sqlite3.Connection, job: CronJob, revision: str) -> tuple[str, ...] | None:
    if job.completion_criteria is not None:
        return job.completion_criteria
    row = db.execute(
        "SELECT items FROM criteria WHERE job=? AND revision=?", (job.id, revision)
    ).fetchone()
    return tuple(json.loads(row[0])) if row else None


def _update_evidence(
    db: sqlite3.Connection,
    job: str,
    revision: str,
    criteria: tuple[str, ...] | None,
    report: ObjectiveReport,
) -> None:
    for criterion, reference in report.evidence.items():
        if criteria and criterion not in criteria:
            continue
        if not reference or not reference.strip():
            db.execute(
                "DELETE FROM evidence WHERE job=? AND revision=? AND criterion=?",
                (job, revision, criterion),
            )
        elif report.status in {"partial", "checked", "complete", "blocked"}:
            db.execute(
                "INSERT OR REPLACE INTO evidence VALUES(?,?,?,?)",
                (job, revision, criterion, reference.strip()[:2000]),
            )


def _check_coverage(job: CronJob, report: ObjectiveReport, reads: set[str]) -> ObjectiveReport:
    missing = sorted(set(job.required_sources) - reads)
    if missing:
        gap = "Missing required reads: " + ", ".join(missing)
    elif job.read_sources is not None and not reads:
        gap = "No successful scoped read."
    else:
        return report
    status = "blocked" if report.status == "blocked" else "unverified"
    retractions = {
        key: value for key, value in report.evidence.items() if not (value or "").strip()
    }
    return report.model_copy(
        update={"status": status, "summary": f"{gap} {report.summary}", "evidence": retractions}
    )


def _check_completion(
    report: ObjectiveReport, criteria: tuple[str, ...] | None, evidence: dict[str, str]
) -> ObjectiveReport:
    if criteria is None:
        gap = "Finish criteria have not been recorded."
    elif not criteria:
        gap = "Ongoing objective."
    elif outstanding := sorted(set(criteria) - evidence.keys()):
        gap = "Missing evidence for: " + ", ".join(outstanding)
    else:
        return report
    return report.model_copy(update={"status": "partial", "summary": f"{gap} {report.summary}"})
