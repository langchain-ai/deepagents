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

from deepagents_talon.cron.actions import CURRENT_ACTIONS, ActionRun

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

    requires_signoff: bool = False
    deadline_risk: str = Field(default="", max_length=1000)
    decision_question: str = Field(default="", max_length=1000)
    decision_unlocks: str = Field(default="", max_length=1000)

    blocker_key: str = Field(default="", max_length=200)

    verified: list[str] = Field(default_factory=list, max_length=64)

    wait_condition: str = Field(default="", max_length=2000)
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
            if deadline and self.next_wake_at:
                return min(deadline, self.next_wake_at)
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

            db.execute(
                "CREATE TABLE IF NOT EXISTS signoff (job TEXT, revision TEXT, "
                "PRIMARY KEY(job,revision))"
            )

            db.execute(
                "CREATE TABLE IF NOT EXISTS decisions (job TEXT, revision TEXT, question TEXT, "
                "unlocks TEXT NOT NULL, answer TEXT, delivered INTEGER DEFAULT 0, message_id TEXT, "
                "PRIMARY KEY(job,revision,question))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS objective_actions (job TEXT, run TEXT, call_id TEXT, "
                "tool TEXT, arguments TEXT, PRIMARY KEY(job, call_id))"
            )
            db.execute("BEGIN IMMEDIATE")
            if "verified" not in {row[1] for row in db.execute("PRAGMA table_info(evidence)")}:
                db.execute("ALTER TABLE evidence ADD COLUMN verified INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def capture_actions(self, job: str, run: str) -> Iterator[ActionRun]:
        """Track real tool execution for an objective until its run ends."""
        actions = ActionRun(self.path, job, run)
        with CURRENT_ACTIONS.set(actions):
            yield actions

    def verification(
        self, job: CronJob
    ) -> tuple[tuple[str, ...] | None, dict[str, dict[str, str | bool]]]:
        """Return standing criteria and evidence, including those learned for older jobs."""
        with closing(sqlite3.connect(self.path)) as db:
            revision = _revision(job)
            criteria = _criteria(db, job, revision)
            rows = db.execute(
                "SELECT criterion,reference,verified FROM evidence WHERE job=? AND revision=? "
                "ORDER BY rowid DESC LIMIT ?",
                (job.id, revision, -1 if criteria else 12),
            )
            return criteria, {
                key: {"reference": ref, "verified": bool(verified)} for key, ref, verified in rows
            }

    def finish_check(
        self, job: CronJob, run: str, report: ObjectiveReport, reads: set[str]
    ) -> ObjectiveReport:
        """Commit coverage, milestone evidence and the effective outcome together."""
        revision = _revision(job)
        with closing(sqlite3.connect(self.path)) as db, db:
            if job.completion_criteria is None and report.completion_criteria is not None:
                db.execute(
                    "INSERT OR IGNORE INTO criteria VALUES(?,?,?)",
                    (job.id, revision, json.dumps(report.completion_criteria)),
                )
            criteria = _criteria(db, job, revision)
            if report.summary.strip() == "[SILENT]":
                work = db.execute("SELECT work FROM checks WHERE run=?", (run,)).fetchone()
                summary = work[0].strip() if work else ""
                if not summary:
                    summary = "No result summary recorded."
                    if reads:
                        summary += f" Read sources: {', '.join(sorted(reads))}."
                report = report.model_copy(update={"summary": summary})
            outstanding = db.execute(
                "SELECT tool, call_id FROM objective_actions WHERE job=? AND run<>? LIMIT 1",
                (job.id, run),
            ).fetchone()
            if outstanding:
                tool_name, call_id = outstanding
                notice = (
                    f"The outcome of {tool_name} ({call_id}) is uncertain. Can you verify the "
                    "result at its destination or decide whether to accept the risk of a duplicate?"
                )
                report = report.model_copy(
                    update={
                        "status": "blocked",
                        "notice": report.notice or notice,
                        "decision_question": notice,
                        "decision_unlocks": "Resolve the previous action before continuing.",
                        "blocker_key": f"uncertain-action:{call_id}",
                    }
                )
            report = _check_coverage(job, report, reads)
            report = _update_evidence(db, job.id, revision, criteria, report)
            if report.requires_signoff:
                db.execute("INSERT OR IGNORE INTO signoff VALUES(?,?)", (job.id, revision))
            if report.status == "complete":
                evidence = {
                    row[0]
                    for row in db.execute(
                        "SELECT criterion FROM evidence WHERE job=? AND revision=? AND verified",
                        (job.id, revision),
                    )
                }
                report = _check_completion(report, criteria, evidence)
                if (
                    report.status == "complete"
                    and db.execute(
                        "SELECT 1 FROM signoff WHERE job=? AND revision=?", (job.id, revision)
                    ).fetchone()
                ):
                    report = self._check_signoff(db, job, revision, report)
            db.execute(
                "UPDATE checks SET status=?,summary=?,read_tools=? WHERE run=?",
                (report.status, report.summary[:4000], ", ".join(sorted(reads))[:2000], run),
            )
        return report

    def _check_signoff(
        self, db: sqlite3.Connection, job: CronJob, revision: str, report: ObjectiveReport
    ) -> ObjectiveReport:
        references = db.execute(
            "SELECT criterion,reference FROM evidence WHERE job=? AND revision=? "
            "AND verified ORDER BY criterion",
            (job.id, revision),
        ).fetchall()
        version = hashlib.sha256(json.dumps(references).encode()).hexdigest()[:10]
        question = (
            f"Do you approve completion of {job.name} (evidence {version})? Reply yes to approve."
        )
        answer = self.decision_answer(job, question)
        if answer and answer.strip().casefold().rstrip(".!") in {
            "yes",
            "approve",
            "approved",
            "i approve",
        }:
            return report
        return report.model_copy(
            update={
                "status": "blocked",
                "notice": question,
                "decision_question": question,
                "decision_unlocks": "Close the verified goal and deliver the completion notice.",
                "blocker_key": f"signoff:{version}",
            }
        )

    def decisions(self, job: CronJob) -> list[tuple[str, str, str | None]]:
        """List questions, the work they unlock, and owner answers for the current revision."""
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute(
                "SELECT question,unlocks,answer FROM decisions WHERE job=? AND revision=? "
                "ORDER BY rowid",
                (job.id, _revision(job)),
            ).fetchall()

    def open_decision(self, job: CronJob, reply_to: str | None = None) -> tuple[str, str] | None:
        """Find the current unanswered question, optionally by its delivered channel message."""
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute(
                "SELECT question,unlocks FROM decisions WHERE job=? AND revision=? AND answer "
                "IS NULL AND (? IS NULL OR message_id=?)",
                (job.id, _revision(job), reply_to, reply_to),
            ).fetchone()

    def decision_answer(self, job: CronJob, question: str) -> str | None:
        """Read the owner's answer to this exact question under the current instructions."""
        return next((answer for q, _, answer in self.decisions(job) if q == question), None)

    def resolve_decision(
        self, job: CronJob, question: str, answer: str, *, wake: Callable[[], bool]
    ) -> bool:
        """Resolve an open question exactly once."""
        with closing(sqlite3.connect(self.path)) as db, db:
            updated = db.execute(
                "UPDATE decisions SET answer=? WHERE job=? AND revision=? AND question=? "
                "AND answer IS NULL",
                (answer, job.id, _revision(job), question),
            ).rowcount
            if updated and wake():
                return True
            db.rollback()
            return False

    def decision_notice(self, job: CronJob, report: ObjectiveReport) -> str:
        """Keep the current question pending until its notice reaches the owner."""
        question = report.decision_question.strip()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "DELETE FROM decisions WHERE job=? AND revision=? AND answer IS NULL "
                "AND question!=?",
                (job.id, _revision(job), question),
            )
            db.execute(
                "INSERT OR IGNORE INTO decisions(job,revision,question,unlocks) VALUES(?,?,?,?)",
                (job.id, _revision(job), question, report.decision_unlocks),
            )
            row = db.execute(
                "SELECT unlocks FROM decisions WHERE job=? AND revision=? AND "
                "question=? AND answer IS NULL AND delivered=0",
                (job.id, _revision(job), question),
            ).fetchone()
            return decision_text(job.name, job.id, question, row[0]) if row else ""

    def blocker_key(self, job: str) -> str:
        """Return the current blocker identity for the next isolated check."""
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT key FROM blockers WHERE job=?", (job,)).fetchone()
            return row[0] if row else ""

    def notice(self, job: CronJob, report: ObjectiveReport) -> str:
        """Keep one notice pending until delivered, or until progress clears the blocker."""
        with closing(sqlite3.connect(self.path)) as db, db:
            risk = report.deadline_risk.strip() if job.target_at or job.until else ""
            if report.status != "blocked" and not risk:
                if report.status in {"checked", "partial", "complete"}:
                    db.execute("DELETE FROM blockers WHERE job=?", (job.id,))
                return ""
            notice = (
                risk
                or report.notice.strip()
                or f"Job {job.name} is blocked. Ask for its saved check results."
            )
            key = (report.blocker_key.strip() or report.notice.strip() or report.summary).casefold()
            if risk:
                key = "deadline:" + risk.casefold()
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

    def mark_delivered(self, job: str, notice: str, message_id: str | None = None) -> None:
        """Acknowledge a blocker only after the channel accepted its message."""
        with closing(sqlite3.connect(self.path)) as db, db:
            for revision, question, unlocks in db.execute(
                "SELECT revision,question,unlocks FROM decisions WHERE job=? AND delivered=0",
                (job,),
            ):
                if notice.endswith(decision_text("", job, question, unlocks)):
                    db.execute(
                        "UPDATE decisions SET delivered=1,message_id=? WHERE job=? AND revision=? "
                        "AND question=?",
                        (message_id, job, revision, question),
                    )
            db.execute(
                "UPDATE blockers SET delivered=1 WHERE job=? AND "
                "(notice=? OR instr(? || char(10), notice || char(10)) > 0)",
                (job, notice, notice),
            )

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
        self,
        job: str | None,
        origin: CronOrigin,
        *,
        offset: int = 0,
        limit: int = 10,
        before_run: str | None = None,
    ) -> list[dict[str, str]]:
        """Return one bounded page, newest first, within the originating owner's thread."""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT job, name, started, status, summary, work, read_tools "
                "FROM checks WHERE channel=? "
                "AND chat=? AND owner=? AND (? IS NULL OR job=?) "
                "AND (? IS NULL OR rowid < (SELECT rowid FROM checks WHERE run=?)) "
                "ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (
                    origin.channel or "",
                    origin.conversation_id,
                    origin.sender_id or "",
                    job,
                    job,
                    before_run,
                    before_run,
                    limit,
                    max(0, offset),
                ),
            )
            return [dict(row) for row in rows]


def decision_text(name: str, job_id: str, question: str, unlocks: str) -> str:
    """Format an owner question with its explicit answer address."""
    return f"{name} [{job_id}] {question} {unlocks} Reply with: answer {job_id}: <your answer>"


def _revision(job: CronJob) -> str:
    instructions = [job.prompt, job.completion_criteria, job.required_sources, job.read_sources]
    return hashlib.sha256(json.dumps(instructions).encode()).hexdigest()


def _criteria(db: sqlite3.Connection, job: CronJob, revision: str) -> tuple[str, ...] | None:
    if job.completion_criteria is not None:
        return job.completion_criteria
    row = db.execute(
        "SELECT items FROM criteria WHERE job=? AND revision=?", (job.id, revision)
    ).fetchone()
    return tuple(json.loads(row[0])) if row else None


def criterion_ids(criteria: Sequence[str]) -> dict[str, str]:
    """Assign stable positions within the job's immutable criteria revision."""
    return {f"C{index}": criterion for index, criterion in enumerate(criteria, start=1)}


def _update_evidence(
    db: sqlite3.Connection,
    job: str,
    revision: str,
    criteria: tuple[str, ...] | None,
    report: ObjectiveReport,
) -> ObjectiveReport:
    aliases = criterion_ids(criteria or ())
    unmatched = []
    for key, reference in report.evidence.items():
        criterion = aliases.get(key, key)
        if criteria and criterion not in criteria:
            unmatched.append(f"{key}: {reference}")
            continue
        if not reference or not reference.strip():
            db.execute(
                "DELETE FROM evidence WHERE job=? AND revision=? AND criterion=?",
                (job, revision, criterion),
            )
        elif report.status in {"partial", "checked", "complete", "blocked"}:
            db.execute(
                "INSERT OR REPLACE INTO evidence(job,revision,criterion,reference,verified) "
                "VALUES(?,?,?,?,?)",
                (job, revision, criterion, reference.strip()[:2000], key in report.verified),
            )

    if unmatched:
        message = "Resubmit evidence with standing criterion IDs: " + "; ".join(unmatched)
        status = "partial" if report.status == "complete" else report.status
        return report.model_copy(
            update={"status": status, "summary": f"{message}. {report.summary}"}
        )
    return report


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
    report: ObjectiveReport, criteria: tuple[str, ...] | None, evidence: set[str]
) -> ObjectiveReport:
    if criteria is None:
        gap = "Finish criteria have not been recorded."
    elif not criteria:
        gap = "Ongoing objective."
    elif outstanding := sorted(set(criteria) - evidence):
        gap = "Missing evidence for: " + ", ".join(outstanding)
    else:
        return report
    return report.model_copy(update={"status": "partial", "summary": f"{gap} {report.summary}"})
