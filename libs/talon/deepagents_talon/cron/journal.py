"""Durable, owner-scoped history of scheduled job runs."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from deepagents_talon.cron.actions import CURRENT_ACTIONS, ActionRun
from deepagents_talon.store_records import StoreRecords, digest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence

    from langgraph.store.base import BaseStore

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.cron.jobs import CronJob, CronOrigin


_SAVE_WORK: ContextVar[Callable[[str], Awaitable[None]] | None] = ContextVar(
    "job_save_work", default=None
)
_NOTE_LIMIT = 2000


@tool
async def record_job_work(note: str) -> str:
    """Checkpoint verified work, evidence references, and the next step during a job run."""
    save = _SAVE_WORK.get()
    if save is None:
        return "Available only during a scheduled job run."
    await save(note)
    if len(note) > _NOTE_LIMIT:
        return "Work saved with its middle truncated. Checkpoint any missing reference separately."
    return "Work saved to job history."


class ObjectiveReport(BaseModel):
    """One check's private work summary and optional owner-facing message."""

    status: Literal["checked", "partial", "blocked", "unverified", "complete"]
    summary: str = Field(min_length=1, max_length=4000)
    notice: str = Field(default="", max_length=2000)

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
    wait = data.get("wait_condition")
    data["wait_condition"] = wait[:2000] if isinstance(wait, str) else ""
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

    def __init__(self, store: BaseStore, assistant_id: str) -> None:
        """Share the configured history backend in a separate journal namespace."""
        self.records = StoreRecords(store, ("talon", assistant_id, "job-history"))

    @classmethod
    @asynccontextmanager
    async def open(cls, config: TalonConfig) -> AsyncGenerator[JobJournal]:
        """Open the configured history store for standalone scheduler callers."""
        from deepagents_talon.history_backends import open_history  # noqa: PLC0415

        if config.history_uri is None:
            config.checkpoint_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            config.checkpoint_path.touch(mode=0o600, exist_ok=True)
        async with open_history(config) as archive:
            yield cls(archive.records.store, config.assistant_id)

    @contextmanager
    def capture_actions(self, job: str, run: str) -> Generator[ActionRun]:
        """Track tool receipts for this objective run."""
        actions = ActionRun(self.records, job, run)
        with CURRENT_ACTIONS.set(actions):
            yield actions

    async def verification(
        self, job: CronJob
    ) -> tuple[tuple[str, ...] | None, dict[str, dict[str, str | bool]]]:
        """Read criteria and evidence for the current objective revision."""
        async with self.records.access():
            state = await self.records.get(_revision_key(job)) or {}
            return _progress(job, state)

    async def finish_check(
        self, job: CronJob, run: str, report: ObjectiveReport, reads: set[str]
    ) -> ObjectiveReport:
        """Commit the effective outcome and evidence in one recoverable mutation."""
        async with self.records.access():
            row = await self.records.get(run)
            if row is None or row["job"] != job.id:
                msg = "Unknown scheduled run"
                raise ValueError(msg)
            key = _revision_key(job)
            state = await self.records.get(key) or {}
            if report.completion_criteria is not None:
                state.setdefault("criteria", report.completion_criteria)
            criteria, evidence = _progress(job, state)
            if report.summary.strip() == "[SILENT]":
                fallback = "No result summary recorded."
                if reads:
                    fallback += f" Read sources: {', '.join(sorted(reads))}."
                report = report.model_copy(update={"summary": str(row["work"]).strip() or fallback})
            actions = await self.records.get("actions:" + job.id) or {}
            for action in actions.values():
                if isinstance(action, dict) and action["run"] != run:
                    call_id = action["call_id"]
                    notice = (
                        f"The outcome of {action['tool']} ({call_id}) is uncertain. "
                        "Can you verify the result at its destination or decide whether to accept "
                        "the risk of a duplicate?"
                    )
                    report = report.model_copy(
                        update={
                            "status": "blocked",
                            "notice": report.notice or notice,
                            "blocker_key": f"uncertain-action:{call_id}",
                        }
                    )
                    break
            report = _update_evidence(evidence, criteria, _check_coverage(job, report, reads))
            if report.status == "complete":
                verified = {k for k, v in evidence.items() if v["verified"]}
                report = _check_completion(report, criteria, verified)
            row.update(
                status=report.status,
                summary=report.summary[:4000],
                read_tools=", ".join(sorted(reads))[:2000],
            )
            state["evidence"] = evidence
            await self.records.commit([(key, state), (run, row)])
            return report

    async def blocker_key(self, job: str) -> str:
        """Read the pending blocker identity."""
        async with self.records.access():
            row = await self.records.get("blocker:" + job) or {}
            return str(row.get("key", ""))

    async def notice(self, job: CronJob, report: ObjectiveReport) -> str:
        """Keep one blocker notice pending until delivery or progress."""
        key = "blocker:" + job.id
        async with self.records.access():
            row = await self.records.get(key)
            if report.status != "blocked":
                if row and report.status in {"checked", "partial", "complete"}:
                    await self.records.commit([(key, None)])
                return ""
            identity = (
                report.blocker_key.strip() or report.notice.strip() or report.summary
            ).casefold()
            if row and row["key"] == identity:
                return "" if row.get("delivered") else str(row["notice"])
            notice = (
                report.notice.strip()
                or f"Job {job.name} is blocked. Ask for its saved check results."
            )
            await self.records.commit([(key, {"key": identity, "notice": notice})])
            return notice

    async def mark_delivered(self, job: str, notice: str) -> None:
        """Acknowledge the matching pending notice."""
        key = "blocker:" + job
        async with self.records.access():
            row = await self.records.get(key)
            if row and row["notice"] == notice:
                await self.records.commit([(key, {**row, "delivered": True})])

    async def start(self, job: CronJob, run: str) -> None:
        """Link a new run into the owner's history before executing it."""
        owner = _owner_key(job.origin)
        async with self.records.access():
            if await self.records.get(run):
                msg = "Scheduled run already exists"
                raise ValueError(msg)
            head = await self.records.get(owner) or {}
            row = {
                "job": job.id,
                "name": job.name,
                "owner": owner,
                "previous": head.get("run"),
                "started": datetime.now(UTC).isoformat(),
                "status": "started",
                "summary": "No outcome recorded yet.",
                "work": "",
                "read_tools": "",
            }
            await self.records.commit([(run, row), (owner, {"run": run})])

    @contextmanager
    def capture_work(self, run: str) -> Generator[None]:
        """Bind checkpoints to the executing check without exposing a writable job ID."""

        async def save(note: str) -> None:
            await self.append_work(run, note)

        token = _SAVE_WORK.set(save)
        try:
            yield
        finally:
            _SAVE_WORK.reset(token)

    async def append_work(self, run: str, note: str) -> None:
        """Keep a bounded note, preserving the next step at the end of long checkpoints."""
        if len(note) > _NOTE_LIMIT:
            marker = "\n[truncated]\n"
            head = _NOTE_LIMIT // 2
            note = note[:head] + marker + note[-(_NOTE_LIMIT - head - len(marker)) :]
        async with self.records.access():
            row = await self.records.get(run)
            if row and row["status"] == "started":
                row["work"] = (str(row["work"]) + "\n" + note)[-4000:]
                await self.records.commit([(run, row)])

    async def finish(
        self, run: str, status: str, summary: str, *, reads: Sequence[str] = ()
    ) -> None:
        """Persist the last known outcome."""
        async with self.records.access():
            row = await self.records.get(run)
            if row:
                row.update(
                    status=status, summary=summary[:4000], read_tools=", ".join(reads)[:2000]
                )
                await self.records.commit([(run, row)])

    async def history(
        self,
        job: str | None,
        origin: CronOrigin,
        *,
        offset: int = 0,
        limit: int = 10,
        before_run: str | None = None,
    ) -> list[dict[str, str]]:
        """Return newest matching checks in the requesting owner's thread."""
        owner = _owner_key(origin)
        async with self.records.access():
            head = await self.records.get(before_run or owner) or {}
            cursor = (
                head.get("previous")
                if before_run and head.get("owner") == owner
                else head.get("run")
            )
            result = []
            while cursor and len(result) < limit:
                row = await self.records.get(str(cursor))
                if not row or row["owner"] != owner:
                    break
                if job is None or row["job"] == job:
                    if offset > 0:
                        offset -= 1
                    else:
                        result.append(
                            {k: str(v) for k, v in row.items() if k not in {"owner", "previous"}}
                        )
                cursor = row.get("previous")
            return result


def _owner_key(origin: CronOrigin) -> str:
    return "owner:" + digest(origin.channel or "", origin.conversation_id, origin.sender_id or "")


def _revision_key(job: CronJob) -> str:
    instructions = [job.prompt, job.completion_criteria, job.required_sources, job.read_sources]
    return "progress:" + digest(job.id, _owner_key(job.origin), json.dumps(instructions))


def _progress(
    job: CronJob, state: dict[str, object]
) -> tuple[tuple[str, ...] | None, dict[str, dict[str, str | bool]]]:
    criteria = job.completion_criteria
    if criteria is None and isinstance(state.get("criteria"), list):
        criteria = tuple(cast("list[str]", state["criteria"]))
    return criteria, cast("dict[str, dict[str, str | bool]]", state.get("evidence", {}))


def criterion_ids(criteria: Sequence[str]) -> dict[str, str]:
    """Assign stable positions within the job's immutable criteria revision."""
    return {f"C{index}": criterion for index, criterion in enumerate(criteria, start=1)}


def _update_evidence(
    evidence: dict[str, dict[str, str | bool]],
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
            evidence.pop(criterion, None)
        elif report.status in {"partial", "checked", "complete", "blocked"}:
            evidence[criterion] = {
                "reference": reference.strip()[:2000],
                "verified": key in report.verified,
            }

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
