"""Read the retired goal ledger once when upgrading to scheduled chat turns."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from deepagents_talon.cron.jobs import CronJob, CronOrigin, CronRepeat, CronSchedule


class _Scope(BaseModel):
    channel: str | None = None
    conversation_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)


class _Charter(BaseModel):
    model_config = ConfigDict(extra="allow")
    outcome: str = Field(min_length=1)
    sources: tuple[str, ...] = ()
    cadence: str = "every 24h"


class _Goal(BaseModel):
    id: str
    scope: _Scope
    state: Literal[
        "draft", "awaiting_confirmation", "active", "paused", "needs_attention", "removed"
    ]
    charter: _Charter
    confirmed_charter: _Charter | None = None
    goal_job_id: str | None = None
    created_at: AwareDatetime


class _Record(BaseModel):
    model_config = ConfigDict(extra="allow")
    goal_id: str


class _Ledger(BaseModel):
    version: Literal[1]
    goals: list[_Goal]
    items: list[_Record] = Field(default_factory=list)
    decisions: list[_Record] = Field(default_factory=list)
    observations: list[_Record] = Field(default_factory=list)


def import_goals(raw: bytes, jobs: list[CronJob], assistant_id: str) -> list[CronJob]:
    """Convert ledger entries, retaining schedules, confirmed charters and prior work."""
    ledger = _Ledger.model_validate_json(raw)
    by_id = {job.id: job for job in jobs}
    # An orphaned or removed legacy job has no authorized work to perform.
    converted = {
        key: job
        for key, job in by_id.items()
        if job.legacy_goal_id is None or job.legacy_goal_id == job.id
    }
    for goal in ledger.goals:
        if goal.state == "removed":
            continue
        charter = goal.confirmed_charter or goal.charter
        existing = by_id.get(goal.goal_job_id or goal.id)
        origin = CronOrigin(
            channel=goal.scope.channel,
            conversation_id=goal.scope.conversation_id,
            sender_id=goal.scope.owner_id,
        )
        if existing is not None and (
            existing.legacy_goal_id != goal.id
            or existing.origin.channel != origin.channel
            or existing.origin.conversation_id != origin.conversation_id
            or existing.origin.sender_id != origin.sender_id
        ):
            msg = "Legacy goal does not match the owner or identity of its scheduled job."
            raise ValueError(msg)
        schedule = CronSchedule.parse(charter.cadence)
        job = existing or CronJob(
            id=goal.id,
            assistant_id=assistant_id,
            mode="objective",
            origin=origin,
            schedule=schedule,
            repeat=CronRepeat(),
            created_at=goal.created_at,
            next_run_at=schedule.next_after(goal.created_at),
            last_run_at=None,
            enabled=False,
            last_status=None,
            last_error=None,
            name="",
            prompt="",
            deliver_to="thread",
        )
        history = {
            "charter": charter.model_dump(exclude_none=True),
            **{
                name: [row.model_dump() for row in rows if row.goal_id == goal.id]
                for name, rows in (
                    ("items", ledger.items),
                    ("decisions", ledger.decisions),
                    ("observations", ledger.observations),
                )
            },
        }
        if goal.confirmed_charter and goal.charter != goal.confirmed_charter:
            history["unconfirmed_changes"] = goal.charter.model_dump(exclude_none=True)
        converted[job.id] = replace(
            job,
            name=charter.outcome.splitlines()[0][:100],
            prompt="Work from the saved charter. Prior work, proposals and unconfirmed edits "
            "are history. Check uncertain outcomes before retrying. Ask about pending decisions "
            "in chat. Summarize progress without quoting private evidence in shared chats.\n"
            + json.dumps(history, ensure_ascii=False),
            enabled=job.enabled and goal.state == "active" and goal.confirmed_charter is not None,
            read_sources=charter.sources,
            legacy_goal_id=None,
        )
    return list(converted.values())
