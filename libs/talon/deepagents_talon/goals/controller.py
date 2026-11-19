"""Owner `/goal` commands and the goal's recurring cron job.

Talon is an experimental runtime and is subject to change or removal at any time.

The controller is the only writer of goal state and of goal-owned cron jobs.
Commands never reach the model: charter fields are set by the owner's own
text, so retrieved content or model output cannot rewrite a goal's scope or
confirm it. Goal cron fires are handled here too, without a model turn.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from pydantic import ValidationError

from deepagents_talon.cron.errors import CronJobError
from deepagents_talon.cron.jobs import CronSchedule
from deepagents_talon.cron.scheduler import SILENT_SENTINEL
from deepagents_talon.goals.errors import GoalLedgerCorruptError, GoalLedgerError
from deepagents_talon.goals.models import GoalCharter, GoalScope
from deepagents_talon.observability import log_event

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents_talon.cron.jobs import CronJob, CronJobStore, CronOrigin
    from deepagents_talon.goals.ledger import GoalLedger
    from deepagents_talon.goals.models import Goal

logger = logging.getLogger(__name__)

GOAL_USAGE = (
    "Goals keep working on an outcome for this chat.\n"
    "/goal <outcome> — start a draft goal.\n"
    "/goal edit <field>: <value> — one field per line. Fields: "
    "outcome, sources, exclusions, progress, kpi, baseline, horizon, review, cadence, "
    "decisions. Separate list entries with `;`; send `none` to clear a field.\n"
    "/goal confirm <revision> — activate or update the goal from that exact revision.\n"
    "/goal status — show the goal, its schedule, and its work.\n"
    "/goal remove — stop the goal and remove its scheduled jobs."
)
_SCOPE_NOTICE = (
    "Confirming lets the goal read only the listed sources and post status updates and "
    "decision requests in this chat. Every other write still needs your approval."
)
_FIELDS: dict[str, str] = {
    "outcome": "outcome",
    "sources": "sources",
    "exclusions": "exclusions",
    "progress": "progress_test",
    "kpi": "kpi",
    "baseline": "baseline",
    "horizon": "horizon",
    "review": "review_on",
    "cadence": "cadence",
    "decisions": "decisions_required",
}
_LIST_FIELDS = frozenset({"sources", "exclusions", "decisions_required"})
_CLEAR = frozenset({"none", "-"})
_JOB_PREFIX = "goal:"


@dataclass(frozen=True, slots=True)
class GoalRequest:
    """One `/goal` command from a trusted channel message.

    Args:
        scope: Conversation and sender the host derived from the message.
        origin: Cron origin for jobs the goal owns.
        argument: Text after `/goal`.
    """

    scope: GoalScope
    origin: CronOrigin
    argument: str


class GoalController:
    """Applies owner commands to the ledger and keeps the goal job in step."""

    def __init__(
        self,
        *,
        ledger: GoalLedger,
        store: CronJobStore,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the controller.

        Args:
            ledger: Durable goal state.
            store: Cron store that holds the goal's recurring job.
            now: Clock override for deterministic tests.
        """
        self.ledger = ledger
        self.store = store
        self.now = now or (lambda: datetime.now(UTC))

    def handle(self, request: GoalRequest) -> str:
        """Run one `/goal` command and return the reply text.

        Args:
            request: Parsed command.

        Returns:
            Reply for the starting chat.
        """
        verb, _, rest = request.argument.strip().partition(" ")
        if "\n" in verb:
            verb, _, more = verb.partition("\n")
            rest = f"{more}\n{rest}"
        handler = {
            "": self._status,
            "status": self._status,
            "help": lambda *_: GOAL_USAGE,
            "edit": self._edit,
            "confirm": self._confirm,
            "remove": self._remove,
        }.get(verb.lower())
        try:
            if handler is None:
                return self._start(request, request.argument.strip())
            return handler(request, rest.strip())
        except GoalLedgerCorruptError:
            logger.exception("Goal ledger is unreadable")
            return "The goal ledger could not be read, so nothing changed. Check Talon logs."
        except (GoalLedgerError, CronJobError) as exc:
            return str(exc)

    def on_fire(self, job: CronJob) -> str:
        """Handle a goal job firing; never invokes the model.

        Scheduled work turns are not wired yet, so a live goal only records a
        quiet check. A job whose goal is gone or removed is disabled.

        Args:
            job: Claimed goal-owned cron job.

        Returns:
            The silent sentinel, so the scheduler delivers nothing.
        """
        goal = self.ledger.get_goal(job.goal_id or "")
        if goal is None or _orphaned(goal, job):
            self.store.edit_job(job.id, origin=job.origin, enabled=False)
            log_event(logger, "goal.orphan_job_disabled", job_id=job.id, goal_id=job.goal_id)
            return SILENT_SENTINEL
        goal = self._mirror_job(goal)
        log_event(logger, "goal.quiet_check", goal_id=goal.id, state=goal.state)
        return SILENT_SENTINEL

    def _start(self, request: GoalRequest, outcome: str) -> str:
        if self.ledger.goal_for_conversation(request.scope) is not None:
            return "This chat already has a goal. Send /goal status, /goal edit, or /goal remove."
        charter = _validated(GoalCharter, {"outcome": outcome})
        goal = self.ledger.create_draft(scope=request.scope, charter=charter, now=self.now())
        return _review(goal)

    def _edit(self, request: GoalRequest, text: str) -> str:
        goal = self._owned_goal(request)
        charter = _apply_edits(goal.charter, text)
        owner = request.scope.owner_id
        goal = self.ledger.revise_charter(
            goal.id,
            owner_id=owner,
            charter=charter,
            expected_revision=goal.revision,
            now=self.now(),
        )
        if goal.state == "draft" and not charter.unresolved():
            goal = self.ledger.submit_for_confirmation(
                goal.id, owner_id=owner, expected_revision=goal.revision, now=self.now()
            )
        return _review(goal)

    def _confirm(self, request: GoalRequest, text: str) -> str:
        if not text.isdigit():
            return "Send /goal confirm <revision> with the revision number you reviewed."
        goal = self._owned_goal(request)
        goal = self.ledger.confirm(
            goal.id, owner_id=request.scope.owner_id, revision=int(text), now=self.now()
        )
        if not goal.activation_pending:
            self._sync_cadence(goal)
            return f"Confirmed revision {goal.revision}. The goal now works from it."
        job = self._ensure_job(goal, request.origin)
        goal = self.ledger.activate(goal.id, goal_job_id=job.id, now=self.now())
        return f"Goal active from revision {goal.revision}. Next check: {_when(job.next_run_at)}."

    def _status(self, request: GoalRequest, _text: str) -> str:
        goal = self._mirror_job(self._owned_goal(request))
        return "\n".join(
            [
                _headline(goal),
                _charter_text(goal.confirmed_charter or goal.charter),
                self._schedule_line(goal),
                *self._work_lines(goal),
            ]
        )

    def _remove(self, request: GoalRequest, _text: str) -> str:
        goal = self._owned_goal(request)
        jobs = self.ledger.owned_job_ids(goal.id)
        self.ledger.remove(goal.id, owner_id=request.scope.owner_id, now=self.now())
        failed = [job_id for job_id in jobs if not self._drop_job(job_id)]
        if failed:
            return (
                f"Removed the goal, but could not remove job(s) {', '.join(failed)}. "
                "They will be disabled if they fire."
            )
        return "Removed the goal and its scheduled jobs. Its record is kept for status history."

    def _owned_goal(self, request: GoalRequest) -> Goal:
        goal = self.ledger.goal_for_conversation(request.scope)
        if goal is None:
            msg = "This chat has no goal. Start one with /goal <outcome>."
            raise GoalLedgerError(msg)
        if goal.scope.owner_id != request.scope.owner_id:
            msg = "Only the person who started this chat's goal can view or change it."
            raise GoalLedgerError(msg)
        return goal

    def _ensure_job(self, goal: Goal, origin: CronOrigin) -> CronJob:
        """Return the goal's recurring job, creating it at most once."""
        for job in self.store.list_jobs():
            if job.goal_id == goal.id and job.name == f"{_JOB_PREFIX}{goal.id}":
                return job
        charter = goal.confirmed_charter or goal.charter
        return self.store.create_job(
            prompt=f"Goal check for {goal.id}; handled by the goal controller.",
            schedule=CronSchedule.parse(charter.cadence),
            origin=origin,
            name=f"{_JOB_PREFIX}{goal.id}",
            deliver_to="thread",
            goal_id=goal.id,
            now=self.now(),
        )

    def _sync_cadence(self, goal: Goal) -> None:
        job = self._goal_job(goal)
        cadence = (goal.confirmed_charter or goal.charter).cadence
        if job is not None and job.schedule.display != cadence:
            schedule = CronSchedule.parse(cadence)
            self.store.edit_job(job.id, origin=job.origin, schedule=schedule, now=self.now())

    def _mirror_job(self, goal: Goal) -> Goal:
        """Reflect a pause or resume made through the existing cron `edit_job`."""
        if goal.state not in {"active", "paused"}:
            return goal
        job = self._goal_job(goal)
        if job is None and goal.state == "active":
            reason = "The goal's scheduled job is missing."
            return self.ledger.set_state(goal.id, "needs_attention", reason=reason, now=self.now())
        if job is not None and job.enabled == (goal.state == "paused"):
            state = "active" if job.enabled else "paused"
            return self.ledger.set_state(goal.id, state, now=self.now())
        return goal

    def _goal_job(self, goal: Goal) -> CronJob | None:
        return self.store.get_job(goal.goal_job_id) if goal.goal_job_id else None

    def _drop_job(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)
        if job is None:
            return True
        try:
            self.store.remove_job(job_id, origin=job.origin)
        except CronJobError, OSError:
            logger.warning("Could not remove goal job %s", job_id, exc_info=True)
            return False
        return True

    def _schedule_line(self, goal: Goal) -> str:
        job = self._goal_job(goal)
        if job is None:
            return "Schedule: none yet." if goal.goal_job_id is None else "Schedule: job missing."
        if not job.enabled:
            return f"Schedule: {job.schedule.display}, paused."
        return f"Schedule: {job.schedule.display}, next check {_when(job.next_run_at)}."

    def _work_lines(self, goal: Goal) -> list[str]:
        counts = Counter(item.status for item in self.ledger.list_items(goal.id))
        work = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
        decisions = len(self.ledger.list_decisions(goal.id, states={"pending"}))
        unsent = len(self.ledger.list_outbox(goal_id=goal.id, states={"pending", "unknown"}))
        lines = [f"Work: {work or 'none yet'}."]
        if decisions:
            lines.append(f"Waiting on you: {decisions} decision(s).")
        if unsent:
            lines.append(f"Updates not confirmed delivered: {unsent}.")
        lines.extend(self._kpi_lines(goal))
        return lines

    def _kpi_lines(self, goal: Goal) -> list[str]:
        latest = {obs.definition: obs for obs in self.ledger.list_observations(goal.id)}
        return [
            f"KPI {definition}: "
            + ("unmeasured" if obs.value is None else f"{obs.value} ({obs.source})")
            + f" as of {_when(obs.measured_at)}."
            for definition, obs in latest.items()
        ]


def _orphaned(goal: Goal, job: CronJob) -> bool:
    """Whether a goal job no longer belongs to a live goal.

    A job created just before a crash, while its goal still awaits activation,
    is kept so that retrying `/goal confirm` can adopt it.
    """
    if goal.state == "removed":
        return True
    return goal.goal_job_id is not None and goal.goal_job_id != job.id


def _apply_edits(charter: GoalCharter, text: str) -> GoalCharter:
    """Apply `field: value` lines from the owner to a charter."""
    updates: dict[str, object] = {}
    for line in filter(None, (line.strip() for line in text.splitlines())):
        name, sep, value = line.partition(":")
        field = _FIELDS.get(name.strip().lower())
        if not sep or field is None:
            msg = f"Could not read `{line}`. Use `field: value` with one of: {', '.join(_FIELDS)}."
            raise GoalLedgerError(msg)
        updates[field] = _field_value(field, value.strip())
    if not updates:
        msg = "Send /goal edit with at least one `field: value` line."
        raise GoalLedgerError(msg)
    return _validated(GoalCharter, {**charter.model_dump(), **updates})


def _field_value(field: str, value: str) -> object:
    cleared = value.lower() in _CLEAR
    if field in _LIST_FIELDS:
        return () if cleared else tuple(part for part in value.split(";") if part.strip())
    if cleared and field in {"outcome", "cadence"}:
        msg = f"`{field}` cannot be cleared."
        raise GoalLedgerError(msg)
    if field == "review_on" and not cleared:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            msg = "`review` must be a date like 2026-12-31."
            raise GoalLedgerError(msg) from exc
    return None if cleared else value


def _validated(model: type[GoalCharter], data: dict[str, object]) -> GoalCharter:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'charter'}: {error['msg']}"
            for error in exc.errors()
        )
        msg = f"That charter is not valid: {problems}."
        raise GoalLedgerError(msg) from exc


def _review(goal: Goal) -> str:
    parts = [_headline(goal), _charter_text(goal.charter)]
    missing = goal.charter.unresolved()
    if missing:
        names = ", ".join("progress" if name == "progress_test" else name for name in missing)
        parts.append(f"Still needed: {names}. Set them with /goal edit, for example:")
        parts.append("/goal edit sources: linear:SEC; github:org/repo")
    elif goal.revision != goal.confirmed_revision:
        parts.append(_SCOPE_NOTICE)
        parts.append(f"Send /goal confirm {goal.revision} to use this revision.")
    return "\n".join(parts)


def _headline(goal: Goal) -> str:
    state = goal.state.replace("_", " ")
    confirmed = goal.confirmed_revision
    if confirmed is not None and confirmed != goal.revision:
        return f"Goal ({state}), revision {goal.revision}; working from revision {confirmed}."
    if goal.attention_reason and goal.state == "needs_attention":
        return f"Goal ({state}), revision {goal.revision}: {goal.attention_reason}"
    return f"Goal ({state}), revision {goal.revision}."


def _charter_text(charter: GoalCharter) -> str:
    rows = {
        "Outcome": charter.outcome,
        "Sources": "; ".join(charter.sources) or "not set",
        "Excluded": "; ".join(charter.exclusions) or None,
        "Progress test": charter.progress_test or "not set",
        "KPI": charter.kpi,
        "Baseline": charter.baseline,
        "Horizon": charter.horizon,
        "Review on": charter.review_on.isoformat() if charter.review_on else None,
        "Cadence": charter.cadence,
        "Ask before": "; ".join(charter.decisions_required) or None,
    }
    return "\n".join(f"{label}: {value}" for label, value in rows.items() if value)


def _when(value: datetime | None) -> str:
    return (
        "not scheduled" if value is None else value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    )
