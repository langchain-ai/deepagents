"""Agent-facing cron job tool helpers.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, tool

from deepagents_talon.cron.jobs import (
    CronJob,
    CronJobStore,
    CronOrigin,
    CronSchedule,
    DeliverTo,
    parse_until,
)

OriginFactory = Callable[[], CronOrigin]
_PAIRED_QUOTE_LENGTH = 2
_UPCOMING_COUNT = 3

SCHEDULE_HELP = """Use `in 30m`, `every 6h`, `at YYYY-MM-DD HH:MM TZ`, `daily at HH:MM TZ`,
or `cron <five fields or @macro> TZ`. Use an IANA timezone such as America/New_York for TZ.
When cron restricts both day-of-month and day-of-week, either can match.
For first Tuesday or last Friday, use `*` for day-of-month and `2#1` or `5L` for day-of-week."""

UNTIL_HELP = """For recurring jobs, repeat_times caps runs and until sets an inclusive deadline
(`YYYY-MM-DD HH:MM TZ`). Use target_at in the same format for a soft target.
A missed soft target is reported and work continues."""

DELIVER_TO_HELP = """In Slack channel threads and public Discord threads, prefer channel delivery
for later days and thread delivery for short follow-ups."""

SKILLS_HELP = "Pin up to 8 installed skills by name."

_JOB_HELP = f"""{SCHEDULE_HELP}
{UNTIL_HELP}
{DELIVER_TO_HELP}
{SKILLS_HELP}
Check the returned `upcoming` run times and fix the schedule if they disagree with the request."""

_CREATE_DESCRIPTION = f"""Schedule a self-contained prompt to run later in this conversation.
{_JOB_HELP}"""

_EDIT_DESCRIPTION = f"""Edit a job in this conversation by ID from create_job or list_jobs.
Supply only changes. Set enabled to false to pause or true to resume.
Clear until with an empty string and unpin all skills with an empty list.
{_JOB_HELP}"""


class CronTools:
    """Tools scoped to Slack channels, public Discord parent channels, or conversations.

    Args:
        store: Persistent job store.
        origin: Callable returning the current conversation origin.
    """

    def __init__(self, *, store: CronJobStore, origin: OriginFactory) -> None:
        """Initialize tool helpers."""
        self.store = store
        self.origin = origin

    def create_job(  # noqa: PLR0913  # agent tool exposes optional job fields
        self,
        *,
        prompt: str,
        schedule: str,
        name: str = "",
        repeat_times: int | None = None,
        until: str | None = None,
        deliver_to: DeliverTo = "channel",
        skills: list[str] | None = None,
        target_at: str | None = None,
    ) -> dict[str, Any]:
        """Create a scheduled job in the current conversation.

        Args:
            prompt: Prompt to run when the job fires.
            schedule: Schedule text, in any form `SCHEDULE_HELP` lists.
            name: Optional human-readable label.
            repeat_times: Optional cap for recurring schedules.
            target_at: Soft target in the same format as until.
            until: Optional end time, in the form `UNTIL_HELP` describes.
            deliver_to: Where results post, as `DELIVER_TO_HELP` describes.
            skills: Skills to pin to each run, as `SKILLS_HELP` describes.

        Returns:
            Created job as a JSON-compatible dictionary, with `upcoming` runs.
        """
        job = self.store.create_job(
            prompt=prompt,
            schedule=CronSchedule.parse(schedule),
            origin=self.origin(),
            name=name,
            repeat_times=repeat_times,
            target_at=parse_until(target_at) if target_at else None,
            until=parse_until(until) if until else None,
            deliver_to=deliver_to,
            skills=skills or (),
        )
        return _tool_job(job, upcoming=True)

    def list_jobs(self) -> list[dict[str, Any]]:
        """List jobs in the current conversation.

        Returns:
            Scoped jobs as JSON-compatible dictionaries.
        """
        return [_tool_job(job) for job in self.store.list_jobs(origin=self.origin())]

    def edit_job(  # noqa: PLR0913  # agent tool exposes optional editable fields
        self,
        job_id: str,
        *,
        name: str | None = None,
        prompt: str | None = None,
        schedule: str | None = None,
        enabled: bool | None = None,
        repeat_times: int | None = None,
        until: str | None = None,
        deliver_to: DeliverTo | None = None,
        skills: list[str] | None = None,
        target_at: str | None = None,
    ) -> dict[str, Any]:
        """Edit a scheduled job in the current conversation.

        Args:
            job_id: Job identifier.
            name: Optional replacement label.
            prompt: Optional replacement prompt.
            schedule: Optional replacement schedule text, in any form
                `SCHEDULE_HELP` lists.
            enabled: Optional enabled flag.
            repeat_times: Optional replacement repeat cap for recurring jobs.
            target_at: Soft target in the same format as until.
            until: Optional replacement end time, in the form `UNTIL_HELP`
                describes. An empty string removes the end time.
            deliver_to: Optional replacement delivery target.
            skills: Optional replacement pinned skills; an empty list unpins all.

        Returns:
            Updated job as a JSON-compatible dictionary, with `upcoming` runs.
        """
        parsed = None if schedule is None else CronSchedule.parse(schedule)
        job = self.store.edit_job(
            job_id,
            origin=self.origin(),
            name=name,
            prompt=prompt,
            schedule=parsed,
            enabled=enabled,
            repeat_times=repeat_times,
            target_at=parse_until(target_at) if target_at else None,
            until=parse_until(until) if until else None,
            clear_target=target_at == "",
            clear_until=until == "",
            deliver_to=deliver_to,
            skills=skills,
        )
        return _tool_job(job, upcoming=True)

    def remove_job(self, job_id: str) -> dict[str, Any]:
        """Remove a scheduled job from the current conversation.

        Args:
            job_id: Job identifier.

        Returns:
            Removed job as a JSON-compatible dictionary.
        """
        return _tool_job(self.store.remove_job(job_id, origin=self.origin()))

    def as_langchain_tools(self) -> list[BaseTool]:
        """Return LangChain tools bound to this cron helper.

        Returns:
            Tools for creating, listing, editing, and removing scoped cron jobs.
        """
        return build_cron_tools(self)


def build_cron_tools(cron: CronTools) -> list[BaseTool]:
    """Build agent-facing cron management tools.

    The create and edit tools take their descriptions from shared constants so
    the schedule grammar is written once.

    Args:
        cron: Conversation-scoped cron helper.

    Returns:
        LangChain tools for cron job management.
    """

    @tool(description=_CREATE_DESCRIPTION)
    def create_job(  # noqa: PLR0913  # agent tool exposes optional job fields
        prompt: str,
        schedule: str,
        name: str = "",
        repeat_times: int | None = None,
        until: str | None = None,
        deliver_to: DeliverTo = "channel",
        skills: list[str] | None = None,
        *,
        target_at: str | None = None,
    ) -> dict[str, Any]:
        try:
            return cron.create_job(
                prompt=prompt,
                schedule=schedule,
                name=name,
                repeat_times=repeat_times,
                target_at=_strip_optional_quotes(target_at),
                until=_strip_optional_quotes(until),
                deliver_to=deliver_to,
                skills=skills,
            )
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)

    @tool
    def list_jobs() -> list[dict[str, Any]] | dict[str, str]:
        """List jobs in this Slack/public Discord channel, or this private conversation.

        Returns:
            Scoped job details, or an error dictionary when no origin is active.
        """
        try:
            return cron.list_jobs()
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)

    @tool(description=_EDIT_DESCRIPTION)
    def edit_job(  # noqa: PLR0913  # agent tool exposes optional editable fields
        job_id: str,
        name: str | None = None,
        prompt: str | None = None,
        schedule: str | None = None,
        *,
        enabled: bool | None = None,
        repeat_times: int | None = None,
        until: str | None = None,
        deliver_to: DeliverTo | None = None,
        skills: list[str] | None = None,
        target_at: str | None = None,
    ) -> dict[str, Any]:
        try:
            return cron.edit_job(
                _strip_quotes(job_id),
                name=_strip_optional_quotes(name),
                prompt=_strip_optional_quotes(prompt),
                schedule=_strip_optional_quotes(schedule),
                enabled=enabled,
                repeat_times=repeat_times,
                target_at=_strip_optional_quotes(target_at),
                until=_strip_optional_quotes(until),
                deliver_to=deliver_to,
                skills=skills,
            )
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)

    @tool
    def remove_job(job_id: str) -> dict[str, Any]:
        """Delete a scheduled job from this conversation.

        Args:
            job_id: Job id returned by the create or list tool.

        Returns:
            Removed job details, or an error dictionary when no scoped job matches.
        """
        try:
            return cron.remove_job(_strip_quotes(job_id))
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)

    return [create_job, list_jobs, edit_job, remove_job]


def _tool_job(job: CronJob, *, upcoming: bool = False) -> dict[str, Any]:
    data = job.to_wire()
    payload: dict[str, Any] = {
        "id": data["id"],
        "name": data["name"],
        "prompt": data["prompt"],
        "schedule": data["schedule"],
        "repeat": data["repeat"],
        "enabled": data["enabled"],
        "next_run_at": data["next_run_at"],
        "last_run_at": data["last_run_at"],
        "last_status": data["last_status"],
        "last_error": data["last_error"],
        "until": data["until"],
        **({"target_at": data["target_at"]} if data["target_at"] else {}),
        "deliver_to": data["deliver_to"],
        "skills": data["skills"],
    }
    if upcoming:
        payload["upcoming"] = [run.isoformat() for run in job.upcoming(_UPCOMING_COUNT)]
    return payload


def _tool_error(exc: Exception) -> dict[str, str]:
    return {"error": str(exc)}


def _strip_optional_quotes(value: str | None) -> str | None:
    if value is None:
        return None
    return _strip_quotes(value)


def _strip_quotes(value: str) -> str:
    stripped = value.strip()
    if (
        len(stripped) >= _PAIRED_QUOTE_LENGTH
        and stripped[0] == stripped[-1]
        and stripped[0] in {'"', "'"}
    ):
        return stripped[1:-1].strip()
    return stripped
