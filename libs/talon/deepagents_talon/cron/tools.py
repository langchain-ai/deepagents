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
    parse_until,
)

OriginFactory = Callable[[], CronOrigin]
_PAIRED_QUOTE_LENGTH = 2
_UPCOMING_COUNT = 3

SCHEDULE_HELP = """Schedule text. One of:
        `in 30m` / `in 2h` (one-shot, relative),
        `every 15m` / `every 6h` (recurring, relative),
        `at 2026-09-04 13:30 America/New_York` (one-shot, wall clock),
        `daily at 08:00 America/New_York` (recurring, same local time each day),
        `cron <minute> <hour> <day-of-month> <month> <day-of-week> <timezone>`
        (recurring, standard five-field cron evaluated in local time).
        Wall-clock and cron forms require an explicit IANA timezone name and
        keep firing at that local time across daylight-saving changes.
        Cron fields accept `*`, `5`, `1-5`, `*/15`, `9-17/2`, and comma lists;
        months `jan`-`dec` and weekdays `sun`-`sat` (0 or 7 is Sunday) work as
        names. Day-of-month also accepts `L` (last day), `LW` (last weekday),
        and `15W` (weekday nearest the 15th); day-of-week accepts `5L` (last
        Friday) and `2#1` (first Tuesday). Macros: `cron @hourly <timezone>`,
        and likewise `@daily`, `@weekly`, `@monthly`, `@yearly`. When both day
        fields are restricted (neither starts with `*`), a day matches if
        EITHER does, as in standard cron.
        Examples: every 15 minutes on June weekdays is
        `cron */15 * * jun mon-fri America/New_York`; weekends at noon is
        `cron 0 12 * * sat,sun America/New_York`; the last Friday of each
        month at 17:00 is `cron 0 17 * * 5L America/New_York`."""

UNTIL_HELP = """Optional last local time a recurring job may run, inclusive,
        as `YYYY-MM-DD HH:MM <timezone>`. The job deletes itself once it
        passes, whether or not it ran; a run missed while the host was down is
        dropped rather than delivered after it. For "the next three months",
        compute the date from `current_time`."""

_RESULT_HELP = """The job's `upcoming` field lists its next few run times; check them
    against what the user asked for and fix the schedule if they disagree."""

_CREATE_DESCRIPTION = f"""Schedule a background task that will later deliver to this conversation.

Args:
    prompt: Self-contained prompt to run when the job fires.
    schedule: {SCHEDULE_HELP}
    name: Optional human-readable label for the job.
    repeat_times: Optional cap for recurring schedules. `1` runs a recurring
        schedule once, at its next match.
    until: {UNTIL_HELP}

Returns:
    Created job details, or an error dictionary for invalid input.
    {_RESULT_HELP}
"""

_EDIT_DESCRIPTION = f"""Update one or more settings on a scheduled job from this conversation.

Args:
    job_id: Job id returned by the create or list tool.
    name: Optional replacement label.
    prompt: Optional replacement prompt.
    schedule: Optional replacement. {SCHEDULE_HELP}
    enabled: Optional enabled flag. Use `False` to pause, `True` to resume.
    repeat_times: Optional replacement repeat cap for recurring schedules.
    until: Optional replacement end time. {UNTIL_HELP}
        Pass an empty string to remove the end time.

Returns:
    Updated job details, or an error dictionary for invalid input.
    {_RESULT_HELP}
"""


class CronTools:
    """Conversation-scoped tools for managing cron jobs.

    Args:
        store: Persistent job store.
        origin: Callable returning the current conversation origin.
    """

    def __init__(self, *, store: CronJobStore, origin: OriginFactory) -> None:
        """Initialize tool helpers."""
        self.store = store
        self.origin = origin

    def create_job(
        self,
        *,
        prompt: str,
        schedule: str,
        name: str = "",
        repeat_times: int | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Create a scheduled job in the current conversation.

        Args:
            prompt: Prompt to run when the job fires.
            schedule: Schedule text, in any form `SCHEDULE_HELP` lists.
            name: Optional human-readable label.
            repeat_times: Optional cap for recurring schedules.
            until: Optional end time, in the form `UNTIL_HELP` describes.

        Returns:
            Created job as a JSON-compatible dictionary, with `upcoming` runs.
        """
        job = self.store.create_job(
            prompt=prompt,
            schedule=CronSchedule.parse(schedule),
            origin=self.origin(),
            name=name,
            repeat_times=repeat_times,
            until=parse_until(until) if until else None,
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
            until: Optional replacement end time, in the form `UNTIL_HELP`
                describes. An empty string removes the end time.

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
            until=parse_until(until) if until else None,
            clear_until=until == "",
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
    def create_job(
        prompt: str,
        schedule: str,
        name: str = "",
        repeat_times: int | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        try:
            return cron.create_job(
                prompt=prompt,
                schedule=schedule,
                name=name,
                repeat_times=repeat_times,
                until=_strip_optional_quotes(until),
            )
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)

    @tool
    def list_jobs() -> list[dict[str, Any]] | dict[str, str]:
        """List scheduled jobs created from this conversation.

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
    ) -> dict[str, Any]:
        try:
            return cron.edit_job(
                _strip_quotes(job_id),
                name=_strip_optional_quotes(name),
                prompt=_strip_optional_quotes(prompt),
                schedule=_strip_optional_quotes(schedule),
                enabled=enabled,
                repeat_times=repeat_times,
                until=_strip_optional_quotes(until),
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
