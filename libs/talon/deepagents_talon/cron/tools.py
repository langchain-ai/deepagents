"""Agent-facing cron job tool helpers.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, tool

from deepagents_talon.cron.jobs import CronJob, CronJobStore, CronOrigin, CronSchedule

OriginFactory = Callable[[], CronOrigin]
_PAIRED_QUOTE_LENGTH = 2


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
    ) -> dict[str, Any]:
        """Create a scheduled job in the current conversation.

        Args:
            prompt: Prompt to run when the job fires.
            schedule: Schedule text such as `in 30m` or `every 15m`.
            name: Optional human-readable label.
            repeat_times: Optional cap for recurring schedules.

        Returns:
            Created job as a JSON-compatible dictionary.
        """
        job = self.store.create_job(
            prompt=prompt,
            schedule=CronSchedule.parse(schedule),
            origin=self.origin(),
            name=name,
            repeat_times=repeat_times,
        )
        return _tool_job(job)

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
    ) -> dict[str, Any]:
        """Edit a scheduled job in the current conversation.

        Args:
            job_id: Job identifier.
            name: Optional replacement label.
            prompt: Optional replacement prompt.
            schedule: Optional replacement schedule text.
            enabled: Optional enabled flag.
            repeat_times: Optional replacement repeat cap for recurring jobs.

        Returns:
            Updated job as a JSON-compatible dictionary.
        """
        parsed = None if schedule is None else CronSchedule.parse(schedule)
        return _tool_job(
            self.store.edit_job(
                job_id,
                origin=self.origin(),
                name=name,
                prompt=prompt,
                schedule=parsed,
                enabled=enabled,
                repeat_times=repeat_times,
            ),
        )

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

    Args:
        cron: Conversation-scoped cron helper.

    Returns:
        LangChain tools for cron job management.
    """

    @tool
    def create_job(
        prompt: str,
        schedule: str,
        name: str = "",
        repeat_times: int | None = None,
    ) -> dict[str, Any]:
        """Schedule a background task that will later deliver to this conversation.

        Args:
            prompt: Self-contained prompt to run when the job fires.
            schedule: Schedule text such as `in 30m` or `every 15m`.
            name: Optional human-readable label for the job.
            repeat_times: Optional cap for recurring schedules.

        Returns:
            Created job details, or an error dictionary for invalid input.
        """
        try:
            return cron.create_job(
                prompt=prompt,
                schedule=schedule,
                name=name,
                repeat_times=repeat_times,
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

    @tool
    def edit_job(  # noqa: PLR0913  # agent tool exposes optional editable fields
        job_id: str,
        name: str | None = None,
        prompt: str | None = None,
        schedule: str | None = None,
        *,
        enabled: bool | None = None,
        repeat_times: int | None = None,
    ) -> dict[str, Any]:
        """Update one or more settings on a scheduled job from this conversation.

        Args:
            job_id: Job id returned by the create or list tool.
            name: Optional replacement label.
            prompt: Optional replacement prompt.
            schedule: Optional replacement schedule text.
            enabled: Optional enabled flag. Use `False` to pause, `True` to resume.
            repeat_times: Optional replacement repeat cap for recurring schedules.

        Returns:
            Updated job details, or an error dictionary for invalid input.
        """
        try:
            return cron.edit_job(
                _strip_quotes(job_id),
                name=_strip_optional_quotes(name),
                prompt=_strip_optional_quotes(prompt),
                schedule=_strip_optional_quotes(schedule),
                enabled=enabled,
                repeat_times=repeat_times,
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


def _tool_job(job: CronJob) -> dict[str, Any]:
    data = job.to_dict()
    return {
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
    }


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
