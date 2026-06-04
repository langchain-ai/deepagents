"""Agent-facing cron job tool helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from deepagents_talon.cron.jobs import CronJob, CronJobStore, CronOrigin, CronSchedule

OriginFactory = Callable[[], CronOrigin]


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
