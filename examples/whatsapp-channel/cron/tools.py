"""Agent-facing cron tools.

The tools read the current chat's origin from a module-level ``ContextVar``
that ``main.py`` sets before dispatching the agent for each inbound message.
The agent therefore never needs to pass a ``chat_id`` — it's always the chat
the user is already in.
"""

from __future__ import annotations

import contextvars
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from cron.jobs import (
    create_job as _db_create_job,
    list_jobs_for_chat as _db_list_jobs_for_chat,
    remove_job as _db_remove_job,
    update_job as _db_update_job,
)

# Set by main.py before each agent dispatch.
#   {"chat_id": "...", "message_id": "..." or None}
# Default is an empty dict so tools can detect "no context" without LookupError.
origin_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "whatsapp_cron_origin_ctx", default={},
)


def _current_origin() -> dict[str, Any] | None:
    origin = origin_ctx.get()
    if not origin or not origin.get("chat_id"):
        return None
    return origin


def build_cron_tools(jobs_path: Path) -> list:
    """Return the three LangChain tools bound to *jobs_path*."""
    jobs_path = Path(jobs_path)

    @tool
    def create_job(
        prompt: str,
        schedule: str,
        name: str | None = None,
        repeat: int | None = None,
    ) -> dict[str, Any]:
        """Schedule a background task that will run later and send its result to this chat.

        The scheduled run is a fresh, isolated agent invocation — it does not
        see the current conversation. Write ``prompt`` so it is self-contained.

        Args:
            prompt: The prompt the scheduled run should execute. Must be self-contained.
            schedule: Either a duration for a one-shot (``"1m"``, ``"30m"``, ``"2h"``,
                ``"1d"``) or an interval prefix (``"every 1m"``, ``"every 15m"``,
                ``"every 2h"``, ``"every 1d"``). Any positive integer paired with
                ``m``/``h``/``d`` works; the minimum is ``1m``. Firing can be up
                to one scheduler tick late (default tick: 60s). Cron expressions
                and absolute timestamps are not supported.
            name: Optional short label for this job. Defaults to the first 50
                characters of ``prompt``.
            repeat: For interval schedules, how many times to run before the job
                is removed. Defaults to forever. Ignored for one-shot schedules
                (which always run exactly once).

        Returns:
            ``{"id", "name", "schedule_display", "next_run_at"}`` on success,
            or ``{"error": "..."}`` on invalid schedule / missing chat context.

        Examples:
            create_job(prompt="Check the build", schedule="every 1m")
            create_job(prompt="Summarize top 3 HN stories", schedule="every 6h")
            create_job(prompt="Remind me to take the bread out", schedule="45m")
        """
        origin = _current_origin()
        if origin is None:
            return {"error": "Cron tools must be called from within a chat context."}
        try:
            job = _db_create_job(
                jobs_path,
                prompt=prompt,
                schedule=schedule,
                origin=origin,
                name=name,
                repeat=repeat,
            )
        except ValueError as e:
            return {"error": str(e)}
        return {
            "id": job["id"],
            "name": job["name"],
            "schedule_display": job["schedule"]["display"],
            "next_run_at": job["next_run_at"],
        }

    @tool
    def list_jobs() -> list[dict[str, Any]] | dict[str, str]:
        """List scheduled jobs created from this chat.

        Returns the full stored payload for each job — ``id``, ``name``,
        ``prompt``, ``schedule``, ``repeat``, ``enabled``, ``next_run_at``,
        ``last_run_at``, ``last_status``, ``last_error``, ``created_at``, and
        ``origin``. Use ``edit_job`` to change settings on a specific job.
        """
        origin = _current_origin()
        if origin is None:
            return {"error": "Cron tools must be called from within a chat context."}
        return _db_list_jobs_for_chat(jobs_path, origin["chat_id"])

    @tool
    def edit_job(
        job_id: str,
        name: str | None = None,
        prompt: str | None = None,
        schedule: str | None = None,
        repeat: int | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Update one or more settings on an existing scheduled job.

        Only fields you pass are changed; omitted fields keep their current
        value. Only jobs scheduled from this chat can be edited.

        Args:
            job_id: The job id returned by ``create_job`` or ``list_jobs``.
            name: Replace the short label.
            prompt: Replace the prompt the scheduled run will execute. Must be
                self-contained — the run does not see the current conversation.
            schedule: Replace the schedule (``"1m"``, ``"30m"``, ``"every 2h"``,
                etc.). Same rules as ``create_job.schedule``; minimum is ``1m``.
                Resets ``next_run_at`` to the new first run.
            repeat: Cap on remaining runs for interval schedules. Pass a
                positive integer to cap, or ``0`` to clear the cap (run forever).
            enabled: Pass ``False`` to pause the job, ``True`` to resume.

        Returns:
            The updated full job payload on success, or ``{"error": "..."}``
            on invalid input, unknown id, or cross-chat access.
        """
        origin = _current_origin()
        if origin is None:
            return {"error": "Cron tools must be called from within a chat context."}
        if (
            name is None
            and prompt is None
            and schedule is None
            and repeat is None
            and enabled is None
        ):
            return {"error": "Pass at least one field to edit."}
        try:
            updated = _db_update_job(
                jobs_path,
                job_id,
                chat_id=origin["chat_id"],
                name=name,
                prompt=prompt,
                schedule=schedule,
                repeat=repeat,
                enabled=enabled,
            )
        except ValueError as e:
            return {"error": str(e)}
        if updated is None:
            return {"error": "not found"}
        return updated

    @tool
    def remove_job(job_id: str) -> dict[str, Any]:
        """Delete a scheduled job by id.

        Only jobs scheduled from this chat can be removed. The same "not found"
        response is returned for ids that don't exist and for ids that belong
        to other chats.
        """
        origin = _current_origin()
        if origin is None:
            return {"error": "Cron tools must be called from within a chat context."}
        removed = _db_remove_job(jobs_path, job_id, chat_id=origin["chat_id"])
        if removed:
            return {"removed": True, "id": job_id}
        return {"removed": False, "reason": "not found"}

    return [create_job, list_jobs, edit_job, remove_job]
