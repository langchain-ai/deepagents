"""Route chat replies to durable objective decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_talon.cron.jobs import CronJobStore, CronOrigin
from deepagents_talon.cron.journal import JobJournal, decision_text

if TYPE_CHECKING:
    from deepagents_talon.cron.jobs import CronJob
    from deepagents_talon.interfaces import ChannelMessage


def decision_reply(
    store: CronJobStore | None, message: ChannelMessage, provider: str | None
) -> str | None:
    """Route explicit decision answers without consuming ordinary chat."""
    text = message.text.strip()
    intent = text.casefold().rstrip("?.!")
    listing = intent == "what needs me"
    address, separator, answer = text.partition(":")
    explicit = address.casefold().startswith("answer ") and bool(separator)
    quoted = message.metadata.get("quoted_message_id") or message.metadata.get(
        "reply_to_message_id"
    )
    reply_to = quoted if isinstance(quoted, str) else None
    if not (
        listing or explicit or reply_to or intent in {"yes", "no", "approve", "decline", "approved"}
    ):
        return None
    if not isinstance(store, CronJobStore) or not message.sender_id:
        return None
    origin = CronOrigin(
        channel=provider, conversation_id=message.conversation_id, sender_id=message.sender_id
    )
    journal = JobJournal(store.cron_dir)
    pending = [
        (job, *question)
        for job in store.list_jobs(origin=origin)
        if job.mode == "objective" and job.enabled
        if (question := journal.open_decision(job, reply_to if not listing else None))
    ]
    if listing:
        return _choices(pending) or "No open decisions in this conversation."
    return _apply_answer(
        store,
        journal,
        pending,
        address[7:].strip() if explicit else None,
        answer.strip() if explicit else text,
    )


def _apply_answer(
    store: CronJobStore,
    journal: JobJournal,
    pending: list[tuple[CronJob, str, str]],
    job_id: str | None,
    answer: str,
) -> str | None:
    if job_id is not None:
        pending = [item for item in pending if item[0].id == job_id]
        if not pending or not answer.strip():
            return "Use an open job ID and a nonempty answer."
    elif not pending:
        return None
    if len(pending) != 1:
        return f"Which decision are you answering?\n{_choices(pending)}"
    job, question, _ = pending[0]
    if not journal.resolve_decision(
        job,
        question,
        answer,
        wake=lambda: store.wake_for_answer(job),
    ):
        return "That decision changed. Check what needs you again."
    return f"Answer recorded for {job.name}."


def _choices(pending: list[tuple[CronJob, str, str]]) -> str:
    return "\n".join(decision_text(job.name, job.id, q, unlocks) for job, q, unlocks in pending)
