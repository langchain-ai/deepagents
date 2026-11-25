"""Route chat replies to durable objective decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_talon.cron.jobs import CronJobStore, CronOrigin
from deepagents_talon.cron.journal import ACTION_CHOICES, JobJournal, decision_text

if TYPE_CHECKING:
    from deepagents_talon.cron.jobs import CronJob
    from deepagents_talon.interfaces import ChannelMessage


async def decision_reply(
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
    root = message.metadata.get("thread_root_message_id")
    reply_to = quoted if isinstance(quoted, str) else None
    if not (
        listing
        or explicit
        or reply_to
        or root
        or intent in {"yes", "no", "approve", "decline", "approved"}
    ):
        return None
    if not isinstance(store, CronJobStore) or not message.sender_id:
        return None
    origin = CronOrigin(
        channel=provider, conversation_id=message.conversation_id, sender_id=message.sender_id
    )
    account = message.metadata.get("channel_account_id")
    private = (provider == "slack" and message.metadata.get("is_dm") is True) or (
        provider == "discord" and message.metadata.get("private_owner_dm") is True
    )
    jobs = [
        job
        for job in store.list_jobs(origin=origin)
        if not job.origin.account_id or job.origin.account_id == account
    ]
    if (
        private
        and isinstance(account, str)
        and account
        and (listing or explicit or not (reply_to or root))
    ):
        jobs.extend(
            job
            for job in store.list_jobs()
            if job.origin.channel == provider
            and job.origin.sender_id == message.sender_id
            and job.origin.account_id == account
            and job.origin.conversation_id != origin.conversation_id
        )
    async with store.open_journal() as journal:
        pending = [
            (job, *question)
            for job in jobs
            if job.mode == "objective" and job.enabled
            if (
                question := await journal.open_decision(
                    job, None if listing or explicit else reply_to
                )
            )
        ]
        if isinstance(root, str) and not (listing or explicit or reply_to):
            addressed = [item for item in pending if await journal.open_decision(item[0], root)]
            same_thread = provider == "slack" and message.conversation_id.endswith(f":{root}")
            if (
                addressed
                or not same_thread
                or intent not in {"yes", "no", "approve", "decline", "approved"}
            ):
                pending = addressed
        if listing:
            return _choices(pending) or "No open decisions in this conversation."
        return await _apply_answer(
            store,
            journal,
            pending,
            address[7:].strip() if explicit else None,
            answer.strip() if explicit else text,
        )


async def _apply_answer(
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
    job, question, unlocks = pending[0]
    if not await store.answer_decision(job, journal, question, answer):
        if unlocks == ACTION_CHOICES:
            return ACTION_CHOICES
        return "That decision changed. Check what needs you again."
    return f"Answer recorded for {job.name}."


def _choices(pending: list[tuple[CronJob, str, str]]) -> str:
    return "\n".join(decision_text(job.name, job.id, q, unlocks) for job, q, unlocks in pending)
