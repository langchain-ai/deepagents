"""Route chat replies to durable objective decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_talon.cron.jobs import CronJobStore, CronOrigin

if TYPE_CHECKING:
    from deepagents_talon.interfaces import ChannelMessage


async def decision_reply(
    store: CronJobStore | None, message: ChannelMessage, provider: str | None
) -> str | None:
    """Resume the owner's delivered question in its original conversation."""
    if not isinstance(store, CronJobStore) or not message.sender_id:
        return None
    origin = CronOrigin(
        channel=provider, conversation_id=message.conversation_id, sender_id=message.sender_id
    )
    root = message.metadata.get("thread_root_message_id")
    if (
        isinstance(root, str)
        and provider == "slack"
        and message.conversation_id.endswith(f":{root}")
    ):
        root = None
    reply_to = (
        message.metadata.get("quoted_message_id")
        or message.metadata.get("reply_to_message_id")
        or root
    )
    async with store.open_journal() as journal:
        pending = [
            (job, question)
            for job in store.list_jobs(origin=origin)
            if job.mode == "objective"
            and job.enabled
            and (
                not job.origin.account_id
                or job.origin.account_id == message.metadata.get("channel_account_id")
            )
            if (
                question := await journal.open_decision(
                    job, reply_to if isinstance(reply_to, str) else None
                )
            )
        ]
        if not pending:
            return None
        job, question = pending[0]
        if len(pending) > 1:
            if message.text.strip().casefold() == "cancel":
                for owner_job, prompt in pending:
                    await journal.reply_to_question(owner_job, prompt, "cancel", wake=lambda: False)
                return "Questions skipped. You can continue chatting."
            if prompt := await journal.focus_question(job, question):
                return prompt
        return await journal.reply_to_question(
            job, question, message.text, wake=lambda: store.wake_for_answer(job)
        )
