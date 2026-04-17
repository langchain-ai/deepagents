"""Cron tick loop and single-job runner for the WhatsApp channel example."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage

from cron.jobs import (
    advance_next_run,
    get_due_jobs,
    mark_job_run,
)

logger = logging.getLogger(__name__)

SILENT_MARKER = "[SILENT]"

_CRON_HINT = (
    "[SYSTEM: You are running as a scheduled cron job. "
    "DELIVERY: Your final response will be automatically delivered to the "
    "user — do not call tools that send messages yourself. "
    "SILENT: If there is genuinely nothing new to report, respond with "
    f"exactly \"{SILENT_MARKER}\" (nothing else) to suppress delivery.]\n\n"
)


def _build_prompt(user_prompt: str) -> str:
    """Prepend the cron-execution hint to *user_prompt*."""
    return _CRON_HINT + (user_prompt or "")


def _extract_final_text(agent_output: Any) -> str:
    """Extract the final ``AIMessage`` text from an agent output dict.

    Mirrors the extraction logic in ``main.py`` so cron-run results look like
    live-chat responses.
    """
    if not agent_output:
        return ""
    messages = agent_output.get("messages", []) if isinstance(agent_output, dict) else []
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if not content:
            continue
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, str):
                    return block
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
    return ""


async def _run_job(
    agent: Any,
    job: dict[str, Any],
) -> tuple[bool, str, str | None]:
    """Run the agent against *job*'s prompt in a fresh session.

    Returns ``(success, final_text, error_message)``. On success ``error_message``
    is ``None``; on failure ``final_text`` is ``""`` and ``error_message`` is
    a short string combining the exception class and message.
    """
    try:
        prompt = _build_prompt(job.get("prompt", ""))
        output = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        return True, _extract_final_text(output), None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.exception("cron: job %s failed: %s", job.get("id"), err)
        return False, "", err


async def _deliver_success(
    adapter: Any,
    job: dict[str, Any],
    text: str,
) -> str | None:
    """Send *text* to the job's origin chat. Returns an error string on failure."""
    if text.strip().upper() == SILENT_MARKER:
        logger.info("cron: job %s returned %s — skipping delivery",
                    job.get("id"), SILENT_MARKER)
        return None
    chat_id = job["origin"]["chat_id"]
    reply_to = job["origin"].get("message_id")
    result = await adapter.send(chat_id, text, reply_to=reply_to)
    if not getattr(result, "success", True):
        err = getattr(result, "error", "unknown") or "unknown"
        logger.warning("cron: delivery to %s failed: %s", chat_id, err)
        return err
    return None


async def _deliver_failure(
    adapter: Any,
    job: dict[str, Any],
    error: str,
) -> None:
    """Send a user-visible error notice for a failed job."""
    chat_id = job["origin"]["chat_id"]
    text = f"⚠️ Scheduled task '{job.get('name', job['id'])}' failed: {error}"
    try:
        await adapter.send(chat_id, text, reply_to=None)
    except Exception as e:
        logger.warning("cron: failure-notice delivery to %s failed: %s", chat_id, e)


async def _tick_once(
    jobs_path: Path,
    adapter: Any,
    agent: Any,
    chat_locks: dict[str, asyncio.Lock],
) -> int:
    """Run one pass over due jobs. Returns how many were executed."""
    due = get_due_jobs(jobs_path)
    if not due:
        return 0

    executed = 0
    for job in due:
        job_id = job["id"]
        chat_id = job.get("origin", {}).get("chat_id")
        if not chat_id:
            logger.warning("cron: job %s has no origin.chat_id; skipping", job_id)
            continue

        # Advance interval jobs BEFORE running so a crash mid-run doesn't re-fire.
        advance_next_run(jobs_path, job_id)

        lock = chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            run_coro = _run_job(agent, job)
            try:
                success, text, error = await asyncio.shield(run_coro)
            except asyncio.CancelledError:
                # Ticker is being shut down; the shielded run may still be completing,
                # but we don't wait for it here.
                raise
            except Exception as e:
                success, text, error = False, "", f"{type(e).__name__}: {e}"

            delivery_error: str | None = None
            if success and text:
                delivery_error = await _deliver_success(adapter, job, text)
            elif not success:
                await _deliver_failure(adapter, job, error or "unknown error")

            # If the run succeeded but delivery failed, surface the delivery
            # error through last_error so the user / logs can see it.
            final_error = error
            if success and delivery_error:
                final_error = f"delivery failed: {delivery_error}"
                mark_job_run(jobs_path, job_id, success=False, error=final_error)
            else:
                mark_job_run(jobs_path, job_id, success=success, error=error)
            executed += 1

    return executed


def start_ticker(
    jobs_path: Path,
    adapter: Any,
    agent: Any,
    chat_locks: dict[str, asyncio.Lock],
    *,
    tick_interval: float = 60.0,
) -> asyncio.Task:
    """Launch the cron ticker as an asyncio task on the running loop.

    The returned task runs ``_tick_once`` every *tick_interval* seconds. Any
    exception raised by a tick (corrupt jobs file, unexpected error) is logged
    and swallowed so the loop keeps running. Cancel the task on shutdown.
    """
    async def _loop() -> None:
        logger.info("cron: ticker started (interval=%ss, path=%s)",
                    tick_interval, jobs_path)
        try:
            while True:
                try:
                    await _tick_once(jobs_path, adapter, agent, chat_locks)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("cron: tick failed")
                await asyncio.sleep(tick_interval)
        except asyncio.CancelledError:
            logger.info("cron: ticker cancelled")
            raise

    return asyncio.create_task(_loop(), name="whatsapp-cron-ticker")
