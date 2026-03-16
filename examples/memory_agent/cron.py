"""Sleep-time memory consolidation cron job.

Runs on a schedule (e.g., nightly) to process recent conversation threads
and consolidate learnings into global and per-user memory. This is the
"sleep-time compute" that makes the agent improve over time.

The primary unit of work is `consolidate_user` — one user, one invocation.
In production (LangGraph Cloud), this runs as a per-user cron job so each
user's consolidation is an independent, horizontally scalable invocation.

For local dev/testing, `run_all_users` is a convenience that discovers all
users and consolidates them (in parallel via thread pool).

Usage:
    # Single user (production pattern)
    python cron.py --user alice

    # All users (local dev / small scale)
    python cron.py --all

    # With a real scheduler (e.g., crontab), one job per user:
    # 0 3 * * * cd /path/to/memory_agent && python cron.py --user alice
    # 0 3 * * * cd /path/to/memory_agent && python cron.py --user bob
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from memory_agent.prompts import CRON_CONSOLIDATION_PROMPT

logger = logging.getLogger(__name__)

GLOBAL_MEMORY_NAMESPACE = ("memory", "global")
USER_MEMORY_NAMESPACE_PREFIX = ("memory", "users")
MEMORY_KEY = "system_prompt"
THREADS_NAMESPACE_PREFIX = ("threads",)


def _load_memory(store: BaseStore, namespace: tuple[str, ...], key: str) -> str:
    """Load a memory document from the store."""
    item = store.get(namespace, key)
    if item is None:
        return ""
    return item.value.get("content", "")


def _save_memory(
    store: BaseStore, namespace: tuple[str, ...], key: str, content: str
) -> None:
    """Save a memory document to the store."""
    store.put(namespace, key, {"content": content})


def _format_thread(thread: dict[str, Any]) -> str:
    """Format a conversation thread for the consolidation prompt."""
    messages = thread.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"**{role}**: {content}")
    return "\n".join(parts)


def collect_threads(
    store: BaseStore,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Collect recent conversation threads from the store.

    In production, this would query LangSmith traces or the checkpointer
    for recent threads. For this example, we read from a threads namespace.

    Args:
        store: The LangGraph store.
        user_id: If provided, only collect threads for this user.
    """
    namespace = THREADS_NAMESPACE_PREFIX
    if user_id:
        namespace = (*namespace, user_id)

    items = store.search(namespace, limit=100)
    return [item.value for item in items]


def consolidate_user(
    *,
    store: BaseStore,
    user_id: str,
    model: str = "anthropic:claude-sonnet-4-6",
    threads: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Consolidate memories for a single user.

    This is the primary unit of work — designed to run as one independent
    invocation per user. In production, each user gets their own cron
    trigger so consolidation scales horizontally.

    Reads the user's recent threads, analyzes them with an LLM, and
    updates both global memory and the user's personal memory.

    Args:
        store: The LangGraph store for reading/writing memories.
        user_id: The user whose threads to consolidate.
        model: Model to use for consolidation.
        threads: Pre-collected threads. If None, collects from store.

    Returns:
        Dict with `global_memory` and `user_memory` keys containing updated content.
    """
    if threads is None:
        threads = collect_threads(store, user_id)

    if not threads:
        logger.info("No threads to consolidate for user=%s", user_id)
        return {"global_memory": "", "user_memory": ""}

    current_global = _load_memory(store, GLOBAL_MEMORY_NAMESPACE, MEMORY_KEY)
    user_namespace = (*USER_MEMORY_NAMESPACE_PREFIX, user_id)
    current_user = _load_memory(store, user_namespace, MEMORY_KEY)

    formatted_threads = "\n\n---\n\n".join(_format_thread(t) for t in threads)

    user_message = (
        f"## Current Global Memory\n\n{current_global or '(empty)'}\n\n"
        f"## Current User Memory (user: {user_id})\n\n{current_user or '(empty)'}\n\n"
        f"## Recent Conversation Threads\n\n{formatted_threads}"
    )

    chat_model = init_chat_model(model)
    response = chat_model.invoke(
        [
            SystemMessage(content=CRON_CONSOLIDATION_PROMPT),
            HumanMessage(content=user_message),
        ]
    )

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        content = response.content if isinstance(response.content, str) else ""
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(content[start:end])
        else:
            logger.error("Failed to parse consolidation response")
            return {"global_memory": current_global, "user_memory": current_user}

    new_global = result.get("global_memory", current_global)
    new_user = result.get("user_memory", current_user)

    _save_memory(store, GLOBAL_MEMORY_NAMESPACE, MEMORY_KEY, new_global)
    _save_memory(store, user_namespace, MEMORY_KEY, new_user)

    logger.info(
        "Consolidated memories for user=%s (global: %d chars, user: %d chars)",
        user_id,
        len(new_global),
        len(new_user),
    )

    return {"global_memory": new_global, "user_memory": new_user}


def run_all_users(
    *,
    store: BaseStore | None = None,
    model: str = "anthropic:claude-sonnet-4-6",
    max_workers: int = 5,
) -> dict[str, dict[str, str]]:
    """Consolidate all users in parallel (convenience for local dev).

    For local dev and small-scale deployments. Discovers all users from
    the store and runs `consolidate_user` for each via a thread pool.

    For large-scale production, use per-user cron invocations instead —
    each user gets their own scheduled job that calls `consolidate_user`
    directly, so consolidation scales horizontally across workers.

    Args:
        store: The LangGraph store. Defaults to InMemoryStore.
        model: Model to use for consolidation.
        max_workers: Max parallel user consolidations.

    Returns:
        Dict mapping user_id to their consolidation results.
    """
    if store is None:
        store = InMemoryStore()

    namespaces = store.list_namespaces(prefix=USER_MEMORY_NAMESPACE_PREFIX, max_depth=3)
    user_ids = list({ns[2] for ns in namespaces if len(ns) > 2})

    if not user_ids:
        user_ids = ["default"]

    if len(user_ids) == 1:
        return {
            user_ids[0]: consolidate_user(store=store, user_id=user_ids[0], model=model)
        }

    logger.info(
        "Consolidating %d users with max_workers=%d", len(user_ids), max_workers
    )
    results: dict[str, dict[str, str]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                consolidate_user, store=store, user_id=uid, model=model
            ): uid
            for uid in user_ids
        }
        for future in as_completed(futures):
            uid = futures[future]
            try:
                results[uid] = future.result()
            except Exception:
                logger.exception("Failed to consolidate user=%s", uid)
                results[uid] = {"global_memory": "", "user_memory": ""}

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Run memory consolidation")
    parser.add_argument(
        "--user", type=str, default=None, help="Consolidate a single user"
    )
    parser.add_argument(
        "--all", action="store_true", help="Consolidate all users (local dev)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Max parallel consolidations (with --all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic:claude-sonnet-4-6",
        help="Model to use",
    )
    args = parser.parse_args()

    if args.user:
        store = InMemoryStore()
        consolidate_user(store=store, user_id=args.user, model=args.model)
    elif args.all:
        run_all_users(model=args.model, max_workers=args.max_workers)
    else:
        parser.error("Specify --user <id> or --all")
