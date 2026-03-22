"""Thread management using LangGraph's built-in checkpoint persistence."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import aiosqlite
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from deepagents_cli.output import OutputFormat

logger = logging.getLogger(__name__)

_aiosqlite_patched = False
_jsonplus_serializer: JsonPlusSerializer | None = None
_message_count_cache: dict[str, tuple[str | None, int]] = {}
_MAX_MESSAGE_COUNT_CACHE = 4096
_initial_prompt_cache: dict[str, tuple[str | None, str | None]] = {}
_MAX_INITIAL_PROMPT_CACHE = 4096
_recent_threads_cache: dict[tuple[str | None, int], list[ThreadInfo]] = {}
_MAX_RECENT_THREADS_CACHE_KEYS = 16


def _patch_aiosqlite() -> None:
    """Patch aiosqlite.Connection with `is_alive()` if missing.

    Required by langgraph-checkpoint>=2.1.0.
    See: https://github.com/langchain-ai/langgraph/issues/6583
    """
    global _aiosqlite_patched  # noqa: PLW0603  # Module-level flag requires global statement
    if _aiosqlite_patched:
        return

    import aiosqlite as _aiosqlite

    if not hasattr(_aiosqlite.Connection, "is_alive"):

        def _is_alive(self: _aiosqlite.Connection) -> bool:
            """Check if the connection is still alive.

            Returns:
                True if connection is alive, False otherwise.
            """
            return bool(self._running and self._connection is not None)

        # Dynamically adding a method to aiosqlite.Connection at runtime.
        # Type checkers can't understand this monkey-patch, so we suppress the
        # "attr-defined" error that would otherwise be raised.
        _aiosqlite.Connection.is_alive = _is_alive  # type: ignore[attr-defined]

    _aiosqlite_patched = True


@asynccontextmanager
async def _connect() -> AsyncIterator[aiosqlite.Connection]:
    """Import aiosqlite, apply the compatibility patch, and connect.

    Centralizes the deferred import + patch + connect sequence used by every
    database function in this module.

    Yields:
        An open aiosqlite connection to the sessions database.
    """
    import aiosqlite as _aiosqlite

    _patch_aiosqlite()

    async with _aiosqlite.connect(str(get_db_path()), timeout=30.0) as conn:
        yield conn


class ThreadInfo(TypedDict):
    """Thread metadata returned by `list_threads`."""

    thread_id: str
    """Unique identifier for the thread."""

    agent_name: str | None
    """Name of the agent that owns the thread."""

    updated_at: str | None
    """ISO timestamp of the last update."""

    created_at: NotRequired[str | None]
    """ISO timestamp of thread creation (earliest checkpoint)."""

    git_branch: NotRequired[str | None]
    """Git branch active when the thread was created."""

    initial_prompt: NotRequired[str | None]
    """First human message in the thread."""

    message_count: NotRequired[int]
    """Number of messages in the thread."""

    latest_checkpoint_id: NotRequired[str | None]
    """Most recent checkpoint ID for cache invalidation."""

    cwd: NotRequired[str | None]
    """Working directory where the thread was last used."""


class _CheckpointSummary(NamedTuple):
    """Structured data extracted from a thread's latest checkpoint."""

    message_count: int
    """Number of messages in the latest checkpoint."""

    initial_prompt: str | None
    """First human prompt recovered from the latest checkpoint."""


def format_timestamp(iso_timestamp: str | None) -> str:
    """Format ISO timestamp for display (e.g., 'Dec 30, 6:10pm').

    Args:
        iso_timestamp: ISO 8601 timestamp string, or `None`.

    Returns:
        Formatted timestamp string or empty string if invalid.
    """
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp).astimezone()
        return (
            dt.strftime("%b %d, %-I:%M%p")
            .lower()
            .replace("am", "am")
            .replace("pm", "pm")
        )
    except (ValueError, TypeError):
        logger.debug(
            "Failed to parse timestamp %r; displaying as blank",
            iso_timestamp,
            exc_info=True,
        )
        return ""


def format_relative_timestamp(iso_timestamp: str | None) -> str:
    """Format ISO timestamp as relative time (e.g., '5m ago', '2h ago').

    Args:
        iso_timestamp: ISO 8601 timestamp string, or `None`.

    Returns:
        Relative time string or empty string if invalid.
    """
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp).astimezone()
    except (ValueError, TypeError):
        logger.debug(
            "Failed to parse timestamp %r; displaying as blank",
            iso_timestamp,
            exc_info=True,
        )
        return ""

    delta = datetime.now(tz=dt.tzinfo) - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:  # noqa: PLR2004
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:  # noqa: PLR2004
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:  # noqa: PLR2004
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:  # noqa: PLR2004
        return f"{days}d ago"
    months = days // 30
    if months < 12:  # noqa: PLR2004
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"


def format_path(path: str | None) -> str:
    """Format a filesystem path for display.

    Paths under the user's home directory are shown relative to `~`.
    All other paths are returned as-is.

    Args:
        path: Absolute filesystem path, or `None`.

    Returns:
        Formatted path string, or empty string if path is falsy.
    """
    if not path:
        return ""
    try:
        home = str(Path.home())
        if path == home:
            return "~"
        prefix = home + "/"
        if path.startswith(prefix):
            return "~/" + path[len(prefix) :]
    except (RuntimeError, KeyError, OSError):
        logger.debug(
            "Could not resolve home directory for path formatting", exc_info=True
        )
        return path
    else:
        return path


def get_db_path() -> Path:
    """Get path to global database.

    Returns:
        Path to the SQLite database file.
    """
    db_dir = Path.home() / ".deepagents"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sessions.db"


def generate_thread_id() -> str:
    """Generate a new thread ID as a full UUID7 string.

    Returns:
        UUID7 string (time-ordered for natural sort by creation time).
    """
    from uuid_utils import uuid7

    return str(uuid7())


async def _table_exists(conn: aiosqlite.Connection, table: str) -> bool:
    """Check if a table exists in the database.

    Returns:
        True if table exists, False otherwise.
    """
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    async with conn.execute(query, (table,)) as cursor:
        return await cursor.fetchone() is not None


async def list_threads(
    agent_name: str | None = None,
    limit: int = 20,
    include_message_count: bool = False,
    sort_by: str = "updated",
    branch: str | None = None,
) -> list[ThreadInfo]:
    """List threads from checkpoints table.

    Args:
        agent_name: Optional filter by agent name.
        limit: Maximum number of threads to return.
        include_message_count: Whether to include message counts.
        sort_by: Sort field — `"updated"` or `"created"`.
        branch: Optional filter by git branch name.

    Returns:
        List of `ThreadInfo` dicts with `thread_id`, `agent_name`,
            `updated_at`, `created_at`, `latest_checkpoint_id`, `git_branch`,
            `cwd`, and optionally `message_count`.

    Raises:
        ValueError: If `sort_by` is not `"updated"` or `"created"`.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []

        if sort_by not in {"updated", "created"}:
            msg = f"Invalid sort_by {sort_by!r}; expected 'updated' or 'created'"
            raise ValueError(msg)
        order_col = "created_at" if sort_by == "created" else "updated_at"

        where_clauses: list[str] = []
        params_list: list[str | int] = []

        if agent_name:
            where_clauses.append("json_extract(metadata, '$.agent_name') = ?")
            params_list.append(agent_name)
        if branch:
            where_clauses.append("json_extract(metadata, '$.git_branch') = ?")
            params_list.append(branch)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
            SELECT thread_id,
                   json_extract(metadata, '$.agent_name') as agent_name,
                   MAX(json_extract(metadata, '$.updated_at')) as updated_at,
                   MAX(checkpoint_id) as latest_checkpoint_id,
                   MIN(json_extract(metadata, '$.updated_at')) as created_at,
                   MAX(json_extract(metadata, '$.git_branch')) as git_branch,
                   MAX(json_extract(metadata, '$.cwd')) as cwd
            FROM checkpoints
            {where_sql}
            GROUP BY thread_id
            ORDER BY {order_col} DESC
            LIMIT ?
        """  # noqa: S608  # where_sql/order_col derived from controlled internal values; user values use ? placeholders
        params: tuple = (*params_list, limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            threads: list[ThreadInfo] = [
                ThreadInfo(
                    thread_id=r[0],
                    agent_name=r[1],
                    updated_at=r[2],
                    latest_checkpoint_id=r[3],
                    created_at=r[4],
                    git_branch=r[5],
                    cwd=r[6],
                )
                for r in rows
            ]

        # Fetch message counts if requested
        if include_message_count and threads:
            await _populate_message_counts(conn, threads)

        # Only cache unfiltered results so the thread selector modal
        # doesn't receive branch-filtered or differently-sorted data.
        if sort_by == "updated" and branch is None:
            _cache_recent_threads(agent_name, limit, threads)
        return threads


async def populate_thread_message_counts(threads: list[ThreadInfo]) -> list[ThreadInfo]:
    """Populate `message_count` for an existing thread list.

    This is used by the `/threads` modal to render rows quickly, then backfill
    counts in the background without issuing a second thread-list query.

    Args:
        threads: Thread rows to enrich in place.

    Returns:
        The same list object with `message_count` values populated.
    """
    if not threads:
        return threads

    async with _connect() as conn:
        await _populate_message_counts(conn, threads)
    return threads


async def populate_thread_checkpoint_details(
    threads: list[ThreadInfo],
    *,
    include_message_count: bool = True,
    include_initial_prompt: bool = True,
) -> list[ThreadInfo]:
    """Populate checkpoint-derived fields for an existing thread list.

    This is used by the `/threads` modal to enrich rows in one background pass,
    so the latest checkpoint is fetched and deserialized at most once per row.

    Args:
        threads: Thread rows to enrich in place.
        include_message_count: Whether to populate `message_count`.
        include_initial_prompt: Whether to populate `initial_prompt`.

    Returns:
        The same list object with missing checkpoint-derived fields populated.
    """
    if not threads or (not include_message_count and not include_initial_prompt):
        return threads

    async with _connect() as conn:
        await _populate_checkpoint_fields(
            conn,
            threads,
            include_message_count=include_message_count,
            include_initial_prompt=include_initial_prompt,
        )
    return threads


async def prewarm_thread_message_counts(limit: int | None = None) -> None:
    """Prewarm thread selector cache for faster `/threads` open.

    Fetches a bounded list of recent threads and populates checkpoint-derived
    fields for currently visible columns into the in-memory cache. Intended to
    run in a background worker during app startup.

    Args:
        limit: Maximum threads to prewarm. Uses `get_thread_limit()` when `None`.
    """
    thread_limit = limit if limit is not None else get_thread_limit()
    if thread_limit < 1:
        return

    try:
        from deepagents_cli.model_config import load_thread_config

        cfg = load_thread_config()
        threads = await list_threads(limit=thread_limit, include_message_count=False)
        if threads:
            await populate_thread_checkpoint_details(
                threads,
                include_message_count=cfg.columns.get("messages", False),
                include_initial_prompt=cfg.columns.get("initial_prompt", False),
            )
        _cache_recent_threads(None, thread_limit, threads)
    except (OSError, sqlite3.Error):
        logger.debug("Could not prewarm thread selector cache", exc_info=True)
    except Exception:
        logger.warning(
            "Unexpected error while prewarming thread selector cache",
            exc_info=True,
        )


def get_cached_threads(
    agent_name: str | None = None,
    limit: int | None = None,
) -> list[ThreadInfo] | None:
    """Get cached recent threads, if available.

    Args:
        agent_name: Optional agent-name filter key.
        limit: Maximum rows requested. Uses `get_thread_limit()` when `None`.

    Returns:
        Copy of cached rows when available, otherwise `None`.
    """

    def _copy_with_cached_counts(rows: list[ThreadInfo]) -> list[ThreadInfo]:
        copied_rows = _copy_threads(rows)
        apply_cached_thread_message_counts(copied_rows)
        apply_cached_thread_initial_prompts(copied_rows)
        return copied_rows

    thread_limit = limit if limit is not None else get_thread_limit()
    if thread_limit < 1:
        return None

    exact = _recent_threads_cache.get((agent_name, thread_limit))
    if exact is not None:
        return _copy_with_cached_counts(exact)

    best_key: tuple[str | None, int] | None = None
    for key in _recent_threads_cache:
        cache_agent, cache_limit = key
        if cache_agent != agent_name or cache_limit < thread_limit:
            continue
        if best_key is None or cache_limit < best_key[1]:
            best_key = key

    if best_key is None:
        return None

    return _copy_with_cached_counts(_recent_threads_cache[best_key][:thread_limit])


def apply_cached_thread_message_counts(threads: list[ThreadInfo]) -> int:
    """Apply cached message counts onto thread rows when freshness matches.

    Args:
        threads: Thread rows to mutate in place.

    Returns:
        Number of rows that were populated from cache.
    """
    populated = 0
    for thread in threads:
        if "message_count" in thread:
            continue
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        cached = _message_count_cache.get(thread_id)
        if cached is None or cached[0] != freshness:
            continue
        thread["message_count"] = cached[1]
        populated += 1
    return populated


def apply_cached_thread_initial_prompts(threads: list[ThreadInfo]) -> int:
    """Apply cached initial prompts onto thread rows when freshness matches.

    Args:
        threads: Thread rows to mutate in place.

    Returns:
        Number of rows that were populated from cache.
    """
    populated = 0
    for thread in threads:
        if "initial_prompt" in thread:
            continue
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        cached = _initial_prompt_cache.get(thread_id)
        if cached is None or cached[0] != freshness:
            continue
        thread["initial_prompt"] = cached[1]
        populated += 1
    return populated


async def _populate_message_counts(
    conn: aiosqlite.Connection,
    threads: list[ThreadInfo],
) -> None:
    """Fill `message_count` on thread rows with cache-aware lookup."""
    await _populate_checkpoint_fields(
        conn,
        threads,
        include_message_count=True,
        include_initial_prompt=False,
    )


async def _get_jsonplus_serializer() -> JsonPlusSerializer:
    """Return a cached JsonPlus serializer, loading it off the UI loop."""
    global _jsonplus_serializer  # noqa: PLW0603  # Module-level cache requires global statement
    if _jsonplus_serializer is not None:
        return _jsonplus_serializer

    loop = asyncio.get_running_loop()
    _jsonplus_serializer = await loop.run_in_executor(None, _create_jsonplus_serializer)
    return _jsonplus_serializer


def _create_jsonplus_serializer() -> JsonPlusSerializer:
    """Import and create a JsonPlus serializer.

    Returns:
        A ready `JsonPlusSerializer` instance.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer()


def _cache_message_count(thread_id: str, freshness: str | None, count: int) -> None:
    """Cache a thread's message count with a freshness token."""
    if len(_message_count_cache) >= _MAX_MESSAGE_COUNT_CACHE and (
        thread_id not in _message_count_cache
    ):
        oldest = next(iter(_message_count_cache))
        _message_count_cache.pop(oldest, None)
    _message_count_cache[thread_id] = (freshness, count)


def _cache_initial_prompt(
    thread_id: str,
    freshness: str | None,
    initial_prompt: str | None,
) -> None:
    """Cache a thread's initial prompt with a freshness token."""
    if len(_initial_prompt_cache) >= _MAX_INITIAL_PROMPT_CACHE and (
        thread_id not in _initial_prompt_cache
    ):
        oldest = next(iter(_initial_prompt_cache))
        _initial_prompt_cache.pop(oldest, None)
    _initial_prompt_cache[thread_id] = (freshness, initial_prompt)


def _thread_freshness(thread: ThreadInfo) -> str | None:
    """Return a cache freshness token for a thread row."""
    return thread.get("latest_checkpoint_id") or thread.get("updated_at")


def _cache_recent_threads(
    agent_name: str | None,
    limit: int,
    threads: list[ThreadInfo],
) -> None:
    """Store a copy of recent thread rows for fast selector startup."""
    key = (agent_name, max(1, limit))
    if len(_recent_threads_cache) >= _MAX_RECENT_THREADS_CACHE_KEYS and (
        key not in _recent_threads_cache
    ):
        _recent_threads_cache.clear()
    _recent_threads_cache[key] = _copy_threads(threads)


def _copy_threads(threads: list[ThreadInfo]) -> list[ThreadInfo]:
    """Return shallow-copied thread rows."""
    return [ThreadInfo(**thread) for thread in threads]


async def _count_messages_from_checkpoint(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> int:
    """Count messages from the most recent checkpoint blob.

    With `durability='exit'`, messages are stored in the checkpoint blob, not in
    the writes table. This function deserializes the checkpoint and counts the
    messages in channel_values.

    Args:
        conn: Database connection.
        thread_id: The thread ID to count messages for.
        serde: Serializer for decoding checkpoint data.

    Returns:
        Number of messages in the checkpoint, or 0 if not found.
    """
    return (await _load_latest_checkpoint_summary(conn, thread_id, serde)).message_count


async def _extract_initial_prompt(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> str | None:
    """Extract the first human message from the latest checkpoint.

    Args:
        conn: Database connection.
        thread_id: The thread ID to extract from.
        serde: Serializer for decoding checkpoint data.

    Returns:
        First human message content, or None if not found.
    """
    summary = await _load_latest_checkpoint_summary(conn, thread_id, serde)
    return summary.initial_prompt


async def populate_thread_initial_prompts(threads: list[ThreadInfo]) -> None:
    """Populate `initial_prompt` for thread rows in the background.

    Args:
        threads: Thread rows to enrich in place.
    """
    if not threads:
        return

    async with _connect() as conn:
        await _populate_checkpoint_fields(
            conn,
            threads,
            include_message_count=False,
            include_initial_prompt=True,
        )


async def _populate_checkpoint_fields(
    conn: aiosqlite.Connection,
    threads: list[ThreadInfo],
    *,
    include_message_count: bool,
    include_initial_prompt: bool,
) -> None:
    """Populate checkpoint-derived thread fields with a batched latest-row pass."""
    serde = await _get_jsonplus_serializer()

    # Phase 1: apply cache hits, collect threads that need DB fetch.
    uncached: list[ThreadInfo] = []
    for thread in threads:
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        needs_count = False
        needs_prompt = False

        if include_message_count:
            cached = _message_count_cache.get(thread_id)
            if cached is not None and cached[0] == freshness:
                thread["message_count"] = cached[1]
            else:
                needs_count = True

        if include_initial_prompt and "initial_prompt" not in thread:
            cached_prompt = _initial_prompt_cache.get(thread_id)
            if cached_prompt is not None and cached_prompt[0] == freshness:
                thread["initial_prompt"] = cached_prompt[1]
            else:
                needs_prompt = True

        if needs_count or needs_prompt:
            uncached.append(thread)

    if not uncached:
        return

    # Phase 2: batch-fetch all uncached threads.
    uncached_ids = [t["thread_id"] for t in uncached]
    batch_results = await _load_latest_checkpoint_summaries_batch(
        conn, uncached_ids, serde
    )

    # Phase 3: apply results and update caches.
    for thread in uncached:
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        summary = batch_results.get(thread_id, _CheckpointSummary(0, None))

        if include_message_count and "message_count" not in thread:
            thread["message_count"] = summary.message_count
            _cache_message_count(thread_id, freshness, summary.message_count)
        if include_initial_prompt and "initial_prompt" not in thread:
            thread["initial_prompt"] = summary.initial_prompt
            _cache_initial_prompt(thread_id, freshness, summary.initial_prompt)


_SQLITE_MAX_VARIABLE_NUMBER = 500
"""Max `?` placeholders per SQL query.

SQLite limits how many `?` parameters a single query can have (default 999,
lower on some builds). If a user accumulates hundreds of threads and the
`/threads` modal fetches them all at once, the `IN (?, ?, ...)` clause could
exceed that limit. We chunk to this size to stay safe.
"""


async def _load_latest_checkpoint_summaries_batch(
    conn: aiosqlite.Connection,
    thread_ids: list[str],
    serde: JsonPlusSerializer,
) -> dict[str, _CheckpointSummary]:
    """Batch-load the latest checkpoint summary for multiple threads.

    Uses a window function to fetch the latest checkpoint per thread, issuing
    one query per chunk for SQLite variable-limit safety.

    Args:
        conn: Database connection.
        thread_ids: Thread IDs to look up.
        serde: Serializer for decoding checkpoint blobs.

    Returns:
        Dict mapping thread IDs to their checkpoint summaries.
    """
    if not thread_ids:
        return {}

    results: dict[str, _CheckpointSummary] = {}

    for start in range(0, len(thread_ids), _SQLITE_MAX_VARIABLE_NUMBER):
        chunk = thread_ids[start : start + _SQLITE_MAX_VARIABLE_NUMBER]
        placeholders = ",".join("?" * len(chunk))
        query = f"""
            SELECT thread_id, type, checkpoint FROM (
                SELECT thread_id, type, checkpoint,
                       ROW_NUMBER() OVER (
                           PARTITION BY thread_id ORDER BY checkpoint_id DESC
                       ) AS rn
                FROM checkpoints
                WHERE thread_id IN ({placeholders})
            ) WHERE rn = 1
        """  # noqa: S608  # placeholders built from len(chunk); user values use ? params
        async with conn.execute(query, chunk) as cursor:
            rows = await cursor.fetchall()

        loop = asyncio.get_running_loop()
        for row in rows:
            tid, type_str, checkpoint_blob = row
            if not type_str or not checkpoint_blob:
                results[tid] = _CheckpointSummary(message_count=0, initial_prompt=None)
                continue
            try:
                data = await loop.run_in_executor(
                    None, serde.loads_typed, (type_str, checkpoint_blob)
                )
                results[tid] = _summarize_checkpoint(data)
            except Exception:
                logger.warning(
                    "Failed to deserialize checkpoint for thread %s; "
                    "message count and initial prompt may be incomplete",
                    tid,
                    exc_info=True,
                )
                results[tid] = _CheckpointSummary(message_count=0, initial_prompt=None)

    return results


async def _load_latest_checkpoint_summary(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> _CheckpointSummary:
    """Load checkpoint-derived summary data from the latest checkpoint row.

    Returns:
        Message-count and prompt data extracted from the latest checkpoint row.
    """
    query = """
        SELECT type, checkpoint
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY checkpoint_id DESC
        LIMIT 1
    """
    async with conn.execute(query, (thread_id,)) as cursor:
        row = await cursor.fetchone()
        if not row or not row[0] or not row[1]:
            return _CheckpointSummary(message_count=0, initial_prompt=None)

        type_str, checkpoint_blob = row
        try:
            data = serde.loads_typed((type_str, checkpoint_blob))
        except (ValueError, TypeError, KeyError, AttributeError):
            logger.warning(
                "Failed to deserialize checkpoint for thread %s; "
                "message count and initial prompt may be incomplete",
                thread_id,
                exc_info=True,
            )
            return _CheckpointSummary(message_count=0, initial_prompt=None)

    return _summarize_checkpoint(data)


def _summarize_checkpoint(data: object) -> _CheckpointSummary:
    """Extract message count and initial human prompt from checkpoint data.

    Returns:
        Structured summary for the decoded checkpoint payload.
    """
    messages = _checkpoint_messages(data)
    return _CheckpointSummary(
        message_count=len(messages),
        initial_prompt=_initial_prompt_from_messages(messages),
    )


def _checkpoint_messages(data: object) -> list[object]:
    """Return checkpoint messages when the decoded payload has the expected shape."""
    if not isinstance(data, dict):
        return []

    payload = cast("dict[str, object]", data)
    channel_values = payload.get("channel_values")
    if not isinstance(channel_values, dict):
        return []

    channel_values_dict = cast("dict[str, object]", channel_values)
    messages = channel_values_dict.get("messages")
    if not isinstance(messages, list):
        return []

    return cast("list[object]", messages)


def _initial_prompt_from_messages(messages: list[object]) -> str | None:
    """Return the first human message content from a checkpoint message list."""
    for msg in messages:
        if getattr(msg, "type", None) == "human":
            return _coerce_prompt_text(getattr(msg, "content", None))
    return None


def _coerce_prompt_text(content: object) -> str | None:
    """Normalize checkpoint message content into displayable text.

    Returns:
        Displayable prompt text, or `None` when the content is empty.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_dict = cast("dict[str, object]", part)
                text = part_dict.get("text")
                parts.append(text if isinstance(text, str) else "")
            else:
                parts.append(str(part))
        joined = " ".join(parts).strip()
        return joined or None
    if content is None:
        return None
    return str(content)


async def get_most_recent(agent_name: str | None = None) -> str | None:
    """Get most recent thread_id, optionally filtered by agent.

    Returns:
        Most recent thread_id or None if no threads exist.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        if agent_name:
            query = """
                SELECT thread_id FROM checkpoints
                WHERE json_extract(metadata, '$.agent_name') = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """
            params: tuple = (agent_name,)
        else:
            query = (
                "SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
            )
            params = ()

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def get_thread_agent(thread_id: str) -> str | None:
    """Get agent_name for a thread.

    Returns:
        Agent name associated with the thread, or None if not found.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        query = """
            SELECT json_extract(metadata, '$.agent_name')
            FROM checkpoints
            WHERE thread_id = ?
            LIMIT 1
        """
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def thread_exists(thread_id: str) -> bool:
    """Check if a thread exists in checkpoints.

    Returns:
        True if thread exists, False otherwise.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False

        query = "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1"
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row is not None


async def find_similar_threads(thread_id: str, limit: int = 3) -> list[str]:
    """Find threads whose IDs start with the given prefix.

    Args:
        thread_id: Prefix to match against thread IDs.
        limit: Maximum number of matching threads to return.

    Returns:
        List of thread IDs that begin with the given prefix.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []

        query = """
            SELECT DISTINCT thread_id
            FROM checkpoints
            WHERE thread_id LIKE ?
            ORDER BY thread_id
            LIMIT ?
        """
        prefix = thread_id + "%"
        async with conn.execute(query, (prefix, limit)) as cursor:
            rows = await cursor.fetchall()
            return [r[0] for r in rows]


async def delete_thread(thread_id: str) -> bool:
    """Delete thread checkpoints.

    Returns:
        True if thread was deleted, False if not found.
    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False

        cursor = await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        deleted = cursor.rowcount > 0
        if await _table_exists(conn, "writes"):
            await conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        await conn.commit()
        if deleted:
            _message_count_cache.pop(thread_id, None)
            for key, rows in list(_recent_threads_cache.items()):
                filtered = [row for row in rows if row["thread_id"] != thread_id]
                _recent_threads_cache[key] = filtered
        return deleted


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Get AsyncSqliteSaver for the global database.

    Yields:
        AsyncSqliteSaver instance for checkpoint persistence.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    _patch_aiosqlite()

    async with AsyncSqliteSaver.from_conn_string(str(get_db_path())) as checkpointer:
        yield checkpointer


_DEFAULT_THREAD_LIMIT = 20


def get_thread_limit() -> int:
    """Read the thread listing limit from `DA_CLI_RECENT_THREADS`.

    Falls back to `_DEFAULT_THREAD_LIMIT` when the variable is unset or contains
    a non-integer value. The result is clamped to a minimum of 1.

    Returns:
        Number of threads to display.
    """
    import os

    raw = os.environ.get("DA_CLI_RECENT_THREADS")
    if raw is None:
        return _DEFAULT_THREAD_LIMIT
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid DA_CLI_RECENT_THREADS value %r, using default %d",
            raw,
            _DEFAULT_THREAD_LIMIT,
        )
        return _DEFAULT_THREAD_LIMIT


async def list_threads_command(
    agent_name: str | None = None,
    limit: int | None = None,
    sort_by: str | None = None,
    branch: str | None = None,
    verbose: bool = False,
    relative: bool | None = None,
    *,
    output_format: OutputFormat = "text",
) -> None:
    """CLI handler for `deepagents threads list`.

    Fetches and displays a table of recent conversation threads, optionally
    filtered by agent name or git branch.

    Args:
        agent_name: Only show threads belonging to this agent.

            When `None`, threads for all agents are shown.
        limit: Maximum number of threads to display.

            When `None`, reads from `DA_CLI_RECENT_THREADS` or falls back to
            the default.
        sort_by: Sort field — `"updated"` or `"created"`.

            When `None`, reads from config (`~/.deepagents/config.toml`).
        branch: Only show threads from this git branch.
        verbose: When `True`, show all columns (branch, created, prompt).
        relative: Show timestamps as relative time (e.g., '5m ago').

            When `None`, reads from config (`~/.deepagents/config.toml`).
        output_format: Output format — `'text'` (Rich) or `'json'`.
    """
    from deepagents_cli.model_config import (
        load_thread_relative_time,
        load_thread_sort_order,
    )

    if sort_by is None:
        raw = load_thread_sort_order()
        sort_by = "created" if raw == "created_at" else "updated"
    if relative is None:
        relative = load_thread_relative_time()

    fmt_ts = format_relative_timestamp if relative else format_timestamp

    limit = get_thread_limit() if limit is None else max(1, limit)

    threads = await list_threads(
        agent_name,
        limit=limit,
        include_message_count=True,
        sort_by=sort_by,
        branch=branch,
    )

    if verbose and threads:
        await populate_thread_checkpoint_details(
            threads, include_message_count=False, include_initial_prompt=True
        )

    if output_format == "json":
        from deepagents_cli.output import write_json

        write_json("threads list", list(threads))
        return

    from rich.markup import escape as escape_markup
    from rich.table import Table

    from deepagents_cli import theme
    from deepagents_cli.config import console

    if not threads:
        filters = []
        if agent_name:
            filters.append(f"agent '{escape_markup(agent_name)}'")
        if branch:
            filters.append(f"branch '{escape_markup(branch)}'")
        if filters:
            console.print(
                f"[yellow]No threads found for {' and '.join(filters)}.[/yellow]"
            )
        else:
            console.print("[yellow]No threads found.[/yellow]")
        console.print("[dim]Start a conversation with: deepagents[/dim]")
        return

    title_parts = []
    if agent_name:
        title_parts.append(f"agent '{escape_markup(agent_name)}'")
    if branch:
        title_parts.append(f"branch '{escape_markup(branch)}'")

    title_filter = f" for {' and '.join(title_parts)}" if title_parts else ""
    sort_label = "created" if sort_by == "created" else "updated"
    title = f"Recent Threads{title_filter} (last {limit}, by {sort_label})"

    table = Table(title=title, show_header=True, header_style=f"bold {theme.PRIMARY}")
    table.add_column("Thread ID", style="bold")
    table.add_column("Agent")
    table.add_column("Messages", justify="right")
    if verbose:
        table.add_column("Created")
    table.add_column("Updated" if sort_by == "updated" else "Last Used")
    if verbose:
        table.add_column("Branch")
        table.add_column("Location")
        table.add_column("Prompt", max_width=40, no_wrap=True)

    prompt_max = 40

    for t in threads:
        row: list[str] = [
            t["thread_id"],
            t["agent_name"] or "unknown",
            str(t.get("message_count", 0)),
        ]
        if verbose:
            row.append(fmt_ts(t.get("created_at")))
        row.append(fmt_ts(t.get("updated_at")))
        if verbose:
            prompt = " ".join((t.get("initial_prompt") or "").split())
            if len(prompt) > prompt_max:
                prompt = prompt[: prompt_max - 3] + "..."
            row.extend(
                [
                    t.get("git_branch") or "",
                    format_path(t.get("cwd")),
                    prompt,
                ]
            )
        table.add_row(*row)

    console.print()
    console.print(table)
    if len(threads) >= limit:
        console.print(
            f"[dim]Showing last {limit} threads. "
            "Override with -n/--limit or DA_CLI_RECENT_THREADS.[/dim]"
        )
    console.print()


async def delete_thread_command(
    thread_id: str, *, output_format: OutputFormat = "text"
) -> None:
    """CLI handler for: deepagents threads delete.

    Args:
        thread_id: ID of the thread to delete.
        output_format: Output format — `'text'` (Rich) or `'json'`.
    """
    deleted = await delete_thread(thread_id)

    if output_format == "json":
        from deepagents_cli.output import write_json

        write_json("threads delete", {"thread_id": thread_id, "deleted": deleted})
        return

    from rich.markup import escape as escape_markup

    from deepagents_cli.config import console

    escaped_id = escape_markup(thread_id)
    if deleted:
        console.print(f"[green]Thread '{escaped_id}' deleted.[/green]")
    else:
        console.print(f"[red]Thread '{escaped_id}' not found.[/red]")
