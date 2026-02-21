"""Thread management using LangGraph's built-in checkpoint persistence."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import aiosqlite
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)

_aiosqlite_patched = False
_jsonplus_serializer: JsonPlusSerializer | None = None
_message_count_cache: dict[str, tuple[str | None, int]] = {}
_MAX_MESSAGE_COUNT_CACHE = 4096
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

    message_count: NotRequired[int]
    """Number of messages in the thread."""

    latest_checkpoint_id: NotRequired[str | None]
    """Most recent checkpoint ID for cache invalidation."""


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


def get_db_path() -> Path:
    """Get path to global database.

    Returns:
        Path to the SQLite database file.
    """
    db_dir = Path.home() / ".deepagents"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sessions.db"


def generate_thread_id() -> str:
    """Generate a new 8-char hex thread ID.

    Returns:
        8-character hexadecimal string.
    """
    return uuid.uuid4().hex[:8]


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
) -> list[ThreadInfo]:
    """List threads from checkpoints table.

    Args:
        agent_name: Optional filter by agent name.
        limit: Maximum number of threads to return.
        include_message_count: Whether to include message counts.

    Returns:
        List of `ThreadInfo` dicts with `thread_id`, `agent_name`,
            `updated_at`, and optionally `message_count`.
    """
    async with _connect() as conn:
        # Return empty if table doesn't exist yet (fresh install)
        if not await _table_exists(conn, "checkpoints"):
            return []

        if agent_name:
            query = """
                SELECT thread_id,
                       json_extract(metadata, '$.agent_name') as agent_name,
                       MAX(json_extract(metadata, '$.updated_at')) as updated_at,
                       MAX(checkpoint_id) as latest_checkpoint_id
                FROM checkpoints
                WHERE json_extract(metadata, '$.agent_name') = ?
                GROUP BY thread_id
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params: tuple = (agent_name, limit)
        else:
            query = """
                SELECT thread_id,
                       json_extract(metadata, '$.agent_name') as agent_name,
                       MAX(json_extract(metadata, '$.updated_at')) as updated_at,
                       MAX(checkpoint_id) as latest_checkpoint_id
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params = (limit,)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            threads: list[ThreadInfo] = [
                ThreadInfo(
                    thread_id=r[0],
                    agent_name=r[1],
                    updated_at=r[2],
                    latest_checkpoint_id=r[3],
                )
                for r in rows
            ]

        # Fetch message counts if requested
        if include_message_count and threads:
            await _populate_message_counts(conn, threads)

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


async def prewarm_thread_message_counts(limit: int | None = None) -> None:
    """Prewarm thread message-count cache for faster `/threads` open.

    Fetches a bounded list of recent threads and populates counts into the
    in-memory cache. Intended to run in a background worker during app startup.

    Args:
        limit: Maximum threads to prewarm. Uses `get_thread_limit()` when `None`.
    """
    thread_limit = limit if limit is not None else get_thread_limit()
    if thread_limit < 1:
        return

    try:
        threads = await list_threads(limit=thread_limit, include_message_count=False)
        if threads:
            await populate_thread_message_counts(threads)
        _cache_recent_threads(None, thread_limit, threads)
    except (OSError, sqlite3.Error):
        logger.debug("Could not prewarm thread message counts", exc_info=True)
    except Exception:
        logger.debug(
            "Unexpected error while prewarming thread message counts",
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
    thread_limit = limit if limit is not None else get_thread_limit()
    if thread_limit < 1:
        return None

    exact = _recent_threads_cache.get((agent_name, thread_limit))
    if exact is not None:
        return _copy_threads(exact)

    best_key: tuple[str | None, int] | None = None
    for key in _recent_threads_cache:
        cache_agent, cache_limit = key
        if cache_agent != agent_name or cache_limit < thread_limit:
            continue
        if best_key is None or cache_limit < best_key[1]:
            best_key = key

    if best_key is None:
        return None

    return _copy_threads(_recent_threads_cache[best_key][:thread_limit])


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
        freshness = thread.get("latest_checkpoint_id") or thread.get("updated_at")
        cached = _message_count_cache.get(thread_id)
        if cached is None or cached[0] != freshness:
            continue
        thread["message_count"] = cached[1]
        populated += 1
    return populated


async def _populate_message_counts(
    conn: aiosqlite.Connection,
    threads: list[ThreadInfo],
) -> None:
    """Fill `message_count` on thread rows with cache-aware lookup."""
    serde = await _get_jsonplus_serializer()
    for thread in threads:
        thread_id = thread["thread_id"]
        freshness = thread.get("latest_checkpoint_id") or thread.get("updated_at")
        cached = _message_count_cache.get(thread_id)
        if cached is not None and cached[0] == freshness:
            thread["message_count"] = cached[1]
            continue

        count = await _count_messages_from_checkpoint(conn, thread_id, serde)
        thread["message_count"] = count
        _cache_message_count(thread_id, freshness, count)


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
        _message_count_cache.clear()
    _message_count_cache[thread_id] = (freshness, count)


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

    With durability="exit", messages are stored in the checkpoint blob,
    not in the writes table. This function deserializes the checkpoint
    and counts the messages in channel_values.

    Args:
        conn: Database connection.
        thread_id: The thread ID to count messages for.
        serde: Serializer for decoding checkpoint data.

    Returns:
        Number of messages in the checkpoint, or 0 if not found.
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
            return 0

        type_str, checkpoint_blob = row
        try:
            data = serde.loads_typed((type_str, checkpoint_blob))
            channel_values = data.get("channel_values", {})
            messages = channel_values.get("messages", [])
            return len(messages)
        except (ValueError, TypeError, KeyError):
            logger.warning(
                "Failed to deserialize checkpoint for thread %s; "
                "message count will show as 0",
                thread_id,
                exc_info=True,
            )
            return 0


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
) -> None:
    """CLI handler for `deepagents threads list`.

    Fetches and displays a table of recent conversation threads, optionally
    filtered by agent name.

    Args:
        agent_name: Only show threads belonging to this agent.

            When `None`, threads for all agents are shown.
        limit: Maximum number of threads to display.

            When `None`, reads from `DA_CLI_RECENT_THREADS` or falls back to
            the default.
    """
    from rich.table import Table

    from deepagents_cli.config import COLORS, console

    limit = get_thread_limit() if limit is None else max(1, limit)

    threads = await list_threads(agent_name, limit=limit, include_message_count=True)

    if not threads:
        if agent_name:
            console.print(
                f"[yellow]No threads found for agent '{agent_name}'.[/yellow]"
            )
        else:
            console.print("[yellow]No threads found.[/yellow]")
        console.print("[dim]Start a conversation with: deepagents[/dim]")
        return

    title = (
        f"Recent threads for '{agent_name}' (last {limit})"
        if agent_name
        else f"Recent Threads (last {limit})"
    )

    table = Table(
        title=title, show_header=True, header_style=f"bold {COLORS['primary']}"
    )
    table.add_column("Thread ID", style="bold")
    table.add_column("Agent")
    table.add_column("Messages", justify="right")
    table.add_column("Last Used", style="dim")

    for t in threads:
        table.add_row(
            t["thread_id"],
            t["agent_name"] or "unknown",
            str(t.get("message_count", 0)),
            format_timestamp(t.get("updated_at")),
        )

    console.print()
    console.print(table)
    console.print()


async def delete_thread_command(thread_id: str) -> None:
    """CLI handler for: deepagents threads delete."""
    from deepagents_cli.config import console

    deleted = await delete_thread(thread_id)

    if deleted:
        console.print(f"[green]Thread '{thread_id}' deleted.[/green]")
    else:
        console.print(f"[red]Thread '{thread_id}' not found.[/red]")
