"""Thread management using LangGraph's built-in checkpoint persistence."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.table import Table

from deepagents_cli.config import COLORS, console

# Patch aiosqlite.Connection to add is_alive() method required by
# langgraph-checkpoint>=2.1.0
# See: https://github.com/langchain-ai/langgraph/issues/6583
if not hasattr(aiosqlite.Connection, "is_alive"):

    def _is_alive(self: aiosqlite.Connection) -> bool:
        """Check if the connection is still alive.

        Returns:
            True if connection is alive, False otherwise.
        """
        return self._connection is not None

    aiosqlite.Connection.is_alive = _is_alive


def _format_timestamp(iso_timestamp: str | None) -> str:
    """Format ISO timestamp for display (e.g., 'Dec 30, 6:10pm').

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
) -> list[dict]:
    """List threads from checkpoints table.

    Args:
        agent_name: Optional filter by agent name.
        limit: Maximum number of threads to return.
        include_message_count: Whether to include message counts.

    Returns:
        List of thread dicts with `thread_id`, `agent_name`, `updated_at`,
            and optionally `message_count`.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        # Return empty if table doesn't exist yet (fresh install)
        if not await _table_exists(conn, "checkpoints"):
            return []

        if agent_name:
            query = """
                SELECT thread_id,
                       json_extract(metadata, '$.agent_name') as agent_name,
                       MAX(json_extract(metadata, '$.updated_at')) as updated_at
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
                       MAX(json_extract(metadata, '$.updated_at')) as updated_at
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params = (limit,)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            threads = [
                {"thread_id": r[0], "agent_name": r[1], "updated_at": r[2]}
                for r in rows
            ]

        # Fetch message counts if requested
        if include_message_count and threads:
            writes_exists = await _table_exists(conn, "writes")
            for thread in threads:
                if writes_exists:
                    count_query = """
                        SELECT COUNT(*)
                        FROM writes
                        WHERE thread_id = ? AND channel = 'messages'
                    """
                    async with conn.execute(
                        count_query, (thread["thread_id"],)
                    ) as cursor:
                        row = await cursor.fetchone()
                        thread["message_count"] = row[0] if row else 0
                else:
                    thread["message_count"] = 0

        return threads


async def get_most_recent(agent_name: str | None = None) -> str | None:
    """Get most recent thread_id, optionally filtered by agent.

    Returns:
        Most recent thread_id or None if no threads exist.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
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
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
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
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False

        query = "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1"
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row is not None


async def find_similar_threads(thread_id: str, limit: int = 3) -> list[str]:
    """Find threads with IDs matching prefix of given ID.

    Args:
        thread_id: The thread ID to search for (partial match).
        limit: Maximum number of suggestions to return.

    Returns:
        List of thread IDs that start with the given prefix.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
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


async def get_thread_message_count(thread_id: str) -> int:
    """Count messages in a thread from the writes table.

    Args:
        thread_id: The thread ID to count messages for.

    Returns:
        Number of messages in the thread (0 if thread not found).
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "writes"):
            return 0

        query = """
            SELECT COUNT(*)
            FROM writes
            WHERE thread_id = ? AND channel = 'messages'
        """
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


async def delete_thread(thread_id: str) -> bool:
    """Delete thread checkpoints.

    Returns:
        True if thread was deleted, False if not found.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False

        cursor = await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        deleted = cursor.rowcount > 0
        if await _table_exists(conn, "writes"):
            await conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        await conn.commit()
        return deleted


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Get AsyncSqliteSaver for the global database.

    Yields:
        AsyncSqliteSaver instance for checkpoint persistence.
    """
    async with AsyncSqliteSaver.from_conn_string(str(get_db_path())) as checkpointer:
        yield checkpointer


async def list_threads_command(
    agent_name: str | None = None,
    limit: int = 20,
) -> None:
    """CLI handler for: deepagents threads list."""
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

    title = f"Threads for '{agent_name}'" if agent_name else "All Threads"

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
            _format_timestamp(t.get("updated_at")),
        )

    console.print()
    console.print(table)
    console.print()


async def delete_thread_command(thread_id: str) -> None:
    """CLI handler for: deepagents threads delete."""
    deleted = await delete_thread(thread_id)

    if deleted:
        console.print(f"[green]Thread '{thread_id}' deleted.[/green]")
    else:
        console.print(f"[red]Thread '{thread_id}' not found.[/red]")


async def export_thread(thread_id: str, output_format: str = "markdown") -> str | None:
    """Export thread conversation history as markdown or JSON.

    Exports user/assistant message text only from the local database.
    For full trace data (tool calls, latencies, tokens), use langsmith fetch.

    Args:
        thread_id: The thread ID to export.
        output_format: Output format, either `'markdown'` or `'json'`.

    Returns:
        Exported content as string, or `None` if thread not found.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "writes"):
            return None

        # First check if thread exists
        check_query = "SELECT 1 FROM writes WHERE thread_id = ? LIMIT 1"
        async with conn.execute(check_query, (thread_id,)) as cursor:
            if await cursor.fetchone() is None:
                return None

        # Fetch messages from writes table
        query = """
            SELECT value, idx
            FROM writes
            WHERE thread_id = ? AND channel = 'messages'
            ORDER BY checkpoint_id, idx
        """
        async with conn.execute(query, (thread_id,)) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return None

        # Parse message blobs
        messages = []
        for value_blob, _ in rows:
            try:
                # The value is stored as JSON blob
                if isinstance(value_blob, bytes):
                    msg_data = json.loads(value_blob.decode("utf-8"))
                else:
                    msg_data = json.loads(value_blob)

                # Extract role and content from the message
                msg_type = msg_data.get("type", "")
                content = msg_data.get("kwargs", {}).get("content", "")

                # Map type to role
                if msg_type == "human":
                    role = "user"
                elif msg_type == "ai":
                    role = "assistant"
                else:
                    role = msg_type

                if content:
                    messages.append({"role": role, "content": content})
            except (json.JSONDecodeError, KeyError, TypeError):
                # Skip malformed messages
                continue

        if not messages:
            return None

        # Format output
        if output_format == "json":
            return json.dumps({"thread_id": thread_id, "messages": messages}, indent=2)

        # Markdown format
        lines = [f"# Thread: {thread_id}", ""]
        for msg in messages:
            role_label = "**User:**" if msg["role"] == "user" else "**Assistant:**"
            lines.extend([role_label, "", msg["content"], "", "---", ""])
        return "\n".join(lines)


async def export_thread_command(
    thread_id: str, output_path: str | None, output_format: str
) -> None:
    """CLI handler for: deepagents threads export.

    Args:
        thread_id: Thread ID to export.
        output_path: File path to write output, or None for stdout.
        output_format: Output format (`'markdown'` or `'json'`).
    """
    content = await export_thread(thread_id, output_format=output_format)

    if content is None:
        console.print(f"[red]Thread '{thread_id}' not found or has no messages.[/red]")
        return

    if output_path:
        # Run file write in executor to avoid blocking async
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, partial(Path(output_path).write_text, content, encoding="utf-8")
        )
        console.print(f"[green]Exported to {output_path}[/green]")
    else:
        console.print(content)
