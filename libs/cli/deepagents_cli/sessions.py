"""Thread management using LangGraph's built-in checkpoint persistence."""

import json
import logging
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.table import Table

from deepagents_cli.config import COLORS, console

logger = logging.getLogger(__name__)

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

    # Dynamically adding a method to aiosqlite.Connection at runtime.
    # Type checkers can't understand this monkey-patch, so we suppress the
    # "attr-defined" error that would otherwise be raised.
    aiosqlite.Connection.is_alive = _is_alive  # type: ignore[attr-defined]


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
            serde = JsonPlusSerializer()
            for thread in threads:
                thread["message_count"] = await _count_messages_from_checkpoint(
                    conn, thread["thread_id"], serde
                )

        return threads


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
            # If deserialization fails, fall back to 0
            return 0


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
    """Find threads whose IDs start with the given prefix.

    Args:
        thread_id: Prefix to match against thread IDs.
        limit: Maximum number of matching threads to return.

    Returns:
        List of thread IDs that begin with the given prefix.
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


def _extract_from_content_blocks(content_blocks: list[object]) -> str:
    """Extract text content from content blocks.

    Args:
        content_blocks: List of typed content blocks.

    Returns:
        Concatenated text from all text blocks.
    """
    text_parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        d: dict[str, object] = block  # type: ignore[assignment]
        if d.get("type") == "text":
            text_parts.append(str(d.get("text", "")))
    return "".join(text_parts)


def _parse_message(msg_data: object) -> dict | None:
    """Parse a message into role/content format.

    Handles both plain dicts (from writes table) and LangChain message
    objects (HumanMessage, AIMessage) from checkpoint blob.

    Args:
        msg_data: Message dict or LangChain message object.

    Returns:
        Dict with 'role' and 'content' keys, or None if no content.
    """
    # Handle LangChain message objects (HumanMessage, AIMessage, etc.)
    # Use .content_blocks for typed content (supports multimodal in future)
    if hasattr(msg_data, "type") and hasattr(msg_data, "content_blocks"):
        msg_type = msg_data.type
        blocks: list[object] = msg_data.content_blocks  # type: ignore[assignment]
        content = _extract_from_content_blocks(blocks)
    # Handle plain dicts (from writes table or test fixtures)
    elif isinstance(msg_data, dict):
        d: dict[str, object] = msg_data  # type: ignore[assignment]
        msg_type = str(d.get("type", ""))
        kwargs = d.get("kwargs")
        content = (
            str(dict.get(kwargs, "content", "")) if isinstance(kwargs, dict) else ""
        )
    else:
        return None

    if msg_type == "human":
        role = "user"
    elif msg_type == "ai":
        role = "assistant"
    else:
        role = msg_type

    if content:
        return {"role": role, "content": content}
    return None


async def _get_messages_from_writes(
    conn: aiosqlite.Connection, thread_id: str
) -> list[dict]:
    """Get messages from the writes table.

    Args:
        conn: Database connection.
        thread_id: The thread ID to get messages for.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    if not await _table_exists(conn, "writes"):
        return []

    query = """
        SELECT value, idx
        FROM writes
        WHERE thread_id = ? AND channel = 'messages'
        ORDER BY checkpoint_id, idx
    """
    async with conn.execute(query, (thread_id,)) as cursor:
        rows = await cursor.fetchall()

    messages = []
    for value_blob, _ in rows:
        try:
            if isinstance(value_blob, bytes):
                msg_data = json.loads(value_blob.decode("utf-8"))
            else:
                msg_data = json.loads(value_blob)

            parsed = _parse_message(msg_data)
            if parsed:
                messages.append(parsed)
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as e:
            logger.debug("Skipped malformed message in thread %s: %s", thread_id, e)
            continue

    return messages


async def _get_messages_from_checkpoint(
    conn: aiosqlite.Connection, thread_id: str
) -> list[dict]:
    """Get messages from the checkpoint blob.

    With durability='exit', LangGraph stores messages in the checkpoint blob
    rather than the writes table.

    Args:
        conn: Database connection.
        thread_id: The thread ID to get messages for.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    if not await _table_exists(conn, "checkpoints"):
        return []

    query = """
        SELECT type, checkpoint
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY checkpoint_id DESC
        LIMIT 1
    """
    async with conn.execute(query, (thread_id,)) as cursor:
        row = await cursor.fetchone()

    if not row:
        return []

    type_str, checkpoint_blob = row
    if not checkpoint_blob:
        return []

    try:
        serde = JsonPlusSerializer()
        data = serde.loads_typed((type_str, checkpoint_blob))
        channel_values = data.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])
    except (KeyError, TypeError, ValueError) as e:
        logger.debug("Failed to parse checkpoint blob for thread %s: %s", thread_id, e)
        return []

    messages = []
    for msg_data in raw_messages:
        parsed = _parse_message(msg_data)
        if parsed:
            messages.append(parsed)
    return messages


async def export_thread(thread_id: str, output_format: str = "markdown") -> str | None:
    """Export thread conversation history as markdown or JSON.

    Exports user/assistant message text only from the local database.
    Reads from writes table first, then falls back to checkpoint blob
    (for durability='exit' mode).

    For full trace data (tool calls, latencies, tokens), use LangSmith fetch.

    Args:
        thread_id: The thread ID to export.
        output_format: Output format, either `'markdown'` or `'json'`.

    Returns:
        Exported content as string, or `None` if thread not found, has no
            messages, or all messages are malformed.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        # Try writes table first
        messages = await _get_messages_from_writes(conn, thread_id)

        # Fall back to checkpoint blob (for durability='exit' mode)
        if not messages:
            messages = await _get_messages_from_checkpoint(conn, thread_id)

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


def _write_export(output_path: str, content: str) -> None:
    """Write export content to file with error handling."""
    try:
        Path(output_path).write_text(content, encoding="utf-8")
        console.print(f"[green]Exported to {output_path}[/green]")
    except PermissionError:
        msg = f"Error: Cannot write to '{output_path}' - permission denied."
        console.print(f"[red]{msg}[/red]")
        sys.exit(1)
    except FileNotFoundError:
        msg = f"Error: Parent directory for '{output_path}' does not exist."
        console.print(f"[red]{msg}[/red]")
        sys.exit(1)
    except IsADirectoryError:
        msg = f"Error: '{output_path}' is a directory, not a file."
        console.print(f"[red]{msg}[/red]")
        sys.exit(1)
    except OSError as e:
        console.print(f"[red]Error writing file: {e}[/red]")
        sys.exit(1)


async def export_thread_command(
    thread_id: str, output_path: str | None, output_format: str
) -> None:
    """CLI handler for: deepagents threads export."""
    content = await export_thread(thread_id, output_format=output_format)

    if content is None:
        console.print(f"[red]Thread '{thread_id}' not found or has no messages.[/red]")
        sys.exit(1)
        return  # Unreachable but helps type narrowing

    if output_path:
        _write_export(output_path, content)
    else:
        console.print(content)
