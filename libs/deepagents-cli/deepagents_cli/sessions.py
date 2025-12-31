"""Thread management for persistent conversations."""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepagents_cli.config import COLORS, console

THREADS_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    project_root TEXT,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_threads_agent_used
    ON threads(agent_name, last_used_at DESC);
"""


def get_db_path() -> Path:
    """Get path to global database."""
    db_dir = Path.home() / ".deepagents"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sessions.db"


class ThreadManager:
    """Manages thread metadata in SQLite.

    Each method opens/closes its own connection for simplicity.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize ThreadManager.

        Args:
            db_path: Path to database file. Defaults to ~/.deepagents/sessions.db
        """
        self.db_path = db_path or get_db_path()

    async def _get_conn(self) -> aiosqlite.Connection:
        """Get a new database connection with schema initialized."""
        conn = await aiosqlite.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = aiosqlite.Row
        await conn.executescript(THREADS_SCHEMA)
        await conn.commit()
        return conn

    async def create_thread(
        self,
        agent_name: str,
        project_root: Path | None = None,
    ) -> str:
        """Create new thread.

        Args:
            agent_name: Name of the agent
            project_root: Optional project root path

        Returns:
            thread_id (8-char hex string)
        """
        thread_id = uuid.uuid4().hex[:8]
        now = datetime.now(UTC).isoformat()

        conn = await self._get_conn()
        try:
            await conn.execute(
                """INSERT INTO threads (id, agent_name, project_root, created_at, last_used_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    thread_id,
                    agent_name,
                    str(project_root) if project_root else None,
                    now,
                    now,  # last_used_at = created_at initially
                ),
            )
            await conn.commit()
            return thread_id
        finally:
            await conn.close()

    async def touch_thread(self, thread_id: str) -> None:
        """Update last_used_at to now.

        Args:
            thread_id: The thread ID to touch
        """
        now = datetime.now(UTC).isoformat()
        conn = await self._get_conn()
        try:
            await conn.execute(
                "UPDATE threads SET last_used_at = ? WHERE id = ?",
                (now, thread_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_thread(self, thread_id: str) -> dict | None:
        """Get thread by ID.

        Args:
            thread_id: The thread ID to look up

        Returns:
            Thread dict or None if not found
        """
        conn = await self._get_conn()
        try:
            async with conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None
        finally:
            await conn.close()

    async def get_most_recent(self, agent_name: str | None = None) -> dict | None:
        """Get most recently used thread.

        Args:
            agent_name: Filter by agent name, or None for most recent overall

        Returns:
            Most recently used thread dict or None if no threads exist
        """
        conn = await self._get_conn()
        try:
            if agent_name:
                query = """SELECT * FROM threads
                           WHERE agent_name = ?
                           ORDER BY last_used_at DESC LIMIT 1"""
                params: tuple = (agent_name,)
            else:
                query = "SELECT * FROM threads ORDER BY last_used_at DESC LIMIT 1"
                params = ()

            async with conn.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None
        finally:
            await conn.close()

    async def list_threads(
        self,
        agent_name: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """List threads ordered by last used.

        Args:
            agent_name: Filter by agent name, or None for all agents
            limit: Maximum number of threads to return

        Returns:
            List of thread dicts, most recently used first
        """
        conn = await self._get_conn()
        try:
            if agent_name:
                query = """SELECT * FROM threads
                           WHERE agent_name = ?
                           ORDER BY last_used_at DESC LIMIT ?"""
                params: tuple = (agent_name, limit)
            else:
                query = "SELECT * FROM threads ORDER BY last_used_at DESC LIMIT ?"
                params = (limit,)

            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete thread and its checkpoints.

        Args:
            thread_id: The thread ID to delete

        Returns:
            True if deleted, False if not found
        """
        conn = await self._get_conn()
        try:
            # Delete thread metadata
            cursor = await conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            deleted = cursor.rowcount > 0

            # Full cleanup: delete checkpoint data too
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            await conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))

            await conn.commit()
            return deleted
        finally:
            await conn.close()


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Get AsyncSqliteSaver for the global database.

    Yields:
        AsyncSqliteSaver instance configured to use the global database
    """
    db_path = str(get_db_path())
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        yield checkpointer


# ============ CLI Command Handlers ============


async def list_threads_command(
    agent_name: str | None = None,
    limit: int = 20,
) -> None:
    """Handler for: deepagents threads list.

    Args:
        agent_name: Filter by agent name, or None for all
        limit: Maximum number of threads to show
    """
    tm = ThreadManager()
    threads = await tm.list_threads(agent_name, limit=limit)

    if not threads:
        if agent_name:
            console.print(f"[yellow]No threads found for agent '{agent_name}'.[/yellow]")
        else:
            console.print("[yellow]No threads found.[/yellow]")
        console.print("[dim]Start a conversation with: deepagents[/dim]")
        return

    title = f"Threads for '{agent_name}'" if agent_name else "All Threads"
    console.print(f"\n[bold]{title}:[/bold]\n", style=COLORS["primary"])

    for t in threads:
        thread_id = t["id"]
        agent = t["agent_name"]
        last_used = t["last_used_at"][:16].replace("T", " ")
        project = t.get("project_root") or ""

        console.print(f"  [bold]{thread_id}[/bold] ({agent})", style=COLORS["primary"])
        console.print(f"    Last used: {last_used}", style=COLORS["dim"])
        if project:
            console.print(f"    Project: {project}", style=COLORS["dim"])
        console.print()


async def delete_thread_command(thread_id: str) -> None:
    """Handler for: deepagents threads delete THREAD_ID.

    Args:
        thread_id: The thread ID to delete
    """
    tm = ThreadManager()
    deleted = await tm.delete_thread(thread_id)

    if deleted:
        console.print(f"[green]Thread '{thread_id}' deleted.[/green]")
    else:
        console.print(f"[red]Thread '{thread_id}' not found.[/red]")
