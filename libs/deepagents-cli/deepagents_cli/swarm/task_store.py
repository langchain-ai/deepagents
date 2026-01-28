"""Task persistence layer using SQLite."""

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from deepagents_cli.swarm.types import Task, TaskStatus


class TaskStore:
    """Persistent storage for task board tasks.

    Tasks are stored in SQLite at ~/.deepagents/tasks.db
    Each task is scoped to a session_id (thread_id).
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (Path.home() / ".deepagents" / "tasks.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def _ensure_table(self, conn: aiosqlite.Connection) -> None:
        """Create tasks table if it doesn't exist."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                description TEXT NOT NULL,
                active_form TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                metadata TEXT,
                blocks TEXT DEFAULT '[]',
                blocked_by TEXT DEFAULT '[]',
                owner TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, id)
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)"
        )
        await conn.commit()

    async def create_task(
        self,
        session_id: str,
        subject: str,
        description: str,
        active_form: str | None = None,
        metadata: dict | None = None,
    ) -> Task:
        """Create a new task. Returns task with sequential ID."""
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await self._ensure_table(conn)

            # Get next sequential ID for this session
            async with conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE session_id = ?", (session_id,)
            ) as cursor:
                count = (await cursor.fetchone())[0]

            task_id = str(count + 1)
            now = datetime.now(timezone.utc).isoformat()

            task: Task = {
                "id": task_id,
                "subject": subject,
                "description": description,
                "active_form": active_form,
                "status": TaskStatus.PENDING,
                "metadata": metadata,
                "blocks": [],
                "blocked_by": [],
                "owner": None,
                "created_at": now,
                "updated_at": now,
            }

            await conn.execute(
                """
                INSERT INTO tasks 
                (id, session_id, subject, description, active_form, status, 
                 metadata, blocks, blocked_by, owner, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    session_id,
                    subject,
                    description,
                    active_form,
                    TaskStatus.PENDING.value,
                    json.dumps(metadata),
                    "[]",
                    "[]",
                    None,
                    now,
                    now,
                ),
            )
            await conn.commit()

        return task

    async def get_task(self, session_id: str, task_id: str) -> Task | None:
        """Get a task by ID."""
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await self._ensure_table(conn)
            conn.row_factory = aiosqlite.Row
            async with conn.execute(
                "SELECT * FROM tasks WHERE session_id = ? AND id = ?",
                (session_id, task_id),
            ) as cursor:
                row = await cursor.fetchone()
                return self._row_to_task(row) if row else None

    async def update_task(
        self,
        session_id: str,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        subject: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        owner: str | None = None,
        metadata: dict | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
    ) -> Task | None:
        """Update an existing task."""
        task = await self.get_task(session_id, task_id)
        if not task:
            return None

        if status is not None:
            task["status"] = status
        if subject is not None:
            task["subject"] = subject
        if description is not None:
            task["description"] = description
        if active_form is not None:
            task["active_form"] = active_form
        if owner is not None:
            task["owner"] = owner
        if metadata is not None:
            existing = task["metadata"] or {}
            task["metadata"] = {**existing, **metadata}
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
        if add_blocked_by:
            task["blocked_by"] = list(set(task["blocked_by"] + add_blocked_by))

        task["updated_at"] = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute(
                """
                UPDATE tasks SET 
                    status = ?, subject = ?, description = ?, active_form = ?,
                    owner = ?, metadata = ?, blocks = ?, blocked_by = ?, updated_at = ?
                WHERE session_id = ? AND id = ?
                """,
                (
                    task["status"].value
                    if isinstance(task["status"], TaskStatus)
                    else task["status"],
                    task["subject"],
                    task["description"],
                    task["active_form"],
                    task["owner"],
                    json.dumps(task["metadata"]),
                    json.dumps(task["blocks"]),
                    json.dumps(task["blocked_by"]),
                    task["updated_at"],
                    session_id,
                    task_id,
                ),
            )
            await conn.commit()

        return task

    async def list_tasks(self, session_id: str) -> list[Task]:
        """List all tasks for a session, ordered by ID."""
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await self._ensure_table(conn)
            conn.row_factory = aiosqlite.Row
            async with conn.execute(
                "SELECT * FROM tasks WHERE session_id = ? ORDER BY CAST(id AS INTEGER)",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_task(row) for row in rows]

    def _row_to_task(self, row: aiosqlite.Row) -> Task:
        """Convert a database row to a Task dict."""
        return {
            "id": row["id"],
            "subject": row["subject"],
            "description": row["description"],
            "active_form": row["active_form"],
            "status": TaskStatus(row["status"]),
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "blocks": json.loads(row["blocks"]),
            "blocked_by": json.loads(row["blocked_by"]),
            "owner": row["owner"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
