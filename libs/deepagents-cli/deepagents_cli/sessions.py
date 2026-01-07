"""Thread management using LangGraph's built-in checkpoint persistence."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.table import Table

from deepagents_cli.config import COLORS, console


def _format_timestamp(iso_timestamp: str | None) -> str:
    """Format ISO timestamp for display (e.g., 'Dec 30, 6:10pm')."""
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp).astimezone()
        return dt.strftime("%b %d, %-I:%M%p").lower().replace("am", "am").replace("pm", "pm")
    except (ValueError, TypeError):
        return ""


def get_db_path() -> Path:
    """Get path to global database."""
    db_dir = Path.home() / ".deepagents"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sessions.db"


def generate_thread_id() -> str:
    """Generate a new 8-char hex thread ID."""
    return uuid.uuid4().hex[:8]


async def list_threads(
    agent_name: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List threads from checkpoints table."""
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
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
            return [{"thread_id": r[0], "agent_name": r[1], "updated_at": r[2]} for r in rows]


async def get_most_recent(agent_name: str | None = None) -> str | None:
    """Get most recent thread_id, optionally filtered by agent."""
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if agent_name:
            query = """
                SELECT thread_id FROM checkpoints
                WHERE json_extract(metadata, '$.agent_name') = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """
            params: tuple = (agent_name,)
        else:
            query = "SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
            params = ()

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def get_thread_agent(thread_id: str) -> str | None:
    """Get agent_name for a thread."""
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
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
    """Check if a thread exists in checkpoints."""
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        query = "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1"
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row is not None


async def delete_thread(thread_id: str) -> bool:
    """Delete thread checkpoints. Returns True if deleted."""
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        cursor = await conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        deleted = cursor.rowcount > 0
        await conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        await conn.commit()
        return deleted


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Get AsyncSqliteSaver for the global database."""
    async with AsyncSqliteSaver.from_conn_string(str(get_db_path())) as checkpointer:
        yield checkpointer


async def list_threads_command(
    agent_name: str | None = None,
    limit: int = 20,
) -> None:
    """CLI handler for: deepagents threads list."""
    threads = await list_threads(agent_name, limit=limit)

    if not threads:
        if agent_name:
            console.print(f"[yellow]No threads found for agent '{agent_name}'.[/yellow]")
        else:
            console.print("[yellow]No threads found.[/yellow]")
        console.print("[dim]Start a conversation with: deepagents[/dim]")
        return

    title = f"Threads for '{agent_name}'" if agent_name else "All Threads"

    table = Table(title=title, show_header=True, header_style=f"bold {COLORS['primary']}")
    table.add_column("Thread ID", style="bold")
    table.add_column("Agent")
    table.add_column("Last Used", style="dim")

    for t in threads:
        table.add_row(
            t["thread_id"],
            t["agent_name"] or "unknown",
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


def setup_threads_parser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """Setup the threads subcommand parser with all its subcommands."""
    threads_parser = subparsers.add_parser(
        "threads",
        help="Manage conversation threads",
        description="Manage conversation threads - list, view, and delete threads",
    )
    threads_subparsers = threads_parser.add_subparsers(
        dest="threads_command", help="Threads command"
    )

    # threads list
    list_parser = threads_subparsers.add_parser(
        "list", help="List threads", description="List all conversation threads"
    )
    list_parser.add_argument(
        "--agent",
        default=None,
        help="Filter by agent name (default: show all)",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of threads to show (default: 20)",
    )

    # threads delete
    delete_parser = threads_subparsers.add_parser(
        "delete", help="Delete a thread", description="Delete a specific thread"
    )
    delete_parser.add_argument("thread_id", help="Thread ID to delete")

    return threads_parser


def execute_threads_command(args: argparse.Namespace) -> None:
    """Execute threads subcommands based on parsed arguments.

    Args:
        args: Parsed command line arguments with threads_command attribute
    """
    if args.threads_command == "list":
        asyncio.run(
            list_threads_command(
                agent_name=args.agent,
                limit=args.limit,
            )
        )
    elif args.threads_command == "delete":
        asyncio.run(delete_thread_command(args.thread_id))
    else:
        # No subcommand provided, show help
        console.print("[yellow]Please specify a threads subcommand: list or delete[/yellow]")
        console.print("\n[bold]Usage:[/bold]", style=COLORS["primary"])
        console.print("  deepagents threads <command> [options]\n")
        console.print("[bold]Available commands:[/bold]", style=COLORS["primary"])
        console.print("  list              List all conversation threads")
        console.print("  delete <id>       Delete a specific thread")
        console.print("\n[bold]Examples:[/bold]", style=COLORS["primary"])
        console.print("  deepagents threads list")
        console.print("  deepagents threads list --agent mybot")
        console.print("  deepagents threads delete a1b2c3d4")
        console.print("\n[dim]For more help on a specific command:[/dim]", style=COLORS["dim"])
        console.print("  deepagents threads <command> --help", style=COLORS["dim"])
