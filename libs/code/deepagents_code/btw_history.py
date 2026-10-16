"""Persist `/btw` side exchanges outside the graph's checkpoints.

The sessions database owns each thread's side history, so it survives client
restarts and every client sees the same exchanges. The graph checkpoint is
never read or written here. Each thread row carries an epoch that a clear or
thread deletion advances; an answer that started under an older epoch is
discarded instead of written back into the cleared history.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING, NamedTuple

from deepagents_code.workspace import _database_path

if TYPE_CHECKING:
    import aiosqlite

_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS dcode_btw_history "
    "(thread_id TEXT PRIMARY KEY NOT NULL, epoch INTEGER NOT NULL, "
    "exchanges TEXT NOT NULL)"
)
_RESET = (
    "INSERT INTO dcode_btw_history VALUES (?, 1, '[]') "
    "ON CONFLICT(thread_id) DO UPDATE SET epoch = epoch + 1, exchanges = '[]'"
)

type Exchange = tuple[str, str]
"""A completed side question and the answer returned for it."""


class HistorySnapshot(NamedTuple):
    """Side exchanges paired with the epoch an answer must present to store."""

    exchanges: list[Exchange]
    epoch: int


def _read(conn: sqlite3.Connection, thread_id: str) -> HistorySnapshot | None:
    row = conn.execute(
        "SELECT epoch, exchanges FROM dcode_btw_history WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    msg = f"Stored /btw history for thread {thread_id} is malformed."
    try:
        stored = json.loads(row[1])
    except json.JSONDecodeError as exc:
        raise ValueError(msg) from exc
    match stored:
        case list():
            pairs: list[object] = stored
        case _:
            raise ValueError(msg)
    exchanges: list[Exchange] = []
    for pair in pairs:
        match pair:
            case [str(question), str(answer)]:
                exchanges.append((question, answer))
            case _:
                raise ValueError(msg)
    return HistorySnapshot(exchanges, row[0])


def load_history(thread_id: str) -> HistorySnapshot:
    """Read a thread's side exchanges and the epoch they belong to.

    Args:
        thread_id: Main thread that owns the side conversation.

    Returns:
        Ordered exchanges and the epoch to pass to `append_exchange`.

    Raises:
        ValueError: If the stored history is malformed.
    """  # noqa: DOC502 — raised by _read
    path = _database_path()
    if not path.exists():
        return HistorySnapshot([], 0)
    with closing(
        sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
    ) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'dcode_btw_history'"
        ).fetchone()
        stored = _read(conn, thread_id) if exists else None
    return stored or HistorySnapshot([], 0)


def append_exchange(thread_id: str, question: str, answer: str, *, epoch: int) -> bool:
    """Append one completed exchange unless the history was cleared meanwhile.

    The immediate transaction serializes overlapping answers, including those
    from other server processes, and writes each exchange whole.

    Args:
        thread_id: Main thread that owns the side conversation.
        question: Question the model answered.
        answer: Answer text returned to the client.
        epoch: Epoch returned by the `load_history` call the answer used.

    Returns:
        Whether the exchange was stored.

    Raises:
        ValueError: If the stored history is malformed.
    """  # noqa: DOC502 — raised by _read
    with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(_CREATE_TABLE)
        exchanges, current = _read(conn, thread_id) or HistorySnapshot([], 0)
        if current != epoch:
            return False
        exchanges.append((question, answer))
        conn.execute(
            "INSERT INTO dcode_btw_history VALUES (?, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET exchanges = excluded.exchanges",
            (thread_id, epoch, json.dumps(exchanges)),
        )
    return True


def clear_history(thread_id: str) -> None:
    """Empty a thread's side history so answers still in progress are not stored.

    Args:
        thread_id: Main thread that owns the side conversation.
    """
    with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(_CREATE_TABLE)
        conn.execute(_RESET, (thread_id,))


async def delete_history(conn: aiosqlite.Connection, thread_id: str) -> None:
    """Erase side history in the caller's thread-deletion transaction.

    Advancing the epoch keeps answers still in progress from restoring it.

    Args:
        conn: Sessions connection whose transaction owns thread deletion.
        thread_id: Thread whose side history must be erased.
    """
    await conn.execute(_CREATE_TABLE)
    await conn.execute(_RESET, (thread_id,))
