"""Benchmarks for thread-picker checkpoint enrichment queries.

Run locally:  `make benchmark`
Run with CodSpeed:  `make bench`
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deepagents_code import sessions

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module")
def checkpoint_history_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create threads with large historical blobs and small latest checkpoints."""
    import sqlite3

    path = tmp_path_factory.mktemp("thread-loading") / "sessions.db"
    serde = JsonPlusSerializer()
    latest = serde.dumps_typed({"channel_values": {"messages": []}})
    historical = b"x" * (256 * 1024)
    thread_ids = [f"thread-{index}" for index in range(8)]

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE checkpoints "
            "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
            "type TEXT, checkpoint BLOB, metadata TEXT)"
        )
        conn.execute(
            "CREATE INDEX checkpoints_thread_id_idx "
            "ON checkpoints(thread_id, checkpoint_ns, checkpoint_id DESC)"
        )
        for thread_id in thread_ids:
            conn.executemany(
                "INSERT INTO checkpoints VALUES (?, '', ?, 'json', ?, '{}')",
                [
                    (thread_id, f"{checkpoint_id:04d}", historical)
                    for checkpoint_id in range(64)
                ],
            )
            conn.execute(
                "INSERT INTO checkpoints VALUES (?, '', '9999', ?, ?, '{}')",
                (thread_id, latest[0], latest[1]),
            )
        conn.commit()
    finally:
        conn.close()

    return path


@pytest.fixture(scope="module")
def write_history_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create threads with large historical writes and small initial prompts."""
    import sqlite3

    path = tmp_path_factory.mktemp("thread-prompts") / "sessions.db"
    serde = JsonPlusSerializer()
    initial = serde.dumps_typed([{"role": "user", "content": "hello"}])
    historical = b"x" * (64 * 1024)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE writes "
            "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, task_id TEXT, "
            "idx INTEGER, channel TEXT, type TEXT, value BLOB)"
        )
        conn.execute(
            "CREATE INDEX writes_thread_id_idx "
            "ON writes(thread_id, checkpoint_ns, checkpoint_id)"
        )
        for thread_id in (f"thread-{index}" for index in range(8)):
            conn.execute(
                "INSERT INTO writes VALUES (?, '', '0000', '', 0, 'messages', ?, ?)",
                (thread_id, initial[0], initial[1]),
            )
            conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, '', 0, 'messages', 'json', ?)",
                [
                    (thread_id, f"{checkpoint_id:04d}", historical)
                    for checkpoint_id in range(1, 65)
                ],
            )
        conn.commit()
    finally:
        conn.close()

    return path


def test_latest_checkpoint_loading_ignores_large_history(
    benchmark: BenchmarkFixture,
    checkpoint_history_db: Path,
) -> None:
    """Measure latest-checkpoint loading with 128 MiB of historical payloads."""
    thread_ids = [f"thread-{index}" for index in range(8)]
    serde = JsonPlusSerializer()

    async def load() -> dict[str, sessions._CheckpointSummary]:
        import aiosqlite

        conn = aiosqlite.connect(checkpoint_history_db)
        sessions._guard_sqlite_handle(conn)
        try:
            async with conn as opened:
                return await sessions._load_latest_checkpoint_summaries_batch(
                    opened, thread_ids, serde
                )
        finally:
            await sessions._drain_aiosqlite_worker(conn)

    result = benchmark.pedantic(
        lambda: asyncio.run(load()),
        rounds=5,
        warmup_rounds=1,
        iterations=1,
    )

    assert set(result) == set(thread_ids)
    assert all(summary.message_count == 0 for summary in result.values())


def test_initial_prompt_loading_ignores_large_history(
    benchmark: BenchmarkFixture,
    write_history_db: Path,
) -> None:
    """Measure prompt loading with 32 MiB of later message-write payloads."""
    thread_ids = [f"thread-{index}" for index in range(8)]
    serde = JsonPlusSerializer()

    async def load() -> dict[str, str | None]:
        import aiosqlite

        conn = aiosqlite.connect(write_history_db)
        sessions._guard_sqlite_handle(conn)
        try:
            async with conn as opened:
                return await sessions._load_initial_prompts_from_writes_batch(
                    opened, thread_ids, serde
                )
        finally:
            await sessions._drain_aiosqlite_worker(conn)

    result = benchmark.pedantic(
        lambda: asyncio.run(load()),
        rounds=5,
        warmup_rounds=1,
        iterations=1,
    )

    assert result == dict.fromkeys(thread_ids, "hello")
