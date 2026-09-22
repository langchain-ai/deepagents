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
