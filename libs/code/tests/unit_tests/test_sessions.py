"""Tests for session/thread management."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deepagents_code import sessions
from deepagents_code.app import TextualSessionState

if TYPE_CHECKING:
    import aiosqlite


class TestGenerateThreadId:
    """Tests for generate_thread_id function."""


class TestMixedThreadIdFormats:
    """Old 8-char hex IDs and new UUID7 IDs coexist in the database."""

    def test_list_returns_both_formats(self, tmp_path: Path) -> None:
        """list_threads returns threads regardless of ID format."""
        db_path = tmp_path / "mixed.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        old_id = "a1b2c3d4"
        new_id = sessions.generate_thread_id()
        now = datetime.now(UTC).isoformat()
        for tid in (old_id, new_id):
            meta = json.dumps({"agent_name": "agent", "updated_at": now})
            conn.execute(
                "INSERT INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, metadata) "
                "VALUES (?, '', ?, ?)",
                (tid, f"cp_{tid}", meta),
            )
        conn.commit()
        conn.close()

        with patch.object(sessions, "get_db_path", return_value=db_path):
            threads = asyncio.run(sessions.list_threads())
            returned_ids = {t["thread_id"] for t in threads}
            assert old_id in returned_ids
            assert new_id in returned_ids


class TestThreadFunctions:
    """Tests for thread query functions."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database with test data."""
        db_path = tmp_path / "test_sessions.db"

        # Create tables and insert test data
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
        """)

        # Insert test threads with metadata as JSON
        now = datetime.now(UTC).isoformat()
        earlier = "2024-01-01T10:00:00+00:00"

        threads = [
            ("thread1", "agent1", now, "/home/user/project-a"),
            ("thread2", "agent2", earlier, "/tmp/workspace"),
            ("thread3", "agent1", earlier, None),
        ]

        for tid, agent, updated, cwd in threads:
            meta: dict[str, str] = {"agent_name": agent, "updated_at": updated}
            if cwd is not None:
                meta["cwd"] = cwd
            metadata = json.dumps(meta)
            conn.execute(
                "INSERT INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, metadata) "
                "VALUES (?, '', ?, ?)",
                (tid, f"cp_{tid}", metadata),
            )

        conn.commit()
        conn.close()

        return db_path

    def test_list_threads_empty(self, tmp_path):
        """List returns empty when no threads exist."""
        db_path = tmp_path / "empty.db"
        # Create empty db with table structure
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        conn.commit()
        conn.close()
        with patch.object(sessions, "get_db_path", return_value=db_path):
            threads = asyncio.run(sessions.list_threads())
            assert threads == []

    def test_get_most_recent_empty(self, tmp_path):
        """Get most recent returns None when empty."""
        db_path = tmp_path / "empty.db"
        # Create empty db with table structure
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        conn.commit()
        conn.close()
        with patch.object(sessions, "get_db_path", return_value=db_path):
            tid = asyncio.run(sessions.get_most_recent())
            assert tid is None

    def test_get_thread_updated_at_returns_latest_value(self, temp_db):
        """The resume-policy lookup uses the thread's newest update time."""
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, metadata) "
            "VALUES (?, '', ?, ?)",
            (
                "thread2",
                "zz_latest",
                json.dumps({"updated_at": "2026-06-01T12:00:00+00:00"}),
            ),
        )
        conn.commit()
        conn.close()

        with patch.object(sessions, "get_db_path", return_value=temp_db):
            updated_at = asyncio.run(sessions.get_thread_updated_at("thread2"))

        assert updated_at == "2026-06-01T12:00:00+00:00"

    def test_get_thread_updated_at_returns_none_when_missing(self, temp_db):
        """A thread with no update timestamp cannot be verified."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            updated_at = asyncio.run(sessions.get_thread_updated_at("missing"))

        assert updated_at is None

    def test_get_thread_cwd(self, temp_db):
        """Get thread cwd returns the stored working directory."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            cwd = asyncio.run(sessions.get_thread_cwd("thread1"))
            assert cwd == "/home/user/project-a"

    def test_get_thread_cwd_returns_latest_value(self, temp_db):
        """Get thread cwd uses the most recent checkpoint metadata."""
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, metadata) "
            "VALUES (?, '', ?, ?)",
            (
                "thread1",
                "zz_latest",
                json.dumps({"agent_name": "agent1", "cwd": "/tmp/new-cwd"}),
            ),
        )
        conn.commit()
        conn.close()
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            cwd = asyncio.run(sessions.get_thread_cwd("thread1"))
            assert cwd == "/tmp/new-cwd"

    def test_get_thread_cwd_not_found(self, temp_db):
        """Get thread cwd returns None when missing."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            cwd = asyncio.run(sessions.get_thread_cwd("thread3"))
            assert cwd is None

    def test_get_thread_cwd_ignores_empty_string(self, temp_db):
        """An empty stored cwd is treated as missing rather than returned."""
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, metadata) "
            "VALUES (?, '', ?, ?)",
            ("thread-empty", "c1", json.dumps({"cwd": ""})),
        )
        conn.commit()
        conn.close()
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            cwd = asyncio.run(sessions.get_thread_cwd("thread-empty"))
            assert cwd is None

    def test_delete_thread(self, temp_db, monkeypatch, tmp_path):
        """Delete thread removes thread."""
        monkeypatch.setattr(
            "deepagents_code.offload.get_deepagents_home",
            lambda: tmp_path / ".deepagents",
        )
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            result = asyncio.run(sessions.delete_thread("thread1"))
            assert result is True
            assert asyncio.run(sessions.thread_exists("thread1")) is False

    def test_delete_thread_not_found(self, temp_db, monkeypatch, tmp_path):
        """Delete thread returns False for non-existing thread."""
        monkeypatch.setattr(
            "deepagents_code.offload.get_deepagents_home",
            lambda: tmp_path / ".deepagents",
        )
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            result = asyncio.run(sessions.delete_thread("nonexistent"))
            assert result is False

    def test_delete_thread_cleans_offloaded_history(
        self, temp_db, monkeypatch, tmp_path
    ):
        """Deleting a thread removes its offloaded conversation history."""
        monkeypatch.setattr(
            "deepagents_code.offload.get_deepagents_home",
            lambda: tmp_path / ".deepagents",
        )
        archive_dir = tmp_path / ".deepagents" / "conversation_history"
        archive_dir.mkdir(parents=True)
        archive = archive_dir / "thread1.md"
        archive.write_text("history")
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            result = asyncio.run(sessions.delete_thread("thread1"))
            assert result is True
        assert not archive.exists()

    def test_delete_thread_succeeds_when_history_cleanup_fails(
        self, temp_db, monkeypatch, tmp_path
    ):
        """Checkpoint deletion drives the result; history cleanup is best-effort."""
        monkeypatch.setattr(
            "deepagents_code.offload.get_deepagents_home",
            lambda: tmp_path / ".deepagents",
        )
        # `delete_thread` imports the helper from `offload` at call time, so
        # patching it there simulates a failed (but swallowed) cleanup.
        from deepagents_code import offload

        monkeypatch.setattr(
            offload, "delete_offloaded_history", MagicMock(return_value=False)
        )
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            result = asyncio.run(sessions.delete_thread("thread1"))
        assert result is True

    def test_delete_thread_removes_orphan_archive_without_checkpoints(
        self, temp_db, monkeypatch, tmp_path
    ):
        """A thread with no checkpoints still has its orphaned archive cleaned."""
        monkeypatch.setattr(
            "deepagents_code.offload.get_deepagents_home",
            lambda: tmp_path / ".deepagents",
        )
        archive_dir = tmp_path / ".deepagents" / "conversation_history"
        archive_dir.mkdir(parents=True)
        archive = archive_dir / "orphan-thread.md"
        archive.write_text("history")
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            result = asyncio.run(sessions.delete_thread("orphan-thread"))
        # No checkpoint rows existed, so the thread is reported "not found"...
        assert result is False
        # ...but its stranded archive is removed regardless.
        assert not archive.exists()


class TestGetCheckpointer:
    """Tests for get_checkpointer async context manager."""

    def test_returns_async_sqlite_saver(self, tmp_path):
        """Get checkpointer returns AsyncSqliteSaver."""

        async def _test() -> None:
            db_path = tmp_path / "test.db"
            with patch.object(sessions, "get_db_path", return_value=db_path):
                async with sessions.get_checkpointer() as cp:
                    assert "AsyncSqliteSaver" in type(cp).__name__

        asyncio.run(_test())


class TestFormatTimestamp:
    """Tests for format_timestamp helper."""

    # Naive inputs keep these exact: `astimezone` attaches the local zone to a
    # naive datetime without shifting the wall clock, so the rendering is the
    # same in every `TZ`. Offset-bearing inputs would not be.
    RENDERINGS: ClassVar = [
        ("2024-12-30T00:30:00", "dec 30, 12:30am"),  # midnight is 12, not 0
        ("2024-12-30T12:30:00", "dec 30, 12:30pm"),  # noon is 12, not 0
        ("2024-12-30T09:05:00", "dec 30, 9:05am"),  # hour unpadded, day padded
        ("2024-12-05T21:18:00", "dec 05, 9:18pm"),  # 12-hour clock, not 24
    ]

    def test_renders_expected_clock(self):
        """Pins the hand-derived 12-hour clock at every boundary."""
        for iso_timestamp, expected in self.RENDERINGS:
            assert sessions.format_timestamp(iso_timestamp) == expected

    def test_renders_where_strftime_rejects_dash_flag(self):
        """Renders identically where the platform strftime has no `-` flag.

        MSVC's CRT documents `#` as its only strftime flag and treats any
        other flag as an invalid formatting code; CPython surfaces that as
        `ValueError`. The stand-in reproduces that on any host, so the
        regression is caught without a Windows runner.
        """

        class _NoDashFlagDatetime(datetime):
            def strftime(self, format: str) -> str:  # noqa: A002  # matches `date.strftime`
                if "%-" in format:
                    msg = "Invalid format string"
                    raise ValueError(msg)
                return super().strftime(format)

        with patch.object(sessions, "datetime", _NoDashFlagDatetime):
            for iso_timestamp, expected in self.RENDERINGS:
                assert sessions.format_timestamp(iso_timestamp) == expected


class TestFormatRelativeTimestamp:
    """Tests for format_relative_timestamp helper."""

    def test_boundary_360_to_364_days_shows_months(self) -> None:
        """360-364 days old should show months, never the bogus '0y ago'."""
        for days in (360, 362, 364):
            ts = (datetime.now(tz=UTC) - timedelta(days=days, hours=1)).isoformat()
            result = sessions.format_relative_timestamp(ts)
            assert result == "12mo ago"

    def test_boundary_365_days_shows_years(self) -> None:
        """At exactly 365 days, the year bucket takes over with '1y ago'."""
        ts = (datetime.now(tz=UTC) - timedelta(days=365, hours=1)).isoformat()
        result = sessions.format_relative_timestamp(ts)
        assert result == "1y ago"


class TestFormatPath:
    """Tests for format_path helper."""


class TestTextualSessionState:
    """Tests for TextualSessionState from app.py."""

    def test_reset_thread_clears_approval_mode_key(self):
        """A new thread must not inherit the prior thread's live approval key.

        The key is hashed per thread; leaving a stale key would point the
        interrupt predicate at the previous thread's mode.
        """
        state = TextualSessionState(thread_id="original")
        state.approval_mode_key = "stale"
        state.reset_thread()
        assert state.approval_mode_key is None

    def test_thread_switch_resets_turn_markers(self):
        """Assigning a different thread_id must not carry the prior turn count.

        `/threads` switches and resume injection set `thread_id` directly
        (not via `reset_thread`); the per-thread turn sequence has to restart so
        the switched-to thread's traces aren't ordered under the previous
        thread's turn_number/turn_id.
        """
        state = TextualSessionState(thread_id="thread-a")
        state.advance_turn()
        state.advance_turn()
        assert state.turn_number == 2

        state.thread_id = "thread-b"
        assert state.turn_number == 0
        assert state.turn_id is None

        turn_id, turn_number = state.advance_turn()
        assert turn_number == 1
        assert state.turn_id == turn_id


class TestFindSimilarThreads:
    """Tests for find_similar_threads function."""

    @pytest.fixture
    def temp_db_with_threads(self, tmp_path: Path) -> Path:
        """Create a temporary database with test threads."""
        db_path = tmp_path / "test_sessions.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)

        # Insert threads with various IDs
        threads = ["abc12345", "abc99999", "abcdef00", "xyz12345"]
        for tid in threads:
            metadata = json.dumps({"agent_name": "agent1", "updated_at": "2024-01-01"})
            conn.execute(
                "INSERT INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, metadata) "
                "VALUES (?, '', ?, ?)",
                (tid, f"cp_{tid}", metadata),
            )

        conn.commit()
        conn.close()
        return db_path


class TestListThreadsWithMessageCount:
    """Tests for list_threads with message count."""

    @pytest.fixture
    def temp_db_with_messages(self, tmp_path: Path) -> Path:
        """Create a temporary database with threads and messages in checkpoint blob."""
        db_path = tmp_path / "test_sessions.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
        """)

        # Create checkpoint with messages in the blob
        serde = JsonPlusSerializer()
        checkpoint_data = {
            "v": 1,
            "ts": "2024-01-01T00:00:00+00:00",
            "id": "test-checkpoint-id",
            "channel_values": {
                "messages": [
                    {"type": "human", "content": "msg1"},
                    {"type": "ai", "content": "msg2"},
                    {"type": "human", "content": "msg3"},
                ],
            },
            "channel_versions": {},
            "versions_seen": {},
            "updated_channels": [],
        }
        type_str, checkpoint_blob = serde.dumps_typed(checkpoint_data)
        metadata = json.dumps({"agent_name": "agent1", "updated_at": "2024-01-01"})
        conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) "
            "VALUES (?, '', ?, ?, ?, ?)",
            ("thread1", "cp_1", type_str, checkpoint_blob, metadata),
        )

        conn.commit()
        conn.close()
        return db_path

    def test_message_count_uses_cache_for_unchanged_thread(
        self, temp_db_with_messages: Path
    ) -> None:
        """Second call should reuse cached count for unchanged checkpoint."""
        sessions._message_count_cache.clear()
        try:
            with (
                patch.object(
                    sessions, "get_db_path", return_value=temp_db_with_messages
                ),
                patch.object(
                    sessions,
                    "_get_jsonplus_serializer",
                    new_callable=AsyncMock,
                    return_value=object(),
                ),
                patch.object(
                    sessions,
                    "_load_latest_checkpoint_summaries_batch",
                    new_callable=AsyncMock,
                    return_value={
                        "thread1": sessions._CheckpointSummary(
                            message_count=3,
                            initial_prompt=None,
                        ),
                    },
                ) as mock_batch,
            ):
                first = asyncio.run(sessions.list_threads(include_message_count=True))
                second = asyncio.run(sessions.list_threads(include_message_count=True))

                assert first[0]["message_count"] == 3
                assert second[0]["message_count"] == 3
                assert mock_batch.await_count == 1
        finally:
            sessions._message_count_cache.clear()

    def test_message_count_cache_invalidates_on_new_checkpoint(
        self, temp_db_with_messages: Path
    ) -> None:
        """A newer checkpoint should invalidate cached message count."""
        sessions._message_count_cache.clear()
        call_count = 0
        try:
            with (
                patch.object(
                    sessions, "get_db_path", return_value=temp_db_with_messages
                ),
                patch.object(
                    sessions,
                    "_get_jsonplus_serializer",
                    new_callable=AsyncMock,
                    return_value=object(),
                ),
            ):
                results = [
                    {"thread1": sessions._CheckpointSummary(3, None)},
                    {"thread1": sessions._CheckpointSummary(4, None)},
                ]

                def _batch_side_effect(
                    *_args: object, **_kwargs: object
                ) -> dict[str, sessions._CheckpointSummary]:
                    nonlocal call_count
                    idx = min(call_count, len(results) - 1)
                    call_count += 1
                    return results[idx]

                with patch.object(
                    sessions,
                    "_load_latest_checkpoint_summaries_batch",
                    new_callable=AsyncMock,
                    side_effect=_batch_side_effect,
                ) as mock_batch:
                    first = asyncio.run(
                        sessions.list_threads(include_message_count=True)
                    )
                    assert first[0]["message_count"] == 3

                    conn = sqlite3.connect(str(temp_db_with_messages))
                    type_str, checkpoint_blob, metadata = conn.execute(
                        "SELECT type, checkpoint, metadata FROM checkpoints "
                        "WHERE thread_id = ? AND checkpoint_id = ?",
                        ("thread1", "cp_1"),
                    ).fetchone()
                    conn.execute(
                        "INSERT INTO checkpoints "
                        "(thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, "
                        "metadata) "
                        "VALUES (?, '', ?, ?, ?, ?)",
                        ("thread1", "cp_2", type_str, checkpoint_blob, metadata),
                    )
                    conn.commit()
                    conn.close()

                    second = asyncio.run(
                        sessions.list_threads(include_message_count=True)
                    )
                    assert second[0]["message_count"] == 4
                    assert mock_batch.await_count == 2
        finally:
            sessions._message_count_cache.clear()


class TestPopulateThreadCheckpointDetails:
    """Tests for combined checkpoint-detail enrichment."""

    async def test_populates_count_and_prompt_from_separate_sources(self) -> None:
        """Counts come from the checkpoint summary, prompts from the writes table."""
        threads: list[sessions.ThreadInfo] = [
            {
                "thread_id": "thread-a",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "latest_checkpoint_id": "cp_1",
            }
        ]

        with (
            patch.object(
                sessions,
                "_get_jsonplus_serializer",
                new_callable=AsyncMock,
                return_value=object(),
            ),
            patch.object(
                sessions,
                "_load_latest_checkpoint_summaries_batch",
                new_callable=AsyncMock,
                return_value={
                    "thread-a": sessions._CheckpointSummary(
                        message_count=4,
                        initial_prompt=None,
                    ),
                },
            ) as mock_summary,
            patch.object(
                sessions,
                "_load_initial_prompts_from_writes_batch",
                new_callable=AsyncMock,
                return_value={"thread-a": "hello world"},
            ) as mock_prompts,
        ):
            await sessions._populate_checkpoint_fields(  # pyright: ignore[reportPrivateUsage]
                cast(
                    "aiosqlite.Connection",
                    object(),  # connection is unused by the mocked loaders
                ),
                threads,
                include_message_count=True,
                include_initial_prompt=True,
            )

        assert threads[0]["message_count"] == 4
        assert threads[0]["initial_prompt"] == "hello world"
        assert mock_summary.await_count == 1
        assert mock_prompts.await_count == 1

    async def test_skips_prompt_query_when_not_requested(self) -> None:
        """The writes-table prompt query should be skipped when prompts are off."""
        threads: list[sessions.ThreadInfo] = [
            {
                "thread_id": "thread-a",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "latest_checkpoint_id": "cp_1",
            }
        ]

        with (
            patch.object(
                sessions,
                "_get_jsonplus_serializer",
                new_callable=AsyncMock,
                return_value=object(),
            ),
            patch.object(
                sessions,
                "_load_latest_checkpoint_summaries_batch",
                new_callable=AsyncMock,
                return_value={
                    "thread-a": sessions._CheckpointSummary(2, None),
                },
            ),
            patch.object(
                sessions,
                "_load_initial_prompts_from_writes_batch",
                new_callable=AsyncMock,
            ) as mock_prompts,
        ):
            await sessions._populate_checkpoint_fields(  # pyright: ignore[reportPrivateUsage]
                cast("aiosqlite.Connection", object()),
                threads,
                include_message_count=True,
                include_initial_prompt=False,
            )

        mock_prompts.assert_not_awaited()

    async def test_falls_back_to_checkpoint_prompt_when_writes_omit_thread(
        self,
    ) -> None:
        """Sessions without message writes still show checkpoint-backed prompts."""
        sessions._initial_prompt_cache.clear()
        try:
            threads: list[sessions.ThreadInfo] = [
                {
                    "thread_id": "thread-a",
                    "agent_name": "agent",
                    "updated_at": "2026-03-08T02:00:00+00:00",
                    "latest_checkpoint_id": "cp_1",
                }
            ]

            with (
                patch.object(
                    sessions,
                    "_get_jsonplus_serializer",
                    new_callable=AsyncMock,
                    return_value=object(),
                ),
                patch.object(
                    sessions,
                    "_load_latest_checkpoint_summaries_batch",
                    new_callable=AsyncMock,
                    return_value={
                        "thread-a": sessions._CheckpointSummary(
                            message_count=2,
                            initial_prompt="hello from checkpoint",
                        ),
                    },
                ) as mock_summary,
                patch.object(
                    sessions,
                    "_load_initial_prompts_from_writes_batch",
                    new_callable=AsyncMock,
                    return_value={},
                ) as mock_prompts,
            ):
                await sessions._populate_checkpoint_fields(  # pyright: ignore[reportPrivateUsage]
                    cast("aiosqlite.Connection", object()),
                    threads,
                    include_message_count=False,
                    include_initial_prompt=True,
                )

            assert threads[0]["initial_prompt"] == "hello from checkpoint"
            mock_summary.assert_awaited_once()
            mock_prompts.assert_awaited_once()
        finally:
            sessions._initial_prompt_cache.clear()


class TestApplyCachedThreadMessageCounts:
    """Tests for applying cached thread counts to rows."""

    def test_populates_rows_from_cache(self) -> None:
        """Rows with matching freshness should get counts from cache."""
        sessions._message_count_cache.clear()
        try:
            sessions._message_count_cache["thread-a"] = ("cp_1", 7)
            threads: list[sessions.ThreadInfo] = [
                {
                    "thread_id": "thread-a",
                    "agent_name": "agent1",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "latest_checkpoint_id": "cp_1",
                },
                {
                    "thread_id": "thread-b",
                    "agent_name": "agent2",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "latest_checkpoint_id": "cp_1",
                },
            ]

            populated = sessions.apply_cached_thread_message_counts(threads)

            assert populated == 1
            assert threads[0]["message_count"] == 7
            assert "message_count" not in threads[1]
        finally:
            sessions._message_count_cache.clear()

    def test_skips_stale_cache_entries(self) -> None:
        """Rows should not use cache when freshness token changes."""
        sessions._message_count_cache.clear()
        try:
            sessions._message_count_cache["thread-a"] = ("cp_1", 7)
            threads: list[sessions.ThreadInfo] = [
                {
                    "thread_id": "thread-a",
                    "agent_name": "agent1",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "latest_checkpoint_id": "cp_2",
                }
            ]

            populated = sessions.apply_cached_thread_message_counts(threads)

            assert populated == 0
            assert "message_count" not in threads[0]
        finally:
            sessions._message_count_cache.clear()


class TestApplyCachedThreadInitialPrompts:
    """Tests for applying cached thread prompts to rows."""

    def test_populates_rows_from_cache(self) -> None:
        """Rows with matching freshness should get prompts from cache."""
        sessions._initial_prompt_cache.clear()
        try:
            sessions._initial_prompt_cache["thread-a"] = ("cp_1", "hello world")
            threads: list[sessions.ThreadInfo] = [
                {
                    "thread_id": "thread-a",
                    "agent_name": "agent1",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "latest_checkpoint_id": "cp_1",
                },
                {
                    "thread_id": "thread-b",
                    "agent_name": "agent2",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "latest_checkpoint_id": "cp_1",
                },
            ]

            populated = sessions.apply_cached_thread_initial_prompts(threads)

            assert populated == 1
            assert threads[0]["initial_prompt"] == "hello world"
            assert "initial_prompt" not in threads[1]
        finally:
            sessions._initial_prompt_cache.clear()

    def test_skips_stale_cache_entries(self) -> None:
        """Rows should not use prompt cache when freshness token changes."""
        sessions._initial_prompt_cache.clear()
        try:
            sessions._initial_prompt_cache["thread-a"] = ("cp_1", "hello world")
            threads: list[sessions.ThreadInfo] = [
                {
                    "thread_id": "thread-a",
                    "agent_name": "agent1",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "latest_checkpoint_id": "cp_2",
                }
            ]

            populated = sessions.apply_cached_thread_initial_prompts(threads)

            assert populated == 0
            assert "initial_prompt" not in threads[0]
        finally:
            sessions._initial_prompt_cache.clear()


class TestPrewarmThreadMessageCounts:
    """Tests for thread-selector cache prewarming."""

    async def test_prewarm_respects_visible_thread_columns(self) -> None:
        """Prewarm should only fetch checkpoint fields for visible columns."""
        from deepagents_code.model_config import ThreadConfig

        threads: list[sessions.ThreadInfo] = [
            {
                "thread_id": "thread-a",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
            }
        ]

        with (
            patch.object(
                sessions,
                "list_threads",
                new_callable=AsyncMock,
                return_value=threads,
            ),
            patch(
                "deepagents_code.model_config.load_thread_config",
                return_value=ThreadConfig(
                    columns={
                        "thread_id": False,
                        "messages": True,
                        "created_at": True,
                        "updated_at": True,
                        "git_branch": False,
                        "initial_prompt": False,
                        "agent_name": False,
                    },
                    relative_time=True,
                    sort_order="updated_at",
                    scope="cwd",
                ),
            ),
            patch.object(
                sessions,
                "populate_thread_checkpoint_details",
                new_callable=AsyncMock,
                return_value=threads,
            ) as mock_populate,
        ):
            await sessions.prewarm_thread_message_counts(limit=3)

        mock_populate.assert_awaited_once_with(
            threads,
            include_message_count=True,
            include_initial_prompt=False,
        )


class TestCacheMessageCount:
    """Tests for message-count cache eviction behavior."""


class TestMessageCountFromCheckpointBlob:
    """Tests for counting messages from checkpoint blob (not writes table).

    With durability="exit", LangGraph stores messages in the checkpoint blob
    but does NOT write individual entries to the writes table. The message
    count should still be accurate.
    """

    @pytest.fixture
    def temp_db_with_checkpoint_messages(self, tmp_path: Path) -> Path:
        """Create a database with messages in checkpoint blob, no writes."""
        db_path = tmp_path / "test_sessions.db"
        conn = sqlite3.connect(str(db_path))

        # Create tables matching LangGraph schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
        """)

        # Create checkpoint blob with messages (simulating real LangGraph data)
        serde = JsonPlusSerializer()
        checkpoint_data = {
            "v": 1,
            "ts": "2024-01-01T00:00:00+00:00",
            "id": "test-checkpoint-id",
            "channel_values": {
                "messages": [
                    {"type": "human", "content": "hello"},
                    {"type": "ai", "content": "hi there"},
                    {"type": "human", "content": "how are you?"},
                    {"type": "ai", "content": "I'm doing well!"},
                ],
            },
            "channel_versions": {},
            "versions_seen": {},
            "updated_channels": [],
        }
        type_str, checkpoint_blob = serde.dumps_typed(checkpoint_data)
        metadata = json.dumps({"agent_name": "agent1", "updated_at": "2024-01-01"})

        conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) "
            "VALUES (?, '', ?, ?, ?, ?)",
            ("thread_with_messages", "cp_1", type_str, checkpoint_blob, metadata),
        )

        # Note: NO entries in writes table - this simulates durability="exit"

        conn.commit()
        conn.close()
        return db_path

    def test_counts_messages_from_checkpoint_blob(
        self, temp_db_with_checkpoint_messages: Path
    ) -> None:
        """Message count should reflect messages in checkpoint blob.

        This test reproduces the bug where threads show 0 messages even
        though they have messages in the checkpoint blob. With durability="exit",
        messages are stored in the checkpoint but NOT in the writes table.
        """
        with patch.object(
            sessions, "get_db_path", return_value=temp_db_with_checkpoint_messages
        ):
            threads = asyncio.run(sessions.list_threads(include_message_count=True))
            assert len(threads) == 1
            # BUG: Currently returns 0 because it looks at writes table
            # EXPECTED: 4 messages from checkpoint blob
            assert threads[0]["message_count"] == 4

    @pytest.fixture
    def temp_db_delta_channel(self, tmp_path: Path) -> Path:
        """DB whose latest checkpoint omits `messages` (DeltaChannel, SDK >= 0.6).

        The full message list is not inlined in `channel_values`; it lives only
        as per-message deltas in the `writes` table, exactly as the deepagents
        SDK's `DeltaChannel` messages channel stores it between snapshots.
        """
        db_path = tmp_path / "delta.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT "
            "DEFAULT '', checkpoint_id TEXT, parent_checkpoint_id TEXT, type "
            "TEXT, checkpoint BLOB, metadata BLOB, "
            "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
        )
        conn.execute(
            "CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT DEFAULT '', "
            "checkpoint_id TEXT, task_id TEXT, idx INTEGER, channel TEXT, type "
            "TEXT, value BLOB, PRIMARY KEY "
            "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx))"
        )

        serde = JsonPlusSerializer()
        # Latest checkpoint: messages absent from channel_values (DeltaChannel
        # only snapshots periodically); other channels still present.
        checkpoint_data = {
            "v": 1,
            "ts": "2024-01-01T00:00:00+00:00",
            "id": "cp_latest",
            "channel_values": {"_context_tokens": 5},
            "channel_versions": {"messages": "00000000000000000000000000000002"},
            "versions_seen": {},
            "updated_channels": [],
        }
        type_str, checkpoint_blob = serde.dumps_typed(checkpoint_data)
        metadata = json.dumps({"agent_name": "agent1", "updated_at": "2024-01-02"})
        conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
            "type, checkpoint, metadata) VALUES (?, '', ?, ?, ?, ?)",
            ("delta_thread", "cp_latest", type_str, checkpoint_blob, metadata),
        )

        # Three message deltas across two checkpoints (one human, one ai, one
        # tool) — the true current count is 3.
        deltas = [
            ("cp_a", "task1", 0, [{"type": "human", "content": "hi", "id": "h1"}]),
            ("cp_a", "task2", 0, [{"type": "ai", "content": "hello", "id": "a1"}]),
            (
                "cp_b",
                "task3",
                0,
                [{"type": "tool", "content": "ok", "id": "t1", "tool_call_id": "c1"}],
            ),
        ]
        for cid, task, idx, value in deltas:
            vtype, vblob = serde.dumps_typed(value)
            conn.execute(
                "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, "
                "task_id, idx, channel, type, value) VALUES (?, '', ?, ?, ?, "
                "'messages', ?, ?)",
                ("delta_thread", cid, task, idx, vtype, vblob),
            )

        conn.commit()
        conn.close()
        return db_path

    def test_counts_messages_from_writes_when_not_inlined(
        self, temp_db_delta_channel: Path
    ) -> None:
        """Regression: DeltaChannel threads reconstruct the count from writes.

        The latest checkpoint omits `messages` from `channel_values`, so the
        count must be rebuilt by replaying the `messages` writes. Before the
        fix this returned 0 for every such thread.
        """
        sessions._message_count_cache.clear()  # pyright: ignore[reportPrivateUsage]
        try:
            with patch.object(
                sessions, "get_db_path", return_value=temp_db_delta_channel
            ):
                threads = asyncio.run(sessions.list_threads(include_message_count=True))
            assert len(threads) == 1
            assert threads[0]["message_count"] == 3
        finally:
            sessions._message_count_cache.clear()  # pyright: ignore[reportPrivateUsage]

    def test_inlined_messages_take_precedence_over_writes(
        self, temp_db_with_checkpoint_messages: Path
    ) -> None:
        """When the latest checkpoint inlines messages, writes are not replayed.

        The fixture inlines 4 messages and has no writes; the count must come
        from the checkpoint without falling back to (here, empty) writes.
        """
        sessions._message_count_cache.clear()  # pyright: ignore[reportPrivateUsage]
        try:
            with (
                patch.object(
                    sessions,
                    "get_db_path",
                    return_value=temp_db_with_checkpoint_messages,
                ),
                patch.object(
                    sessions,
                    "_load_message_counts_from_writes_batch",
                    new_callable=AsyncMock,
                ) as mock_writes,
            ):
                threads = asyncio.run(sessions.list_threads(include_message_count=True))
            assert threads[0]["message_count"] == 4
            mock_writes.assert_not_awaited()
        finally:
            sessions._message_count_cache.clear()  # pyright: ignore[reportPrivateUsage]

    def test_writes_reconstructed_count_is_cached(
        self, temp_db_delta_channel: Path
    ) -> None:
        """A delta-channel count is cached so the writes replay runs only once.

        The freshness-keyed cache is what keeps the `/threads` modal cheap; a
        second open must not re-query the `writes` table.
        """
        sessions._message_count_cache.clear()  # pyright: ignore[reportPrivateUsage]
        try:
            with patch.object(
                sessions, "get_db_path", return_value=temp_db_delta_channel
            ):
                first = asyncio.run(sessions.list_threads(include_message_count=True))
                assert first[0]["message_count"] == 3

                # Second open: the reconstructed count is served from cache, so
                # the writes-replay loader must not run again.
                with patch.object(
                    sessions,
                    "_load_message_counts_from_writes_batch",
                    new_callable=AsyncMock,
                ) as mock_writes:
                    second = asyncio.run(
                        sessions.list_threads(include_message_count=True)
                    )
                assert second[0]["message_count"] == 3
                mock_writes.assert_not_awaited()
        finally:
            sessions._message_count_cache.clear()  # pyright: ignore[reportPrivateUsage]


class TestGetThreadLimit:
    """Tests for get_thread_limit() env var parsing."""


class TestListThreadsSortAndBranch:
    """Tests for sort_by and branch params on list_threads."""

    @pytest.fixture
    def db_with_branches(self, tmp_path: Path) -> Path:
        """Create a database with threads on different branches.

        thread_a: created 2025-01-01, updated 2025-06-01 (on main)
        thread_b: created 2025-03-01, updated 2025-05-15 (on feat)

        sort_by="updated" → thread_a first (June > May)
        sort_by="created" → thread_b first (March > January)
        """
        db_path = tmp_path / "branches.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        conn.execute("""
            CREATE TABLE writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
        """)

        ins = (
            "INSERT INTO checkpoints"
            " (thread_id, checkpoint_ns, checkpoint_id, metadata)"
            " VALUES (?, '', ?, ?)"
        )

        # thread_a: created 2025-01-01, updated 2025-06-01, on main
        conn.execute(
            ins,
            (
                "thread_a",
                "cp1a",
                json.dumps(
                    {
                        "agent_name": "bot",
                        "updated_at": "2025-01-01T12:00:00+00:00",
                        "git_branch": "main",
                    }
                ),
            ),
        )
        # Second checkpoint for thread_a with a later updated_at
        conn.execute(
            ins,
            (
                "thread_a",
                "cp1b",
                json.dumps(
                    {
                        "agent_name": "bot",
                        "updated_at": "2025-06-01T12:00:00+00:00",
                        "git_branch": "main",
                    }
                ),
            ),
        )
        # thread_b: created 2025-03-01, updated 2025-05-15, on feat
        conn.execute(
            ins,
            (
                "thread_b",
                "cp2",
                json.dumps(
                    {
                        "agent_name": "bot",
                        "updated_at": "2025-03-01T12:00:00+00:00",
                        "git_branch": "feat",
                    }
                ),
            ),
        )
        # Second checkpoint for thread_b with a later updated_at
        conn.execute(
            ins,
            (
                "thread_b",
                "cp2b",
                json.dumps(
                    {
                        "agent_name": "bot",
                        "updated_at": "2025-05-15T12:00:00+00:00",
                        "git_branch": "feat",
                    }
                ),
            ),
        )
        conn.commit()
        conn.close()
        return db_path


class TestListThreadsCwdFilter:
    """Tests for the `cwd` filter on `list_threads`."""

    @pytest.fixture
    def db_with_cwds(self, tmp_path: Path) -> Path:
        """Database with threads stored under distinct cwd values.

        thread_a: cwd=/home/user/project-a, branch=main, agent=bot
        thread_b: cwd=/tmp/workspace, branch=feat, agent=bot
        thread_c: no cwd metadata (legacy), branch=main, agent=bot
        """
        db_path = tmp_path / "cwds.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        ins = (
            "INSERT INTO checkpoints"
            " (thread_id, checkpoint_ns, checkpoint_id, metadata)"
            " VALUES (?, '', ?, ?)"
        )
        conn.execute(
            ins,
            (
                "thread_a",
                "cp_a",
                json.dumps(
                    {
                        "agent_name": "bot",
                        "updated_at": "2025-06-01T12:00:00+00:00",
                        "git_branch": "main",
                        "cwd": "/home/user/project-a",
                    }
                ),
            ),
        )
        conn.execute(
            ins,
            (
                "thread_b",
                "cp_b",
                json.dumps(
                    {
                        "agent_name": "bot",
                        "updated_at": "2025-05-15T12:00:00+00:00",
                        "git_branch": "feat",
                        "cwd": "/tmp/workspace",
                    }
                ),
            ),
        )
        conn.execute(
            ins,
            (
                "thread_c",
                "cp_c",
                json.dumps(
                    {
                        "agent_name": "bot",
                        "updated_at": "2025-04-01T12:00:00+00:00",
                        "git_branch": "main",
                    }
                ),
            ),
        )
        conn.commit()
        conn.close()
        return db_path

    def test_filter_by_cwd_excludes_legacy_rows(self, db_with_cwds: Path) -> None:
        """Threads without stored cwd metadata are dropped by the filter."""
        with patch.object(sessions, "get_db_path", return_value=db_with_cwds):
            threads = asyncio.run(sessions.list_threads(cwd="/tmp/workspace"))
        ids = {t["thread_id"] for t in threads}
        # thread_c (no cwd) must not leak through; only thread_b matches.
        assert ids == {"thread_b"}


class TestListThreadsCommandConfigDefaults:
    """Tests for list_threads_command reading config defaults."""

    _THREAD: ClassVar[dict[str, str | int]] = {
        "thread_id": "abc123",
        "agent_name": "bot",
        "message_count": 2,
        "updated_at": "2025-06-01T12:00:00+00:00",
        "created_at": "2025-05-30T10:00:00+00:00",
    }


class TestListThreadsCommandJson:
    """Tests for list_threads_command JSON output."""

    _THREAD: ClassVar[dict] = {
        "thread_id": "abc12345",
        "agent_name": "agent",
        "updated_at": "2025-01-01T12:00:00",
        "created_at": "2025-01-01T11:00:00",
        "latest_checkpoint_id": "cp1",
        "git_branch": None,
        "cwd": "/tmp",
        "message_count": 5,
    }


class TestDeleteThreadCommandJson:
    """Tests for delete_thread_command JSON output."""


class TestBatchCheckpointSummaries:
    """Tests for _load_latest_checkpoint_summaries_batch."""

    async def test_batch_returns_summaries_for_multiple_threads(self) -> None:
        """Batch query should return summaries keyed by thread_id."""
        serde = JsonPlusSerializer()
        from langchain_core.messages import HumanMessage

        checkpoint_data = {
            "channel_values": {"messages": [HumanMessage(content="hello")]},
        }
        blob = serde.dumps_typed(checkpoint_data)

        import aiosqlite

        db_path = ":memory:"
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                "CREATE TABLE checkpoints "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "type TEXT, checkpoint BLOB, metadata TEXT)"
            )
            for tid, cpid in [("t1", "cp_1"), ("t1", "cp_2"), ("t2", "cp_1")]:
                await conn.execute(
                    "INSERT INTO checkpoints VALUES (?, '', ?, ?, ?, '{}')",
                    (tid, cpid, blob[0], blob[1]),
                )
            await conn.commit()

            results = await sessions._load_latest_checkpoint_summaries_batch(
                conn, ["t1", "t2"], serde
            )

        assert "t1" in results
        assert "t2" in results
        assert results["t1"].message_count == 1
        assert results["t1"].initial_prompt == "hello"
        assert results["t2"].message_count == 1

    async def test_batch_chunking_returns_all_results(self) -> None:
        """Chunking across multiple batches should merge all results."""
        serde = JsonPlusSerializer()
        from langchain_core.messages import HumanMessage

        checkpoint_data = {
            "channel_values": {"messages": [HumanMessage(content="hi")]},
        }
        blob = serde.dumps_typed(checkpoint_data)

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE checkpoints "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "type TEXT, checkpoint BLOB, metadata TEXT)"
            )
            thread_ids = [f"t{i}" for i in range(5)]
            for tid in thread_ids:
                await conn.execute(
                    "INSERT INTO checkpoints VALUES (?, '', 'cp1', ?, ?, '{}')",
                    (tid, blob[0], blob[1]),
                )
            await conn.commit()

            with patch.object(sessions, "_SQLITE_MAX_VARIABLE_NUMBER", 2):
                results = await sessions._load_latest_checkpoint_summaries_batch(
                    conn, thread_ids, serde
                )

        assert set(results.keys()) == set(thread_ids)
        for tid in thread_ids:
            assert results[tid].message_count == 1

    async def test_batch_empty_ids_returns_empty_dict(self) -> None:
        """Empty thread_ids list should return empty dict without querying."""
        serde = JsonPlusSerializer()
        result = await sessions._load_latest_checkpoint_summaries_batch(
            None,  # ty: ignore  # connection not used
            [],
            serde,
        )
        assert result == {}


class TestLoadInitialPromptsFromWritesBatch:
    """Tests for the writes-table initial-prompt loader."""

    async def test_returns_prompt_from_dict_message(self) -> None:
        """Initial input is an OpenAI-shape dict; helper should extract its text."""
        serde = JsonPlusSerializer()
        blob = serde.dumps_typed([{"role": "user", "content": "hi there"}])

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.execute(
                "INSERT INTO writes VALUES (?, '', ?, '', 0, 'messages', ?, ?)",
                ("t1", "cp_a", blob[0], blob[1]),
            )
            await conn.commit()

            results = await sessions._load_initial_prompts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {"t1": "hi there"}

    async def test_picks_earliest_messages_write(self) -> None:
        """Earliest write by (checkpoint_id, idx) should win over later ones."""
        serde = JsonPlusSerializer()
        first_blob = serde.dumps_typed([{"role": "user", "content": "first"}])
        later_blob = serde.dumps_typed([{"role": "user", "content": "second"}])

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, '', ?, 'messages', ?, ?)",
                [
                    ("t1", "cp_b", 0, later_blob[0], later_blob[1]),
                    ("t1", "cp_a", 0, first_blob[0], first_blob[1]),
                ],
            )
            await conn.commit()

            results = await sessions._load_initial_prompts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {"t1": "first"}

    async def test_omits_threads_with_no_messages_writes(self) -> None:
        """Threads without any messages-channel write should be absent from result."""
        serde = JsonPlusSerializer()

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )

            results = await sessions._load_initial_prompts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1", "t2"], serde
            )

        assert results == {}

    async def test_empty_input_returns_empty(self) -> None:
        """Empty thread list should short-circuit without touching the connection."""
        serde = JsonPlusSerializer()
        result = await sessions._load_initial_prompts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
            None,  # ty: ignore  # connection not used
            [],
            serde,
        )
        assert result == {}

    async def test_corrupt_blob_is_skipped_without_raising(self) -> None:
        """A row with valid type but undecodable bytes should be omitted, not raised."""
        serde = JsonPlusSerializer()

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.execute(
                "INSERT INTO writes VALUES (?, '', ?, '', 0, 'messages', ?, ?)",
                ("t1", "cp_a", "msgpack", b"\xff\xff garbage"),
            )
            await conn.commit()

            results = await sessions._load_initial_prompts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {}


class TestLoadMessageCountsFromWritesBatch:
    """Tests for the writes-table message-count reconstruction loader."""

    async def test_reconstructs_count_across_checkpoints(self) -> None:
        """Deltas spread across checkpoints fold into the full message count."""
        serde = JsonPlusSerializer()
        first = serde.dumps_typed([{"type": "human", "content": "hi", "id": "h1"}])
        second = serde.dumps_typed([{"type": "ai", "content": "yo", "id": "a1"}])

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, ?, 0, 'messages', ?, ?)",
                [
                    ("t1", "cp_b", "task2", second[0], second[1]),
                    ("t1", "cp_a", "task1", first[0], first[1]),
                ],
            )
            await conn.commit()

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {"t1": 2}

    async def test_dedups_updates_by_id(self) -> None:
        """A later write that reuses a message ID updates, not appends."""
        serde = JsonPlusSerializer()
        original = serde.dumps_typed([{"type": "ai", "content": "draft", "id": "a1"}])
        updated = serde.dumps_typed([{"type": "ai", "content": "final", "id": "a1"}])

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, ?, 0, 'messages', ?, ?)",
                [
                    ("t1", "cp_a", "task1", original[0], original[1]),
                    ("t1", "cp_b", "task2", updated[0], updated[1]),
                ],
            )
            await conn.commit()

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {"t1": 1}

    async def test_remove_message_tombstone_decrements(self) -> None:
        """A `RemoveMessage` write drops the matching message from the count."""
        from langchain_core.messages import RemoveMessage

        serde = JsonPlusSerializer()
        add_two = serde.dumps_typed(
            [
                {"type": "human", "content": "hi", "id": "h1"},
                {"type": "ai", "content": "yo", "id": "a1"},
            ]
        )
        remove_one = serde.dumps_typed([RemoveMessage(id="a1")])

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, ?, 0, 'messages', ?, ?)",
                [
                    ("t1", "cp_a", "task1", add_two[0], add_two[1]),
                    ("t1", "cp_b", "task2", remove_one[0], remove_one[1]),
                ],
            )
            await conn.commit()

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {"t1": 1}

    async def test_overwrite_resets_accumulator(self) -> None:
        """An `Overwrite` write replaces the accumulated list as a reset point."""
        from langgraph.types import Overwrite

        serde = JsonPlusSerializer()
        seed = serde.dumps_typed(
            [
                {"type": "human", "content": "a", "id": "h1"},
                {"type": "ai", "content": "b", "id": "a1"},
                {"type": "human", "content": "c", "id": "h2"},
            ]
        )
        overwrite = serde.dumps_typed(
            Overwrite(value=[{"type": "human", "content": "fresh", "id": "h9"}])
        )

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, ?, 0, 'messages', ?, ?)",
                [
                    ("t1", "cp_a", "task1", seed[0], seed[1]),
                    ("t1", "cp_b", "task2", overwrite[0], overwrite[1]),
                ],
            )
            await conn.commit()

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {"t1": 1}

    async def test_remove_all_messages_resets_then_appends(self) -> None:
        """`REMOVE_ALL_MESSAGES` clears the list; later deltas rebuild from there.

        The deepagents SDK uses this for compaction/reset, so it is a live path
        for the threads this loader targets.
        """
        from langchain_core.messages import RemoveMessage
        from langgraph.graph.message import REMOVE_ALL_MESSAGES

        serde = JsonPlusSerializer()
        seed = serde.dumps_typed(
            [
                {"type": "human", "content": "a", "id": "h1"},
                {"type": "ai", "content": "b", "id": "a1"},
            ]
        )
        # Clear everything, then add a single fresh message in the same delta.
        reset = serde.dumps_typed(
            [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                {"type": "human", "content": "fresh", "id": "h9"},
            ]
        )

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, ?, 0, 'messages', ?, ?)",
                [
                    ("t1", "cp_a", "task1", seed[0], seed[1]),
                    ("t1", "cp_b", "task2", reset[0], reset[1]),
                ],
            )
            await conn.commit()

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {"t1": 1}

    async def test_excludes_subgraph_namespace_writes(self) -> None:
        """Only root-namespace writes count; subagent (`checkpoint_ns`) excluded.

        Subagents persist their own `messages` writes under the same
        `thread_id` with a non-empty `checkpoint_ns`; those must not inflate the
        root conversation's count.
        """
        serde = JsonPlusSerializer()
        root = serde.dumps_typed([{"type": "human", "content": "hi", "id": "h1"}])
        sub_a = serde.dumps_typed([{"type": "ai", "content": "x", "id": "s1"}])
        sub_b = serde.dumps_typed([{"type": "tool", "content": "y", "id": "s2"}])

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.executemany(
                "INSERT INTO writes VALUES (?, ?, ?, ?, 0, 'messages', ?, ?)",
                [
                    ("t1", "", "cp_a", "task1", root[0], root[1]),
                    ("t1", "subagent:abc", "cp_a", "task2", sub_a[0], sub_a[1]),
                    ("t1", "subagent:abc", "cp_b", "task3", sub_b[0], sub_b[1]),
                ],
            )
            await conn.commit()

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {"t1": 1}

    async def test_counts_each_thread_independently(self) -> None:
        """Multiple threads in one batch fold separately."""
        serde = JsonPlusSerializer()
        one = serde.dumps_typed([{"type": "human", "content": "hi", "id": "h1"}])
        two = serde.dumps_typed(
            [
                {"type": "human", "content": "x", "id": "h2"},
                {"type": "ai", "content": "y", "id": "a2"},
            ]
        )

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, ?, 0, 'messages', ?, ?)",
                [
                    ("t1", "cp_a", "task1", one[0], one[1]),
                    ("t2", "cp_a", "task1", two[0], two[1]),
                ],
            )
            await conn.commit()

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1", "t2"], serde
            )

        assert results == {"t1": 1, "t2": 2}

    async def test_omits_threads_with_no_messages_writes(self) -> None:
        """Threads without any messages-channel write are absent from result."""
        serde = JsonPlusSerializer()

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1", "t2"], serde
            )

        assert results == {}

    async def test_empty_input_returns_empty(self) -> None:
        """Empty thread list short-circuits without touching the connection."""
        serde = JsonPlusSerializer()
        result = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
            None,  # ty: ignore  # connection not used
            [],
            serde,
        )
        assert result == {}

    async def test_corrupt_blob_is_skipped_without_raising(self) -> None:
        """A row with undecodable bytes is skipped, not raised."""
        serde = JsonPlusSerializer()
        good = serde.dumps_typed([{"type": "human", "content": "hi", "id": "h1"}])

        import aiosqlite

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            await conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, ?, 0, 'messages', ?, ?)",
                [
                    ("t1", "cp_a", "task1", good[0], good[1]),
                    ("t1", "cp_b", "task2", "msgpack", b"\xff\xff garbage"),
                ],
            )
            await conn.commit()

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        # The good write still counts; the corrupt one is skipped.
        assert results == {"t1": 1}

    async def test_large_append_history_counts_correctly(self) -> None:
        """A long append-only history counts correctly via the one-pass fold.

        Correctness-at-scale check for the `threads list` speedup. Note this
        does not guard against a perf regression on its own: the old O(n^2)
        fold returns the same count (just slowly), so a revert to the quadratic
        path would still pass. Wall-clock assertions are too flaky for CI; a
        codspeed benchmark would be the real regression guard.
        """
        serde = JsonPlusSerializer()

        import aiosqlite

        n = 4000
        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute(
                "CREATE TABLE writes "
                "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
            )
            rows = []
            for i in range(n):
                type_str, blob = serde.dumps_typed(
                    [{"type": "human", "content": "x", "id": f"m{i}"}]
                )
                rows.append(("t1", f"cp_{i:06d}", "task1", type_str, blob))
            await conn.executemany(
                "INSERT INTO writes VALUES (?, '', ?, ?, 0, 'messages', ?, ?)",
                rows,
            )
            await conn.commit()

            results = await sessions._load_message_counts_from_writes_batch(  # pyright: ignore[reportPrivateUsage]
                conn, ["t1"], serde
            )

        assert results == {"t1": n}


class TestCountMessagesFromDeltas:
    """Tests for the delta-folding message counter and its exact fallback."""

    def test_internal_messages_do_not_inflate_count(self) -> None:
        """Metadata-marked local and remote notices are not user-visible rows."""
        from langchain_core.messages import HumanMessage

        deltas = [
            [HumanMessage(content="real", id="h1")],
            [
                HumanMessage(
                    content="hidden",
                    id="h2",
                    additional_kwargs={"lc_source": "goal_state"},
                )
            ],
            [
                {
                    "type": "human",
                    "content": "hidden remote",
                    "id": "h3",
                    "additional_kwargs": {"lc_source": "goal_state"},
                }
            ],
            [
                HumanMessage(
                    content="hidden continuation",
                    id="h4",
                    additional_kwargs={"lc_source": "goal_control"},
                )
            ],
            [
                HumanMessage(
                    content="hidden summary",
                    id="h5",
                    additional_kwargs={"lc_source": "summarization"},
                )
            ],
            [
                HumanMessage(
                    content="hidden local context",
                    id="h6",
                    additional_kwargs={"lc_source": "local_context"},
                )
            ],
        ]

        assert sessions._count_messages_from_deltas(deltas) == 1  # pyright: ignore[reportPrivateUsage]


def test_inlined_checkpoint_count_excludes_internal_messages() -> None:
    """Checkpoint summaries count only user-visible messages."""
    from langchain_core.messages import HumanMessage

    summary = sessions._summarize_checkpoint(  # pyright: ignore[reportPrivateUsage]
        {
            "channel_values": {
                "messages": [
                    HumanMessage(content="real", id="h1"),
                    HumanMessage(
                        content="hidden",
                        id="h2",
                        additional_kwargs={"lc_source": "goal_state"},
                    ),
                    HumanMessage(
                        content="hidden context",
                        id="h3",
                        additional_kwargs={"lc_source": "local_context"},
                    ),
                ]
            }
        }
    )

    assert summary.message_count == 1
    assert summary.initial_prompt == "real"


class TestInitialPromptFromMessages:
    """Tests for the message-list parser used by the writes-table reader."""

    def test_handles_dict_with_user_role(self) -> None:
        """OpenAI-shape dicts (the initial-input format) should match."""
        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [{"role": "user", "content": "hello"}]
        )
        assert result == "hello"

    def test_handles_human_message_object(self) -> None:
        """LangChain `HumanMessage` objects continue to work."""
        from langchain_core.messages import AIMessage, HumanMessage

        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [AIMessage(content="hi"), HumanMessage(content="hello")]
        )
        assert result == "hello"

    def test_returns_none_for_no_human_message(self) -> None:
        """Lists without any human/user message should return None."""
        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [{"role": "assistant", "content": "ack"}]
        )
        assert result is None

    def test_skips_system_prefixed_human_message(self) -> None:
        """Synthetic `[SYSTEM]` interrupt notices are not used as the prompt."""
        from langchain_core.messages import HumanMessage

        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [
                HumanMessage(
                    content="[SYSTEM] Task interrupted by user. "
                    "Previous operation was cancelled."
                ),
                HumanMessage(content="real prompt"),
            ]
        )
        assert result == "real prompt"

    def test_skips_system_prefixed_dict_message(self) -> None:
        """`[SYSTEM]`-prefixed OpenAI-shape dicts are skipped too."""
        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [
                {"role": "user", "content": "[SYSTEM] Task interrupted by user."},
                {"role": "user", "content": "real prompt"},
            ]
        )
        assert result == "real prompt"

    def test_skips_metadata_marked_local_and_remote_messages(self) -> None:
        """`lc_source` prevents hidden notices from becoming thread titles."""
        from langchain_core.messages import HumanMessage

        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [
                HumanMessage(
                    content="hidden local",
                    additional_kwargs={"lc_source": "goal_state"},
                ),
                {
                    "type": "human",
                    "content": "hidden remote",
                    "additional_kwargs": {"lc_source": "goal_state"},
                },
                HumanMessage(
                    content="hidden continuation",
                    additional_kwargs={"lc_source": "goal_control"},
                ),
                {"role": "user", "content": "real prompt"},
            ]
        )

        assert result == "real prompt"

    def test_skips_rendered_goal_context_with_objective_and_criteria(self) -> None:
        """Detailed model context must not leak into a thread title."""
        from deepagents_code.goal_state_notice import build_goal_state_notice

        notice = build_goal_state_notice(
            {
                "_goal_objective": "Keep the SSO migration confidential",
                "_goal_status": "active",
                "_goal_rubric": "Do not expose the customer rollout list",
            }
        )

        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [notice, {"role": "user", "content": "continue the migration"}]
        )

        assert result == "continue the migration"

    def test_unknown_source_can_be_the_initial_prompt(self) -> None:
        from langchain_core.messages import HumanMessage

        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [
                HumanMessage(
                    content="connector prompt",
                    additional_kwargs={"lc_source": "slack"},
                )
            ]
        )

        assert result == "connector prompt"

    def test_returns_none_when_only_system_message(self) -> None:
        """A lone `[SYSTEM]` message yields no displayable prompt."""
        from langchain_core.messages import HumanMessage

        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [HumanMessage(content="[SYSTEM] Task interrupted by user.")]
        )
        assert result is None

    def test_skips_system_message_across_mixed_shapes(self) -> None:
        """Skipping advances across object/dict shapes, not just within one.

        The first write to `messages` is a raw dict; later writes are serialized
        `BaseMessage` instances, so a real thread mixes the two shapes. The skip
        must carry over from a `[SYSTEM]` dict to a real `HumanMessage` object
        and vice versa.
        """
        from langchain_core.messages import HumanMessage

        dict_then_object = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [
                {"role": "user", "content": "[SYSTEM] Task interrupted by user."},
                HumanMessage(content="real prompt"),
            ]
        )
        assert dict_then_object == "real prompt"

        object_then_dict = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [
                HumanMessage(content="[SYSTEM] Task interrupted by user."),
                {"role": "user", "content": "real prompt"},
            ]
        )
        assert object_then_dict == "real prompt"

    def test_skips_consecutive_system_messages(self) -> None:
        """A run of several `[SYSTEM]` messages is skipped, not just the first."""
        from langchain_core.messages import HumanMessage

        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [
                HumanMessage(content="[SYSTEM] first notice"),
                HumanMessage(content="[SYSTEM] second notice"),
                HumanMessage(content="real prompt"),
            ]
        )
        assert result == "real prompt"

    def test_empty_first_content_returns_without_falling_through(self) -> None:
        """Empty (non-`[SYSTEM]`) first content returns as-is, not skipped.

        Only `[SYSTEM]`-prefixed content is skipped; an empty-string first human
        message is returned verbatim (`""`) rather than falling through to a
        later message. This pins the `prompt is not None` guard so a future
        "skip empties" change cannot silently alter behavior.
        """
        from langchain_core.messages import HumanMessage

        result = sessions._initial_prompt_from_messages(  # pyright: ignore[reportPrivateUsage]
            [
                HumanMessage(content=""),
                HumanMessage(content="real prompt"),
            ]
        )
        assert result == ""
