"""Tests for session/thread management."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deepagents_cli import sessions
from deepagents_cli.app import TextualSessionState
from deepagents_cli.sessions import get_thread_limit

if TYPE_CHECKING:
    import aiosqlite


class TestGenerateThreadId:
    """Tests for generate_thread_id function."""

    def test_length(self):
        """Thread IDs are 8 characters."""
        tid = sessions.generate_thread_id()
        assert len(tid) == 8

    def test_hex(self):
        """Thread IDs are valid hex strings."""
        tid = sessions.generate_thread_id()
        # Should not raise
        int(tid, 16)

    def test_unique(self):
        """Thread IDs are unique."""
        ids = {sessions.generate_thread_id() for _ in range(100)}
        assert len(ids) == 100


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

    def test_list_threads(self, temp_db):
        """List returns all threads with cwd."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            threads = asyncio.run(sessions.list_threads())
            assert len(threads) == 3
            by_id = {t["thread_id"]: t for t in threads}
            assert by_id["thread1"]["cwd"] == "/home/user/project-a"
            assert by_id["thread2"]["cwd"] == "/tmp/workspace"
            assert by_id["thread3"]["cwd"] is None

    def test_list_threads_filter_by_agent(self, temp_db):
        """List filters by agent name."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            threads = asyncio.run(sessions.list_threads(agent_name="agent1"))
            assert len(threads) == 2
            assert all(t["agent_name"] == "agent1" for t in threads)

    def test_list_threads_limit(self, temp_db):
        """List respects limit."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            threads = asyncio.run(sessions.list_threads(limit=2))
            assert len(threads) == 2

    def test_get_most_recent(self, temp_db):
        """Get most recent returns latest thread."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            tid = asyncio.run(sessions.get_most_recent())
            assert tid is not None

    def test_get_most_recent_filter(self, temp_db):
        """Get most recent filters by agent."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            tid = asyncio.run(sessions.get_most_recent(agent_name="agent2"))
            assert tid == "thread2"

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

    def test_thread_exists(self, temp_db):
        """Thread exists returns True for existing thread."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            assert asyncio.run(sessions.thread_exists("thread1")) is True

    def test_thread_not_exists(self, temp_db):
        """Thread exists returns False for non-existing thread."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            assert asyncio.run(sessions.thread_exists("nonexistent")) is False

    def test_get_thread_agent(self, temp_db):
        """Get thread agent returns correct agent name."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            agent = asyncio.run(sessions.get_thread_agent("thread1"))
            assert agent == "agent1"

    def test_get_thread_agent_not_found(self, temp_db):
        """Get thread agent returns None for non-existing thread."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            agent = asyncio.run(sessions.get_thread_agent("nonexistent"))
            assert agent is None

    def test_delete_thread(self, temp_db):
        """Delete thread removes thread."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            result = asyncio.run(sessions.delete_thread("thread1"))
            assert result is True
            assert asyncio.run(sessions.thread_exists("thread1")) is False

    def test_delete_thread_not_found(self, temp_db):
        """Delete thread returns False for non-existing thread."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            result = asyncio.run(sessions.delete_thread("nonexistent"))
            assert result is False


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

    def test_valid_timestamp(self):
        """Formats valid ISO timestamp."""
        result = sessions.format_timestamp("2024-12-30T21:18:00+00:00")
        assert result  # Non-empty string
        assert "dec" in result.lower()

    def test_none(self):
        """Returns empty for None."""
        result = sessions.format_timestamp(None)
        assert result == ""

    def test_invalid(self):
        """Returns empty for invalid timestamp."""
        result = sessions.format_timestamp("not a timestamp")
        assert result == ""


class TestFormatRelativeTimestamp:
    """Tests for format_relative_timestamp helper."""

    def test_none_returns_empty(self) -> None:
        """Returns empty string for None input."""
        assert sessions.format_relative_timestamp(None) == ""

    def test_empty_returns_empty(self) -> None:
        """Returns empty string for empty string input."""
        assert sessions.format_relative_timestamp("") == ""

    def test_invalid_returns_empty(self) -> None:
        """Returns empty string for invalid timestamp."""
        assert sessions.format_relative_timestamp("not a timestamp") == ""

    def test_seconds_ago(self) -> None:
        """Recent timestamps show seconds."""
        ts = (datetime.now(tz=UTC) - timedelta(seconds=30)).isoformat()
        result = sessions.format_relative_timestamp(ts)
        assert result.endswith("s ago")

    def test_minutes_ago(self) -> None:
        """Timestamps within the hour show minutes."""
        ts = (datetime.now(tz=UTC) - timedelta(minutes=5)).isoformat()
        result = sessions.format_relative_timestamp(ts)
        assert result.endswith("m ago")

    def test_hours_ago(self) -> None:
        """Timestamps within the day show hours."""
        ts = (datetime.now(tz=UTC) - timedelta(hours=3)).isoformat()
        result = sessions.format_relative_timestamp(ts)
        assert result.endswith("h ago")

    def test_days_ago(self) -> None:
        """Timestamps within the month show days."""
        ts = (datetime.now(tz=UTC) - timedelta(days=10)).isoformat()
        result = sessions.format_relative_timestamp(ts)
        assert result.endswith("d ago")

    def test_months_ago(self) -> None:
        """Timestamps within the year show months."""
        ts = (datetime.now(tz=UTC) - timedelta(days=90)).isoformat()
        result = sessions.format_relative_timestamp(ts)
        assert result.endswith("mo ago")

    def test_years_ago(self) -> None:
        """Timestamps over a year show years."""
        ts = (datetime.now(tz=UTC) - timedelta(days=400)).isoformat()
        result = sessions.format_relative_timestamp(ts)
        assert result.endswith("y ago")

    def test_future_timestamp_returns_just_now(self) -> None:
        """Future timestamps return 'just now'."""
        ts = (datetime.now(tz=UTC) + timedelta(minutes=5)).isoformat()
        assert sessions.format_relative_timestamp(ts) == "just now"

    def test_boundary_60_seconds(self) -> None:
        """At exactly 60 seconds, should show 1m ago."""
        ts = (datetime.now(tz=UTC) - timedelta(seconds=60)).isoformat()
        result = sessions.format_relative_timestamp(ts)
        assert result == "1m ago"

    def test_boundary_59_seconds(self) -> None:
        """At 59 seconds, should still show seconds."""
        ts = (datetime.now(tz=UTC) - timedelta(seconds=59)).isoformat()
        result = sessions.format_relative_timestamp(ts)
        assert result.endswith("s ago")


class TestFormatPath:
    """Tests for format_path helper."""

    def test_none(self):
        """Returns empty for None."""
        assert sessions.format_path(None) == ""

    def test_empty_string(self):
        """Returns empty for empty string."""
        assert sessions.format_path("") == ""

    def test_home_directory(self):
        """Home directory is shown as ~."""
        home = str(Path.home())
        assert sessions.format_path(home) == "~"

    def test_path_under_home(self):
        """Paths under home are shown relative to ~."""
        home = str(Path.home())
        path = home + "/projects/my-app"
        assert sessions.format_path(path) == "~/projects/my-app"

    def test_path_outside_home(self):
        """Paths outside home are shown as-is."""
        assert sessions.format_path("/tmp/workspace") == "/tmp/workspace"

    def test_path_with_similar_prefix(self):
        """Paths that start like home but aren't under it are shown as-is."""
        home = str(Path.home())
        path = home + "-other/projects"
        assert sessions.format_path(path) == path


class TestTextualSessionState:
    """Tests for TextualSessionState from app.py."""

    def test_stores_provided_thread_id(self):
        """TextualSessionState stores provided thread_id."""
        tid = sessions.generate_thread_id()
        state = TextualSessionState(thread_id=tid)
        assert state.thread_id == tid

    def test_generates_id_if_none(self):
        """TextualSessionState generates ID if none provided."""
        state = TextualSessionState(thread_id=None)
        assert state.thread_id is not None
        assert len(state.thread_id) == 8

    def test_reset_thread(self):
        """reset_thread generates a new thread ID."""
        state = TextualSessionState(thread_id="original")
        old_id = state.thread_id
        new_id = state.reset_thread()
        assert new_id != old_id
        assert len(new_id) == 8
        assert state.thread_id == new_id


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

    def test_finds_matching_prefix(self, temp_db_with_threads: Path) -> None:
        """Find threads that start with given prefix."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            results = asyncio.run(sessions.find_similar_threads("abc"))
            assert len(results) == 3
            assert all(r.startswith("abc") for r in results)

    def test_no_matches(self, temp_db_with_threads: Path) -> None:
        """Return empty list when no matches found."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            results = asyncio.run(sessions.find_similar_threads("zzz"))
            assert results == []

    def test_respects_limit(self, temp_db_with_threads: Path) -> None:
        """Respects the limit parameter."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            results = asyncio.run(sessions.find_similar_threads("abc", limit=2))
            assert len(results) == 2

    def test_empty_db(self, tmp_path: Path) -> None:
        """Return empty list for empty database."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()
        with patch.object(sessions, "get_db_path", return_value=db_path):
            results = asyncio.run(sessions.find_similar_threads("abc"))
            assert results == []


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

    def test_includes_message_count(self, temp_db_with_messages: Path) -> None:
        """List threads includes message count when requested."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_messages):
            threads = asyncio.run(sessions.list_threads(include_message_count=True))
            assert len(threads) == 1
            assert threads[0]["message_count"] == 3

    def test_no_message_count_by_default(self, temp_db_with_messages: Path) -> None:
        """List threads does not include message count by default."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_messages):
            threads = asyncio.run(sessions.list_threads())
            assert len(threads) == 1
            assert "message_count" not in threads[0]

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
                    "_load_latest_checkpoint_summary",
                    new_callable=AsyncMock,
                    return_value=sessions._CheckpointSummary(
                        message_count=3,
                        initial_prompt=None,
                    ),
                ) as mock_summary,
            ):
                first = asyncio.run(sessions.list_threads(include_message_count=True))
                second = asyncio.run(sessions.list_threads(include_message_count=True))

                assert first[0]["message_count"] == 3
                assert second[0]["message_count"] == 3
                assert mock_summary.await_count == 1
        finally:
            sessions._message_count_cache.clear()

    def test_message_count_cache_invalidates_on_new_checkpoint(
        self, temp_db_with_messages: Path
    ) -> None:
        """A newer checkpoint should invalidate cached message count."""
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
                    "_load_latest_checkpoint_summary",
                    new_callable=AsyncMock,
                    side_effect=[
                        sessions._CheckpointSummary(
                            message_count=3,
                            initial_prompt=None,
                        ),
                        sessions._CheckpointSummary(
                            message_count=4,
                            initial_prompt=None,
                        ),
                    ],
                ) as mock_summary,
            ):
                first = asyncio.run(sessions.list_threads(include_message_count=True))
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

                second = asyncio.run(sessions.list_threads(include_message_count=True))
                assert second[0]["message_count"] == 4
                assert mock_summary.await_count == 2
        finally:
            sessions._message_count_cache.clear()


class TestPopulateThreadCheckpointDetails:
    """Tests for combined checkpoint-detail enrichment."""

    async def test_shared_summary_populates_count_and_prompt_once(self) -> None:
        """One summary lookup should fill both fields for a thread row."""
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
                "_load_latest_checkpoint_summary",
                new_callable=AsyncMock,
                return_value=sessions._CheckpointSummary(
                    message_count=4,
                    initial_prompt="hello world",
                ),
            ) as mock_summary,
        ):
            await sessions._populate_checkpoint_fields(  # pyright: ignore[reportPrivateUsage]
                cast(
                    "aiosqlite.Connection",
                    object(),  # connection is unused by the mocked loader
                ),
                threads,
                include_message_count=True,
                include_initial_prompt=True,
            )

        assert threads[0]["message_count"] == 4
        assert threads[0]["initial_prompt"] == "hello world"
        assert mock_summary.await_count == 1


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


class TestGetCachedThreads:
    """Tests for cached thread snapshot retrieval."""

    def test_returns_exact_cached_limit(self) -> None:
        """Exact cache key should return copied rows."""
        sessions._recent_threads_cache.clear()
        try:
            sessions._recent_threads_cache[None, 5] = [
                {
                    "thread_id": "thread-a",
                    "agent_name": "agent1",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "message_count": 3,
                }
            ]
            rows = sessions.get_cached_threads(limit=5)
            assert rows is not None
            assert len(rows) == 1
            assert rows[0]["thread_id"] == "thread-a"
            rows[0]["thread_id"] = "mutated"
            assert sessions._recent_threads_cache[None, 5][0]["thread_id"] == "thread-a"
        finally:
            sessions._recent_threads_cache.clear()

    def test_uses_larger_cached_limit(self) -> None:
        """Larger cached window should satisfy smaller requested limit."""
        sessions._recent_threads_cache.clear()
        try:
            sessions._recent_threads_cache[None, 20] = [
                {
                    "thread_id": "thread-1",
                    "agent_name": "agent1",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                },
                {
                    "thread_id": "thread-2",
                    "agent_name": "agent1",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                },
            ]
            rows = sessions.get_cached_threads(limit=1)
            assert rows is not None
            assert len(rows) == 1
            assert rows[0]["thread_id"] == "thread-1"
        finally:
            sessions._recent_threads_cache.clear()

    def test_applies_cached_message_counts_to_snapshot(self) -> None:
        """Returned snapshot should hydrate counts from message-count cache."""
        sessions._recent_threads_cache.clear()
        sessions._message_count_cache.clear()
        try:
            sessions._recent_threads_cache[None, 5] = [
                {
                    "thread_id": "thread-a",
                    "agent_name": "agent1",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "latest_checkpoint_id": "cp_1",
                }
            ]
            sessions._message_count_cache["thread-a"] = ("cp_1", 9)

            rows = sessions.get_cached_threads(limit=5)

            assert rows is not None
            assert rows[0]["message_count"] == 9
            assert "message_count" not in sessions._recent_threads_cache[None, 5][0]
        finally:
            sessions._recent_threads_cache.clear()
            sessions._message_count_cache.clear()

    def test_applies_cached_initial_prompts_to_snapshot(self) -> None:
        """Returned snapshot should hydrate prompts from prompt cache."""
        sessions._recent_threads_cache.clear()
        sessions._initial_prompt_cache.clear()
        try:
            sessions._recent_threads_cache[None, 5] = [
                {
                    "thread_id": "thread-a",
                    "agent_name": "agent1",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "latest_checkpoint_id": "cp_1",
                }
            ]
            sessions._initial_prompt_cache["thread-a"] = ("cp_1", "hello world")

            rows = sessions.get_cached_threads(limit=5)

            assert rows is not None
            assert rows[0]["initial_prompt"] == "hello world"
            assert "initial_prompt" not in sessions._recent_threads_cache[None, 5][0]
        finally:
            sessions._recent_threads_cache.clear()
            sessions._initial_prompt_cache.clear()


class TestPrewarmThreadMessageCounts:
    """Tests for thread-selector cache prewarming."""

    async def test_prewarm_respects_visible_thread_columns(self) -> None:
        """Prewarm should only fetch checkpoint fields for visible columns."""
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
                "deepagents_cli.model_config.load_thread_columns",
                return_value={
                    "thread_id": False,
                    "messages": True,
                    "created_at": True,
                    "updated_at": True,
                    "git_branch": False,
                    "initial_prompt": False,
                    "agent_name": False,
                },
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

    async def test_prewarm_populates_checkpoint_details_before_caching(self) -> None:
        """Prefetched rows should include prompt/count data in the recent cache."""
        sessions._recent_threads_cache.clear()
        threads: list[sessions.ThreadInfo] = [
            {
                "thread_id": "thread-a",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
            }
        ]

        async def _populate(
            rows: list[sessions.ThreadInfo],
            *,
            include_message_count: bool,
            include_initial_prompt: bool,
        ) -> list[sessions.ThreadInfo]:
            await asyncio.sleep(0)
            assert include_message_count is True
            assert include_initial_prompt is True
            rows[0]["message_count"] = 6
            rows[0]["initial_prompt"] = "hello world"
            return rows

        try:
            with (
                patch.object(
                    sessions,
                    "list_threads",
                    new_callable=AsyncMock,
                    return_value=threads,
                ),
                patch.object(
                    sessions,
                    "populate_thread_checkpoint_details",
                    new_callable=AsyncMock,
                    side_effect=_populate,
                ) as mock_populate,
            ):
                await sessions.prewarm_thread_message_counts(limit=3)

            mock_populate.assert_awaited_once()
            cached = sessions.get_cached_threads(limit=3)
            assert cached is not None
            assert cached[0]["message_count"] == 6
            assert cached[0]["initial_prompt"] == "hello world"
        finally:
            sessions._recent_threads_cache.clear()

    async def test_unexpected_errors_log_warning(self) -> None:
        """Unexpected prewarm failures should be visible at warning level."""
        with (
            patch(
                "deepagents_cli.sessions.list_threads",
                new_callable=AsyncMock,
                side_effect=RuntimeError("unexpected type mismatch"),
            ),
            patch.object(sessions.logger, "warning") as mock_warning,
        ):
            await sessions.prewarm_thread_message_counts(limit=3)

        mock_warning.assert_called_once()


class TestCacheMessageCount:
    """Tests for message-count cache eviction behavior."""

    def test_overflow_evicts_oldest_entry_only(self) -> None:
        """Cache overflow should evict only the oldest key, not clear all keys."""
        sessions._message_count_cache.clear()
        try:
            with patch.object(sessions, "_MAX_MESSAGE_COUNT_CACHE", 2):
                sessions._cache_message_count("thread-1", "cp_1", 1)
                sessions._cache_message_count("thread-2", "cp_2", 2)
                sessions._cache_message_count("thread-3", "cp_3", 3)

            assert "thread-1" not in sessions._message_count_cache
            assert sessions._message_count_cache["thread-2"] == ("cp_2", 2)
            assert sessions._message_count_cache["thread-3"] == ("cp_3", 3)
        finally:
            sessions._message_count_cache.clear()


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


class TestGetThreadLimit:
    """Tests for get_thread_limit() env var parsing."""

    def test_default_when_unset(self) -> None:
        """Returns default limit when DA_CLI_RECENT_THREADS is not set."""
        env = {
            k: v
            for k, v in __import__("os").environ.items()
            if k != "DA_CLI_RECENT_THREADS"
        }
        with patch.dict("os.environ", env, clear=True):
            assert get_thread_limit() == 20

    def test_custom_value(self) -> None:
        """Returns parsed integer from DA_CLI_RECENT_THREADS."""
        with patch.dict("os.environ", {"DA_CLI_RECENT_THREADS": "50"}):
            assert get_thread_limit() == 50

    def test_invalid_value_falls_back(self) -> None:
        """Returns default when DA_CLI_RECENT_THREADS is not a valid integer."""
        with patch.dict("os.environ", {"DA_CLI_RECENT_THREADS": "abc"}):
            assert get_thread_limit() == 20

    def test_zero_clamps_to_one(self) -> None:
        """Returns 1 when DA_CLI_RECENT_THREADS is 0."""
        with patch.dict("os.environ", {"DA_CLI_RECENT_THREADS": "0"}):
            assert get_thread_limit() == 1

    def test_negative_clamps_to_one(self) -> None:
        """Returns 1 when DA_CLI_RECENT_THREADS is negative."""
        with patch.dict("os.environ", {"DA_CLI_RECENT_THREADS": "-5"}):
            assert get_thread_limit() == 1


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

    def test_sort_by_updated(self, db_with_branches: Path) -> None:
        """Default sort returns most recently updated first."""
        with patch.object(sessions, "get_db_path", return_value=db_with_branches):
            threads = asyncio.run(sessions.list_threads(sort_by="updated"))
            assert threads[0]["thread_id"] == "thread_a"

    def test_sort_by_created(self, db_with_branches: Path) -> None:
        """Most recently created first (thread_b: March > thread_a: Jan)."""
        with patch.object(sessions, "get_db_path", return_value=db_with_branches):
            threads = asyncio.run(sessions.list_threads(sort_by="created"))
            assert threads[0]["thread_id"] == "thread_b"

    def test_filter_by_branch(self, db_with_branches: Path) -> None:
        """Branch filter returns only matching threads."""
        with patch.object(sessions, "get_db_path", return_value=db_with_branches):
            threads = asyncio.run(sessions.list_threads(branch="feat"))
            assert len(threads) == 1
            assert threads[0]["thread_id"] == "thread_b"
            assert threads[0]["git_branch"] == "feat"

    def test_filter_by_branch_no_match(self, db_with_branches: Path) -> None:
        """Branch filter returns empty list when no match."""
        with patch.object(sessions, "get_db_path", return_value=db_with_branches):
            threads = asyncio.run(sessions.list_threads(branch="nonexistent"))
            assert threads == []

    def test_combined_agent_and_branch_filter(self, db_with_branches: Path) -> None:
        """Agent + branch filters combine with AND."""
        with patch.object(sessions, "get_db_path", return_value=db_with_branches):
            threads = asyncio.run(
                sessions.list_threads(agent_name="bot", branch="main")
            )
            assert len(threads) == 1
            assert threads[0]["thread_id"] == "thread_a"


class TestListThreadsCommandConfigDefaults:
    """Tests for list_threads_command reading config defaults."""

    _THREAD: ClassVar[dict[str, str | int]] = {
        "thread_id": "abc123",
        "agent_name": "bot",
        "message_count": 2,
        "updated_at": "2025-06-01T12:00:00+00:00",
        "created_at": "2025-05-30T10:00:00+00:00",
    }

    def test_sort_reads_config_when_not_specified(self) -> None:
        """sort_by=None falls back to config value."""
        with (
            patch(
                "deepagents_cli.model_config.load_thread_sort_order",
                return_value="created_at",
            ),
            patch(
                "deepagents_cli.model_config.load_thread_relative_time",
                return_value=False,
            ),
            patch(
                "deepagents_cli.sessions.list_threads",
                new_callable=AsyncMock,
                return_value=[self._THREAD],
            ) as mock_list,
            patch("deepagents_cli.sessions.format_timestamp", side_effect=str),
            patch("deepagents_cli.config.console"),
        ):
            asyncio.run(sessions.list_threads_command())
            mock_list.assert_called_once()
            assert mock_list.call_args.kwargs["sort_by"] == "created"

    def test_sort_flag_overrides_config(self) -> None:
        """Explicit sort_by overrides config."""
        with (
            patch(
                "deepagents_cli.model_config.load_thread_sort_order",
                return_value="created_at",
            ),
            patch(
                "deepagents_cli.model_config.load_thread_relative_time",
                return_value=False,
            ),
            patch(
                "deepagents_cli.sessions.list_threads",
                new_callable=AsyncMock,
                return_value=[self._THREAD],
            ) as mock_list,
            patch("deepagents_cli.sessions.format_timestamp", side_effect=str),
            patch("deepagents_cli.config.console"),
        ):
            asyncio.run(sessions.list_threads_command(sort_by="updated"))
            mock_list.assert_called_once()
            assert mock_list.call_args.kwargs["sort_by"] == "updated"

    def test_relative_reads_config_when_not_specified(self) -> None:
        """relative=None falls back to config value."""
        with (
            patch(
                "deepagents_cli.model_config.load_thread_sort_order",
                return_value="updated_at",
            ),
            patch(
                "deepagents_cli.model_config.load_thread_relative_time",
                return_value=True,
            ),
            patch(
                "deepagents_cli.sessions.list_threads",
                new_callable=AsyncMock,
                return_value=[self._THREAD],
            ),
            patch(
                "deepagents_cli.sessions.format_relative_timestamp",
                side_effect=str,
            ) as mock_rel,
            patch("deepagents_cli.sessions.format_timestamp") as mock_abs,
            patch("deepagents_cli.config.console"),
        ):
            asyncio.run(sessions.list_threads_command())
            assert mock_rel.call_count > 0
            assert mock_abs.call_count == 0

    def test_relative_flag_overrides_config(self) -> None:
        """Explicit relative=False overrides config True."""
        with (
            patch(
                "deepagents_cli.model_config.load_thread_sort_order",
                return_value="updated_at",
            ),
            patch(
                "deepagents_cli.model_config.load_thread_relative_time",
                return_value=True,
            ),
            patch(
                "deepagents_cli.sessions.list_threads",
                new_callable=AsyncMock,
                return_value=[self._THREAD],
            ),
            patch(
                "deepagents_cli.sessions.format_relative_timestamp",
            ) as mock_rel,
            patch(
                "deepagents_cli.sessions.format_timestamp",
                side_effect=str,
            ) as mock_abs,
            patch("deepagents_cli.config.console"),
        ):
            asyncio.run(sessions.list_threads_command(relative=False))
            assert mock_abs.call_count > 0
            assert mock_rel.call_count == 0

    def test_branch_forwarded_to_list_threads(self) -> None:
        """Branch parameter is passed through to list_threads."""
        with (
            patch(
                "deepagents_cli.model_config.load_thread_sort_order",
                return_value="updated_at",
            ),
            patch(
                "deepagents_cli.model_config.load_thread_relative_time",
                return_value=False,
            ),
            patch(
                "deepagents_cli.sessions.list_threads",
                new_callable=AsyncMock,
                return_value=[self._THREAD],
            ) as mock_list,
            patch("deepagents_cli.sessions.format_timestamp", side_effect=str),
            patch("deepagents_cli.config.console"),
        ):
            asyncio.run(sessions.list_threads_command(branch="main"))
            mock_list.assert_called_once()
            assert mock_list.call_args.kwargs["branch"] == "main"

    def test_verbose_calls_populate_details(self) -> None:
        """verbose=True triggers populate_thread_checkpoint_details."""
        with (
            patch(
                "deepagents_cli.model_config.load_thread_sort_order",
                return_value="updated_at",
            ),
            patch(
                "deepagents_cli.model_config.load_thread_relative_time",
                return_value=False,
            ),
            patch(
                "deepagents_cli.sessions.list_threads",
                new_callable=AsyncMock,
                return_value=[{**self._THREAD, "git_branch": "main"}],
            ),
            patch(
                "deepagents_cli.sessions.populate_thread_checkpoint_details",
                new_callable=AsyncMock,
            ) as mock_populate,
            patch("deepagents_cli.sessions.format_timestamp", side_effect=str),
            patch("deepagents_cli.config.console"),
        ):
            asyncio.run(sessions.list_threads_command(verbose=True))
            mock_populate.assert_called_once()
            assert mock_populate.call_args.kwargs["include_initial_prompt"] is True
