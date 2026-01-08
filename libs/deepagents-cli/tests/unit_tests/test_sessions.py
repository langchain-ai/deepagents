"""Tests for session/thread management."""

import asyncio
import json
import sqlite3
from unittest.mock import patch

import pytest

from deepagents_cli import sessions


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
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        earlier = "2024-01-01T10:00:00+00:00"

        threads = [
            ("thread1", "agent1", now),
            ("thread2", "agent2", earlier),
            ("thread3", "agent1", earlier),
        ]

        for tid, agent, updated in threads:
            metadata = json.dumps({"agent_name": agent, "updated_at": updated})
            conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
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
        """List returns all threads."""
        with patch.object(sessions, "get_db_path", return_value=temp_db):
            threads = asyncio.run(sessions.list_threads())
            assert len(threads) == 3

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

        async def _test():
            db_path = tmp_path / "test.db"
            with patch.object(sessions, "get_db_path", return_value=db_path):
                async with sessions.get_checkpointer() as cp:
                    assert "AsyncSqliteSaver" in type(cp).__name__

        asyncio.run(_test())


class TestFormatTimestamp:
    """Tests for _format_timestamp helper."""

    def test_valid_timestamp(self):
        """Formats valid ISO timestamp."""
        result = sessions._format_timestamp("2024-12-30T21:18:00+00:00")
        assert result  # Non-empty string
        assert "dec" in result.lower()

    def test_none(self):
        """Returns empty for None."""
        result = sessions._format_timestamp(None)
        assert result == ""

    def test_invalid(self):
        """Returns empty for invalid timestamp."""
        result = sessions._format_timestamp("not a timestamp")
        assert result == ""


class TestTextualSessionState:
    """Tests for TextualSessionState from app.py."""

    def test_stores_provided_thread_id(self):
        """TextualSessionState stores provided thread_id."""
        from deepagents_cli.app import TextualSessionState

        tid = sessions.generate_thread_id()
        state = TextualSessionState(thread_id=tid)
        assert state.thread_id == tid

    def test_generates_id_if_none(self):
        """TextualSessionState generates ID if none provided."""
        from deepagents_cli.app import TextualSessionState

        state = TextualSessionState(thread_id=None)
        assert state.thread_id is not None
        assert len(state.thread_id) == 8

    def test_reset_thread(self):
        """reset_thread generates a new thread ID."""
        from deepagents_cli.app import TextualSessionState

        state = TextualSessionState(thread_id="original")
        old_id = state.thread_id
        new_id = state.reset_thread()
        assert new_id != old_id
        assert len(new_id) == 8
        assert state.thread_id == new_id
