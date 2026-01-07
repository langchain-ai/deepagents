"""Tests for sessions module - thread management functionality."""

import re
from pathlib import Path
from unittest import mock

import pytest

from deepagents_cli.sessions import (
    _format_timestamp,
    delete_thread,
    generate_thread_id,
    get_db_path,
    get_most_recent,
    get_thread_agent,
    list_threads,
    thread_exists,
)


class TestGenerateThreadId:
    """Test thread ID generation."""

    def test_generates_8_char_hex(self) -> None:
        """Test that generated IDs are 8 character hex strings."""
        thread_id = generate_thread_id()
        assert len(thread_id) == 8
        assert re.match(r"^[0-9a-f]{8}$", thread_id)

    def test_generates_unique_ids(self) -> None:
        """Test that generated IDs are unique."""
        ids = [generate_thread_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestGetDbPath:
    """Test database path resolution."""

    def test_returns_path_in_deepagents_dir(self) -> None:
        """Test that db path is in ~/.deepagents directory."""
        db_path = get_db_path()
        assert db_path.parent == Path.home() / ".deepagents"
        assert db_path.name == "sessions.db"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that parent directory is created if it doesn't exist."""
        with mock.patch.object(Path, "home", return_value=tmp_path):
            db_path = get_db_path()
            assert db_path.parent.exists()


class TestFormatTimestamp:
    """Test timestamp formatting."""

    def test_formats_valid_timestamp(self) -> None:
        """Test formatting a valid ISO timestamp."""
        result = _format_timestamp("2025-12-30T18:10:00+00:00")
        # Result should contain month and time
        assert "dec" in result.lower()
        assert "30" in result

    def test_returns_empty_for_none(self) -> None:
        """Test that None returns empty string."""
        assert _format_timestamp(None) == ""

    def test_returns_empty_for_invalid(self) -> None:
        """Test that invalid timestamp returns empty string."""
        assert _format_timestamp("not-a-timestamp") == ""


@pytest.fixture
def temp_db_path(tmp_path: Path):
    """Fixture to create a temporary database path."""
    db_dir = tmp_path / ".deepagents"
    db_dir.mkdir(parents=True)
    return db_dir / "sessions.db"


@pytest.fixture
def mock_db_path(temp_db_path: Path):
    """Fixture to mock get_db_path to use temp directory."""
    with mock.patch("deepagents_cli.sessions.get_db_path", return_value=temp_db_path):
        yield temp_db_path


@pytest.mark.asyncio
class TestListThreads:
    """Test listing threads from database."""

    async def test_returns_empty_list_when_no_db(self, mock_db_path: Path) -> None:
        """Test that empty list is returned when no database exists."""
        # Database doesn't exist yet - should handle gracefully
        # Note: aiosqlite will create the file, but the table won't exist
        # We should verify the function handles this case
        # This test may need adjustment based on actual error handling

    async def test_returns_empty_list_for_empty_db(self, mock_db_path: Path) -> None:
        """Test that empty list is returned when database has no threads."""
        import aiosqlite

        # Create the database with proper schema
        async with aiosqlite.connect(str(mock_db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            await conn.commit()

        threads = await list_threads()
        assert threads == []


@pytest.mark.asyncio
class TestThreadExists:
    """Test thread existence checking."""

    async def test_returns_false_for_nonexistent_thread(self, mock_db_path: Path) -> None:
        """Test that False is returned for non-existent thread."""
        import aiosqlite

        async with aiosqlite.connect(str(mock_db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            await conn.commit()

        exists = await thread_exists("nonexistent")
        assert exists is False

    async def test_returns_true_for_existing_thread(self, mock_db_path: Path) -> None:
        """Test that True is returned for existing thread."""
        import aiosqlite

        async with aiosqlite.connect(str(mock_db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id, metadata) VALUES (?, ?, ?)",
                ("abc12345", "cp1", '{"agent_name": "test-agent"}'),
            )
            await conn.commit()

        exists = await thread_exists("abc12345")
        assert exists is True


@pytest.mark.asyncio
class TestGetMostRecent:
    """Test getting most recent thread."""

    async def test_returns_none_when_no_threads(self, mock_db_path: Path) -> None:
        """Test that None is returned when no threads exist."""
        import aiosqlite

        async with aiosqlite.connect(str(mock_db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            await conn.commit()

        result = await get_most_recent()
        assert result is None

    async def test_returns_most_recent_thread(self, mock_db_path: Path) -> None:
        """Test that most recent thread is returned."""
        import aiosqlite

        async with aiosqlite.connect(str(mock_db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            # Insert threads - later checkpoint_id = more recent
            await conn.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?)",
                ("thread1", "cp1", '{"agent_name": "agent1"}'),
            )
            await conn.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?)",
                ("thread2", "cp2", '{"agent_name": "agent2"}'),
            )
            await conn.commit()

        result = await get_most_recent()
        assert result == "thread2"


@pytest.mark.asyncio
class TestGetThreadAgent:
    """Test getting agent name for a thread."""

    async def test_returns_none_for_nonexistent_thread(self, mock_db_path: Path) -> None:
        """Test that None is returned for non-existent thread."""
        import aiosqlite

        async with aiosqlite.connect(str(mock_db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            await conn.commit()

        result = await get_thread_agent("nonexistent")
        assert result is None

    async def test_returns_agent_name_for_existing_thread(self, mock_db_path: Path) -> None:
        """Test that agent name is returned for existing thread."""
        import aiosqlite

        async with aiosqlite.connect(str(mock_db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            await conn.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?)",
                ("mythread", "cp1", '{"agent_name": "my-custom-agent"}'),
            )
            await conn.commit()

        result = await get_thread_agent("mythread")
        assert result == "my-custom-agent"


@pytest.mark.asyncio
class TestDeleteThread:
    """Test thread deletion."""

    async def test_returns_false_for_nonexistent_thread(self, mock_db_path: Path) -> None:
        """Test that False is returned when thread doesn't exist."""
        import aiosqlite

        async with aiosqlite.connect(str(mock_db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT,
                    data TEXT
                )
            """)
            await conn.commit()

        result = await delete_thread("nonexistent")
        assert result is False

    async def test_deletes_existing_thread(self, mock_db_path: Path) -> None:
        """Test that existing thread is deleted."""
        import aiosqlite

        async with aiosqlite.connect(str(mock_db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT,
                    data TEXT
                )
            """)
            await conn.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?)",
                ("to-delete", "cp1", '{"agent_name": "test"}'),
            )
            await conn.commit()

        result = await delete_thread("to-delete")
        assert result is True

        # Verify it's actually deleted
        exists = await thread_exists("to-delete")
        assert exists is False
