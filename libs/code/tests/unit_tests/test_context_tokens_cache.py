"""Unit tests for `deepagents_code.context_tokens_cache`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_code import context_tokens_cache

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point the cache at a fresh temp directory for each test."""
    cache_dir = tmp_path / "context_tokens"
    monkeypatch.setattr(context_tokens_cache, "CONTEXT_TOKENS_DIR", cache_dir)
    return cache_dir


class TestReadContextTokens:
    """`read_context_tokens` behavior."""

    def test_returns_none_when_file_missing(self) -> None:
        """A thread with no cache file should report `None` (cache miss)."""
        assert context_tokens_cache.read_context_tokens("missing-thread") is None

    def test_returns_persisted_value(self) -> None:
        """A previously written value should round-trip."""
        context_tokens_cache.write_context_tokens("t-1", 5000)
        assert context_tokens_cache.read_context_tokens("t-1") == 5000

    def test_returns_zero_for_legitimately_persisted_zero(self) -> None:
        """A persisted `0` should round-trip as `0` (distinct from cache miss)."""
        context_tokens_cache.write_context_tokens("t-zero", 0)
        assert context_tokens_cache.read_context_tokens("t-zero") == 0

    def test_returns_none_for_corrupt_file(self, cache_dir: Path) -> None:
        """A non-JSON file should not crash; return `None`."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "t-bad.json").write_text("not json", encoding="utf-8")
        assert context_tokens_cache.read_context_tokens("t-bad") is None

    def test_returns_none_for_negative_value(
        self, cache_dir: Path
    ) -> None:
        """A persisted negative value should be treated as cache miss."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "t-neg.json").write_text(
            '{"tokens": -1}', encoding="utf-8"
        )
        assert context_tokens_cache.read_context_tokens("t-neg") is None

    def test_returns_none_for_unexpected_shape(self, cache_dir: Path) -> None:
        """A JSON value with the wrong shape should be treated as cache miss."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "t-shape.json").write_text('["unexpected"]', encoding="utf-8")
        assert context_tokens_cache.read_context_tokens("t-shape") is None

    def test_unsafe_thread_id_returns_none(self) -> None:
        """Path-traversal-y thread IDs should refuse to read, not escape."""
        assert context_tokens_cache.read_context_tokens("../escape") is None
        assert context_tokens_cache.read_context_tokens("") is None


class TestWriteContextTokens:
    """`write_context_tokens` behavior."""

    def test_creates_directory(self, cache_dir: Path) -> None:
        """Writes should create the cache directory if absent."""
        assert not cache_dir.exists()
        context_tokens_cache.write_context_tokens("t-1", 42)
        assert cache_dir.is_dir()

    def test_overwrites_previous_value(self, cache_dir: Path) -> None:
        """Successive writes should update the persisted value."""
        context_tokens_cache.write_context_tokens("t-1", 100)
        context_tokens_cache.write_context_tokens("t-1", 200)
        assert context_tokens_cache.read_context_tokens("t-1") == 200
        # The atomic-write pattern should never leave a `.tmp` behind.
        assert not list(cache_dir.glob("*.tmp"))

    def test_clamps_negative_to_zero(self) -> None:
        """Negative inputs should be normalized to 0 on write."""
        context_tokens_cache.write_context_tokens("t-1", -5)
        assert context_tokens_cache.read_context_tokens("t-1") == 0

    def test_unsafe_thread_id_does_not_write(self, cache_dir: Path) -> None:
        """Path-traversal-y thread IDs should refuse to write, not escape."""
        context_tokens_cache.write_context_tokens("../escape", 123)
        assert not cache_dir.exists() or not any(cache_dir.iterdir())

    def test_swallows_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Filesystem errors during write must be best-effort, never raise."""
        # Point the cache dir at a file path so mkdir(parents=True) fails
        # with NotADirectoryError (an OSError subclass).
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(
            context_tokens_cache, "CONTEXT_TOKENS_DIR", blocked / "ctx"
        )

        context_tokens_cache.write_context_tokens("t-1", 42)  # must not raise


class TestDeleteContextTokens:
    """`delete_context_tokens` behavior."""

    def test_removes_existing_file(self) -> None:
        context_tokens_cache.write_context_tokens("t-1", 42)
        context_tokens_cache.delete_context_tokens("t-1")
        assert context_tokens_cache.read_context_tokens("t-1") is None

    def test_missing_file_is_noop(self) -> None:
        """Deleting a nonexistent thread's cache should not raise."""
        context_tokens_cache.delete_context_tokens("never-existed")
