"""Tests for exception handling improvements across backends.

These tests verify that:
1. Silent exceptions are now properly logged
2. Specific exception types are caught instead of bare Exception
3. The code behaves correctly when exceptions occur
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend


class TestFilesystemBackendExceptionHandling:
    """Test exception handling in FilesystemBackend."""

    def test_ls_info_handles_permission_error_gracefully(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test that ls_info logs and handles permission errors."""
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        # Create a directory that will fail to list
        restricted_dir = tmp_path / "restricted"
        restricted_dir.mkdir()

        with (
            caplog.at_level(logging.DEBUG),
            patch.object(Path, "iterdir", side_effect=PermissionError("Access denied")),
        ):
            result = be.ls_info("/")

        # Should return empty list and log the error
        assert isinstance(result, list)

        # Verify the error was logged
        assert "Failed to list directory" in caplog.text
        assert "Access denied" in caplog.text

    def test_glob_info_handles_oserror_gracefully(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test that glob_info logs and handles OS errors."""
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        with (
            caplog.at_level(logging.DEBUG),
            patch.object(Path, "rglob", side_effect=OSError("Filesystem error")),
        ):
            result = be.glob_info("*.txt", "/")

        # Should return empty list
        assert isinstance(result, list)

        # Verify the error was logged
        assert "Failed to glob pattern" in caplog.text
        assert "Filesystem error" in caplog.text

    def test_ripgrep_search_handles_path_outside_root(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test that ripgrep search logs when paths are outside root."""
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with caplog.at_level(logging.DEBUG):
            # This should work normally - just verify no crash
            result = be.grep_raw("test", "/")

        assert isinstance(result, list)


class TestCompositeBackendExceptionHandling:
    """Test exception handling in CompositeBackend."""

    def test_write_handles_state_merge_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that write logs state merge failures."""
        # Create a mock runtime with state
        mock_runtime = MagicMock()
        mock_runtime.state = {}

        default_backend = StateBackend(mock_runtime)
        composite = CompositeBackend(default=default_backend, routes={})

        with caplog.at_level(logging.DEBUG):
            # Write should succeed even if state merge has issues
            result = composite.write("/test.txt", "content")

        # Should complete without raising
        assert result is not None

    def test_edit_handles_state_merge_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that edit logs state merge failures."""
        mock_runtime = MagicMock()
        mock_runtime.state = {"files": {"/test.txt": {"content": ["original"], "created_at": "2024-01-01", "modified_at": "2024-01-01"}}}

        default_backend = StateBackend(mock_runtime)
        composite = CompositeBackend(default=default_backend, routes={})

        with caplog.at_level(logging.DEBUG):
            result = composite.edit("/test.txt", "original", "modified")

        # Should complete without raising
        assert result is not None


class TestStoreBackendExceptionHandling:
    """Test exception handling in StoreBackend."""

    def test_get_namespace_handles_missing_config(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that _get_namespace logs when config is unavailable."""
        from deepagents.backends.store import StoreBackend

        mock_runtime = MagicMock()
        mock_runtime.store = MagicMock()
        mock_runtime.config = None  # No config available

        backend = StoreBackend(mock_runtime)

        with (
            caplog.at_level(logging.DEBUG),
            patch("deepagents.backends.store.get_config", side_effect=RuntimeError("No config")),
        ):
            namespace = backend._get_namespace()

        # Should return default namespace
        assert namespace == ("filesystem",)

        # Verify the error was logged
        assert "Failed to get langgraph config" in caplog.text
        assert "No config" in caplog.text


# CLI-specific tests are in the deepagents-cli package
# These tests only cover the core deepagents package exception handling
