"""Unit tests for _validate_path() function."""

import pytest

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import _validate_path


class TestValidatePath:
    """Test cases for path validation and normalization."""

    def test_relative_path_normalization(self):
        """Test that relative paths get normalized with leading slash."""
        assert _validate_path("foo/bar") == "/foo/bar"
        assert _validate_path("relative/path.txt") == "/relative/path.txt"

    def test_absolute_path_normalization(self):
        """Test that absolute virtual paths are preserved."""
        assert _validate_path("/workspace/file.txt") == "/workspace/file.txt"
        assert _validate_path("/output/report.csv") == "/output/report.csv"

    def test_path_normalization_removes_redundant_separators(self):
        """Test that redundant path separators are normalized."""
        assert _validate_path("/./foo//bar") == "/foo/bar"
        assert _validate_path("foo/./bar") == "/foo/bar"

    def test_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("foo/../../etc/passwd")

    def test_home_directory_expansion_rejected(self):
        """Test that home directory expansion is rejected."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("~/secret.txt")

    def test_windows_absolute_path_allowed_when_virtual_mode_false(self):
        """Test that Windows absolute paths are allowed when virtual_mode=False."""
        backend = FilesystemBackend(virtual_mode=False)
        # Windows paths should pass through unchanged when virtual_mode=False
        assert _validate_path("C:\\Users\\Documents\\file.txt", backend=backend) == "C:\\Users\\Documents\\file.txt"
        assert _validate_path("F:\\git\\project\\file.txt", backend=backend) == "F:\\git\\project\\file.txt"
        assert _validate_path("C:/Users/Documents/file.txt", backend=backend) == "C:/Users/Documents/file.txt"
        assert _validate_path("D:/data/output.csv", backend=backend) == "D:/data/output.csv"

    def test_windows_absolute_path_allowed_when_no_backend(self):
        """Test that Windows absolute paths are allowed when no backend is provided (default behavior)."""
        # When no backend is provided, Windows paths should pass through (treated as virtual_mode=False)
        assert _validate_path("C:\\Users\\Documents\\file.txt") == "C:\\Users\\Documents\\file.txt"
        assert _validate_path("F:\\git\\project\\file.txt") == "F:\\git\\project\\file.txt"
        assert _validate_path("C:/Users/Documents/file.txt") == "C:/Users/Documents/file.txt"
        assert _validate_path("D:/data/output.csv") == "D:/data/output.csv"

    def test_windows_absolute_path_rejected_when_virtual_mode_true(self):
        """Test that Windows absolute paths are rejected when virtual_mode=True."""
        backend = FilesystemBackend(virtual_mode=True)
        with pytest.raises(ValueError, match="Windows absolute paths are not supported in virtual_mode"):
            _validate_path("C:\\Users\\Documents\\file.txt", backend=backend)

        with pytest.raises(ValueError, match="Windows absolute paths are not supported in virtual_mode"):
            _validate_path("F:\\git\\project\\file.txt", backend=backend)

        with pytest.raises(ValueError, match="Windows absolute paths are not supported in virtual_mode"):
            _validate_path("C:/Users/Documents/file.txt", backend=backend)

        with pytest.raises(ValueError, match="Windows absolute paths are not supported in virtual_mode"):
            _validate_path("D:/data/output.csv", backend=backend)

    def test_allowed_prefixes_enforcement(self):
        """Test that allowed_prefixes parameter is enforced."""
        # Should pass when prefix matches
        result = _validate_path("/workspace/file.txt", allowed_prefixes=["/workspace/"])
        assert result == "/workspace/file.txt"

        # Should fail when prefix doesn't match
        with pytest.raises(ValueError, match="Path must start with one of"):
            _validate_path("/etc/file.txt", allowed_prefixes=["/workspace/"])

    def test_backslash_normalization(self):
        """Test that backslashes in relative paths are normalized to forward slashes."""
        # Relative paths with backslashes should be normalized
        assert _validate_path("foo\\bar\\baz") == "/foo/bar/baz"
