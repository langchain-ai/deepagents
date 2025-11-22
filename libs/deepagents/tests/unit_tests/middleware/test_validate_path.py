"""Unit tests for _validate_path() function."""

import pytest

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

    def test_windows_absolute_path_rejected_backslash(self):
        """Test that Windows absolute paths with backslashes are rejected."""
        with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
            _validate_path("C:\\Users\\Documents\\file.txt")

        with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
            _validate_path("F:\\git\\project\\file.txt")

    def test_windows_absolute_path_rejected_forward_slash(self):
        """Test that Windows absolute paths with forward slashes are rejected."""
        with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
            _validate_path("C:/Users/Documents/file.txt")

        with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
            _validate_path("D:/data/output.csv")

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
