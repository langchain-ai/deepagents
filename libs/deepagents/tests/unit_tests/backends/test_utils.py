"""Tests for backends/utils.py utility functions."""

from typing import Any

import pytest

from deepagents.backends.utils import _glob_search_files, validate_path


class TestValidatePath:
    """Tests for validate_path - the canonical path validation function."""

    @pytest.mark.parametrize(
        ("input_path", "expected"),
        [
            ("foo/bar", "/foo/bar"),
            ("/workspace/file.txt", "/workspace/file.txt"),
            ("/./foo//bar", "/foo/bar"),
            ("foo\\bar\\baz", "/foo/bar/baz"),
            ("foo/bar\\baz/qux", "/foo/bar/baz/qux"),
        ],
    )
    def test_path_normalization(self, input_path: str, expected: str) -> None:
        """Test various path normalization scenarios."""
        assert validate_path(input_path) == expected

    @pytest.mark.parametrize(
        ("invalid_path", "error_match"),
        [
            ("../etc/passwd", "Path traversal not allowed"),
            ("foo/../../etc", "Path traversal not allowed"),
            ("~/secret.txt", "Path traversal not allowed"),
            ("C:\\Users\\file.txt", "Windows absolute paths are not supported"),
            ("D:/data/file.txt", "Windows absolute paths are not supported"),
        ],
    )
    def test_invalid_paths_rejected(self, invalid_path: str, error_match: str) -> None:
        """Test that dangerous paths are rejected."""
        with pytest.raises(ValueError, match=error_match):
            validate_path(invalid_path)

    def test_allowed_prefixes_enforced(self) -> None:
        """Test allowed_prefixes parameter."""
        assert validate_path("/workspace/file.txt", allowed_prefixes=["/workspace/"]) == "/workspace/file.txt"

        with pytest.raises(ValueError, match="Path must start with one of"):
            validate_path("/etc/passwd", allowed_prefixes=["/workspace/"])

    def test_no_backslashes_in_output(self) -> None:
        """Test that output never contains backslashes."""
        paths = ["foo\\bar", "a\\b\\c\\d", "mixed/path\\here"]
        for path in paths:
            result = validate_path(path)
            assert "\\" not in result, f"Backslash in output for input '{path}': {result}"

    def test_root_path(self) -> None:
        """Test that root path normalizes correctly."""
        assert validate_path("/") == "/"

    def test_double_dots_in_filename_allowed(self) -> None:
        """Test that filenames containing `'..'` as a substring are not rejected.

        Only `'..'` as a path component (directory traversal) should be rejected.
        """
        assert validate_path("foo..bar.txt") == "/foo..bar.txt"
        assert validate_path("backup..2024/data.csv") == "/backup..2024/data.csv"
        assert validate_path("v2..0/release") == "/v2..0/release"

    def test_allowed_prefixes_boundary(self) -> None:
        """Test that prefix matching requires exact directory boundary.

        `'/workspace-evil/file'` should NOT match prefix `'/workspace/'`.
        """
        with pytest.raises(ValueError, match="Path must start with one of"):
            validate_path("/workspace-evil/file", allowed_prefixes=["/workspace/"])

    def test_traversal_as_path_component_rejected(self) -> None:
        """Test that `'..'` as a path component is still rejected."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            validate_path("foo/../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal not allowed"):
            validate_path("/workspace/../../../etc/shadow")

    def test_dot_and_empty_string_normalize_to_slash_dot(self) -> None:
        """Document that `'.'` and `''` normalize to `'/.'` via `os.path.normpath`."""
        assert validate_path(".") == "/."
        assert validate_path("") == "/."


class TestGlobSearchFiles:
    """Tests for _glob_search_files."""

    @pytest.fixture
    def sample_files(self) -> dict[str, Any]:
        """Sample files dict."""
        return {
            "/src/main.py": {"modified_at": "2024-01-01T10:00:00"},
            "/src/utils/helper.py": {"modified_at": "2024-01-01T11:00:00"},
            "/src/utils/common.py": {"modified_at": "2024-01-01T09:00:00"},
            "/docs/readme.md": {"modified_at": "2024-01-01T08:00:00"},
            "/test.py": {"modified_at": "2024-01-01T12:00:00"},
        }

    def test_basic_glob(self, sample_files: dict[str, Any]) -> None:
        """Test basic glob matching."""
        result = _glob_search_files(sample_files, "*.py", "/")
        assert "/test.py" in result

    def test_recursive_glob(self, sample_files: dict[str, Any]) -> None:
        """Test recursive glob pattern."""
        result = _glob_search_files(sample_files, "**/*.py", "/")
        assert "/src/main.py" in result
        assert "/src/utils/helper.py" in result

    def test_path_filter(self, sample_files: dict[str, Any]) -> None:
        """Test glob respects path parameter."""
        result = _glob_search_files(sample_files, "*.py", "/src/utils/")
        assert "/src/utils/helper.py" in result
        assert "/src/main.py" not in result

    def test_no_matches(self, sample_files: dict[str, Any]) -> None:
        """Test no matches returns message."""
        assert _glob_search_files(sample_files, "*.xyz", "/") == "No files found"

    def test_sorted_by_modification_time(self, sample_files: dict[str, Any]) -> None:
        """Test results sorted by modification time (most recent first)."""
        result = _glob_search_files(sample_files, "**/*.py", "/")
        assert result.strip().split("\n")[0] == "/test.py"

    def test_path_traversal_rejected(self, sample_files: dict[str, Any]) -> None:
        """Test that path traversal in path parameter is rejected."""
        result = _glob_search_files(sample_files, "*.py", "../etc/")
        assert result == "No files found"
