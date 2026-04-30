"""Tests for backends/utils.py utility functions."""

from typing import Any

import pytest
from langchain_core.messages.content import ContentBlock
from pydantic import TypeAdapter

from deepagents.backends.protocol import FileData, ReadResult
from deepagents.backends.utils import (
    _EXTENSION_TO_FILE_TYPE,
    _get_file_type,
    _glob_search_files,
    perform_string_replacement,
    slice_read_response,
    to_posix_path,
    validate_path,
)


class TestToPosixPath:
    """`to_posix_path` is the load-bearing primitive for Windows path handling."""

    @pytest.mark.parametrize(
        ("input_path", "expected"),
        [
            (r"C:\Users\project\file.txt", "C:/Users/project/file.txt"),
            (r"C:\Users\project\skills\my-skill\\", "C:/Users/project/skills/my-skill//"),
            ("already/posix/path", "already/posix/path"),
            (r"mixed/sep\path/with\backslash", "mixed/sep/path/with/backslash"),
            (r"\\server\share\file", "//server/share/file"),
            ("", ""),
            ("/", "/"),
            (r"\\", "//"),
            ("/foo/bar", "/foo/bar"),
        ],
        ids=[
            "windows-drive",
            "trailing-backslash",
            "already-posix",
            "mixed-separators",
            "unc",
            "empty",
            "root",
            "bare-backslashes",
            "posix-absolute",
        ],
    )
    def test_normalizes(self, input_path: str, expected: str) -> None:
        assert to_posix_path(input_path) == expected


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

    def test_leading_slash_in_pattern(self, sample_files: dict[str, Any]) -> None:
        """Patterns with a leading slash should still match (models often produce them)."""
        result = _glob_search_files(sample_files, "/src/**/*.py", "/")
        assert "/src/main.py" in result
        assert "/src/utils/helper.py" in result

    def test_leading_slash_pattern_with_subdir_path(self) -> None:
        """Leading-slash pattern scoped to a subdirectory path."""
        files = {
            "/foo/a.md": {"modified_at": "2024-01-01T10:00:00"},
            "/foo/b.txt": {"modified_at": "2024-01-01T09:00:00"},
            "/foo/c.md": {"modified_at": "2024-01-01T08:00:00"},
        }
        result = _glob_search_files(files, "/foo/**/*.md", "/")
        assert "/foo/a.md" in result
        assert "/foo/c.md" in result
        assert "/foo/b.txt" not in result


_content_block_adapter = TypeAdapter(ContentBlock)


def test_get_file_type_returns_text_for_unknown_extensions() -> None:
    assert _get_file_type("/foo/bar.txt") == "text"
    assert _get_file_type("/foo/bar.py") == "text"
    assert _get_file_type("/foo/bar") == "text"


def test_get_file_type_non_text_values_are_valid_content_block_types() -> None:
    """Every non-text file type must be accepted as a ContentBlock `type`."""
    for file_type in _EXTENSION_TO_FILE_TYPE.values():
        block = {"type": file_type, "base64": "dGVzdA==", "mime_type": "application/octet-stream"}
        _content_block_adapter.validate_python(block)


class TestPerformStringReplacement:
    """`perform_string_replacement` underpins every backend's `edit()` path."""

    def test_basic_single_replacement(self) -> None:
        result = perform_string_replacement("hello world", "world", "there")
        assert result == ("hello there", 1)

    def test_not_found_returns_error_string(self) -> None:
        result = perform_string_replacement("hello world", "missing", "x")
        assert isinstance(result, str)
        assert "not found" in result

    def test_multiple_matches_without_replace_all(self) -> None:
        result = perform_string_replacement("a a a", "a", "b")
        assert isinstance(result, str)
        assert "appears 3 times" in result

    def test_multiple_matches_with_replace_all(self) -> None:
        result = perform_string_replacement("a a a", "a", "b", replace_all=True)
        assert result == ("b b b", 3)

    def test_eof_newline_mismatch_returns_actionable_error(self) -> None:
        """Trailing-newline mismatch at EOF must surface a precise hint.

        Models infer terminators on what looks like a well-formed line. When
        the file lacks one, exact-match must hold; the caller needs an
        actionable error so it can self-correct rather than loop on a
        generic "not found".
        """
        content = "# Agent Role:\nyou are an assistant"
        old_string = "# Agent Role:\nyou are an assistant\n"
        new_string = "# Agent Role:\nyou are an assistant\nYou can do anything\n"
        result = perform_string_replacement(content, old_string, new_string)
        assert isinstance(result, str)
        assert "old_string ends with a newline" in result
        assert "Retry with the trailing newline removed" in result

    def test_eof_newline_mismatch_reports_ambiguous_stripped_match(self) -> None:
        """When the stripped key would also be ambiguous, the hint must say so.

        Otherwise the caller fixes the trailing newline, retries, and hits a
        separate `appears N times` error — two round-trips for one cause.
        """
        content = "x x x"
        old_string = "x\n"
        new_string = "Y\n"
        result = perform_string_replacement(content, old_string, new_string)
        assert isinstance(result, str)
        assert "old_string ends with a newline" in result
        assert "appear 3 times" in result
        assert "add surrounding context" in result

    def test_eof_newline_mismatch_does_not_fire_when_match_succeeds(self) -> None:
        """Primary match wins; the EOF-mismatch hint stays dormant."""
        content = "alpha\nbeta\n"
        old_string = "beta\n"
        new_string = "BETA\n"
        result = perform_string_replacement(content, old_string, new_string)
        assert result == ("alpha\nBETA\n", 1)

    def test_eof_newline_mismatch_does_not_fire_for_lone_newline(self) -> None:
        """A lone-newline `old_string` falls through to the generic error."""
        result = perform_string_replacement("hello", "\n", "x")
        assert isinstance(result, str)
        assert "not found" in result
        assert "old_string ends with a newline" not in result

    def test_eof_newline_mismatch_does_not_fire_for_interior_prefix(self) -> None:
        """Interior prefix matches must not trigger the EOF hint.

        `old_string="return foo\n"` against `content="return foobar"`: the
        stripped key matches mid-content, not at EOF. Caller gets the
        generic "not found" error, never a misleading EOF hint that would
        invite a corrupting retry.
        """  # noqa: D301
        content = "return foobar"
        old_string = "return foo\n"
        new_string = "return baz\n"
        result = perform_string_replacement(content, old_string, new_string)
        assert isinstance(result, str)
        assert "not found" in result
        assert "old_string ends with a newline" not in result

    def test_eof_newline_mismatch_does_not_fire_when_eof_text_differs(self) -> None:
        """If file's EOF text doesn't match the stripped key, no EOF hint."""
        content = "x foo y"
        old_string = "x foo\n"
        new_string = "REPLACED\n"
        result = perform_string_replacement(content, old_string, new_string)
        assert isinstance(result, str)
        assert "not found" in result
        assert "old_string ends with a newline" not in result


class TestSliceReadResponse:
    """`slice_read_response` must round-trip the file's trailing-newline state.

    That state is the load-bearing input to `perform_string_replacement`'s
    EOF-mismatch detection. If it gets dropped here, the EOF hint can't
    fire and callers fall back to the generic "not found" loop.
    """

    @staticmethod
    def _file(content: str) -> FileData:
        return FileData(content=content, encoding="utf-8")

    def test_preserves_trailing_newline_when_file_has_one(self) -> None:
        result = slice_read_response(self._file("foo\nbar\n"), offset=0, limit=2000)
        assert result == "foo\nbar\n"

    def test_preserves_no_trailing_newline_when_file_lacks_one(self) -> None:
        result = slice_read_response(self._file("foo\nbar"), offset=0, limit=2000)
        assert result == "foo\nbar"

    def test_normalizes_crlf_to_lf(self) -> None:
        """State/Store callers may carry CRLF; downstream tooling assumes LF."""
        result = slice_read_response(self._file("foo\r\nbar\r\n"), offset=0, limit=2000)
        assert isinstance(result, str)
        assert "\r" not in result
        assert result == "foo\nbar\n"

    def test_normalizes_bare_cr_to_lf(self) -> None:
        result = slice_read_response(self._file("foo\rbar\r"), offset=0, limit=2000)
        assert isinstance(result, str)
        assert "\r" not in result
        assert result == "foo\nbar\n"

    def test_partial_window_keeps_terminator_on_internal_lines(self) -> None:
        """A window ending on a non-terminal line still ends with that line's terminator."""
        result = slice_read_response(self._file("a\nb\nc\nd\n"), offset=1, limit=2)
        assert result == "b\nc\n"

    def test_partial_window_ending_on_unterminated_last_line(self) -> None:
        """A window covering the last line keeps that line's missing-terminator state."""
        result = slice_read_response(self._file("a\nb\nc"), offset=2, limit=1)
        assert result == "c"

    def test_offset_beyond_file_returns_error_result(self) -> None:
        result = slice_read_response(self._file("a\nb"), offset=10, limit=5)
        assert isinstance(result, ReadResult)
        assert result.error is not None
        assert "exceeds file length" in result.error
