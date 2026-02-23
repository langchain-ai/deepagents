"""Shared utility functions for memory backend implementations.

This module contains both user-facing string formatters and structured
helpers used by backends and the composite router. Structured helpers
enable composition without fragile string parsing.
"""

import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, overload

import wcmatch.glob as wcglob

from deepagents.backends.protocol import FileInfo as _FileInfo, GrepMatch as _GrepMatch

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 5000
LINE_NUMBER_WIDTH = 6
TOOL_RESULT_TOKEN_LIMIT = 20000  # Same threshold as eviction
TRUNCATION_GUIDANCE = "... [results truncated, try being more specific with your parameters]"

# Re-export protocol types for backwards compatibility
FileInfo = _FileInfo
GrepMatch = _GrepMatch


def sanitize_tool_call_id(tool_call_id: str) -> str:
    r"""Sanitize tool_call_id to prevent path traversal and separator issues.

    Replaces dangerous characters (., /, \) with underscores.
    """
    return tool_call_id.replace(".", "_").replace("/", "_").replace("\\", "_")


def format_content_with_line_numbers(
    content: str | list[str],
    start_line: int = 1,
) -> str:
    """Format file content with line numbers (cat -n style).

    Chunks lines longer than MAX_LINE_LENGTH with continuation markers (e.g., 5.1, 5.2).

    Args:
        content: File content as string or list of lines
        start_line: Starting line number (default: 1)

    Returns:
        Formatted content with line numbers and continuation markers
    """
    if isinstance(content, str):
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
    else:
        lines = content

    result_lines = []
    for i, line in enumerate(lines):
        line_num = i + start_line

        if len(line) <= MAX_LINE_LENGTH:
            result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{line}")
        else:
            # Split long line into chunks with continuation markers
            num_chunks = (len(line) + MAX_LINE_LENGTH - 1) // MAX_LINE_LENGTH
            for chunk_idx in range(num_chunks):
                start = chunk_idx * MAX_LINE_LENGTH
                end = min(start + MAX_LINE_LENGTH, len(line))
                chunk = line[start:end]
                if chunk_idx == 0:
                    # First chunk: use normal line number
                    result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{chunk}")
                else:
                    # Continuation chunks: use decimal notation (e.g., 5.1, 5.2)
                    continuation_marker = f"{line_num}.{chunk_idx}"
                    result_lines.append(f"{continuation_marker:>{LINE_NUMBER_WIDTH}}\t{chunk}")

    return "\n".join(result_lines)


def check_empty_content(content: str) -> str | None:
    """Check if content is empty and return warning message.

    Args:
        content: Content to check

    Returns:
        Warning message if empty, None otherwise
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None


def file_data_to_string(file_data: dict[str, Any]) -> str:
    """Convert FileData to plain string content.

    Args:
        file_data: FileData dict with 'content' key

    Returns:
        Content as string with lines joined by newlines
    """
    return "\n".join(file_data["content"])


def create_file_data(content: str, created_at: str | None = None) -> dict[str, Any]:
    """Create a FileData object with timestamps.

    Args:
        content: File content as string
        created_at: Optional creation timestamp (ISO format)

    Returns:
        FileData dict with content and timestamps
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(UTC).isoformat()

    return {
        "content": lines,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(file_data: dict[str, Any], content: str) -> dict[str, Any]:
    """Update FileData with new content, preserving creation timestamp.

    Args:
        file_data: Existing FileData dict
        content: New content as string

    Returns:
        Updated FileData dict
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(UTC).isoformat()

    return {
        "content": lines,
        "created_at": file_data["created_at"],
        "modified_at": now,
    }


def format_read_response(
    file_data: dict[str, Any],
    offset: int,
    limit: int,
) -> str:
    """Format file data for read response with line numbers.

    Args:
        file_data: FileData dict
        offset: Line offset (0-indexed)
        limit: Maximum number of lines

    Returns:
        Formatted content or error message
    """
    content = file_data_to_string(file_data)
    empty_msg = check_empty_content(content)
    if empty_msg:
        return empty_msg

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    selected_lines = lines[start_idx:end_idx]
    return format_content_with_line_numbers(selected_lines, start_line=start_idx + 1)


def perform_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,  # noqa: FBT001, FBT002
) -> tuple[str, int] | str:
    """Perform string replacement with occurrence validation.

    Args:
        content: Original content
        old_string: String to replace
        new_string: Replacement string
        replace_all: Whether to replace all occurrences

    Returns:
        Tuple of (new_content, occurrences) on success, or error message string
    """
    occurrences = content.count(old_string)

    if occurrences == 0:
        return f"Error: String not found in file: '{old_string}'"

    if occurrences > 1 and not replace_all:
        return (
            f"Error: String '{old_string}' appears {occurrences} times in file. "
            f"Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        )

    new_content = content.replace(old_string, new_string)
    return new_content, occurrences


@overload
def truncate_if_too_long(result: list[str]) -> list[str]: ...


@overload
def truncate_if_too_long(result: str) -> str: ...


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """Truncate list or string result if it exceeds token limit (rough estimate: 4 chars/token)."""
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT * 4:
            return result[: len(result) * TOOL_RESULT_TOKEN_LIMIT * 4 // total_chars] + [TRUNCATION_GUIDANCE]  # noqa: RUF005  # Concatenation preferred for clarity
        return result
    # string
    if len(result) > TOOL_RESULT_TOKEN_LIMIT * 4:
        return result[: TOOL_RESULT_TOKEN_LIMIT * 4] + "\n" + TRUNCATION_GUIDANCE
    return result


def validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    r"""Validate and normalize file path for security.

    Ensures paths are safe to use by preventing directory traversal attacks
    and enforcing consistent formatting. All paths are normalized to use
    forward slashes and start with a leading slash.

    This function is designed for virtual filesystem paths and rejects
    Windows absolute paths (e.g., `C:/...`, `F:/...`) to maintain consistency
    and prevent path format ambiguity.

    Args:
        path: The path to validate and normalize.
        allowed_prefixes: Optional list of allowed path prefixes.

            If provided, the normalized path must start with one of
            these prefixes.

    Returns:
        Normalized canonical path starting with `/` and using forward slashes.

    Raises:
        ValueError: If path contains traversal sequences (`..` or `~`), is a
            Windows absolute path (e.g., `C:/...`), or does not start with an
            allowed prefix when `allowed_prefixes` is specified.

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path(r"C:\\Users\\file.txt")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    # Check for traversal as a path component (not substring) to avoid
    # false-positive rejection of legitimate filenames like "foo..bar.txt"
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if ".." in parts or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Reject Windows absolute paths (e.g., C:\..., D:/...)
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    # Defense-in-depth: verify normpath didn't produce traversal
    if ".." in normalized.split("/"):
        msg = f"Path traversal detected after normalization: {path} -> {normalized}"
        raise ValueError(msg)

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


def _normalize_path(path: str | None) -> str:
    """Normalize a path to canonical form.

    Converts path to absolute form starting with /, removes trailing slashes
    (except for root), and validates that the path is not empty.

    Args:
        path: Path to normalize (None defaults to "/")

    Returns:
        Normalized path starting with / (without trailing slash unless it's root)

    Raises:
        ValueError: If path is invalid (empty string after strip)

    Example:
        _normalize_path(None) -> "/"
        _normalize_path("/dir/") -> "/dir"
        _normalize_path("dir") -> "/dir"
        _normalize_path("/") -> "/"
    """
    path = path or "/"
    if not path or path.strip() == "":
        msg = "Path cannot be empty"
        raise ValueError(msg)

    normalized = path if path.startswith("/") else "/" + path

    # Only root should have trailing slash
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")

    return normalized


def _filter_files_by_path(files: dict[str, Any], normalized_path: str) -> dict[str, Any]:
    """Filter files dict by normalized path, handling exact file matches and directory prefixes.

    Expects a normalized path from _normalize_path (no trailing slash except root).

    Args:
        files: Dictionary mapping file paths to file data
        normalized_path: Normalized path from _normalize_path (e.g., "/", "/dir", "/dir/file")

    Returns:
        Filtered dictionary of files matching the path

    Example:
        files = {"/dir/file": {...}, "/dir/other": {...}}
        _filter_files_by_path(files, "/dir/file")  # Returns {"/dir/file": {...}}
        _filter_files_by_path(files, "/dir")       # Returns both files
    """
    # Check if path matches an exact file
    if normalized_path in files:
        return {normalized_path: files[normalized_path]}

    # Otherwise treat as directory prefix
    if normalized_path == "/":
        # Root directory - match all files starting with /
        return {fp: fd for fp, fd in files.items() if fp.startswith("/")}
    # Non-root directory - add trailing slash for prefix matching
    dir_prefix = normalized_path + "/"
    return {fp: fd for fp, fd in files.items() if fp.startswith(dir_prefix)}


def _glob_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str = "/",
) -> str:
    r"""Search files dict for paths matching glob pattern.

    Args:
        files: Dictionary of file paths to FileData.
        pattern: Glob pattern (e.g., "*.py", "**/*.ts").
        path: Base path to search from.

    Returns:
        Newline-separated file paths, sorted by modification time (most recent first).
        Returns "No files found" if no matches.

    Example:
        ```python
        files = {"/src/main.py": FileData(...), "/test.py": FileData(...)}
        _glob_search_files(files, "*.py", "/")
        # Returns: "/test.py\n/src/main.py" (sorted by modified_at)
        ```
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return "No files found"

    filtered = _filter_files_by_path(files, normalized_path)

    # Respect standard glob semantics:
    # - Patterns without path separators (e.g., "*.py") match only in the current
    #   directory (non-recursive) relative to `path`.
    # - Use "**" explicitly for recursive matching.
    effective_pattern = pattern

    matches = []
    for file_path, file_data in filtered.items():
        # Compute relative path for glob matching
        # If normalized_path is "/dir", we want "/dir/file.txt" -> "file.txt"
        # If normalized_path is "/dir/file.txt" (exact file), we want "file.txt"
        if normalized_path == "/":
            relative = file_path[1:]  # Remove leading slash
        elif file_path == normalized_path:
            # Exact file match - use just the filename
            relative = file_path.split("/")[-1]
        else:
            # Directory prefix - strip the directory path
            relative = file_path[len(normalized_path) + 1 :]  # +1 for the slash

        if wcglob.globmatch(relative, effective_pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR):
            matches.append((file_path, file_data["modified_at"]))

    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return "No files found"

    return "\n".join(fp for fp, _ in matches)


def _format_grep_results(
    results: dict[str, list[tuple[int, str]]],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """Format grep search results based on output mode.

    Args:
        results: Dictionary mapping file paths to list of (line_num, line_content) tuples
        output_mode: Output format - "files_with_matches", "content", or "count"

    Returns:
        Formatted string output
    """
    if output_mode == "files_with_matches":
        return "\n".join(sorted(results.keys()))
    if output_mode == "count":
        lines = []
        for file_path in sorted(results.keys()):
            count = len(results[file_path])
            lines.append(f"{file_path}: {count}")
        return "\n".join(lines)
    lines = []
    for file_path in sorted(results.keys()):
        lines.append(f"{file_path}:")
        for line_num, line in results[file_path]:
            lines.append(f"  {line_num}: {line}")
    return "\n".join(lines)


def _grep_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    """Search file contents for regex pattern.

    Args:
        files: Dictionary of file paths to FileData.
        pattern: Regex pattern to search for.
        path: Base path to search from.
        glob: Optional glob pattern to filter files (e.g., "*.py").
        output_mode: Output format - "files_with_matches", "content", or "count".

    Returns:
        Formatted search results. Returns "No matches found" if no results.

    Example:
        ```python
        files = {"/file.py": FileData(content=["import os", "print('hi')"], ...)}
        _grep_search_files(files, "import", "/")
        # Returns: "/file.py" (with output_mode="files_with_matches")
        ```
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return "No matches found"

    filtered = _filter_files_by_path(files, normalized_path)

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    results: dict[str, list[tuple[int, str]]] = {}
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if regex.search(line):
                if file_path not in results:
                    results[file_path] = []
                results[file_path].append((line_num, line))

    if not results:
        return "No matches found"
    return _format_grep_results(results, output_mode)


# -------- Structured helpers for composition --------


def grep_matches_from_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> list[GrepMatch] | str:
    """Return structured grep matches from an in-memory files mapping.

    Performs literal text search (not regex).

    Returns a list of GrepMatch on success, or a string for invalid inputs.
    We deliberately do not raise here to keep backends non-throwing in tool
    contexts and preserve user-facing error messages.
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return []

    filtered = _filter_files_by_path(files, normalized_path)

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    matches: list[GrepMatch] = []
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if pattern in line:  # Simple substring search for literal matching
                matches.append({"path": file_path, "line": int(line_num), "text": line})
    return matches


def build_grep_results_dict(matches: list[GrepMatch]) -> dict[str, list[tuple[int, str]]]:
    """Group structured matches into the legacy dict form used by formatters."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    for m in matches:
        grouped.setdefault(m["path"], []).append((m["line"], m["text"]))
    return grouped


def format_grep_matches(
    matches: list[GrepMatch],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """Format structured grep matches using existing formatting logic."""
    if not matches:
        return "No matches found"
    return _format_grep_results(build_grep_results_dict(matches), output_mode)
