"""Shared utility functions for memory backend implementations.

This module contains both user-facing string formatters and structured
helpers used by backends and the composite router. Structured helpers
enable composition without fragile string parsing.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import wcmatch.glob as wcglob

from deepagents.backends.protocol import FileInfo as _FileInfo
from deepagents.backends.protocol import GrepMatch as _GrepMatch

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 10000
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
    sanitized = tool_call_id.replace(".", "_").replace("/", "_").replace("\\", "_")
    return sanitized


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
    replace_all: bool,
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
        return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."

    new_content = content.replace(old_string, new_string)
    return new_content, occurrences


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """Truncate list or string result if it exceeds token limit (rough estimate: 4 chars/token)."""
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT * 4:
            return result[: len(result) * TOOL_RESULT_TOKEN_LIMIT * 4 // total_chars] + [TRUNCATION_GUIDANCE]
        return result
    # string
    if len(result) > TOOL_RESULT_TOKEN_LIMIT * 4:
        return result[: TOOL_RESULT_TOKEN_LIMIT * 4] + "\n" + TRUNCATION_GUIDANCE
    return result


def _validate_path(path: str | None) -> str:
    """Validate and normalize a path.

    Args:
        path: Path to validate

    Returns:
        Normalized path starting with /

    Raises:
        ValueError: If path is invalid
    """
    path = path or "/"
    if not path or path.strip() == "":
        raise ValueError("Path cannot be empty")

    normalized = path if path.startswith("/") else "/" + path

    if not normalized.endswith("/"):
        normalized += "/"

    return normalized


def _glob_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str = "/",
) -> str:
    """Search files dict for paths matching glob pattern.

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
        normalized_path = _validate_path(path)
    except ValueError:
        return "No files found"

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    # Respect standard glob semantics:
    # - Patterns without path separators (e.g., "*.py") match only in the current
    #   directory (non-recursive) relative to `path`.
    # - Use "**" explicitly for recursive matching.
    effective_pattern = pattern

    matches = []
    for file_path, file_data in filtered.items():
        relative = file_path[len(normalized_path) :].lstrip("/")
        if not relative:
            relative = file_path.split("/")[-1]

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
        normalized_path = _validate_path(path)
    except ValueError:
        return "No matches found"

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

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

    Returns a list of GrepMatch on success, or a string for invalid inputs
    (e.g., invalid regex). We deliberately do not raise here to keep backends
    non-throwing in tool contexts and preserve user-facing error messages.
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        normalized_path = _validate_path(path)
    except ValueError:
        return []

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    matches: list[GrepMatch] = []
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if regex.search(line):
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
