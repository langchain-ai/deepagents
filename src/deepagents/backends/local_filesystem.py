"""File search middleware for Anthropic text editor and memory tools.

This module provides Glob and Grep search tools that operate on files stored
in state or filesystem.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from deepagents.backends.fs import FileSystem


def _expand_include_patterns(pattern: str) -> list[str] | None:
    """Expand brace patterns like ``*.{py,pyi}`` into a list of globs."""
    if "}" in pattern and "{" not in pattern:
        return None

    expanded: list[str] = []

    def _expand(current: str) -> None:
        start = current.find("{")
        if start == -1:
            expanded.append(current)
            return

        end = current.find("}", start)
        if end == -1:
            raise ValueError

        prefix = current[:start]
        suffix = current[end + 1 :]
        inner = current[start + 1 : end]
        if not inner:
            raise ValueError

        for option in inner.split(","):
            _expand(prefix + option + suffix)

    try:
        _expand(pattern)
    except ValueError:
        return None

    return expanded


def _is_valid_include_pattern(pattern: str) -> bool:
    """Validate glob pattern used for include filters."""
    if not pattern:
        return False

    if any(char in pattern for char in ("\x00", "\n", "\r")):
        return False

    expanded = _expand_include_patterns(pattern)
    if expanded is None:
        return False

    try:
        for candidate in expanded:
            re.compile(fnmatch.translate(candidate))
    except re.error:
        return False

    return True


def _match_include_pattern(basename: str, pattern: str) -> bool:
    """Return True if the basename matches the include pattern."""
    expanded = _expand_include_patterns(pattern)
    if not expanded:
        return False

    return any(fnmatch.fnmatch(basename, candidate) for candidate in expanded)


class LocalFileSystem(FileSystem):
    """Local filesystem implementation of the FileSystem protocol.

    Provides direct access to files on the local filesystem with path validation
    and sandboxing to a root directory.

    Example:
        ```python
        fs = LocalFileSystem(root_path="/workspace")

        # List files
        files = fs.ls(prefix="/src")

        # Read file
        content = fs.read("/src/main.py")

        # Search files
        results = fs.grep("TODO", path="/src", include="*.py")
        ```
    """

    def __init__(
        self,
        *,
        root_path: str,
        use_ripgrep: bool = True,
        max_file_size_mb: int = 10,
    ) -> None:
        """Initialize the local filesystem.

        Args:
            root_path: Root directory for all file operations.
            use_ripgrep: Whether to use ripgrep for search (default: True).
                Falls back to Python if ripgrep unavailable.
            max_file_size_mb: Maximum file size to process in MB (default: 10).
        """
        self.root_path = Path(root_path).resolve()
        self.use_ripgrep = use_ripgrep
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _validate_and_resolve_path(self, path: str) -> Path:
        """Validate and resolve a virtual path to filesystem path."""
        # Normalize path
        if not path.startswith("/"):
            path = "/" + path

        # Check for path traversal
        if ".." in path or "~" in path:
            msg = "Path traversal not allowed"
            raise ValueError(msg)

        # Convert virtual path to filesystem path
        relative = path.lstrip("/")
        full_path = (self.root_path / relative).resolve()

        # Ensure path is within root
        try:
            full_path.relative_to(self.root_path)
        except ValueError:
            msg = f"Path outside root directory: {path}"
            raise ValueError(msg) from None

        return full_path

    def ls(self, prefix: str | None = None) -> list[dict]:
        """List all file paths, optionally filtered by prefix.

        Args:
            prefix: Optional path prefix to filter results (e.g., "/subdir/")
                   If None, returns all files.

        Returns:
            List of FileInfo dicts with path, kind, size, and timestamps.
        """
        try:
            base_path = self._validate_and_resolve_path(prefix or "/")
        except ValueError:
            return []

        if not base_path.exists():
            return []

        result: list[dict] = []

        if base_path.is_file():
            # Single file
            stat = base_path.stat()
            result.append(
                {
                    "path": "/" + str(base_path.relative_to(self.root_path)),
                    "kind": "file",
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                    "created_at": datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat(),
                }
            )
        else:
            # Directory - list contents recursively
            for item in base_path.rglob("*"):
                try:
                    stat = item.stat()
                    result.append(
                        {
                            "path": "/" + str(item.relative_to(self.root_path)),
                            "kind": "dir" if item.is_dir() else "file",
                            "size": stat.st_size if item.is_file() else 0,
                            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat(),
                        }
                    )
                except (PermissionError, OSError):
                    continue

        return result

    def upload_file(self, file: bytes, path: str, *, timeout: int = 30 * 60) -> None:
        """Upload a file to the filesystem.

        Args:
            file: File content as bytes.
            path: Virtual path where to write the file.
            timeout: Ignored for local filesystem.
        """
        try:
            full_path = self._validate_and_resolve_path(path)
        except ValueError as e:
            raise ValueError(f"Invalid path: {e}") from e

        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file with O_NOFOLLOW to prevent symlink attacks
        try:
            fd = os.open(full_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, "wb") as f:
                f.write(file)
        except OSError as e:
            raise ValueError(f"Cannot write file '{path}': {e}") from e

    def download_file(self, path: str, *, timeout: int = 30 * 60) -> bytes:
        """Download a file from the filesystem.

        Args:
            path: Virtual path to the file.
            timeout: Ignored for local filesystem.

        Returns:
            File content as bytes.
        """
        try:
            full_path = self._validate_and_resolve_path(path)
        except ValueError as e:
            raise FileNotFoundError(f"File not found: {path}") from e

        if not full_path.exists() or not full_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        # Open with O_NOFOLLOW to prevent symlink attacks
        try:
            fd = os.open(full_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as f:
                return f.read()
        except OSError as e:
            raise FileNotFoundError(f"Cannot read file: {path}") from e

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Virtual path to the file.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers (cat -n style), or error message.
        """
        try:
            full_path = self._validate_and_resolve_path(file_path)
        except ValueError:
            return f"Error: File '{file_path}' not found"

        if not full_path.exists() or not full_path.is_file():
            return f"Error: File '{file_path}' not found"

        # Open with O_NOFOLLOW to prevent symlink attacks
        try:
            fd = os.open(full_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError, PermissionError) as e:
            return f"Error: Cannot read file '{file_path}': {e}"

        if not content:
            return "System reminder: File exists but has empty contents"

        lines = content.splitlines()

        # Apply offset and limit
        selected_lines = lines[offset : offset + limit]

        # Format with line numbers (cat -n style)
        result = []
        for i, line in enumerate(selected_lines, start=offset + 1):
            result.append(f"     {i:>d}\t{line}")

        return "\n".join(result)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Virtual path to the file.
            old_string: String to find and replace.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences; if False, require unique match.

        Returns:
            Success message or error message.
        """
        try:
            full_path = self._validate_and_resolve_path(file_path)
        except ValueError:
            return f"Error: File '{file_path}' not found"

        if not full_path.exists() or not full_path.is_file():
            return f"Error: File '{file_path}' not found"

        # Open with O_NOFOLLOW to prevent symlink attacks
        try:
            fd = os.open(full_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError, PermissionError) as e:
            return f"Error: Cannot read file '{file_path}': {e}"

        # Count occurrences
        count = content.count(old_string)

        if count == 0:
            return f"Error: String not found in file: '{old_string}'"

        if count > 1 and not replace_all:
            return f"Error: String '{old_string}' appears {count} times. Use replace_all=True to replace all occurrences."

        # Perform replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        # Write back with O_NOFOLLOW to prevent symlink attacks
        try:
            fd = os.open(full_path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
        except OSError as e:
            return f"Error: Cannot write file '{file_path}': {e}"

        return f"Successfully replaced {count} occurrence(s) in {file_path}"

    def delete(self, file_path: str) -> None:
        """Delete a file by path.

        Args:
            file_path: Virtual path to the file to delete.
        """
        try:
            full_path = self._validate_and_resolve_path(file_path)
        except ValueError as e:
            raise FileNotFoundError(f"File not found: {file_path}") from e

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if full_path.is_dir():
            full_path.rmdir()
        else:
            full_path.unlink()

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: str | None = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files.

        Args:
            pattern: Regular expression pattern to search for.
            path: Path to search in (default "/").
            include: Optional glob pattern to filter files (e.g., "*.py").
            output_mode: Output format - "files_with_matches", "content", or "count".

        Returns:
            Formatted search results based on output_mode.
        """
        # Compile regex pattern (for validation)
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        if include and not _is_valid_include_pattern(include):
            return "Invalid include pattern"

        # Try ripgrep first if enabled
        results = None
        if self.use_ripgrep:
            with suppress(
                FileNotFoundError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                results = self._ripgrep_search(pattern, path, include)

        # Python fallback if ripgrep failed or is disabled
        if results is None:
            results = self._python_search(pattern, path, include)

        if not results:
            return "No matches found"

        # Format output based on mode
        return self._format_grep_results(results, output_mode)

    def _ripgrep_search(self, pattern: str, base_path: str, include: str | None) -> dict[str, list[tuple[int, str]]]:
        """Search using ripgrep subprocess."""
        try:
            base_full = self._validate_and_resolve_path(base_path)
        except ValueError:
            return {}

        if not base_full.exists():
            return {}

        # Build ripgrep command
        cmd = ["rg", "--json"]

        if include:
            # Convert glob pattern to ripgrep glob
            cmd.extend(["--glob", include])

        cmd.extend(["--", pattern, str(base_full)])

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to Python search if ripgrep unavailable or times out
            return self._python_search(pattern, base_path, include)

        # Parse ripgrep JSON output
        results: dict[str, list[tuple[int, str]]] = {}
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if data["type"] == "match":
                    path = data["data"]["path"]["text"]
                    # Convert to virtual path
                    virtual_path = "/" + str(Path(path).relative_to(self.root_path))
                    line_num = data["data"]["line_number"]
                    line_text = data["data"]["lines"]["text"].rstrip("\n")

                    if virtual_path not in results:
                        results[virtual_path] = []
                    results[virtual_path].append((line_num, line_text))
            except (json.JSONDecodeError, KeyError):
                continue

        return results

    def _python_search(self, pattern: str, base_path: str, include: str | None) -> dict[str, list[tuple[int, str]]]:
        """Search using Python regex (fallback)."""
        try:
            base_full = self._validate_and_resolve_path(base_path)
        except ValueError:
            return {}

        if not base_full.exists():
            return {}

        regex = re.compile(pattern)
        results: dict[str, list[tuple[int, str]]] = {}

        # Walk directory tree
        for file_path in base_full.rglob("*"):
            if not file_path.is_file():
                continue

            # Check include filter
            if include and not _match_include_pattern(file_path.name, include):
                continue

            # Skip files that are too large
            if file_path.stat().st_size > self.max_file_size_bytes:
                continue

            try:
                content = file_path.read_text()
            except (UnicodeDecodeError, PermissionError):
                continue

            # Search content
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    virtual_path = "/" + str(file_path.relative_to(self.root_path))
                    if virtual_path not in results:
                        results[virtual_path] = []
                    results[virtual_path].append((line_num, line))

        return results

    def _format_grep_results(
        self,
        results: dict[str, list[tuple[int, str]]],
        output_mode: str,
    ) -> str:
        """Format grep results based on output mode."""
        if output_mode == "files_with_matches":
            # Just return file paths
            return "\n".join(sorted(results.keys()))

        if output_mode == "content":
            # Return file:line:content format
            lines = []
            for file_path in sorted(results.keys()):
                for line_num, line in results[file_path]:
                    lines.append(f"{file_path}:{line_num}:{line}")
            return "\n".join(lines)

        if output_mode == "count":
            # Return file:count format
            lines = []
            for file_path in sorted(results.keys()):
                count = len(results[file_path])
                lines.append(f"{file_path}:{count}")
            return "\n".join(lines)

        # Default to files_with_matches
        return "\n".join(sorted(results.keys()))

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt").
            path: Base path to search from (default "/").

        Returns:
            List of virtual file paths matching the pattern.
        """
        try:
            base_full = self._validate_and_resolve_path(path)
        except ValueError:
            return []

        if not base_full.exists() or not base_full.is_dir():
            return []

        # Use pathlib glob
        matching: list[str] = []
        for match in base_full.glob(pattern):
            if match.is_file():
                # Convert to virtual path
                virtual_path = "/" + str(match.relative_to(self.root_path))
                matching.append(virtual_path)

        return matching

    @property
    def get_capabilities(self) -> dict:
        """Get the filesystem capabilities."""
        return {
            "can_upload": True,
            "can_download": True,
            "can_list_files": True,
            "can_read": True,
            "can_edit": True,
            "can_delete": True,
            "can_grep": True,
            "can_glob": True,
        }


__all__ = [
    "LocalFileSystem",
]
