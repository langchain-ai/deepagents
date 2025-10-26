"""FilesystemBackend: Read and write files directly from the filesystem.

Security and search upgrades:
- Secure path resolution with root containment when in virtual_mode (sandboxed to cwd)
- Prevent symlink-following on file I/O using O_NOFOLLOW when available
- Ripgrep-powered grep with JSON parsing, plus Python fallback with regex
  and optional glob include filtering, while preserving virtual path behavior
"""

import os
import re
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from langgraph.types import Command

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

from .utils import (
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
    _format_grep_results,
    truncate_if_too_long
)
import wcmatch.glob as wcglob



class FilesystemBackend:
    """Backend that reads and writes files directly from the filesystem.

    Files are accessed using their actual filesystem paths. Relative paths are
    resolved relative to the current working directory. Content is read/written
    as plain text, and metadata (timestamps) are derived from filesystem stats.
    """

    def __init__(
        self, 
        root_dir: Optional[str | Path] = None,
        virtual_mode: bool = False,
        max_file_size_mb: int = 10,
    ) -> None:
        """Initialize filesystem backend.
        
        Args:
            root_dir: Optional root directory for file operations. If provided,
                     all file paths will be resolved relative to this directory.
                     If not provided, uses the current working directory.
        """
        self.cwd = Path(root_dir) if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _resolve_path(self, key: str) -> Path:
        """Resolve a file path with security checks.

        When virtual_mode=True, treat incoming paths as virtual absolute paths under
        self.cwd, disallow traversal (.., ~) and ensure resolved path stays within root.
        When virtual_mode=False, preserve legacy behavior: absolute paths are allowed
        as-is; relative paths resolve under cwd.

        Args:
            key: File path (absolute, relative, or virtual when virtual_mode=True)

        Returns:
            Resolved absolute Path object
        """
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                raise ValueError("Path traversal not allowed")
            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                raise ValueError(f"Path outside root directory: {key}") from None
            return full

        path = Path(key)
        if path.is_absolute():
            return path
        return (self.cwd / path).resolve()

    def ls(self, path: str) -> list[str]:
        """List files from filesystem.

        Args:
            path: Absolute directory path to list files from.
        
        Returns:
            List of absolute file paths.
        """
        dir_path = self._resolve_path(path)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        results: list[str] = []

        # Convert cwd to string for comparison
        cwd_str = str(self.cwd)
        if not cwd_str.endswith("/"):
            cwd_str += "/"

        # Walk the directory tree
        try:
            for path in dir_path.rglob("*"):
                if path.is_file():
                    abs_path = str(path)
                    if not self.virtual_mode:
                        results.append(abs_path)
                        continue
                    # Strip the cwd prefix if present
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str):]
                    elif abs_path.startswith(str(self.cwd)):
                        # Handle case where cwd doesn't end with /
                        relative_path = abs_path[len(str(self.cwd)):].lstrip("/")
                    else:
                        # Path is outside cwd, return as-is or skip
                        relative_path = abs_path

                    results.append("/" + relative_path)
        except (OSError, PermissionError):
            pass

        return truncate_if_too_long(sorted(results))
    
    def read(
        self, 
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.
        
        Args:
            file_path: Absolute or relative file path
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to readReturns:
            Formatted file content with line numbers, or error message.
        """
        resolved_path = self._resolve_path(file_path)
        
        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"
        
        try:
            # Open with O_NOFOLLOW where available to avoid symlink traversal
            try:
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                # Fallback to normal open if O_NOFOLLOW unsupported or fails
                with open(resolved_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
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
        except (OSError, UnicodeDecodeError) as e:
            return f"Error reading file '{file_path}': {e}"
    
    def write(
        self, 
        file_path: str,
        content: str,
    ) -> Command | str:
        """Create a new file with content.
        
        Args:
            file_path: Absolute or relative file path
            content: File content as a stringReturns:
            Success message or error if file already exists.
        """
        resolved_path = self._resolve_path(file_path)
        
        if resolved_path.exists():
            return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        
        try:
            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # Prefer O_NOFOLLOW to avoid writing through symlinks
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"Updated file {file_path}"
        except (OSError, UnicodeEncodeError) as e:
            return f"Error writing file '{file_path}': {e}"
    
    def edit(
        self, 
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> Command | str:
        """Edit a file by replacing string occurrences.
        
        Args:
            file_path: Absolute or relative file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrencesReturns:
            Success message or error message on failure.
        """
        resolved_path = self._resolve_path(file_path)
        
        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"
        
        try:
            # Read securely
            try:
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            result = perform_string_replacement(content, old_string, new_string, replace_all)
            
            if isinstance(result, str):
                return result
            
            new_content, occurrences = result
            
            # Write securely
            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return f"Successfully replaced {occurrences} instance(s) of the string in '{file_path}'"
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return f"Error editing file '{file_path}': {e}"
    
    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files (ripgrep with Python fallback).

        Args:
            pattern: Regex pattern to search for
            path: Path to search in (default "/")
            glob: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"

        Returns:
            Formatted search results based on output_mode.
        """
        # Validate regex
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        # Resolve base path
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return f"No matches found for pattern: '{pattern}'"

        if not base_full.exists():
            return f"No matches found for pattern: '{pattern}'"

        # Try ripgrep first
        results = self._ripgrep_search(pattern, base_full, glob)
        if results is None:
            results = self._python_search(pattern, base_full, glob)

        if not results:
            return f"No matches found for pattern: '{pattern}'"

        return truncate_if_too_long(_format_grep_results(results, output_mode))

    def _ripgrep_search(
        self, pattern: str, base_full: Path, include_glob: Optional[str]
    ) -> Optional[dict[str, list[tuple[int, str]]]]:
        cmd = ["rg", "--json"]
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base_full)])

        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        results: dict[str, list[tuple[int, str]]] = {}
        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            p = Path(ftext)
            if self.virtual_mode:
                try:
                    virt = "/" + str(p.resolve().relative_to(self.cwd))
                except Exception:
                    continue
            else:
                virt = str(p)
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(virt, []).append((int(ln), lt))

        return results

    def _python_search(
        self, pattern: str, base_full: Path, include_glob: Optional[str]
    ) -> dict[str, list[tuple[int, str]]]:
        try:
            regex = re.compile(pattern)
        except re.error:
            return {}

        results: dict[str, list[tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent

        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            if include_glob and not wcglob.globmatch(fp.name, include_glob, flags=wcglob.BRACE):
                continue
            try:
                if fp.stat().st_size > self.max_file_size_bytes:
                    continue
            except OSError:
                continue
            try:
                content = fp.read_text()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    if self.virtual_mode:
                        try:
                            virt_path = "/" + str(fp.resolve().relative_to(self.cwd))
                        except Exception:
                            continue
                    else:
                        virt_path = str(fp)
                    results.setdefault(virt_path, []).append((line_num, line))

        return results
    
    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")Returns:
            List of absolute file paths matching the pattern.
        """
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")
        
        if path == "/":
            search_path = self.cwd
        else:
            search_path = self._resolve_path(path)
        
        if not search_path.exists() or not search_path.is_dir():
            return []
        
        results = []
        
        try:
            for matched_path in search_path.glob(pattern):
                if matched_path.is_file():
                    abs_path = str(matched_path)
                    if not self.virtual_mode:
                        results.append(abs_path)
                        continue
                    
                    cwd_str = str(self.cwd)
                    if not cwd_str.endswith("/"):
                        cwd_str += "/"
                    
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str):]
                    elif abs_path.startswith(str(self.cwd)):
                        relative_path = abs_path[len(str(self.cwd)):].lstrip("/")
                    else:
                        relative_path = abs_path
                    
                    results.append("/" + relative_path)
        except (OSError, ValueError):
            pass

        return truncate_if_too_long(sorted(results))
