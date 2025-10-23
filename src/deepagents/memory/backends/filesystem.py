"""FilesystemBackend: Read and write files directly from the filesystem."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from langgraph.types import Command

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

from .utils import check_empty_content, format_content_with_line_numbers, perform_string_replacement


class FilesystemBackend:
    """Backend that reads and writes files directly from the filesystem.

    Files are accessed using their actual filesystem paths. Relative paths are
    resolved relative to the current working directory. Content is read/written
    as plain text, and metadata (timestamps) are derived from filesystem stats.
    """

    def __init__(
        self, 
        root_dir: Optional[str | Path] = None,
        virtual_mode: bool = False
    ) -> None:
        """Initialize filesystem backend.
        
        Args:
            root_dir: Optional root directory for file operations. If provided,
                     all file paths will be resolved relative to this directory.
                     If not provided, uses the current working directory.
        """
        self.cwd = Path(root_dir) if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode

    def _resolve_path(self, key: str) -> Path:
        """Resolve a file path relative to cwd if not absolute.

        Args:
            key: File path (absolute or relative)

        Returns:
            Resolved absolute Path object
        """
        if self.virtual_mode:
            return self.cwd / key.lstrip('/')
        path = Path(key)
        if path.is_absolute():
            return path
        return self.cwd / path

    def ls(self, prefix: Optional[str] = None, runtime: Optional["ToolRuntime"] = None) -> list[str]:
        """List files from filesystem.

        Args:
            prefix: Optional directory path to list files from (absolute or relative to cwd).
                   Defaults to current working directory if not provided.
            runtime: Optional ToolRuntime (ignored by FilesystemBackend).

        Returns:
            List of absolute file paths.
        """
        if prefix is None:
            # Default to current working directory
            dir_path = self.cwd
        else:
            dir_path = self._resolve_path(prefix)
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

        return sorted(results)
    
    def read(
        self, 
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        runtime: Optional["ToolRuntime"] = None,
    ) -> str:
        """Read file content with line numbers.
        
        Args:
            file_path: Absolute or relative file path
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read
            runtime: Optional ToolRuntime (ignored by FilesystemBackend).
        
        Returns:
            Formatted file content with line numbers, or error message.
        """
        resolved_path = self._resolve_path(file_path)
        
        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"
        
        try:
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
        runtime: Optional["ToolRuntime"] = None,
    ) -> Command | str:
        """Create a new file with content.
        
        Args:
            file_path: Absolute or relative file path
            content: File content as a string
            runtime: Optional ToolRuntime (ignored by FilesystemBackend).
        
        Returns:
            Success message or error if file already exists.
        """
        resolved_path = self._resolve_path(file_path)
        
        if resolved_path.exists():
            return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        
        try:
            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(resolved_path, "w", encoding="utf-8") as f:
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
        runtime: Optional["ToolRuntime"] = None,
    ) -> Command | str:
        """Edit a file by replacing string occurrences.
        
        Args:
            file_path: Absolute or relative file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences
            runtime: Optional ToolRuntime (ignored by FilesystemBackend).
        
        Returns:
            Success message or error message on failure.
        """
        resolved_path = self._resolve_path(file_path)
        
        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"
        
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            result = perform_string_replacement(content, old_string, new_string, replace_all)
            
            if isinstance(result, str):
                return result
            
            new_content, occurrences = result
            
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return f"Successfully replaced {occurrences} instance(s) of the string in '{file_path}'"
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return f"Error editing file '{file_path}': {e}"

    def delete(self, file_path: str, runtime: Optional["ToolRuntime"] = None) -> Command | None:
        """Delete file from filesystem.

        Args:
            file_path: File path to delete (absolute or relative to cwd)
            runtime: Optional ToolRuntime (ignored by FilesystemBackend).
        
        Returns:
            None (direct filesystem modification)
        """
        resolved_path = self._resolve_path(file_path)

        if resolved_path.exists() and resolved_path.is_file():
            resolved_path.unlink()
        
        return None
    
    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: Optional[str] = None,
        output_mode: str = "files_with_matches",
        runtime: Optional["ToolRuntime"] = None,
    ) -> str:
        """Search for a pattern in files.
        
        Args:
            pattern: String pattern to search for
            path: Path to search in (default "/")
            include: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"
            runtime: Optional ToolRuntime (ignored by FilesystemBackend).
        
        Returns:
            Formatted search results based on output_mode.
        """
        regex = re.compile(re.escape(pattern))
        
        if include:
            files_to_search = self.glob(include, runtime=runtime)
        else:
            files_to_search = self.ls(path if path != "/" else None, runtime=runtime)
        
        if path != "/":
            files_to_search = [f for f in files_to_search if f.startswith(path)]
        
        file_matches = {}
        
        for fp in files_to_search:
            resolved_path = self._resolve_path(fp)
            
            if not resolved_path.exists() or not resolved_path.is_file():
                continue
            
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                matches = []
                for line_num, line in enumerate(lines, start=1):
                    if regex.search(line):
                        matches.append((line_num, line.rstrip()))
                
                if matches:
                    file_matches[fp] = matches
            except (OSError, UnicodeDecodeError):
                continue
        
        if not file_matches:
            return f"No matches found for pattern: '{pattern}'"
        
        if output_mode == "files_with_matches":
            return "\n".join(sorted(file_matches.keys()))
        elif output_mode == "count":
            results = []
            for fp in sorted(file_matches.keys()):
                count = len(file_matches[fp])
                results.append(f"{fp}: {count}")
            return "\n".join(results)
        else:
            results = []
            for fp in sorted(file_matches.keys()):
                results.append(f"{fp}:")
                for line_num, line in file_matches[fp]:
                    results.append(f"  {line_num}: {line}")
            return "\n".join(results)
    
    def glob(self, pattern: str, path: str = "/", runtime: Optional["ToolRuntime"] = None) -> list[str]:
        """Find files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")
            runtime: Optional ToolRuntime (ignored by FilesystemBackend).
        
        Returns:
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
        
        return sorted(results)