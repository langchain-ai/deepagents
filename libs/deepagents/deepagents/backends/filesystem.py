"""`FilesystemBackend`: Read and write files directly from the filesystem."""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
import warnings

import wcmatch.glob as wcglob

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.utils import (
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
)


class FilesystemBackend(BackendProtocol):
    """Backend that reads and writes files directly from the filesystem.

    Files are accessed using their actual filesystem paths. Relative paths are
    resolved relative to the current working directory. Content is read/written
    as plain text, and metadata (timestamps) are derived from filesystem stats.

    !!! warning "Security Warning"

        This backend grants agents direct filesystem read/write access. Use with
        caution and only in appropriate environments.

        **Appropriate use cases:**

        - Local development CLIs (coding assistants, development tools)
        - CI/CD pipelines (see security considerations below)

        **Inappropriate use cases:**

        - Web servers or HTTP APIs - use `StateBackend`, `StoreBackend`, or
            `SandboxBackend` instead

        **Security risks:**

        - Agents can read any accessible file, including secrets (API keys,
            credentials, `.env` files)
        - Combined with network tools, secrets may be exfiltrated via SSRF attacks
        - File modifications are permanent and irreversible

        **Recommended safeguards:**

        1. Enable Human-in-the-Loop (HITL) middleware to review sensitive operations
        2. Exclude secrets from accessible filesystem paths (especially in CI/CD)
        3. Use `SandboxBackend` for production environments requiring filesystem
            interaction
        4. Use `restrict_to_root=True` with `root_dir` to enable path-based
           access restrictions (blocks `..`, `~`, and absolute paths outside root).
           In a future version, `restrict_to_root` will default to `True` for safety.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        virtual_mode: bool | None = None,
        *,
        restrict_to_root: bool | None = None,
        allowed_paths: list[str | Path] | None = None,
        max_file_size_mb: int = 10,
    ) -> None:
        """Initialize filesystem backend.

        Args:
            root_dir: Base directory for resolving relative paths.
                Defaults to current working directory.

            virtual_mode: **DEPRECATED**. Use `restrict_to_root` instead.
                This parameter will be removed in a future version.

            restrict_to_root: Whether to restrict file access to root_dir.
                - True: Restrict to root_dir (blocks traversal, enforces boundary)
                - False: No restrictions, full filesystem access (unless allowed_paths is set)
                - None: Currently defaults to False with a warning.
                  This parameter will default to True in a future version.

            allowed_paths: List of directories to restrict file access to.
                
                When specified, only these directories are accessible (root_dir is NOT
                automatically included unless restrict_to_root=True).
                
                Can be combined with restrict_to_root=True to allow root_dir plus
                these additional directories. If restrict_to_root=False (or None) and
                allowed_paths is specified, ONLY the directories in allowed_paths are
                accessible, not root_dir.

                When path restrictions are enabled (restrict_to_root=True or
                allowed_paths specified), blocks `..` and `~` and verifies all
                resolved paths remain within allowed directories.

            max_file_size_mb: Maximum file size in megabytes for operations like
                grep's Python fallback search.

                Files exceeding this limit are skipped during search. Defaults to 10 MB.

        Raises:
            ValueError: If allowed_paths is an empty list.

        Examples:
            Restrict to project directory (recommended):

            ```python
            backend = FilesystemBackend(
                root_dir="/project",
                restrict_to_root=True
            )
            ```

            Explicitly opt out of restrictions (use with caution):

            ```python
            backend = FilesystemBackend(restrict_to_root=False)
            ```

            Allow root_dir plus additional directories:

            ```python
            backend = FilesystemBackend(
                root_dir="/project",
                restrict_to_root=True,
                allowed_paths=["/data", "/home/user/.config"]
            )
            ```

            Allow only specific directories (not including root):

            ```python
            # Note: root_dir is used for path resolution but is NOT accessible
            # Only /data and /tmp are accessible
            backend = FilesystemBackend(
                root_dir="/project",
                restrict_to_root=False,  # Optional: makes intent explicit
                allowed_paths=["/data", "/tmp"]
            )
            ```
        """
        # Handle parameter migration and defaults
        # Goal: Migrate from virtual_mode AND require explicit restrict_to_root
        
        if restrict_to_root is not None:
            # User explicitly set restrict_to_root - no warnings needed
            # (This is the desired end state)
            pass
        elif virtual_mode is not None:
            # User is using deprecated virtual_mode parameter
            warnings.warn(
                "The 'virtual_mode' parameter is deprecated and will be removed in a future version. "
                "Use 'restrict_to_root' instead for the same behavior.",
                DeprecationWarning,
                stacklevel=2,
            )
            restrict_to_root = virtual_mode
        else:
            # Neither parameter specified - warn about missing explicit choice
            warnings.warn(
                "The default value of 'restrict_to_root' will change to True in a future version for security. "
                "Explicitly set restrict_to_root=False or restrict_to_root=True.",
                FutureWarning,
                stacklevel=2,
            )
            restrict_to_root = False

        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()

        # Build list of allowed directories
        allowed_dirs: list[Path] = []

        if restrict_to_root:
            allowed_dirs.append(self.cwd)

        if allowed_paths is not None:
            if isinstance(allowed_paths, list) and len(allowed_paths) == 0:
                raise ValueError(
                    "allowed_paths cannot be an empty list. "
                    "Use None for no additional paths, or omit the parameter."
                )
            allowed_dirs.extend(Path(p).resolve() for p in allowed_paths)

        # Store as None if no restrictions, otherwise store the list
        self._allowed_dirs: list[Path] | None = allowed_dirs if allowed_dirs else None
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _resolve_path(self, key: str) -> Path:
        """Resolve a file path with security checks.

        When path restrictions are enabled (_allowed_dirs is not None),
        treat incoming paths as virtual paths, block traversal, and verify
        the resolved path is within an allowed directory.

        When unrestricted (_allowed_dirs is None), preserve legacy behavior:
        absolute paths are used as-is, relative paths resolve under cwd.

        Args:
            key: File path (absolute or relative).

        Returns:
            Resolved absolute Path object.

        Raises:
            ValueError: If path traversal is attempted when restrictions are
                enabled, or if the resolved path is outside all allowed directories.
        """
        if self._allowed_dirs is None:
            # Unrestricted mode - legacy behavior
            path = Path(key)
            return path if path.is_absolute() else (self.cwd / path).resolve()

        # Restricted mode - treat as virtual path under root
        vpath = key if key.startswith("/") else "/" + key

        if ".." in vpath or vpath.startswith("~"):
            raise ValueError(
                f"Path traversal not allowed when restrictions are enabled: {key}"
            )

        # Resolve against cwd
        full = (self.cwd / vpath.lstrip("/")).resolve()

        # Verify path is within at least one allowed directory
        if not any(self._is_within_dir(full, allowed) for allowed in self._allowed_dirs):
            allowed_str = ", ".join(str(d) for d in self._allowed_dirs)
            raise ValueError(
                f"Path '{full}' is outside allowed directories: {allowed_str}"
            )

        return full

    def _is_within_dir(self, path: Path, directory: Path) -> bool:
        """Check if path is within directory (after resolving symlinks)."""
        try:
            path.relative_to(directory)
            return True
        except ValueError:
            return False

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute directory path to list files from.

        Returns:
            List of `FileInfo`-like dicts for files and directories directly in the
                directory. Directories have a trailing `/` in their path and
                `is_dir=True`.
        """
        dir_path = self._resolve_path(path)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        results: list[FileInfo] = []

        # Convert cwd to string for comparison
        cwd_str = str(self.cwd)
        if not cwd_str.endswith("/"):
            cwd_str += "/"

        # List only direct children (non-recursive)
        try:
            for child_path in dir_path.iterdir():
                try:
                    is_file = child_path.is_file()
                    is_dir = child_path.is_dir()
                except OSError:
                    continue

                abs_path = str(child_path)

                if self._allowed_dirs is None:
                    # Unrestricted mode: use absolute paths
                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": abs_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                }
                            )
                        except OSError:
                            results.append({"path": abs_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": abs_path + "/",
                                    "is_dir": True,
                                    "size": 0,
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                }
                            )
                        except OSError:
                            results.append({"path": abs_path + "/", "is_dir": True})
                else:
                    # Restricted mode: strip cwd prefix to return virtual paths
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str) :]
                    elif abs_path.startswith(str(self.cwd)):
                        # Handle case where cwd doesn't end with /
                        relative_path = abs_path[len(str(self.cwd)) :].lstrip("/")
                    else:
                        # Path is outside cwd, return as-is or skip
                        relative_path = abs_path

                    virt_path = "/" + relative_path

                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": virt_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                }
                            )
                        except OSError:
                            results.append({"path": virt_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": virt_path + "/",
                                    "is_dir": True,
                                    "size": 0,
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                                }
                            )
                        except OSError:
                            results.append({"path": virt_path + "/", "is_dir": True})
        except (OSError, PermissionError):
            pass

        # Keep deterministic order by path
        results.sort(key=lambda x: x.get("path", ""))
        return results

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute or relative file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or error message.
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"

        try:
            # Open with O_NOFOLLOW where available to avoid symlink traversal
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
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
    ) -> WriteResult:
        """Create a new file with content.

        Args:
            file_path: Path where the new file will be created.
            content: Text content to write to the file.

        Returns:
            `WriteResult` with path on success, or error message if the file
                already exists or write fails. External storage sets `files_update=None`.
        """
        resolved_path = self._resolve_path(file_path)

        if resolved_path.exists():
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

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

            return WriteResult(path=file_path, files_update=None)
        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Path to the file to edit.
            old_string: The text to search for and replace.
            new_string: The replacement text.
            replace_all: If `True`, replace all occurrences. If `False` (default),
                replace only if exactly one occurrence exists.

        Returns:
            `EditResult` with path and occurrence count on success, or error
                message if file not found or replacement fails. External storage sets
                `files_update=None`.
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            # Read securely
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            result = perform_string_replacement(content, old_string, new_string, replace_all)

            if isinstance(result, str):
                return EditResult(error=result)

            new_content, occurrences = result

            # Write securely
            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)

            return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search for a regex pattern in files.

        Uses ripgrep if available, falling back to Python regex search.

        Args:
            pattern: Regular expression pattern to search for.
            path: Directory or file path to search in. Defaults to current directory.
            glob: Optional glob pattern to filter which files to search.

        Returns:
            List of GrepMatch dicts containing path, line number, and matched text.
            Returns an error string if the regex pattern is invalid.
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
            return []

        if not base_full.exists():
            return []

        # Try ripgrep first
        results = self._ripgrep_search(pattern, base_full, glob)
        if results is None:
            results = self._python_search(pattern, base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append({"path": fpath, "line": int(line_num), "text": line_text})
        return matches

    def _ripgrep_search(self, pattern: str, base_full: Path, include_glob: str | None) -> dict[str, list[tuple[int, str]]] | None:
        """Search using ripgrep with JSON output parsing.

        Args:
            pattern: Regex pattern to search for.
            base_full: Resolved base path to search in.
            include_glob: Optional glob pattern to filter files.

        Returns:
            Dict mapping file paths to list of `(line_number, line_text)` tuples.
                Returns `None` if ripgrep is unavailable or times out.
        """
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
            if self._allowed_dirs is not None:
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

    def _python_search(self, pattern: str, base_full: Path, include_glob: str | None) -> dict[str, list[tuple[int, str]]]:
        """Fallback search using Python regex when ripgrep is unavailable.

        Recursively searches files, respecting `max_file_size_bytes` limit.

        Args:
            pattern: Regex pattern to search for.
            base_full: Resolved base path to search in.
            include_glob: Optional glob pattern to filter files by name.

        Returns:
            Dict mapping file paths to list of `(line_number, line_text)` tuples.
        """
        try:
            regex = re.compile(pattern)
        except re.error:
            return {}

        results: dict[str, list[tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent

        for fp in root.rglob("*"):
            try:
                if not fp.is_file():
                    continue
            except (PermissionError, OSError):
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
                    if self._allowed_dirs is not None:
                        try:
                            virt_path = "/" + str(fp.resolve().relative_to(self.cwd))
                        except Exception:
                            continue
                    else:
                        virt_path = str(fp)
                    results.setdefault(virt_path, []).append((line_num, line))

        return results

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern to match files against (e.g., `'*.py'`, `'**/*.txt'`).
            path: Base directory to search from. Defaults to root (`/`).

        Returns:
            List of `FileInfo` dicts for matching files, sorted by path. Each dict
                contains `path`, `is_dir`, `size`, and `modified_at` fields.
        """
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        search_path = self.cwd if path == "/" else self._resolve_path(path)
        if not search_path.exists() or not search_path.is_dir():
            return []

        results: list[FileInfo] = []
        try:
            # Use recursive globbing to match files in subdirectories as tests expect
            for matched_path in search_path.rglob(pattern):
                try:
                    is_file = matched_path.is_file()
                except (PermissionError, OSError):
                    continue
                if not is_file:
                    continue
                abs_path = str(matched_path)
                if self._allowed_dirs is None:
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": abs_path,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                            }
                        )
                    except OSError:
                        results.append({"path": abs_path, "is_dir": False})
                else:
                    cwd_str = str(self.cwd)
                    if not cwd_str.endswith("/"):
                        cwd_str += "/"
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str) :]
                    elif abs_path.startswith(str(self.cwd)):
                        relative_path = abs_path[len(str(self.cwd)) :].lstrip("/")
                    else:
                        relative_path = abs_path
                    virt = "/" + relative_path
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": virt,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                            }
                        )
                    except OSError:
                        results.append({"path": virt, "is_dir": False})
        except (OSError, ValueError):
            pass

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the filesystem.

        Args:
            files: List of (path, content) tuples where content is bytes.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                resolved_path = self._resolve_path(path)

                # Create parent directories if needed
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(resolved_path, flags, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(content)

                responses.append(FileUploadResponse(path=path, error=None))
            except FileNotFoundError:
                responses.append(FileUploadResponse(path=path, error="file_not_found"))
            except PermissionError:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
            except (ValueError, OSError) as e:
                # ValueError from _resolve_path for path traversal, OSError for other file errors
                if isinstance(e, ValueError) or "invalid" in str(e).lower():
                    responses.append(FileUploadResponse(path=path, error="invalid_path"))
                else:
                    # Generic error fallback
                    responses.append(FileUploadResponse(path=path, error="invalid_path"))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the filesystem.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved_path = self._resolve_path(path)
                # Use flags to optionally prevent symlink following if
                # supported by the OS
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except FileNotFoundError:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            except PermissionError:
                responses.append(FileDownloadResponse(path=path, content=None, error="permission_denied"))
            except IsADirectoryError:
                responses.append(FileDownloadResponse(path=path, content=None, error="is_directory"))
            except ValueError:
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
            # Let other errors propagate
        return responses
