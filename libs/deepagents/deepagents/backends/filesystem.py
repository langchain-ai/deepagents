"""`FilesystemBackend`: Read and write files directly from the filesystem."""

import json
import logging
import os
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

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

logger = logging.getLogger(__name__)


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
        3. For production environments, prefer `StateBackend`, `StoreBackend` or `SandboxBackend`

        In general, we expect this backend to be used with Human-in-the-Loop (HITL)
        middleware, or within a properly sandboxed environment if you need to run
        untrusted workloads.

        !!! note

            `virtual_mode=True` is primarily for virtual path semantics (for example with
            `CompositeBackend`). It can also provide path-based guardrails by blocking
            traversal (`..`, `~`) and absolute paths outside `root_dir`, but it does not
            provide sandboxing or process isolation. The default (`virtual_mode=False`)
            provides no security even with `root_dir` set.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        virtual_mode: bool | None = None,  # noqa: FBT001
        max_file_size_mb: int = 10,
    ) -> None:
        """Initialize filesystem backend.

        Args:
            root_dir: Optional root directory for file operations.

                Defaults to the current working directory.

                - When `virtual_mode=False` (default): Only affects relative path resolution.
                - When `virtual_mode=True`: Acts as a virtual root for filesystem operations.

            virtual_mode: Enable virtual path mode.

                **Primary use case:** stable, backend-independent path semantics when
                used with `CompositeBackend`, which strips route prefixes and forwards
                normalized paths to the routed backend.

                When `True`, all paths are treated as virtual paths anchored to
                `root_dir`. Path traversal (`..`, `~`) is blocked and all resolved paths
                are verified to remain within `root_dir`.

                When `False` (default), absolute paths are used as-is and relative paths
                are resolved under `root_dir`. This provides no security against an agent
                choosing paths outside `root_dir`.

                - Absolute paths (e.g., `/etc/passwd`) bypass `root_dir` entirely
                - Relative paths with `..` can escape `root_dir`
                - Agents have unrestricted filesystem access

            max_file_size_mb: Maximum file size in megabytes for operations like
                grep's Python fallback search.

                Files exceeding this limit are skipped during search. Defaults to 10 MB.
        """
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        if virtual_mode is None:
            warnings.warn(
                "FilesystemBackend virtual_mode default will change in deepagents 0.5.0; "
                "please specify virtual_mode explicitly. "
                "Note: virtual_mode is for virtual path semantics (e.g., CompositeBackend routing) and optional path-based guardrails; "
                "it does not provide sandboxing or process isolation. "
                "Security note: leaving virtual_mode=False allows absolute paths and '..' to bypass root_dir. "
                "Consult the API reference for details.",
                DeprecationWarning,
                stacklevel=2,
            )
            virtual_mode = False
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _resolve_path(self, key: str) -> Path:
        """Resolve a file path with security checks.

        When `virtual_mode=True`, treat incoming paths as virtual absolute paths under
        `self.cwd`, disallow traversal (`..`, `~`) and ensure resolved path stays within
        root.

        When `virtual_mode=False`, preserve legacy behavior: absolute paths are allowed
        as-is; relative paths resolve under cwd.

        Args:
            key: File path (absolute, relative, or virtual when `virtual_mode=True`).

        Returns:
            Resolved absolute `Path` object.

        Raises:
            ValueError: If path traversal is attempted in `virtual_mode` or if the
                resolved path escapes the root directory.
        """
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                msg = "Path traversal not allowed"
                raise ValueError(msg)
            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                msg = f"Path:{full} outside root directory: {self.cwd}"
                raise ValueError(msg) from None
            return full

        path = Path(key)
        if path.is_absolute():
            return path
        return (self.cwd / path).resolve()

    def _to_virtual_path(self, path: Path) -> str:
        """Convert a filesystem path to a virtual path relative to cwd.

        Args:
            path: Filesystem path to convert.

        Returns:
            Forward-slash relative path string prefixed with `/`.

        Raises:
            ValueError: If path is outside cwd.
            OSError: If path cannot be resolved (broken symlink, permission denied).
        """
        return "/" + path.resolve().relative_to(self.cwd).as_posix()

    def ls_info(self, path: str) -> list[FileInfo]:  # noqa: C901, PLR0912, PLR0915  # Complex virtual_mode logic
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

                if not self.virtual_mode:
                    # Non-virtual mode: use absolute paths
                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": abs_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
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
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                                }
                            )
                        except OSError:
                            results.append({"path": abs_path + "/", "is_dir": True})
                else:
                    # Virtual mode: strip cwd prefix using Path for cross-platform support
                    try:
                        virt_path = self._to_virtual_path(child_path)
                    except ValueError:
                        logger.debug("Skipping path outside root: %s", child_path)
                        continue
                    except OSError:
                        logger.warning("Could not resolve path: %s", child_path, exc_info=True)
                        continue

                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": virt_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
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
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
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
        replace_all: bool = False,  # noqa: FBT001, FBT002
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
        """Search for a literal text pattern in files.

        Uses ripgrep if available, falling back to Python search.

        Args:
            pattern: Literal string to search for (NOT regex).
            path: Directory or file path to search in. Defaults to current directory.
            glob: Optional glob pattern to filter which files to search.

        Returns:
            List of GrepMatch dicts containing path, line number, and matched text.
        """
        # Resolve base path
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return []

        if not base_full.exists():
            return []

        # Try ripgrep first (with -F flag for literal search)
        results = self._ripgrep_search(pattern, base_full, glob)
        if results is None:
            # Python fallback needs escaped pattern for literal search
            results = self._python_search(re.escape(pattern), base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append({"path": fpath, "line": int(line_num), "text": line_text})
        return matches

    def _ripgrep_search(self, pattern: str, base_full: Path, include_glob: str | None) -> dict[str, list[tuple[int, str]]] | None:  # noqa: C901  # Split except clauses for logging
        """Search using ripgrep with fixed-string (literal) mode.

        Args:
            pattern: Literal string to search for (unescaped).
            base_full: Resolved base path to search in.
            include_glob: Optional glob pattern to filter files.

        Returns:
            Dict mapping file paths to list of `(line_number, line_text)` tuples.
                Returns `None` if ripgrep is unavailable or times out.
        """
        cmd = ["rg", "--json", "-F"]  # -F enables fixed-string (literal) mode
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
                    virt = self._to_virtual_path(p)
                except ValueError:
                    logger.debug("Skipping grep result outside root: %s", p)
                    continue
                except OSError:
                    logger.warning("Could not resolve grep result path: %s", p, exc_info=True)
                    continue
            else:
                virt = str(p)
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(virt, []).append((int(ln), lt))

        return results

    def _python_search(self, pattern: str, base_full: Path, include_glob: str | None) -> dict[str, list[tuple[int, str]]]:  # noqa: C901, PLR0912
        """Fallback search using Python when ripgrep is unavailable.

        Recursively searches files, respecting `max_file_size_bytes` limit.

        Args:
            pattern: Escaped regex pattern (from re.escape) for literal search.
            base_full: Resolved base path to search in.
            include_glob: Optional glob pattern to filter files by name.

        Returns:
            Dict mapping file paths to list of `(line_number, line_text)` tuples.
        """
        # Compile escaped pattern once for efficiency (used in loop)
        regex = re.compile(pattern)

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
                    if self.virtual_mode:
                        try:
                            virt_path = self._to_virtual_path(fp)
                        except ValueError:
                            logger.debug("Skipping grep result outside root: %s", fp)
                            continue
                        except OSError:
                            logger.warning("Could not resolve grep result path: %s", fp, exc_info=True)
                            continue
                    else:
                        virt_path = str(fp)
                    results.setdefault(virt_path, []).append((line_num, line))

        return results

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:  # noqa: C901, PLR0912  # Complex virtual_mode logic
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

        if self.virtual_mode and ".." in Path(pattern).parts:
            msg = "Path traversal not allowed in glob pattern"
            raise ValueError(msg)

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
                if self.virtual_mode:
                    try:
                        matched_path.resolve().relative_to(self.cwd)
                    except ValueError:
                        continue
                abs_path = str(matched_path)
                if not self.virtual_mode:
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": abs_path,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                            }
                        )
                    except OSError:
                        results.append({"path": abs_path, "is_dir": False})
                else:
                    # Virtual mode: use Path for cross-platform support
                    try:
                        virt = self._to_virtual_path(matched_path)
                    except ValueError:
                        logger.debug("Skipping glob result outside root: %s", matched_path)
                        continue
                    except OSError:
                        logger.warning("Could not resolve glob result path: %s", matched_path, exc_info=True)
                        continue
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": virt,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
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
