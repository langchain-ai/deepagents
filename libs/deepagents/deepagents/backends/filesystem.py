"""`FilesystemBackend`: Read and write files directly from the filesystem."""

import base64
import errno
import functools
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from deepagents._api.deprecation import warn_deprecated
from deepagents.backends.protocol import (
    DEFAULT_GREP_TIMEOUT,
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileOperationError,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    MAX_VIDEO_INPUT_BYTES,
    _get_backend_read_file_type,
    check_empty_content,
    compile_grep_include_glob,
    compile_recursive_glob,
    perform_string_replacement,
)

logger = logging.getLogger(__name__)

_DEFAULT_GLOB_TIMEOUT = 5
"""Wall-clock budget in seconds for a single local `glob` walk.

A fixed bound keeps `glob` from hanging on huge or slow trees; when it elapses
the walk returns whatever it found so far with `GlobResult.truncated=True`
rather than erroring. Kept below the middleware's `GLOB_TIMEOUT`
(`deepagents.middleware.filesystem.GLOB_TIMEOUT`, currently 10s) so the backend
returns partial results before that outer net abandons the call; the ordering is
guarded by `test_glob_backend_budget_below_middleware_deadline`.
"""


@functools.cache
def _resolve_ripgrep_path() -> str | None:
    """Locate the `rg` executable on `PATH`, cached for the process lifetime.

    Logs an `INFO`-level message exactly once if ripgrep is not found so
    operators can diagnose silent slow-path searches when `rg` is installed
    but not visible on the agent's `PATH` (common in sandboxed or
    stripped-environment launchers).

    Returns:
        Absolute path to `rg`, or `None` if not on `PATH`.
    """
    path = shutil.which("rg")
    if path is None:
        logger.info(
            "ripgrep ('rg') not found on PATH; using Python grep fallback. Install ripgrep for faster searches and automatic .gitignore handling."
        )
    return path


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
            warn_deprecated(
                since="0.5.0",
                removal="0.6.0",
                message=(
                    "`FilesystemBackend` `virtual_mode` default will change "
                    "in deepagents==0.6.0; please specify `virtual_mode` "
                    "explicitly. Note: `virtual_mode` is for virtual path "
                    "semantics (e.g., `CompositeBackend` routing) and "
                    "optional path-based guardrails; it does not provide "
                    "sandboxing or process isolation. Security note: leaving "
                    "`virtual_mode=False` allows absolute paths and `'..'` "
                    "to bypass `root_dir`. Consult the API reference for "
                    "details."
                ),
                package="deepagents",
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
            OSError: If the path is a symlink loop (`ELOOP`).
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
            _raise_if_symlink_loop(full)
            return full

        path = Path(key)
        if path.is_absolute():
            _raise_if_symlink_loop(path)
            return path
        resolved = (self.cwd / path).resolve()
        _raise_if_symlink_loop(resolved)
        return resolved

    def _to_virtual_path(self, path: Path) -> str:
        """Convert a filesystem path to a virtual path relative to cwd.

        Args:
            path: Filesystem path to convert.

        Returns:
            Forward-slash relative path string prefixed with `/`.

        Raises:
            ValueError: If path is outside cwd.
            OSError: If `Path.resolve()` raises during resolution (e.g.,
                permission denied, or `ELOOP` on Python 3.13+).
            RuntimeError: If `Path.resolve()` detects a symlink loop on
                Python <=3.12 (wraps the underlying `OSError(ELOOP)`).
        """
        return "/" + path.resolve().relative_to(self.cwd).as_posix()

    def _display_path(self, path: Path) -> str:
        """Render a path for agent-visible messages without leaking the real root.

        In `virtual_mode`, surfacing the resolved on-disk path would defeat the
        virtual-path abstraction (and leak `root_dir`), so convert to the virtual
        form; fall back to the bare name (or `/` for a root path with no name
        component) if that conversion fails (e.g., the path escaped the root or
        could not be resolved). In non-virtual mode the real path is already the
        caller's own, so return it unchanged.

        Args:
            path: Filesystem path to render.

        Returns:
            A virtual path string in `virtual_mode`, otherwise the real path.
        """
        if not self.virtual_mode:
            return str(path)
        try:
            return self._to_virtual_path(path)
        except (ValueError, OSError, RuntimeError):
            return path.name or "/"

    def ls(self, path: str) -> LsResult:  # noqa: C901, PLR0912, PLR0915  # Complex virtual_mode logic
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute directory path to list files from.

        Returns:
            `LsResult` with `entries` listing files and directories directly in the
                directory on success.

                Directories have a trailing `/` in their path and `is_dir=True`.

                Missing paths set `error` to `Path '<path>': path_not_found`
                with `entries=None`.

                File paths set `error` to `Path '<path>': not_a_directory`
                with `entries=None`.

                Empty directories return `error=None` and `entries=[]`.
        """
        try:
            dir_path = self._resolve_path(path)
            if not dir_path.exists():
                return LsResult(error=f"Path '{path}': path_not_found", entries=None)
            if not dir_path.is_dir():
                return LsResult(error=f"Path '{path}': not_a_directory", entries=None)
        except (OSError, RuntimeError) as e:
            msg = f"Cannot list '{path}': {e}"
            logger.warning("%s", msg)
            return LsResult(error=msg, entries=None)

        results: list[FileInfo] = []
        errors: list[str] = []

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
                except (OSError, RuntimeError) as e:
                    msg = f"child error: cannot stat '{child_path}': {e}"
                    logger.warning("%s", msg)
                    errors.append(msg)
                    continue

                abs_path = str(child_path)
                if not is_file and not is_dir:
                    # `is_symlink()` itself can raise OSError on stale handles or
                    # mid-walk permission flips; keep it inside the guard.
                    try:
                        if child_path.is_symlink():
                            child_path.resolve()
                            _raise_if_symlink_loop(child_path)
                    except (OSError, RuntimeError) as e:
                        msg = f"child error: cannot resolve '{child_path}': {e}"
                        logger.warning("%s", msg)
                        errors.append(msg)
                    continue

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
                    except (OSError, RuntimeError) as e:
                        msg = f"child error: cannot resolve '{child_path}': {e}"
                        logger.warning("%s", msg)
                        errors.append(msg)
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
        except (OSError, RuntimeError) as e:
            # iterdir() itself can raise mid-iteration (NFS drops, FUSE failures,
            # permission flips). Surface as a top-level abort so partial results
            # are not labeled as authoritative.
            msg = f"Listing of '{path}' aborted: {e}"
            logger.warning("%s", msg)
            errors.append(msg)

        # Keep deterministic order by path
        results.sort(key=lambda x: x.get("path", ""))
        # Sort errors for deterministic output across filesystems (iterdir()
        # ordering varies); newline-join keeps them readable when any individual
        # message contains punctuation.
        error = "\n".join(sorted(errors)) if errors else None
        return LsResult(error=error, entries=results)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content for the requested line range.

        Args:
            file_path: Absolute or relative file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            `ReadResult` with raw (unformatted) content for the requested window.

                Line-number formatting is applied by the middleware.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as e:
            return ReadResult(error=f"Error reading file '{file_path}': {e}")

        try:
            if not resolved_path.exists() or not resolved_path.is_file():
                return ReadResult(error=f"File '{file_path}' not found")

            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                file_type = _get_backend_read_file_type(file_path)
                if file_type != "text":
                    if file_type == "video" and os.fstat(fd).st_size > MAX_VIDEO_INPUT_BYTES:
                        return ReadResult(error=f"Video file exceeds maximum input size of {MAX_VIDEO_INPUT_BYTES} bytes")
                    with os.fdopen(fd, "rb") as f:
                        fd = -1
                        raw = f.read()
                    encoded = base64.standard_b64encode(raw).decode("ascii")
                    file_data = FileData(content=encoded, encoding="base64")
                else:
                    with os.fdopen(fd, "r", encoding="utf-8") as f:
                        fd = -1
                        content = f.read()
            finally:
                if fd >= 0:
                    os.close(fd)

            if file_type == "text":
                empty_msg = check_empty_content(content)
                if empty_msg:
                    file_data = FileData(content=empty_msg, encoding="utf-8")
                else:
                    # `splitlines(keepends=True)` preserves whether the final line
                    # has a terminator; joining with `""` round-trips the file's
                    # trailing-newline state. Required so `edit()` can detect
                    # EOF-newline mismatches in the model's `old_string`.
                    lines = content.splitlines(keepends=True)
                    start_idx = offset
                    end_idx = min(start_idx + limit, len(lines))

                    if start_idx >= len(lines):
                        return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")

                    file_data = FileData(content="".join(lines[start_idx:end_idx]), encoding="utf-8")

            return ReadResult(file_data=file_data)
        except (OSError, UnicodeDecodeError) as e:
            return ReadResult(error=f"Error reading file '{file_path}': {e}")

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Write content to a file, creating it or overwriting it if it already exists.

        Args:
            file_path: Path where the file will be written.
            content: Text content to write to the file.

        Returns:
            `WriteResult` with path on success, or error message on write failure.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

        try:
            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # Prefer O_NOFOLLOW to avoid writing through symlinks
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags, 0o644)
            # newline="" disables Windows CRLF translation so callers that
            # pass LF-only content get LF-only bytes on disk.
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(content)

            return WriteResult(path=file_path)
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
                message if file not found or replacement fails.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

        try:
            if not resolved_path.exists() or not resolved_path.is_file():
                return EditResult(error=f"Error: File '{file_path}' not found")

            # Read securely
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            # Normalize line endings in old_string/new_string to match the
            # text-mode read above. Python universal newlines (the default
            # when newline=None) converts \r\n and bare \r to \n on read.
            # Callers that obtained content via binary-mode reads (e.g.
            # download_files) may pass strings with \r\n or \r that would
            # fail to match the \n-only content.
            old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
            new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")

            result = perform_string_replacement(content, old_string, new_string, replace_all)

            if isinstance(result, str):
                return EditResult(error=result)

            new_content, occurrences = result

            # Write securely
            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(new_content)

            return EditResult(path=file_path, occurrences=int(occurrences))
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a file or directory from the filesystem.

        Files are unlinked. Directories are removed recursively along with all
        of their contents. Symlinks are removed as links and never followed into
        their target (so deleting a symlink to a directory removes only the link).

        Args:
            file_path: Path to the file or directory to delete.

        Returns:
            `DeleteResult` with the deleted path on success, or an error if the
                path does not exist or removal fails. A recursive directory
                removal may delete some entries before failing partway (for
                example when a nested entry is not writable).
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as e:
            return DeleteResult(error=f"Error deleting '{file_path}': {e}")

        try:
            if not resolved_path.exists() and not resolved_path.is_symlink():
                return DeleteResult(error=f"Error: '{file_path}' not found")
            if resolved_path.is_symlink():
                resolved_path.unlink()
            elif resolved_path.is_dir():
                shutil.rmtree(resolved_path)
            else:
                resolved_path.unlink()
            return DeleteResult(path=file_path)
        except (OSError, RuntimeError) as e:
            return DeleteResult(error=f"Error deleting '{file_path}': {e}")

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for a literal text pattern in files.

        Uses ripgrep if available, falling back to Python search.

        Args:
            pattern: Literal string to search for (NOT regex).
            path: Directory or file path to search in. Defaults to current directory.
            glob: Optional glob pattern to filter which files to search.

        Returns:
            `GrepResult` with matches or error.
        """
        # Resolve base path
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return GrepResult(matches=[])
        except (OSError, RuntimeError) as e:
            search_path = path or "."
            return GrepResult(error=f"Error searching path '{search_path}': {e}", matches=[])

        try:
            if not base_full.exists():
                return GrepResult(matches=[])
        except OSError as e:
            search_path = path or "."
            return GrepResult(error=f"Error searching path '{search_path}': {e}", matches=[])

        # Try ripgrep first (with -F flag for literal search)
        results, truncated = self._ripgrep_search(pattern, base_full, glob)
        partial_error: str | None = None
        if results is None:
            # Python fallback does literal substring matching on the raw pattern.
            results, truncated, partial_error = self._python_search(pattern, base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append({"path": fpath, "line": int(line_num), "text": line_text})
        return GrepResult(error=partial_error, matches=matches, truncated=truncated)

    def _ripgrep_search(self, pattern: str, base_full: Path, include_glob: str | None) -> tuple[dict[str, list[tuple[int, str]]] | None, bool]:  # noqa: C901, PLR0912, PLR0915  # except clauses split per-exception for targeted logging (timeout vs exec-race vs ripgrep hard-error)
        """Search using ripgrep with fixed-string (literal) mode.

        Args:
            pattern: Literal string to search for (unescaped).
            base_full: Resolved base path to search in.
            include_glob: Optional glob pattern to filter files.

        Returns:
            A `(results, truncated)` tuple. `results` maps file paths to a list
                of `(line_number, line_text)` tuples, or is `None` when ripgrep
                is unavailable, hard-errored, or timed out before emitting any
                output — in each case the caller should fall back to the Python
                search. `truncated` is `True` when ripgrep timed out but had
                already emitted partial output (returned here instead of falling
                back). Results whose resolved path lies outside `base_full` are
                silently filtered regardless of `virtual_mode`.
        """
        rg_path = _resolve_ripgrep_path()
        if rg_path is None:
            return None, False

        cmd = [rg_path, "--json", "-F"]  # -F enables fixed-string (literal) mode
        if include_glob:
            cmd.extend(["--glob", include_glob])
        # When rg is given an absolute search path, directory-component
        # globs (e.g. "docs/*.md") silently match nothing if the process cwd
        # != search root (#2732). For a directory, set `cwd=base_full` and
        # use `.` as the search path so `--glob` resolves correctly. For a
        # single file, leave `cwd` unset and keep the absolute path —
        # `subprocess.run` would raise `NotADirectoryError` if passed a file
        # path as `cwd`, and globs are irrelevant for single-file searches.
        rg_cwd: str | None = None
        if base_full.is_dir():
            cmd.extend(["--", pattern, "."])
            rg_cwd = str(base_full)
        else:
            cmd.extend(["--", pattern, str(base_full)])

        truncated = False
        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=DEFAULT_GREP_TIMEOUT,
                check=False,
                cwd=rg_cwd,
            )
            stdout = proc.stdout
        except subprocess.TimeoutExpired as exc:
            # `subprocess.run` attaches whatever ripgrep wrote before the kill to
            # `exc.stdout` on both POSIX (drained during `communicate`) and
            # Windows (via a post-`kill` `communicate`), so this path needs no
            # per-platform branch. `--json` is newline-delimited so a truncated
            # trailing frame just fails to parse and is skipped below; the
            # matches that did land are still usable. Only fall back to the
            # (slower) Python search when nothing was captured.
            # `TimeoutExpired.stdout` is bytes even under `text=True`, so decode
            # before the emptiness check or real partial output looks empty.
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if not stdout:
                logger.warning("ripgrep timed out after %ds with no output; using Python grep fallback", DEFAULT_GREP_TIMEOUT)
                return None, False
            logger.warning("ripgrep timed out after %ds; returning partial results", DEFAULT_GREP_TIMEOUT)
            truncated = True
        except (FileNotFoundError, PermissionError, NotADirectoryError) as e:
            # `rg` resolved at cache time but failed at exec — treat as a
            # runtime anomaly (uninstall, permission change, or `which`-vs-exec
            # race) rather than a missing-tool config, hence WARNING instead
            # of the INFO emitted by `_resolve_ripgrep_path`. Drop the cache
            # so the next call re-probes `PATH`.
            logger.warning("ripgrep subprocess failed (%s: %s); using Python grep fallback", type(e).__name__, e)
            _resolve_ripgrep_path.cache_clear()
            return None, False

        # Ripgrep exits 0 on match, 1 on no-match (both expected), 2+ on a hard
        # error (invalid pattern, unreadable directory, malformed glob, etc.).
        # Silently parsing stdout on a hard error reports zero matches to the
        # agent — exactly the silent failure this resolver is meant to avoid.
        # A timeout has no return code yet, so skip this guard for that path.
        if not truncated and proc.returncode not in (0, 1):
            stderr = proc.stderr.strip()[:500] if proc.stderr else ""
            logger.warning("ripgrep exited %d (stderr=%r); using Python grep fallback", proc.returncode, stderr)
            return None, False

        results: dict[str, list[tuple[int, str]]] = {}
        base_resolved = base_full.resolve()
        for line in stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            data_type = data.get("type")
            if data_type == "error":
                # Per-file errors in `--json` mode (e.g., non-UTF-8 file
                # ripgrep refused to read). Surface at DEBUG so debugging is
                # possible without spamming WARNING for every binary file.
                logger.debug("ripgrep per-file error frame: %s", data.get("data"))
                continue
            if data_type != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            # When rg ran from cwd=base_full it emits paths relative to that
            # cwd; join (don't `.resolve()`) so symlink form is preserved for
            # callers. When rg searched a single file it emits the absolute
            # path we passed in.
            raw = Path(ftext)
            p = raw if raw.is_absolute() else (base_full / raw)
            # Defensive containment check: resolve both sides only for the
            # comparison so symlinks that resolve to paths outside `base_full`
            # can't leak results, while `p` itself keeps its original shape.
            # OSError guards against unresolvable symlink targets.
            try:
                p.resolve().relative_to(base_resolved)
            except (ValueError, OSError):
                logger.warning(
                    "Skipping ripgrep result outside search root: path=%s root=%s",
                    p,
                    base_full,
                )
                continue
            if self.virtual_mode:
                try:
                    virt = self._to_virtual_path(p)
                except ValueError:
                    logger.debug("Skipping grep result outside root: %s", p)
                    continue
                except (OSError, RuntimeError):
                    logger.warning("Could not resolve grep result path: %s", p, exc_info=True)
                    continue
            else:
                virt = str(p)
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(virt, []).append((int(ln), lt))

        return results, truncated

    def _python_search(  # noqa: C901, PLR0912, PLR0915
        self,
        pattern: str,
        base_full: Path,
        include_glob: str | None,
        *,
        timeout: int = DEFAULT_GREP_TIMEOUT,
    ) -> tuple[dict[str, list[tuple[int, str]]], bool, str | None]:
        """Fallback search using Python when ripgrep is unavailable.

        Recursively searches files, respecting `max_file_size_bytes` limit
        and a wall-clock timeout.

        Args:
            pattern: Literal string to search for (substring match, not regex).
            base_full: Resolved base path to search in.
            include_glob: Optional glob pattern to filter files by name.
            timeout: Maximum wall-clock seconds before the search is aborted.

        Returns:
            A `(results, truncated, error)` tuple. `results` contains every
                match found before iteration stopped. `truncated` is `True` when
                the wall-clock `timeout` elapsed, leaving `results` valid but
                incomplete. `error` is `None` on a clean walk, otherwise a
                human-readable message when at least one file could not be
                opened or fully read, or the walk aborted early (e.g., a
                directory entry was removed mid-walk).
        """
        deadline = time.monotonic() + timeout
        glob_matcher = compile_grep_include_glob(include_glob) if include_glob else None

        results: dict[str, list[tuple[int, str]]] = {}
        file_errors: list[str] = []
        root = base_full if base_full.is_dir() else base_full.parent

        def _log_timeout() -> None:
            logger.warning(
                "Grep of '%s' timed out after %ss with %d matching file(s); returning partial results",
                self._display_path(base_full),
                timeout,
                len(results),
            )

        def _file_errors_msg() -> str | None:
            if not file_errors:
                return None
            return "One or more files could not be fully searched:\n" + "\n".join(file_errors)

        def _safe_detail(exc: Exception) -> str:
            # Build an agent-safe detail string. `OSError.__str__` embeds the
            # real filename/path, so for those surface only `strerror` (the
            # path-free reason). `UnicodeDecodeError` exposes `.reason`. In
            # virtual mode, generic exception text can still contain the real
            # root path (for example from `Path.rglob`), so keep it out of
            # agent-visible errors.
            if isinstance(exc, OSError):
                detail = exc.strerror
            else:
                detail = getattr(exc, "reason", None)
                if detail is None and not self.virtual_mode:
                    detail = str(exc)
            return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__

        try:
            for fp in root.rglob("*"):
                if time.monotonic() > deadline:
                    _log_timeout()
                    return results, True, None
                try:
                    if not fp.is_file():
                        continue
                except (PermissionError, OSError, RuntimeError):
                    continue
                if glob_matcher is not None:
                    rel_path = fp.relative_to(root).as_posix()
                    if not glob_matcher(rel_path):
                        continue
                try:
                    if fp.stat().st_size > self.max_file_size_bytes:
                        continue
                except (OSError, RuntimeError):
                    continue
                # Stream the file line-by-line so a single huge file neither
                # blows peak memory nor monopolizes the wall-clock budget.
                scanned_lines = 0
                try:
                    if self.virtual_mode:
                        try:
                            virt_path = self._to_virtual_path(fp)
                        except ValueError:
                            logger.debug("Skipping grep result outside root: %s", fp)
                            continue
                        except (OSError, RuntimeError):
                            logger.warning("Could not resolve grep result path: %s", fp, exc_info=True)
                            continue
                    else:
                        virt_path = str(fp)
                    with fp.open(encoding="utf-8", errors="strict") as handle:
                        for line_num, raw_line in enumerate(handle, 1):
                            scanned_lines = line_num
                            if line_num % 2048 == 0 and time.monotonic() > deadline:
                                _log_timeout()
                                return results, True, None
                            if pattern not in raw_line:
                                continue
                            line = raw_line.rstrip("\n")
                            results.setdefault(virt_path, []).append((line_num, line))
                except UnicodeDecodeError as e:
                    # A file that fails to decode before any line is scanned is
                    # treated as binary and skipped silently, mirroring ripgrep's
                    # binary-file skip (and its DEBUG-level per-file error frames).
                    # If decoding only failed partway through, surface the
                    # truncation so the partial result is flagged.
                    if scanned_lines > 0 or virt_path in results:
                        file_errors.append(f"- {virt_path}: {_safe_detail(e)}")
                    else:
                        logger.debug("Skipping undecodable file in grep fallback: %s", fp)
                    continue
                except (OSError, RuntimeError) as e:
                    # Could not open or fully read the file. Unlike an undecodable
                    # binary, this is a file the caller likely expected to search,
                    # so always surface it even when no lines were scanned.
                    file_errors.append(f"- {virt_path}: {_safe_detail(e)}")
                    logger.debug("Could not fully read %s in grep fallback", fp, exc_info=True)
                    continue
        except (OSError, RuntimeError) as e:
            # `rglob` raised mid-iteration. `OSError` covers the common case
            # where a directory entry is unlinked or renamed during the walk
            # (the original `FileNotFoundError` report). `RuntimeError` covers
            # symlink-loop detection on older Python versions. Return the
            # matches already accumulated and surface the abort so callers
            # don't treat the result as complete.
            # `_display_path`/`_safe_detail` keep the real `root_dir` out of the
            # agent-visible error (the raw `rglob` exception can embed it too).
            msg = f"Grep of '{self._display_path(base_full)}' aborted after {len(results)} matching file(s): {_safe_detail(e)}"
            logger.warning("%s", msg, exc_info=True)
            return results, False, msg

        return results, False, _file_errors_msg()

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:  # noqa: C901, PLR0912, PLR0915  # Complex virtual_mode logic
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern to match files against (e.g., `'*.py'`, `'**/*.txt'`).
            path: Base directory to search from.

                Defaults to `root_dir` / `cwd`.

        Returns:
            `GlobResult` with matching files. `truncated` is `True` (and
            `matches` is partial) when the walk exceeded its wall-clock budget.
        """
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        if self.virtual_mode and ".." in Path(pattern).parts:
            msg = "Path traversal not allowed in glob pattern"
            raise ValueError(msg)

        try:
            search_path = self.cwd if path is None or path == "/" else self._resolve_path(path)
            if not search_path.exists() or not search_path.is_dir():
                return GlobResult(matches=[])
        except (OSError, RuntimeError) as e:
            display_path = path if path is not None else "<default>"
            return GlobResult(error=f"Error globbing path '{display_path}': {e}", matches=[])

        # A fixed wall-clock budget keeps a glob over a huge or slow tree from
        # hanging; on expiry we return the matches gathered so far flagged as
        # truncated rather than blocking or erroring.
        deadline = time.monotonic() + _DEFAULT_GLOB_TIMEOUT
        truncated = False
        results: list[FileInfo] = []
        # Walk every entry (`rglob("*")`) and apply the pattern ourselves rather
        # than `rglob(pattern)`: `rglob(pattern)` only surfaces matches, so a
        # sparse or zero-match search over a huge tree traverses the whole tree
        # without ever checking the deadline. `rglob("*")` yields on every entry,
        # letting us honour the deadline while matching with `rglob` semantics.
        try:
            # Compiled inside the try so a malformed pattern (e.g. an unbalanced
            # brace, now that brace expansion is enabled) returns a
            # `GlobResult(error=...)` instead of raising to a direct caller.
            matches_pattern = compile_recursive_glob(pattern)
            for matched_path in search_path.rglob("*"):
                if time.monotonic() > deadline:
                    logger.warning(
                        "Glob of '%s' timed out after %ss with %d match(es); returning partial results",
                        path if path is not None else "<default>",
                        _DEFAULT_GLOB_TIMEOUT,
                        len(results),
                    )
                    truncated = True
                    break
                try:
                    rel_path = matched_path.relative_to(search_path).as_posix()
                except ValueError:
                    continue
                if not matches_pattern(rel_path):
                    continue
                try:
                    is_file = matched_path.is_file()
                except (PermissionError, OSError, RuntimeError):
                    continue
                if not is_file:
                    continue
                if self.virtual_mode:
                    try:
                        matched_path.resolve().relative_to(self.cwd)
                    except (OSError, RuntimeError, ValueError):
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
                    except (OSError, RuntimeError):
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
        except (OSError, RuntimeError, ValueError) as e:
            # The pattern failed to compile, or `rglob()` raised mid-iteration.
            # Return whatever was accumulated but as an error so callers don't
            # trust it as complete.
            display_path = path if path is not None else "<default>"
            msg = f"Glob of '{display_path}' aborted partway: {e}"
            logger.warning("%s", msg, exc_info=True)
            results.sort(key=lambda x: x.get("path", ""))
            return GlobResult(error=msg, matches=results)

        results.sort(key=lambda x: x.get("path", ""))
        return GlobResult(matches=results, truncated=truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the filesystem.

        Args:
            files: List of `(path, content)` tuples where content is bytes.

        Returns:
            List of `FileUploadResponse` objects, one per input file.

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
            except Exception as exc:
                error = _map_exception_to_standard_error(exc)
                if error is None:
                    raise
                responses.append(FileUploadResponse(path=path, error=error))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the filesystem.

        Args:
            paths: List of file paths to download.

        Returns:
            List of `FileDownloadResponse` objects, one per input path.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved_path = self._resolve_path(path)
                if resolved_path.is_dir():
                    responses.append(FileDownloadResponse(path=path, content=None, error=IS_DIRECTORY))
                    continue
                # Use flags to optionally prevent symlink following if
                # supported by the OS
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception as exc:
                error = _map_exception_to_standard_error(exc)
                if error is None:
                    raise
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
        return responses


def _map_exception_to_standard_error(exc: Exception) -> FileOperationError | None:
    """Map a caught exception to a standardized `FileOperationError` code.

    Classification is based on exception type only (stdlib hierarchy).
    Returns `None` for any exception that cannot be classified by type,
    letting callers decide whether to re-raise or fall back to `str(exc)`.

    Args:
        exc: The exception to classify.

    Returns:
        A `FileOperationError` literal, or `None` if unrecognized.
    """
    error: FileOperationError | None = None
    if isinstance(exc, FileNotFoundError):
        error = FILE_NOT_FOUND
    elif _is_symlink_loop_error(exc):
        error = INVALID_PATH
    elif isinstance(exc, PermissionError):
        error = PERMISSION_DENIED
    elif isinstance(exc, IsADirectoryError):
        error = IS_DIRECTORY
    elif isinstance(exc, (NotADirectoryError, FileExistsError, ValueError)):
        error = INVALID_PATH
    return error


# Win32 `ERROR_CANT_RESOLVE_FILENAME`, surfaced by NTFS for reparse-point
# cycles. Python's mapping to `errno.ELOOP` is unreliable on this code path,
# so we match the raw winerror when classifying symlink-loop failures.
_WIN32_ERROR_CANT_RESOLVE_FILENAME = 1921


def _is_eloop_oserror(exc: BaseException | None) -> bool:
    """Return `True` if `exc` is an `OSError` reporting a symlink loop on any platform."""
    return isinstance(exc, OSError) and (exc.errno == errno.ELOOP or getattr(exc, "winerror", None) == _WIN32_ERROR_CANT_RESOLVE_FILENAME)


def _is_symlink_loop_error(exc: Exception) -> bool:
    """Return `True` when an exception came from an `ELOOP` filesystem error."""
    if _is_eloop_oserror(exc):
        return True

    # Python <=3.12 wraps `OSError(errno.ELOOP, ...)` from `Path.resolve()` in
    # `RuntimeError`. The stable signal is the exception context, not the
    # human-readable RuntimeError message.
    return isinstance(exc, RuntimeError) and any(_is_eloop_oserror(chained) for chained in (exc.__cause__, exc.__context__))


def _raise_if_symlink_loop(path: Path) -> None:
    """Raise `OSError(ELOOP)` if `path` is an unresolvable symlink loop.

    Python 3.13+ changed `Path.resolve(strict=False)` to silently return the
    unresolved path for symlink loops instead of raising. This restores the
    pre-3.13 contract by probing with a `stat()` that follows symlinks and
    re-raising loop errors. Other errors (broken target, permission denied)
    are left for downstream existence checks to surface.

    Windows surfaces NTFS reparse-point cycles as `OSError` with
    `winerror=1921` (`ERROR_CANT_RESOLVE_FILENAME`); Python's mapping to
    `errno.ELOOP` is unreliable on this path, so we match the Win32 code
    explicitly via `_is_eloop_oserror`.
    """
    if not path.is_symlink():
        return
    try:
        path.stat()
    except OSError as exc:
        if _is_eloop_oserror(exc):
            raise
