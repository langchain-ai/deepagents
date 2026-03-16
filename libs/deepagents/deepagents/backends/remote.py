"""`RemoteBackend`: Store files in cloud object storage (S3, Azure Blob, GCS).

Uses `fsspec <https://filesystem-spec.readthedocs.io>`_ as the underlying
abstraction layer, so any ``fsspec``-compatible filesystem is supported out of
the box.  Amazon S3 (``s3fs``), Google Cloud Storage (``gcsfs``), and Azure
Blob Storage (``adlfs``) are included as dependencies and available without
any extra install steps.

All agent-facing paths are virtual POSIX paths (starting with ``/``),
identical to ``FilesystemBackend(virtual_mode=True)``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import fsspec
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


def _extract_modified_at(info: dict[str, Any]) -> str:
    """Extract a normalised ISO timestamp from an fsspec ``info`` dict.

    Different providers expose the modification time under different keys
    (``LastModified`` for S3, ``updated`` for GCS, ``last_modified`` for
    Azure, ``mtime`` for SFTP/local, ``created`` for the memory backend).
    """
    for key in ("LastModified", "updated", "last_modified", "mtime", "modified", "created"):
        val = info.get(key)
        if val is not None:
            if isinstance(val, datetime):
                return val.isoformat()
            return str(val)
    return ""


class RemoteBackend(BackendProtocol):
    """Backend that stores files in cloud object storage (S3, Azure Blob, GCS).

    Uses `fsspec <https://filesystem-spec.readthedocs.io>`_ as the underlying
    abstraction  so any ``fsspec``-compatible filesystem (S3, GCS, Azure, SFTP, HTTP, …)
    works without changes to this class.

    Files are stored as plain UTF-8 objects.  All agent-facing paths are
    virtual POSIX paths anchored to the configured ``url`` prefix, identical
    to ``FilesystemBackend(virtual_mode=True)``:

    - Path traversal (``..``, ``~``) is blocked.
    - All resolved paths must stay within the root prefix.
    - ``WriteResult.files_update`` and ``EditResult.files_update`` are always
      ``None`` (external storage; no LangGraph state update required).

    Examples::

        # Amazon S3 (requires s3fs)
        backend = RemoteBackend("s3://my-bucket/agent-workspace/")

        # Google Cloud Storage (requires gcsfs)
        backend = RemoteBackend("gs://my-bucket/agent-workspace/")

        # Azure Blob Storage (requires adlfs)
        backend = RemoteBackend(
            "az://my-container/agent-workspace/",
            storage_options={"account_name": "...", "account_key": "..."},
        )

        # In-process memory store — useful for tests and local development
        backend = RemoteBackend("memory://workspace/files/")

    !!! warning "Security Warning"

        This backend grants agents read/write access to a cloud storage
        prefix.  Use caution and apply appropriate IAM / bucket policies.

        Recommended safeguards:

        1. Scope IAM permissions to the specific prefix used as ``url``.
        2. Enable Human-in-the-Loop (HITL) middleware for sensitive workloads.
        3. For untrusted agents, prefer ``StateBackend`` or ``SandboxBackend``.
    """

    def __init__(
        self,
        url: str,
        *,
        storage_options: dict[str, Any] | None = None,
        max_grep_file_size_mb: int = 10,
    ) -> None:
        """Initialise the remote backend.

        Args:
            url: Cloud storage URL that defines both the provider and the
                root prefix for all virtual paths.  Supported schemes:

                - ``s3://bucket/prefix`` - Amazon S3
                - ``gs://bucket/prefix`` - Google Cloud Storage
                - ``az://container/prefix`` - Azure Blob Storage
                - ``memory://bucket/prefix`` - in-process memory

            storage_options: Provider-specific options forwarded verbatim to
                ``fsspec.url_to_fs``.  Used for credentials, region, timeouts, etc.
                For example:

                - S3: ``{"key": "AKID...", "secret": "...", "region_name": "us-east-1"}``
                - GCS: ``{"token": "/path/to/service-account.json", "project": "my-gcp-project"}``
                - Azure: ``{"account_name": "myaccount", "account_key": "..."}``

            max_grep_file_size_mb: Files larger than this limit (in MB) are
                skipped during ``grep_raw`` to avoid downloading very large
                objects.  Defaults to ``10``.
        """
        self._fs, root = fsspec.url_to_fs(url, **(storage_options or {}))
        # Normalise root: no trailing slash, keep whatever prefix fsspec gives us
        self._root: str = root.rstrip("/")
        self._max_grep_bytes = max_grep_file_size_mb * 1024 * 1024

    # ------------------------------------------------------------------
    # Internal path helpers  (mirrors FilesystemBackend virtual-mode logic)
    # ------------------------------------------------------------------

    def _resolve_path(self, key: str) -> str:
        """Resolve a virtual path to a full cloud-storage path.

        Mirrors ``FilesystemBackend._resolve_path`` for ``virtual_mode=True``:
        path traversal (``..``, ``~``) is blocked and the resolved path must
        remain within ``self._root``.

        Args:
            key: Virtual path (e.g. ``"/src/main.py"`` or ``"src/main.py"``).

        Returns:
            Full path accepted by ``self._fs`` (e.g. ``"bucket/prefix/src/main.py"``).

        Raises:
            ValueError: On path traversal or escape from the root prefix.
        """
        vpath = key if key.startswith("/") else "/" + key
        if ".." in vpath.split("/") or vpath.startswith("~"):
            msg = "Path traversal not allowed"
            raise ValueError(msg)
        clean = vpath.lstrip("/")
        return f"{self._root}/{clean}" if clean else self._root

    def _to_virtual_path(self, fspath: str) -> str:
        """Convert a full cloud-storage path back to a virtual path.

        Mirrors ``FilesystemBackend._to_virtual_path``.

        Args:
            fspath: Full path as returned by ``self._fs`` operations.

        Returns:
            Virtual path starting with ``/`` (e.g. ``"/src/main.py"``).

        Raises:
            ValueError: If *fspath* is outside ``self._root``.
        """
        root_prefix = self._root + "/"
        if fspath.startswith(root_prefix):
            return "/" + fspath[len(root_prefix) :]
        if fspath == self._root:
            return "/"
        msg = f"Path {fspath!r} is outside root {self._root!r}"
        raise ValueError(msg)

    def _grep_file(self, fspath: str, info: dict[str, Any], pattern: str, glob: str | None) -> list[GrepMatch]:
        """Search a single file for *pattern* and return structured matches.

        Skips the file silently (returns ``[]``) if it is a directory, too
        large, filtered out by *glob*, or unreadable.

        Args:
            fspath: Full cloud-storage path of the file.
            info: fsspec ``info`` dict for the file.
            pattern: Literal string to search for.
            glob: Optional filename glob filter.

        Returns:
            List of :class:`GrepMatch` for matching lines, or ``[]`` if skipped.
        """
        if info.get("type") == "directory":
            return []

        size = int(info.get("size", 0))
        if size > self._max_grep_bytes:
            logger.debug("Skipping large file during grep: %s (%d bytes)", fspath, size)
            return []

        if glob:
            filename = fspath.rsplit("/", 1)[-1]
            if not wcglob.globmatch(filename, glob, flags=wcglob.BRACE):
                return []

        try:
            with self._fs.open(fspath, "rb") as f:
                content = f.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            logger.debug("Skipping unreadable file during grep: %s", fspath)
            return []

        try:
            vpath = self._to_virtual_path(fspath)
        except ValueError:
            logger.debug("Skipping grep result outside root: %s", fspath)
            return []

        return [{"path": vpath, "line": line_num, "text": line} for line_num, line in enumerate(content.splitlines(), 1) if pattern in line]

    # ------------------------------------------------------------------
    # BackendProtocol implementation
    # ------------------------------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        """List the direct children of a virtual directory.

        Args:
            path: Virtual directory path (e.g. ``"/"`` or ``"/src"``).

        Returns:
            ``FileInfo`` list for immediate files and sub-directories.
            Directories have a trailing ``/`` in their ``path`` and
            ``is_dir=True``.  Returns an empty list if the path does not
            exist or is not a directory.
        """
        try:
            resolved = self._resolve_path(path)
        except ValueError:
            return []

        try:
            entries = self._fs.ls(resolved, detail=True)
        except Exception:  # noqa: BLE001
            return []

        results: list[FileInfo] = []
        for entry in entries:
            fspath = entry["name"]
            etype = entry.get("type", "file")

            try:
                vpath = self._to_virtual_path(fspath)
            except ValueError:
                logger.debug("Skipping path outside root: %s", fspath)
                continue

            if etype == "directory":
                results.append({"path": vpath + "/", "is_dir": True, "size": 0, "modified_at": ""})
            else:
                results.append(
                    {
                        "path": vpath,
                        "is_dir": False,
                        "size": int(entry.get("size", 0)),
                        "modified_at": _extract_modified_at(entry),
                    }
                )

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read a remote file with line numbers.

        Args:
            file_path: Virtual path of the file to read.
            offset: Line number to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted content with line numbers, or an error string.
        """
        try:
            resolved = self._resolve_path(file_path)
        except ValueError as e:
            return f"Error: {e}"

        try:
            with self._fs.open(resolved, "rb") as f:
                content = f.read().decode("utf-8")
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except (UnicodeDecodeError, Exception) as e:  # noqa: BLE001
            return f"Error reading file '{file_path}': {e}"

        empty_msg = check_empty_content(content)
        if empty_msg:
            return empty_msg

        lines = content.splitlines()
        if offset >= len(lines):
            return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

        return format_content_with_line_numbers(lines[offset : min(offset + limit, len(lines))], start_line=offset + 1)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file in remote storage.

        Args:
            file_path: Virtual path for the new file. Must not already exist.
            content: Text content to write.

        Returns:
            ``WriteResult`` with ``files_update=None`` (external storage).
        """
        try:
            resolved = self._resolve_path(file_path)
        except ValueError as e:
            return WriteResult(error=str(e))

        try:
            if self._fs.exists(resolved):
                return WriteResult(
                    error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
                )
            self._fs.pipe_file(resolved, content.encode("utf-8"))
            return WriteResult(path=file_path, files_update=None)
        except Exception as e:  # noqa: BLE001
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a remote file by replacing string occurrences.

        Args:
            file_path: Virtual path of the file to edit.
            old_string: Exact string to search for and replace.
            new_string: Replacement string.
            replace_all: If ``True``, replace all occurrences.  If ``False``
                (default), ``old_string`` must appear exactly once.

        Returns:
            ``EditResult`` with ``files_update=None`` (external storage).
        """
        try:
            resolved = self._resolve_path(file_path)
        except ValueError as e:
            return EditResult(error=str(e))

        try:
            with self._fs.open(resolved, "rb") as f:
                content = f.read().decode("utf-8")
        except FileNotFoundError:
            return EditResult(error=f"Error: File '{file_path}' not found")
        except (UnicodeDecodeError, Exception) as e:  # noqa: BLE001
            return EditResult(error=f"Error reading file '{file_path}': {e}")

        result = perform_string_replacement(content, old_string, new_string, replace_all)
        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        try:
            self._fs.pipe_file(resolved, new_content.encode("utf-8"))
            return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))
        except Exception as e:  # noqa: BLE001
            return EditResult(error=f"Error writing file '{file_path}': {e}")

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search for a literal text pattern across remote files.

        Downloads each candidate file and performs an in-process literal
        search (analogous to ``FilesystemBackend._python_search``).
        Files larger than ``max_grep_file_size_mb`` are skipped.

        Args:
            pattern: Literal string to search for (NOT regex).
            path: Virtual directory to restrict the search to.
            glob: Optional glob pattern to filter files by filename.

        Returns:
            List of :class:`GrepMatch` dicts on success, or ``[]`` on error.
        """
        try:
            resolved = self._resolve_path(path or "/")
        except ValueError:
            return []

        try:
            # find() returns {fspath: info, …} with detail=True (recursive)
            found: dict[str, Any] = self._fs.find(resolved, detail=True)
        except Exception:  # noqa: BLE001
            return []

        matches: list[GrepMatch] = []
        for fspath, info in found.items():
            matches.extend(self._grep_file(fspath, info, pattern, glob))
        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob pattern in remote storage.

        Args:
            pattern: Glob pattern (e.g. ``"*.py"``, ``"**/*.ts"``).
            path: Base virtual directory to search from. Default: ``"/"`` (root).

        Returns:
            Sorted list of :class:`FileInfo` dicts for matching files.
        """
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        try:
            resolved_base = self._resolve_path(path)
        except ValueError:
            return []

        glob_pattern = f"{resolved_base.rstrip('/')}/{pattern}"

        try:
            matched_paths: list[str] = self._fs.glob(glob_pattern)
        except Exception:  # noqa: BLE001
            return []

        results: list[FileInfo] = []
        for fspath in matched_paths:
            try:
                info = self._fs.info(fspath)
            except Exception:  # noqa: BLE001
                logger.debug("Skipping unresolvable glob result: %s", fspath)
                continue
            if info.get("type") == "directory":
                continue
            try:
                vpath = self._to_virtual_path(fspath)
            except ValueError:
                logger.debug("Skipping glob result outside root: %s", fspath)
                continue
            results.append(
                {
                    "path": vpath,
                    "is_dir": False,
                    "size": int(info.get("size", 0)),
                    "modified_at": _extract_modified_at(info),
                }
            )

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to remote storage.

        Args:
            files: List of ``(virtual_path, content_bytes)`` tuples.

        Returns:
            One :class:`FileUploadResponse` per input file, in input order.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                resolved = self._resolve_path(path)
            except ValueError:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                self._fs.pipe_file(resolved, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except PermissionError:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
            except Exception:  # noqa: BLE001
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from remote storage.

        Args:
            paths: List of virtual file paths to download.

        Returns:
            One :class:`FileDownloadResponse` per input path, in input order.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved = self._resolve_path(path)
            except ValueError:
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue
            try:
                with self._fs.open(resolved, "rb") as f:
                    data: bytes = f.read()
                responses.append(FileDownloadResponse(path=path, content=data, error=None))
            except FileNotFoundError:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            except PermissionError:
                responses.append(FileDownloadResponse(path=path, content=None, error="permission_denied"))
            except IsADirectoryError:
                responses.append(FileDownloadResponse(path=path, content=None, error="is_directory"))
            except Exception:  # noqa: BLE001
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
        return responses
