"""StateBackend: Store files in LangGraph agent state (ephemeral)."""

import base64
import warnings
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph._internal._constants import CONFIG_KEY_READ, CONFIG_KEY_SEND
from langgraph.config import get_config

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
    _glob_search_files,
    create_file_data,
    file_data_to_string,
    format_read_response,
    grep_matches_from_files,
    perform_string_replacement,
    update_file_data,
)


class StateBackend(BackendProtocol):
    """Backend that stores files in agent state (ephemeral).

    Uses LangGraph's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.

    Reads and writes go through LangGraph's `CONFIG_KEY_READ` /
    `CONFIG_KEY_SEND` so that state updates are queued as proper channel
    writes rather than returned as `files_update` dicts.
    """

    def __init__(
        self,
        runtime: object = None,
        *,
        file_format: FileFormat = "v2",
    ) -> None:
        r"""Initialize StateBackend.

        Args:
            runtime: Deprecated - accepted for backward compatibility but
                ignored.  State is now read/written via `get_config()`.
            file_format: Storage format version. `"v1"` stores
                content as `list[str]` (lines split on `\\n`) without an
                `encoding` field.  `"v2"` (default) stores content as a
                plain `str` with an `encoding` field.
        """
        if runtime is not None:
            warnings.warn(
                "Passing `runtime` to StateBackend is deprecated and will be "
                "removed in v0.7. StateBackend now reads and writes "
                "state via `get_config()`. Simply use `StateBackend()` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._file_format = file_format

    # ------------------------------------------------------------------
    # Internal helpers for reading / writing state via config keys
    # ------------------------------------------------------------------

    def _get_config(self) -> RunnableConfig:
        """Return the current LangGraph config, with a clear error if missing."""
        try:
            config = get_config()
        except RuntimeError:
            msg = (
                "StateBackend must be used inside a LangGraph graph execution "
                "(e.g. via create_deep_agent). It cannot read or write state "
                "outside of a graph context. To pre-populate files, pass them "
                'on invoke: agent.invoke({"messages": [...], "files": {...}})'
            )
            raise RuntimeError(msg) from None
        configurable = config.get("configurable", {})
        if CONFIG_KEY_READ not in configurable:
            msg = (
                "StateBackend requires CONFIG_KEY_READ / CONFIG_KEY_SEND in "
                "the LangGraph config. Make sure the backend is used inside "
                "a graph node or tool, not called directly. To pre-populate "
                "files, pass them on invoke: "
                'agent.invoke({"messages": [...], "files": {...}})'
            )
            raise RuntimeError(msg)
        return config

    def _read_files(self) -> dict[str, Any]:
        """Read the current `files` channel via Pregel internals.

        Uses `CONFIG_KEY_READ` to read state directly — this lets us
        initialize StateBackend once and fetch state on demand from any
        graph context (tools, middleware nodes, etc.).

        `fresh=False` reads the value as of the *start* of the current
        superstep (checkpointed value + writes from prior steps). Writes
        queued during the current step aren't applied until the node boundary,
        so every call within the same step sees a consistent snapshot.
        """
        config = self._get_config()
        read = config["configurable"][CONFIG_KEY_READ]
        fresh = False
        return read("files", fresh) or {}

    def _send_files_update(self, update: dict[str, Any]) -> None:
        """Queue a write to the `files` channel via Pregel internals.

        The whole point of this helper is that callers of `backend.write`
        / `backend.edit` don't need to know about or manage state updates
        themselves — the backend handles it internally.

        Uses `CONFIG_KEY_SEND` to enqueue a partial `files` update
        directly — same rationale as `_read_files` for initializing
        StateBackend once and writing from any graph context. `send`
        takes a list of `(channel, value)` tuples; the `files` channel
        uses a dict-merge reducer, so we only need to include changed
        files — unchanged ones are preserved by the reducer.

        Writes are not applied until the node boundary, so they won't be
        visible to other calls in the same step (see `_read_files` and
        its use of `fresh=False`).
        """
        config = self._get_config()
        send = config["configurable"][CONFIG_KEY_SEND]
        send([("files", update)])

    def _prepare_for_storage(self, file_data: FileData) -> dict[str, Any]:
        """Convert FileData to the format used for state storage.

        When `file_format="v1"`, returns the legacy format.
        """
        if self._file_format == "v1":
            return _to_legacy_file_data(file_data)
        return {**file_data}

    def ls(self, path: str) -> LsResult:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute path to directory.

        Returns:
            List of FileInfo-like dicts for files and directories directly in the directory.
            Directories have a trailing / in their path and is_dir=True.
        """
        files = self._read_files()
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # Normalize path to have trailing slash for proper prefix matching
        normalized_path = path if path.endswith("/") else path + "/"

        for k, fd in files.items():
            # Check if file is in the specified directory or a subdirectory
            if not k.startswith(normalized_path):
                continue

            # Get the relative path after the directory
            relative = k[len(normalized_path) :]

            # If relative path contains '/', it's in a subdirectory
            if "/" in relative:
                # Extract the immediate subdirectory name
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # This is a file directly in the current directory
            size = len("\n".join(fd.get("content", [])))
            infos.append(
                {
                    "path": k,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", ""),
                }
            )

        # Add directories to the results
        infos.extend(FileInfo(path=subdir, is_dir=True, size=0, modified_at="") for subdir in sorted(subdirs))

        infos.sort(key=lambda x: x.get("path", ""))
        return infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or error message.
        """
        files = self._read_files()
        file_data = files.get(file_path)

        if file_data is None:
            return f"Error: File '{file_path}' not found"

        if _get_file_type(file_path) != "text":
            return ReadResult(file_data=file_data)

        sliced = slice_read_response(file_data, offset, limit)
        if isinstance(sliced, ReadResult):
            return sliced
        sliced_fd = FileData(
            content=sliced,
            encoding=file_data.get("encoding", "utf-8"),
        )
        if "created_at" in file_data:
            sliced_fd["created_at"] = file_data["created_at"]
        if "modified_at" in file_data:
            sliced_fd["modified_at"] = file_data["modified_at"]
        return ReadResult(file_data=sliced_fd)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content.

        The update is queued directly via `CONFIG_KEY_SEND`.
        """
        files = self._read_files()

        if file_path in files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        new_file_data = create_file_data(content)
        self._send_files_update({file_path: self._prepare_for_storage(new_file_data)})
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        The update is queued directly via `CONFIG_KEY_SEND`.
        """
        files = self._read_files()
        file_data = files.get(file_path)

        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        self._send_files_update({file_path: self._prepare_for_storage(new_file_data)})
        return EditResult(path=file_path, occurrences=int(occurrences))

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search state files for a literal text pattern."""
        files = self._read_files()
        return grep_matches_from_files(files, pattern, path if path is not None else "/", glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Get FileInfo for files matching glob pattern."""
        files = self._read_files()
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            size = len("\n".join(fd.get("content", []))) if fd else 0
            infos.append(
                {
                    "path": p,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", "") if fd else "",
                }
            )
        return infos

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to state.

        Args:
            files: List of (path, content) tuples to upload

        Returns:
            List of FileUploadResponse objects, one per input file
        """
        existing = self._read_files()
        responses: list[FileUploadResponse] = []
        update: dict[str, Any] = {}
        for path, content in files:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = base64.b64encode(content).decode("ascii")

            prev = existing.get(path)
            file_data = update_file_data(prev, text) if prev else create_file_data(text)
            update[path] = {**file_data}
            responses.append(FileUploadResponse(path=path, error=None))

        if update:
            self._send_files_update(update)
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from state.

        Args:
            paths: List of file paths to download

        Returns:
            List of FileDownloadResponse objects, one per input path
        """
        state_files = self._read_files()
        responses: list[FileDownloadResponse] = []

        for path in paths:
            file_data = state_files.get(path)

            if file_data is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            # Convert file data to bytes
            content_str = file_data_to_string(file_data)
            content_bytes = content_str.encode("utf-8")

            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
