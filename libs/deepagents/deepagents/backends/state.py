"""StateBackend: Store files in LangGraph agent state (ephemeral)."""

import base64
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileFormat,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.utils import (
    _glob_search_files,
    _to_legacy_file_data,
    create_file_data,
    file_data_to_string,
    format_read_response,
    grep_matches_from_files,
    perform_string_replacement,
    update_file_data,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime


class StateBackend(BackendProtocol):
    """Backend that stores files in agent state (ephemeral).

    Uses LangGraph's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.

    Special handling: Since LangGraph state must be updated via Command objects
    (not direct mutation), operations return Command objects instead of None.
    This is indicated by the uses_state=True flag.
    """

    def __init__(
        self,
        runtime: "ToolRuntime",
        *,
        file_format: FileFormat = "v1",
    ) -> None:
        r"""Initialize StateBackend with runtime.

        Args:
            runtime: The ToolRuntime instance providing store access and configuration.
            file_format: Storage format version. `"v1"` (default) stores
                content as `list[str]` (lines split on `\\n`) without an
                `encoding` field.  `"v2"` stores content as a plain `str`
                with an `encoding` field.
        """
        self.runtime = runtime
        self._file_format = file_format

    def _prepare_for_storage(self, file_data: FileData) -> dict[str, Any]:
        """Convert FileData to the format used for state storage.

        When `file_format="v1"`, returns the legacy format.
        """
        if self._file_format == "v1":
            return _to_legacy_file_data(file_data)
        return {**file_data}

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute path to directory.

        Returns:
            List of FileInfo-like dicts for files and directories directly in the directory.
            Directories have a trailing / in their path and is_dir=True.
        """
        files = self.runtime.state.get("files", {})
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
            # BACKWARDS COMPAT: handle legacy list[str] content for size computation
            raw = fd.get("content", "")
            size = len("\n".join(raw)) if isinstance(raw, list) else len(raw)
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
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return f"Error: File '{file_path}' not found"

        return format_read_response(file_data, offset, limit)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content.

        Returns WriteResult with files_update to update LangGraph state.
        """
        files = self.runtime.state.get("files", {})

        if file_path in files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        new_file_data = create_file_data(content)
        return WriteResult(path=file_path, files_update={file_path: self._prepare_for_storage(new_file_data)})

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Returns EditResult with files_update and occurrences.
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        return EditResult(path=file_path, files_update={file_path: self._prepare_for_storage(new_file_data)}, occurrences=int(occurrences))

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search state files for a literal text pattern."""
        files = self.runtime.state.get("files", {})
        return grep_matches_from_files(files, pattern, path if path is not None else "/", glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Get FileInfo for files matching glob pattern."""
        files = self.runtime.state.get("files", {})
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            if fd:
                # BACKWARDS COMPAT: handle legacy list[str] content for size computation
                raw = fd.get("content", "")
                size = len("\n".join(raw)) if isinstance(raw, list) else len(raw)
            else:
                size = 0
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
        msg = (
            "StateBackend does not support upload_files yet. You can upload files "
            "directly by passing them in invoke if you're storing files in the memory."
        )
        raise NotImplementedError(msg)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from state.

        Args:
            paths: List of file paths to download

        Returns:
            List of FileDownloadResponse objects, one per input path
        """
        state_files = self.runtime.state.get("files", {})
        responses: list[FileDownloadResponse] = []

        for path in paths:
            file_data = state_files.get(path)

            if file_data is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            content_str = file_data_to_string(file_data)

            encoding = file_data.get("encoding", "utf-8")
            content_bytes = content_str.encode("utf-8") if encoding == "utf-8" else base64.standard_b64decode(content_str)
            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
