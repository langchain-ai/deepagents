"""ContextHubBackend: Store files in a LangSmith Hub agent repo (persistent)."""

from __future__ import annotations

import fnmatch
import logging
import re
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    create_file_data,
    perform_string_replacement,
    slice_read_response,
)

if TYPE_CHECKING:
    from langsmith import Client
    from langsmith.schemas import AgentContext

logger = logging.getLogger(__name__)

_LINKED_ENTRY_WRITE_ERROR = "Cannot write to a linked entry. Linked entries are read-only from this backend."

# Root-level files that configure the agent itself. Not runtime-editable via
# this backend — bootstrap them out-of-band (hub UI or explicit push_agent).
_IMMUTABLE_ROOT_FILES = frozenset({"AGENTS.md"})
_IMMUTABLE_WRITE_ERROR = "'/{}' is read-only. Agent configuration files are not runtime-editable."

# Matches the ":<hash>" suffix appended by langsmith's _build_context_url.
_URL_COMMIT_SUFFIX_RE = re.compile(r":([0-9a-f]{8,64})$")


class ContextHubBackend(BackendProtocol):
    """Backend that stores files in a LangSmith Hub agent repo (persistent).

    Linked agent/skill entries in the repo are read-only; writes that target
    paths under a linked entry fail fast.
    """

    def __init__(
        self,
        identifier: str,
        *,
        client: Client | None = None,
    ) -> None:
        """Initialize ContextHubBackend.

        Args:
            identifier: Hub agent repo, as ``"owner/name"`` or ``"-/name"``.
            client: LangSmith client. Defaults to ``Client()``.
        """
        from langsmith import Client as _Client  # noqa: PLC0415

        self._identifier = identifier
        self._client = client if client is not None else _Client()
        self._cache: dict[str, str] | None = None
        self._linked_entries: dict[str, str] = {}
        self._commit_hash: str | None = None

    def _load_tree(self) -> None:
        """Fetch the file tree. A missing repo is treated as empty (first commit creates it)."""
        from langsmith.utils import LangSmithNotFoundError  # noqa: PLC0415

        try:
            context: AgentContext = self._client.pull_agent(self._identifier)
        except LangSmithNotFoundError:
            self._cache = {}
            self._linked_entries = {}
            self._commit_hash = None
            return

        self._commit_hash = context.commit_hash
        self._cache = {}
        self._linked_entries = {}
        from langsmith.schemas import FileEntry  # noqa: PLC0415

        for path, entry in context.files.items():
            if isinstance(entry, FileEntry):
                self._cache[path] = entry.content
            else:
                self._linked_entries[path] = entry.repo_handle

    def _ensure_cache(self) -> dict[str, str]:
        """Load the file tree if not yet loaded."""
        if self._cache is None:
            self._load_tree()
        return self._cache  # type: ignore[return-value]

    def get_linked_entries(self) -> dict[str, str]:
        """Return linked-entry paths mapped to their repo handles."""
        self._ensure_cache()
        return dict(self._linked_entries)

    def _commit(self, path: str, content: str) -> None:
        """Push a single-file change and update the cache on success."""
        from langsmith.schemas import FileEntry  # noqa: PLC0415

        url = self._client.push_agent(
            self._identifier,
            files={path: FileEntry(type="file", content=content)},
            parent_commit=self._commit_hash,
        )
        match = _URL_COMMIT_SUFFIX_RE.search(url)
        if match:
            self._commit_hash = match.group(1)

        if self._cache is not None:
            self._cache[path] = content

    @staticmethod
    def _strip_prefix(path: str) -> str:
        return path.lstrip("/")

    @staticmethod
    def _is_immutable_path(hub_path: str) -> bool:
        """Return True if ``hub_path`` is a protected root-level config file."""
        return hub_path in _IMMUTABLE_ROOT_FILES

    def _is_under_linked_entry(self, hub_path: str) -> bool:
        """Return True if ``hub_path`` is at or under a linked entry root."""
        self._ensure_cache()
        for linked in self._linked_entries:
            normalized = linked.rstrip("/")
            if hub_path == normalized or hub_path.startswith(normalized + "/"):
                return True
        return False

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read file content for the requested line range.

        Args:
            file_path: Absolute file path.
            offset: 0-indexed starting line.
            limit: Maximum number of lines.

        Returns:
            ReadResult with raw (unformatted) content.
        """
        hub_path = self._strip_prefix(file_path)
        try:
            cache = self._ensure_cache()
        except Exception as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            return ReadResult(error=f"Hub unavailable: {exc}")
        content = cache.get(hub_path)
        if content is None:
            return ReadResult(error=f"File '{file_path}' not found")

        file_data = create_file_data(content)
        sliced = slice_read_response(file_data, offset, limit)
        if isinstance(sliced, ReadResult):
            return sliced
        return ReadResult(
            file_data=FileData(
                content=sliced,
                encoding=file_data.get("encoding", "utf-8"),
                created_at=file_data.get("created_at", ""),
                modified_at=file_data.get("modified_at", ""),
            )
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Commit ``content`` to ``file_path``. Rejects writes under linked entries."""
        hub_path = self._strip_prefix(file_path)
        if self._is_immutable_path(hub_path):
            return WriteResult(error=_IMMUTABLE_WRITE_ERROR.format(hub_path))
        try:
            self._ensure_cache()
            if self._is_under_linked_entry(hub_path):
                return WriteResult(error=_LINKED_ENTRY_WRITE_ERROR)
            self._commit(hub_path, content)
        except Exception as exc:
            logger.exception("Hub write failed for %r", self._identifier)
            self._cache = None
            return WriteResult(error=f"Hub unavailable: {exc}")
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Replace ``old_string`` with ``new_string``. Fails on multiple matches unless ``replace_all=True``."""
        hub_path = self._strip_prefix(file_path)
        if self._is_immutable_path(hub_path):
            return EditResult(error=_IMMUTABLE_WRITE_ERROR.format(hub_path))
        try:
            cache = self._ensure_cache()
            if self._is_under_linked_entry(hub_path):
                return EditResult(error=_LINKED_ENTRY_WRITE_ERROR)

            current = cache.get(hub_path)
            if current is None:
                return EditResult(error=f"Error: File '{file_path}' not found")

            result = perform_string_replacement(current, old_string, new_string, replace_all)
            if isinstance(result, str):
                return EditResult(error=result)

            new_content, occurrences = result
            self._commit(hub_path, new_content)
        except Exception as exc:
            logger.exception("Hub edit failed for %r", self._identifier)
            self._cache = None
            return EditResult(error=f"Hub unavailable: {exc}")
        return EditResult(path=file_path, occurrences=occurrences)

    def ls(self, path: str = "/") -> LsResult:
        """List immediate files and subdirectories under ``path`` (non-recursive)."""
        hub_prefix = self._strip_prefix(path).rstrip("/")
        try:
            cache = self._ensure_cache()
        except Exception as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            return LsResult(error=f"Hub unavailable: {exc}")

        dirs: set[str] = set()
        entries: list[FileInfo] = []

        for file_path in cache:
            if hub_prefix and not file_path.startswith(hub_prefix + "/"):
                continue

            relative = file_path[len(hub_prefix) + 1 :] if hub_prefix else file_path
            if not relative:
                continue

            parts = relative.split("/", 1)
            if len(parts) == 1:
                entries.append(FileInfo(path=f"/{file_path}", is_dir=False))
            else:
                dir_name = parts[0]
                dir_path = f"{hub_prefix}/{dir_name}" if hub_prefix else dir_name
                if dir_path not in dirs:
                    dirs.add(dir_path)
                    entries.append(FileInfo(path=f"/{dir_path}", is_dir=True))

        return LsResult(entries=entries)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search file contents for lines matching ``pattern``, optionally filtered by ``path`` prefix and ``glob``."""
        try:
            cache = self._ensure_cache()
        except Exception as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            return GrepResult(error=f"Hub unavailable: {exc}")
        matches: list[GrepMatch] = []

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return GrepResult(error=f"Invalid regex pattern: {e}")

        prefix = self._strip_prefix(path).rstrip("/") if path else ""

        for file_path, content in cache.items():
            if prefix and not file_path.startswith(prefix):
                continue
            if glob and not fnmatch.fnmatch(file_path, glob):
                continue
            for i, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append(GrepMatch(path=f"/{file_path}", line=i, text=line))

        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:  # noqa: ARG002
        """Return files matching ``pattern`` (``path`` is unused — hub namespace is flat)."""
        try:
            cache = self._ensure_cache()
        except Exception as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            return GlobResult(error=f"Hub unavailable: {exc}")
        results: list[FileInfo] = [
            FileInfo(path=f"/{file_path}", is_dir=False)
            for file_path in cache
            if fnmatch.fnmatch(f"/{file_path}", pattern) or fnmatch.fnmatch(file_path, pattern)
        ]
        return GlobResult(matches=results)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload text files. Non-utf-8 bytes are rejected with ``invalid_path``."""
        results: list[FileUploadResponse] = []
        for path, content in files:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                results.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            res = self.write(path, text)
            if res.error:
                # Backend-specific error string passed through per protocol docs
                # (FileOperationError literal union doesn't cover hub failures).
                results.append(FileUploadResponse(path=path, error=res.error))  # type: ignore[arg-type]
            else:
                results.append(FileUploadResponse(path=path))
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files as raw bytes. Missing paths return ``file_not_found``."""
        try:
            cache = self._ensure_cache()
        except Exception as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            # Backend-specific error string per protocol docs (FileOperationError
            # literal union doesn't cover hub failures).
            return [
                FileDownloadResponse(path=p, error=f"Hub unavailable: {exc}")  # type: ignore[arg-type]
                for p in paths
            ]
        results: list[FileDownloadResponse] = []
        for path in paths:
            hub_path = self._strip_prefix(path)
            content = cache.get(hub_path)
            if content is not None:
                results.append(FileDownloadResponse(path=path, content=content.encode("utf-8")))
            else:
                results.append(FileDownloadResponse(path=path, error="file_not_found"))
        return results
