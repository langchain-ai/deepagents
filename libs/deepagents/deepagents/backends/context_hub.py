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

_LINKED_ENTRY_WRITE_ERROR = (
    "Cannot write to a linked entry. Linked entries are read-only from "
    "this backend."
)

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
            identifier: Hub identifier of the target agent repo, in the form
                ``"owner/name"`` or ``"-/name"`` (for the current tenant).
            client: Optional :class:`langsmith.Client`. When ``None``, a
                default ``Client()`` is constructed and inherits configuration
                from ambient env vars (``LANGSMITH_API_KEY``, etc.).
        """
        from langsmith import Client as _Client  # noqa: PLC0415

        self._identifier = identifier
        self._client = client if client is not None else _Client()
        self._cache: dict[str, str] | None = None
        self._linked_entries: dict[str, str] = {}
        self._commit_hash: str | None = None

    def _load_tree(self) -> None:
        """Fetch the file tree via the SDK.

        Treats ``LangSmithNotFoundError`` as an empty repo (the first commit
        lazily creates it). Any other failure is propagated; the caller is
        expected to surface it as a ``Result(error=...)``.
        """
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
        for path, entry in context.files.items():
            if entry.type == "file":
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

    def _is_under_linked_entry(self, hub_path: str) -> bool:
        """Check whether a write to ``hub_path`` would land inside a linked entry.

        Linked entries (``SkillEntry`` / ``AgentEntry``) reference other hub
        repos authored and versioned separately. Writing to paths under them
        would fight the linked repo's ownership model and the server would
        reject the commit. To edit a linked skill or agent, construct a
        separate ``ContextHubBackend`` pointing at that repo's identifier.
        """
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
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            ReadResult with raw (unformatted) content for the requested window.
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
        """Commit ``content`` to ``file_path`` in the hub repo.

        Args:
            file_path: Absolute file path.
            content: Text content to store.

        Returns:
            WriteResult with the committed path, or an error if the write
            targets a linked entry or the hub call fails.
        """
        hub_path = self._strip_prefix(file_path)
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
        """Replace ``old_string`` with ``new_string`` in ``file_path``.

        Args:
            file_path: Absolute file path.
            old_string: Exact string to replace.
            new_string: Replacement string.
            replace_all: If True, replace every occurrence; otherwise fail
                when more than one match is found.

        Returns:
            EditResult with the path and replacement count, or an error.
        """
        hub_path = self._strip_prefix(file_path)
        try:
            cache = self._ensure_cache()
            if self._is_under_linked_entry(hub_path):
                return EditResult(error=_LINKED_ENTRY_WRITE_ERROR)

            current = cache.get(hub_path)
            if current is None:
                return EditResult(error=f"Error: File '{file_path}' not found")

            result = perform_string_replacement(
                current, old_string, new_string, replace_all
            )
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
        """List files and directories directly under ``path`` (non-recursive).

        Args:
            path: Absolute directory path.

        Returns:
            LsResult with immediate file and subdirectory entries.
        """
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
        """Search file contents for lines matching a regex pattern.

        Args:
            pattern: Regex pattern to search for.
            path: Optional path prefix to restrict the search.
            glob: Optional filename glob to filter which files are searched.

        Returns:
            GrepResult with matching lines, or an error if the pattern is
            invalid or the hub pull fails.
        """
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
        """Return files matching a glob pattern.

        Args:
            pattern: Glob pattern to match against file paths.
            path: Base directory (unused for the flat hub namespace).

        Returns:
            GlobResult with matching FileInfo entries.
        """
        try:
            cache = self._ensure_cache()
        except Exception as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            return GlobResult(error=f"Hub unavailable: {exc}")
        results: list[FileInfo] = [
            FileInfo(path=f"/{file_path}", is_dir=False)
            for file_path in cache
            if fnmatch.fnmatch(f"/{file_path}", pattern)
            or fnmatch.fnmatch(file_path, pattern)
        ]
        return GlobResult(matches=results)

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload text files into the hub repo.

        Args:
            files: List of ``(path, bytes)`` tuples. Content must decode as
                utf-8; binary uploads are rejected with ``invalid_path``.

        Returns:
            FileUploadResponse per input, in order.
        """
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
        """Download files from the hub repo as raw bytes.

        Args:
            paths: List of absolute file paths to download.

        Returns:
            FileDownloadResponse per input, in order.
        """
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
                results.append(
                    FileDownloadResponse(path=path, content=content.encode("utf-8"))
                )
            else:
                results.append(FileDownloadResponse(path=path, error="file_not_found"))
        return results
