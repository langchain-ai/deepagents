"""PromptHub backend.

`BackendProtocol` implementation backed by a single LangSmith Prompt Hub
repo, used as the agent-scoped store for skills, ``AGENTS.md``, ``mcp.json``,
and other agent files that live outside any particular thread and are
shared across every run of an agent.

The backend uses the directories API on a hub repo:

- ``GET  /v1/platform/hub/repos/-/{handle}/directories`` returns
  ``{commit_hash, commit_id, files: {path: {type, content}}}``.
- ``POST /v1/platform/hub/repos/-/{handle}/directories/commits`` writes a
  multi-file commit. Body shape:
  ``{files: {path: {type: "file", content: "..."}}, parent_commit?: hash}``.

The directories API requires the repo to be created with ``repo_type =
"agent"``. `HubBackend._ensure_repo_exists` handles this automatically on
the first write to a fresh repo handle.

A single instance is intended to live for the lifetime of a graph
invocation. Reads serve from a lazy in-memory cache populated on first
access. Writes commit synchronously and refresh the cache + parent commit
hash on success. Use `batch()` to group multiple writes into one commit.
"""

from __future__ import annotations

import contextlib
import fnmatch
import logging
import os
import re
import threading
from typing import TYPE_CHECKING, Any

import httpx

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
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
from deepagents.backends.utils import perform_string_replacement

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import FileData

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30.0
_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


class HubBackend(BackendProtocol):
    """`BackendProtocol` implementation backed by a LangSmith hub repo.

    Construct with `HubBackend.from_env(handle)` for the standard
    ``LANGSMITH_API_KEY`` setup, or pass ``base_url`` / ``api_key``
    explicitly to override. ``base_url`` resolves from
    ``LANGSMITH_HUB_ENDPOINT``, then ``LANGSMITH_ENDPOINT``, and finally
    defaults to the public LangSmith API.

    A single instance is intended to live for the lifetime of a graph
    invocation. Reads serve from a lazy in-memory cache. Writes commit
    synchronously, refreshing cache + ``parent_commit`` on success. Use
    `batch()` to group writes into one commit.
    """

    def __init__(
        self,
        repo_handle: str,
        *,
        base_url: str,
        api_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._handle = repo_handle
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client or httpx.Client(
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            timeout=_HTTP_TIMEOUT,
        )

        self._lock = threading.Lock()
        self._cache: dict[str, str] | None = None
        self._commit_hash: str | None = None
        # When non-None, writes are buffered into this dict instead of
        # being committed individually. Set by ``batch()``.
        self._buffer: dict[str, str] | None = None

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        repo_handle: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> HubBackend:
        """Build a `HubBackend` from environment variables.

        Resolution order for ``base_url``:

        1. Explicit ``base_url`` arg
        2. ``LANGSMITH_HUB_ENDPOINT`` env var
        3. ``LANGSMITH_ENDPOINT`` env var
        4. ``https://api.smith.langchain.com``

        ``LANGSMITH_HUB_ENDPOINT`` is read separately from
        ``LANGSMITH_ENDPOINT`` so deployments can override the hub URL
        independently of other LangSmith integrations.

        Args:
            repo_handle: Bare hub repo handle (no owner — the API
                resolves the owner from the API key via the ``-``
                placeholder).
            base_url: Override for env var resolution.
            api_key: Override for ``LANGSMITH_API_KEY``.
        """
        resolved_url = (
            base_url
            or os.environ.get("LANGSMITH_HUB_ENDPOINT")
            or os.environ.get("LANGSMITH_ENDPOINT")
            or _DEFAULT_ENDPOINT
        )
        resolved_key = api_key or os.environ.get("LANGSMITH_API_KEY")
        if not resolved_key:
            msg = "LANGSMITH_API_KEY is required for HubBackend"
            raise RuntimeError(msg)
        return cls(repo_handle, base_url=resolved_url, api_key=resolved_key)

    # -- HTTP plumbing --------------------------------------------------------

    @property
    def _directory_url(self) -> str:
        return f"{self._base_url}/v1/platform/hub/repos/-/{self._handle}/directories"

    @property
    def _commits_url(self) -> str:
        return f"{self._directory_url}/commits"

    @property
    def _create_repo_url(self) -> str:
        return f"{self._base_url}/api/v1/repos"

    def _ensure_repo_exists(self) -> None:
        """Best-effort create the agent repo on first write.

        The directories endpoint requires ``repo_type = "agent"``. We
        swallow conflicts in case the repo races into existence.
        """
        payload = {
            "repo_handle": self._handle,
            "is_public": False,
            "description": "deepagents agent-scoped files",
            "repo_type": "agent",
        }
        resp = self._client.post(self._create_repo_url, json=payload)
        if resp.status_code in (200, 201):
            return
        if resp.status_code in (400, 409, 422):
            logger.debug(
                "Hub repo %s create returned %s; assuming it already exists.",
                self._handle,
                resp.status_code,
            )
            return
        resp.raise_for_status()

    def _load_tree(self) -> None:
        """Fetch the file tree. Caller must hold ``_lock``.

        404 with ``"commit not found"`` means the repo exists but has no
        commits yet — treat as an empty tree, the next write will create
        the first commit.
        """
        resp = self._client.get(self._directory_url)
        if resp.status_code == 404:
            self._cache = {}
            self._commit_hash = None
            return
        resp.raise_for_status()
        data = resp.json()
        self._commit_hash = data.get("commit_hash")
        self._cache = {}
        for path, entry in (data.get("files") or {}).items():
            if entry.get("type") == "file":
                self._cache[path] = entry.get("content", "")

    def _ensure_cache(self) -> dict[str, str]:
        with self._lock:
            if self._cache is None:
                self._load_tree()
            return self._cache  # type: ignore[return-value]

    def _commit(self, files: dict[str, str]) -> None:
        """POST a multi-file commit and update local state on success."""
        with self._lock:
            if self._cache is None:
                self._load_tree()
            parent = self._commit_hash

        payload: dict[str, Any] = {
            "files": {p: {"type": "file", "content": c} for p, c in files.items()},
        }
        if parent is not None:
            payload["parent_commit"] = parent

        resp = self._client.post(self._commits_url, json=payload)
        if resp.status_code == 404:
            # Repo doesn't exist yet — create it as an agent repo and retry.
            self._ensure_repo_exists()
            resp = self._client.post(self._commits_url, json=payload)
        resp.raise_for_status()

        new_hash = resp.json().get("commit", {}).get("commit_hash")
        with self._lock:
            self._commit_hash = new_hash
            assert self._cache is not None
            self._cache.update(files)

    def _stage_or_commit(self, path: str, content: str) -> None:
        """Apply a single-file write. Buffered if inside ``batch()``."""
        with self._lock:
            if self._cache is None:
                self._load_tree()
            if self._buffer is not None:
                self._buffer[path] = content
                assert self._cache is not None
                self._cache[path] = content
                return
        self._commit({path: content})

    # -- batching -------------------------------------------------------------

    @contextlib.contextmanager
    def batch(self) -> Iterator[None]:
        """Buffer writes into a single multi-file commit on exit.

        Nested calls re-use the outer buffer (only the outermost flushes).
        """
        with self._lock:
            if self._buffer is not None:
                yield
                return
            self._buffer = {}
        try:
            yield
        finally:
            with self._lock:
                buffer = self._buffer or {}
                self._buffer = None
            if buffer:
                self._commit(buffer)

    # -- BackendProtocol ------------------------------------------------------

    @staticmethod
    def _strip_prefix(path: str) -> str:
        return path.lstrip("/")

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        hub_path = self._strip_prefix(file_path)
        cache = self._ensure_cache()
        content = cache.get(hub_path)
        if content is None:
            return ReadResult(error=f"Error: File '{file_path}' not found")
        lines = content.splitlines()
        text = "\n".join(lines[offset : offset + limit])
        file_data: FileData = {"content": text, "encoding": "utf-8"}
        return ReadResult(file_data=file_data)

    def write(self, file_path: str, content: str) -> WriteResult:
        hub_path = self._strip_prefix(file_path)
        try:
            self._stage_or_commit(hub_path, content)
        except httpx.HTTPError as exc:
            return WriteResult(error=f"Failed to write '{file_path}': {exc}")
        return WriteResult(path=f"/{hub_path}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        hub_path = self._strip_prefix(file_path)
        cache = self._ensure_cache()
        current = cache.get(hub_path)
        if current is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        result = perform_string_replacement(current, old_string, new_string, replace_all)
        if isinstance(result, str):
            return EditResult(error=result)
        new_content, occurrences = result

        try:
            self._stage_or_commit(hub_path, new_content)
        except httpx.HTTPError as exc:
            return EditResult(error=f"Failed to edit '{file_path}': {exc}")
        return EditResult(path=f"/{hub_path}", occurrences=occurrences)

    def ls(self, path: str = "/") -> LsResult:
        hub_path = self._strip_prefix(path).rstrip("/")
        cache = self._ensure_cache()

        seen_dirs: set[str] = set()
        entries: list[FileInfo] = []
        for fp in cache:
            if hub_path and not fp.startswith(hub_path + "/"):
                continue
            relative = fp[len(hub_path) + 1 :] if hub_path else fp
            if not relative:
                continue
            head, _, tail = relative.partition("/")
            if not tail:
                entries.append({"path": f"/{fp}", "is_dir": False})
            else:
                dir_path = f"{hub_path}/{head}" if hub_path else head
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    entries.append({"path": f"/{dir_path}", "is_dir": True})
        return LsResult(entries=entries)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return GrepResult(error=f"Invalid regex pattern: {e}")

        cache = self._ensure_cache()
        prefix = self._strip_prefix(path).rstrip("/") if path else ""
        matches: list[GrepMatch] = []
        for fp, content in cache.items():
            if prefix and not fp.startswith(prefix):
                continue
            if glob and not fnmatch.fnmatch(fp, glob):
                continue
            for i, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append({"path": f"/{fp}", "line": i, "text": line})
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        cache = self._ensure_cache()
        results: list[FileInfo] = []
        for fp in cache:
            if fnmatch.fnmatch(f"/{fp}", pattern) or fnmatch.fnmatch(fp, pattern):
                results.append({"path": f"/{fp}", "is_dir": False})
        return GlobResult(matches=results)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload as a single batched commit."""
        responses: list[FileUploadResponse] = []
        with self.batch():
            for path, content in files:
                res = self.write(path, content.decode("utf-8", errors="replace"))
                responses.append(
                    FileUploadResponse(path=path, error=None if not res.error else "invalid_path")
                )
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        cache = self._ensure_cache()
        responses: list[FileDownloadResponse] = []
        for path in paths:
            content = cache.get(self._strip_prefix(path))
            if content is None:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            else:
                responses.append(
                    FileDownloadResponse(path=path, content=content.encode("utf-8"))
                )
        return responses


__all__ = ["HubBackend"]
