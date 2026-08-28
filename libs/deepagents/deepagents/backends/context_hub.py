"""`ContextHubBackend`: Store files in a LangSmith Hub agent repo (persistent)."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from langsmith import Client
from langsmith.schemas import AgentEntry, FileEntry, SkillEntry
from langsmith.utils import (
    LangSmithConflictError,
    LangSmithError,
    LangSmithNotFoundError,
    parse_hub_identifier,
)

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    BackendProtocol,
    DeleteResult,
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
from deepagents.backends.utils import (
    InvalidGlobPatternError,
    compile_grep_include_glob,
    create_file_data,
    perform_string_replacement,
    slice_read_response,
)

if TYPE_CHECKING:
    from langsmith.schemas import AgentContext

logger = logging.getLogger(__name__)

_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{8}")
_BATCH_WINDOW_SECONDS = 0.05
_MAX_CONFLICT_RETRIES = 3
_TreeSnapshot = tuple[dict[str, str], dict[str, str], str | None]


@dataclass(frozen=True)
class _WriteIntent:
    changes: dict[str, str | None]


@dataclass(frozen=True)
class _EditIntent:
    path: str
    old_string: str
    new_string: str
    replace_all: bool


@dataclass(frozen=True)
class _DeleteIntent:
    path: str


@dataclass
class _Mutation:
    """One accepted mutation and the caller waiting for its durability."""

    intent: _WriteIntent | _EditIntent | _DeleteIntent
    changes: dict[str, str | None]
    occurrences: int | None = None
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass
class _MutationQueue:
    """Mutation overlays guarded by one condition lock."""

    condition: threading.Condition = field(default_factory=threading.Condition)
    pending: list[_Mutation] = field(default_factory=list)
    in_flight: list[_Mutation] = field(default_factory=list)
    deadline: float | None = None


class ContextHubBackend(BackendProtocol):
    """Backend that stores files in a LangSmith Hub agent repo (persistent)."""

    def __init__(
        self,
        identifier: str,
        *,
        client: Client | None = None,
    ) -> None:
        """Initialize ContextHubBackend.

        Args:
            identifier: Hub agent repo, as `"owner/name"` or `"-/name"`.
            client: LangSmith client. Defaults to `Client()`.
        """
        self._identifier = identifier
        self._client = client if client is not None else Client()
        self._cache: dict[str, str] | None = None
        self._linked_entries: dict[str, str] = {}
        self._commit_hash: str | None = None
        self._mutations = _MutationQueue()
        self._worker: threading.Thread | None = None

    def _fetch_tree(self) -> _TreeSnapshot:
        """Fetch a remote snapshot; missing repos are treated as empty."""
        try:
            context: AgentContext = self._client.pull_agent(self._identifier)
        except LangSmithNotFoundError:
            return {}, {}, None

        cache: dict[str, str] = {}
        linked_entries: dict[str, str] = {}

        for path, entry in context.files.items():
            if isinstance(entry, FileEntry):
                cache[path] = entry.content
            else:
                linked_entries[path] = entry.repo_handle
        return cache, linked_entries, context.commit_hash

    def _load_tree_locked(self) -> None:
        cache, linked_entries, commit_hash = self._fetch_tree()
        self._cache = cache
        self._linked_entries = linked_entries
        self._commit_hash = commit_hash

    def _ensure_cache_locked(self) -> dict[str, str]:
        if self._cache is None:
            # The state lock makes the cold remote pull single-flight.
            self._load_tree_locked()
        if self._cache is None:
            msg = "Context Hub cache failed to initialize"
            raise RuntimeError(msg)
        return self._cache

    @staticmethod
    def _overlay(cache: dict[str, str], changes: dict[str, str | None]) -> None:
        for path, content in changes.items():
            if content is None:
                cache.pop(path, None)
            else:
                cache[path] = content

    def _visible_cache_locked(self) -> dict[str, str]:
        visible = dict(self._ensure_cache_locked())
        for mutations in (self._mutations.in_flight, self._mutations.pending):
            for mutation in mutations:
                self._overlay(visible, mutation.changes)
        return visible

    def _ensure_cache(self) -> dict[str, str]:
        """Return a snapshot including accepted local mutation overlays."""
        with self._mutations.condition:
            return self._visible_cache_locked()

    def get_linked_entries(self) -> dict[str, str]:
        """Return linked-entry paths mapped to their repo handles."""
        with self._mutations.condition:
            self._ensure_cache_locked()
            return dict(self._linked_entries)

    def has_prior_commits(self) -> bool:
        """Return `True` if the hub repo already exists with at least one commit."""
        with self._mutations.condition:
            self._ensure_cache_locked()
            return self._commit_hash is not None

    def _queue_changes_locked(
        self,
        changes: dict[str, str | None],
        *,
        intent: _WriteIntent | _EditIntent | _DeleteIntent | None = None,
        occurrences: int | None = None,
    ) -> _Mutation:
        accepted_changes = dict(changes)
        mutation = _Mutation(
            intent=intent or _WriteIntent(changes=dict(accepted_changes)),
            changes=accepted_changes,
            occurrences=occurrences,
        )
        queue = self._mutations
        if not queue.pending:
            queue.deadline = time.monotonic() + _BATCH_WINDOW_SECONDS
        queue.pending.append(mutation)

        if self._worker is None:
            try:
                worker = threading.Thread(
                    target=self._drain_mutations,
                    name="context-hub-mutations",
                    daemon=True,
                )
                self._worker = worker
                worker.start()
            except BaseException:  # Construction failure must not orphan the mutation.
                self._worker = None
                queue.pending.remove(mutation)
                if not queue.pending:
                    queue.deadline = None
                raise
        queue.condition.notify_all()
        return mutation

    @staticmethod
    def _wait_for_mutation(mutation: _Mutation) -> None:
        mutation.done.wait()
        if mutation.error is not None:
            raise mutation.error

    def _submit_changes(self, changes: dict[str, str | None]) -> None:
        with self._mutations.condition:
            self._ensure_cache_locked()
            mutation = self._queue_changes_locked(changes)
        self._wait_for_mutation(mutation)

    def _take_batch(self) -> list[_Mutation] | None:
        queue = self._mutations
        with queue.condition:
            while queue.pending:
                if queue.deadline is None:
                    msg = "Context Hub mutation deadline is missing"
                    raise RuntimeError(msg)
                remaining = queue.deadline - time.monotonic()
                if remaining > 0:
                    queue.condition.wait(timeout=remaining)
                    continue
                batch = queue.pending
                queue.pending = []
                queue.deadline = None
                queue.in_flight = batch
                return batch
            self._worker = None
            return None

    @staticmethod
    def _merge_batch(batch: list[_Mutation]) -> dict[str, str | None]:
        changes: dict[str, str | None] = {}
        for mutation in batch:
            changes.update(mutation.changes)
        return changes

    @staticmethod
    def _rematerialize_mutation(
        mutation: _Mutation,
        cache: dict[str, str],
        conflict: LangSmithConflictError,
    ) -> tuple[dict[str, str | None], int | None]:
        intent = mutation.intent
        if isinstance(intent, _WriteIntent):
            return dict(intent.changes), None
        if isinstance(intent, _EditIntent):
            current = cache.get(intent.path)
            if current is None:
                raise conflict
            result = perform_string_replacement(
                current,
                intent.old_string,
                intent.new_string,
                intent.replace_all,
            )
            if isinstance(result, str):
                raise conflict
            content, occurrences = result
            return {intent.path: content}, occurrences

        base = intent.path
        prefix = base + "/"
        paths = [path for path in cache if path == base or path.startswith(prefix)]
        return dict.fromkeys(paths, None), None

    def _rematerialize_locked(
        self,
        mutations: list[_Mutation],
        cache: dict[str, str],
        conflict: LangSmithConflictError,
    ) -> None:
        materialized: list[tuple[_Mutation, dict[str, str | None], int | None]] = []
        visible = dict(cache)
        for mutation in mutations:
            changes, occurrences = self._rematerialize_mutation(mutation, visible, conflict)
            materialized.append((mutation, changes, occurrences))
            self._overlay(visible, changes)

        for mutation, changes, occurrences in materialized:
            mutation.changes = changes
            mutation.occurrences = occurrences

    def _reload_after_conflict(self, conflict: LangSmithConflictError) -> None:
        cache, linked_entries, commit_hash = self._fetch_tree()
        failed_pending: list[_Mutation] = []
        with self._mutations.condition:
            self._rematerialize_locked(self._mutations.in_flight, cache, conflict)
            pending_cache = dict(cache)
            for mutation in self._mutations.in_flight:
                self._overlay(pending_cache, mutation.changes)
            try:
                self._rematerialize_locked(self._mutations.pending, pending_cache, conflict)
            except LangSmithConflictError as exc:
                failed_pending = self._mutations.pending
                self._mutations.pending = []
                self._mutations.deadline = None
                for mutation in failed_pending:
                    mutation.error = exc
            self._cache = cache
            self._linked_entries = linked_entries
            self._commit_hash = commit_hash
        for mutation in failed_pending:
            mutation.done.set()

    def _extract_commit_hash(self, url: str) -> str | None:
        try:
            path = urlsplit(url).path
            owner, name, _ = parse_hub_identifier(self._identifier)
        except ValueError:
            return None

        prefixes = (f"/context/{name}/", f"/hub/{owner}/{name}:")
        for prefix in prefixes:
            if path.startswith(prefix):
                commit_hash = path.removeprefix(prefix)
                if _COMMIT_HASH_RE.fullmatch(commit_hash):
                    return commit_hash
        return None

    def _push_batch(self, batch: list[_Mutation]) -> tuple[dict[str, str | None], str | None, _TreeSnapshot | None]:
        for attempt in range(_MAX_CONFLICT_RETRIES + 1):
            with self._mutations.condition:
                changes = self._merge_batch(batch)
                parent_commit = self._commit_hash
            if not changes:
                return changes, parent_commit, None

            payload: dict[str, FileEntry | AgentEntry | SkillEntry | None] = {
                path: FileEntry(type="file", content=content) if content is not None else None for path, content in changes.items()
            }
            try:
                url = self._client.push_agent(
                    self._identifier,
                    files=payload,
                    parent_commit=parent_commit,
                )
            except LangSmithConflictError as conflict:
                if attempt == _MAX_CONFLICT_RETRIES:
                    raise
                self._reload_after_conflict(conflict)
                continue

            commit_hash = self._extract_commit_hash(url)
            if commit_hash is not None:
                return changes, commit_hash, None

            snapshot = self._fetch_tree()
            commit_hash = snapshot[2]
            if commit_hash is None:
                msg = "Context Hub commit succeeded but its hash could not be resolved"
                raise RuntimeError(msg)
            return changes, commit_hash, snapshot

        msg = "Context Hub conflict retry loop exhausted unexpectedly"
        raise RuntimeError(msg)

    def _complete_batch(
        self,
        batch: list[_Mutation],
        changes: dict[str, str | None],
        commit_hash: str | None,
        *,
        authoritative_snapshot: _TreeSnapshot | None,
    ) -> bool:
        failed_pending: list[_Mutation] = []
        with self._mutations.condition:
            if authoritative_snapshot is None:
                cache = dict(self._ensure_cache_locked())
                self._overlay(cache, changes)
                self._cache = cache
            else:
                cache, linked_entries, _ = authoritative_snapshot
                if self._mutations.pending:
                    msg = "Context Hub changed before a pending mutation could be applied"
                    pending_conflict = LangSmithConflictError(msg)
                    try:
                        self._rematerialize_locked(self._mutations.pending, cache, pending_conflict)
                    except LangSmithConflictError as exc:
                        failed_pending = self._mutations.pending
                        self._mutations.pending = []
                        self._mutations.deadline = None
                        for mutation in failed_pending:
                            mutation.error = exc
                self._cache = cache
                self._linked_entries = linked_entries
            self._commit_hash = commit_hash
            self._mutations.in_flight = []
            drained = not self._mutations.pending
            if drained:
                self._worker = None
        for mutation in batch:
            mutation.done.set()
        for mutation in failed_pending:
            mutation.done.set()
        return drained

    def _fail_burst(self, batch: list[_Mutation], error: BaseException) -> None:
        with self._mutations.condition:
            affected = [*batch, *self._mutations.pending]
            self._mutations.in_flight = []
            self._mutations.pending = []
            self._mutations.deadline = None
            self._cache = None
            self._linked_entries = {}
            self._commit_hash = None
            self._worker = None
            for mutation in affected:
                mutation.error = error
        for mutation in affected:
            mutation.done.set()

    def _drain_mutations(self) -> None:
        while True:
            batch: list[_Mutation] = []
            try:
                next_batch = self._take_batch()
                if next_batch is None:
                    return
                batch = next_batch
                changes, commit_hash, authoritative_snapshot = self._push_batch(batch)
                drained = self._complete_batch(
                    batch,
                    changes,
                    commit_hash,
                    authoritative_snapshot=authoritative_snapshot,
                )
            except BaseException as exc:  # noqa: BLE001  # Every queued waiter must be settled.
                self._fail_burst(batch, exc)
                return
            if drained:
                return

    @staticmethod
    def _strip_prefix(path: str) -> str:
        return path.lstrip("/")

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read file content for the requested line range.

        Args:
            file_path: Absolute file path.
            offset: 0-indexed starting line.
            limit: Maximum number of lines.

        Returns:
            `ReadResult` with raw (unformatted) content.
        """
        hub_path = self._strip_prefix(file_path)
        try:
            cache = self._ensure_cache()
        except LangSmithError as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            return ReadResult(error=f"Hub unavailable: {exc}")
        content = cache.get(hub_path)
        if content is None:
            return ReadResult(error=f"File '{file_path}' not found")

        file_data = create_file_data(content)
        return slice_read_response(file_data, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Commit `content` to `file_path`."""
        hub_path = self._strip_prefix(file_path)
        try:
            self._submit_changes({hub_path: content})
        except LangSmithError as exc:
            logger.exception("Hub write failed for %r", self._identifier)
            return WriteResult(error=f"Hub unavailable: {exc}")
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Replace `old_string` with `new_string` in a file."""
        hub_path = self._strip_prefix(file_path)
        try:
            with self._mutations.condition:
                current = self._visible_cache_locked().get(hub_path)
                if current is None:
                    return EditResult(error=f"Error: File '{file_path}' not found")

                result = perform_string_replacement(current, old_string, new_string, replace_all)
                if isinstance(result, str):
                    return EditResult(error=result)

                new_content, occurrences = result
                mutation = self._queue_changes_locked(
                    {hub_path: new_content},
                    intent=_EditIntent(
                        path=hub_path,
                        old_string=old_string,
                        new_string=new_string,
                        replace_all=replace_all,
                    ),
                    occurrences=occurrences,
                )
            self._wait_for_mutation(mutation)
        except LangSmithError as exc:
            logger.exception("Hub edit failed for %r", self._identifier)
            return EditResult(error=f"Hub unavailable: {exc}")
        return EditResult(path=file_path, occurrences=mutation.occurrences)

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a file or directory by committing its removal from the hub repo.

        Deleting a path removes the exact file plus every nested entry under it
        (the prefix `file_path` + "/"), so a directory is removed recursively.

        Args:
            file_path: Path of the file or directory to delete.

        Returns:
            `DeleteResult` with `file_path` on success, or an error if nothing is
                stored at or under it (or the hub is unavailable).
        """
        hub_path = self._strip_prefix(file_path)
        try:
            with self._mutations.condition:
                cache = self._visible_cache_locked()
                base = hub_path.rstrip("/")
                prefix = base + "/"
                to_delete = [key for key in cache if key == base or key.startswith(prefix)]
                if not to_delete:
                    return DeleteResult(error=f"Error: File '{file_path}' not found")
                mutation = self._queue_changes_locked(
                    dict.fromkeys(to_delete, None),
                    intent=_DeleteIntent(path=base),
                )
            self._wait_for_mutation(mutation)
        except LangSmithError as exc:
            logger.exception("Hub delete failed for %r", self._identifier)
            return DeleteResult(error=f"Hub unavailable: {exc}")
        return DeleteResult(path=file_path)

    def ls(self, path: str = "/") -> LsResult:
        """List immediate files and subdirectories under `path` (non-recursive)."""
        hub_prefix = self._strip_prefix(path).rstrip("/")
        try:
            cache = self._ensure_cache()
        except LangSmithError as exc:
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
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search contents for `pattern` (optional `path` / `glob` filters).

        The `glob` filter uses `fnmatch` semantics and does *not* follow the
        shared include-glob contract that `glob()` on this class uses: `*`
        crosses `/`, and matching is case-normalized. Do not assume the two
        agree on a given pattern.

        When `max_count` is set, at most that many matches are returned; if more
        exist the search stops and the result is flagged `truncated=True`.
        Exactly `max_count` matches with none dropped is reported complete
        (`truncated=False`).
        """
        try:
            cache = self._ensure_cache()
        except LangSmithError as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            return GrepResult(error=f"Hub unavailable: {exc}")
        matches: list[GrepMatch] = []

        prefix = self._strip_prefix(path).rstrip("/") if path else ""

        glob_re = re.compile(fnmatch.translate(os.path.normcase(glob))) if glob else None

        for file_path, content in cache.items():
            if prefix and not file_path.startswith(prefix):
                continue
            if glob_re is not None and not glob_re.match(os.path.normcase(file_path)):
                continue
            for i, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    if max_count is not None and len(matches) >= max_count:
                        # A further match beyond `max_count` proves more exist;
                        # stop and flag truncation. Checked before appending so
                        # exactly `max_count` matches is reported complete.
                        return GrepResult(matches=matches, truncated=True)
                    matches.append(GrepMatch(path=f"/{file_path}", line=i, text=line))

        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:  # noqa: ARG002
        """Return files matching `pattern` (`path` unused — flat namespace).

        Matching uses the shared backend contract (`compile_grep_include_glob`),
        so a bare pattern matches the basename at any depth, a pattern with `/`
        matches the namespace-relative path, and a leading `/` anchors. Plain
        `fnmatch` would diverge on all three: its `*` crosses `/`, so `/*.py`
        would widen to every depth instead of anchoring to the top level.
        """
        try:
            cache = self._ensure_cache()
        except LangSmithError as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            return GlobResult(error=f"Hub unavailable: {exc}")
        try:
            matcher = compile_grep_include_glob(pattern)
        except InvalidGlobPatternError as exc:
            logger.warning("Refused glob pattern %r for %r: %s", pattern, self._identifier, exc)
            return GlobResult(error=str(exc))
        results: list[FileInfo] = [FileInfo(path=f"/{file_path}", is_dir=False) for file_path in cache if matcher(file_path)]
        return GlobResult(matches=results)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload text files in one commit; non-UTF-8 inputs rejected per file."""
        # Decode each input; `None` text means we'll reject this entry as invalid.
        decoded: list[tuple[str, str | None]] = []
        valid_files: dict[str, str | None] = {}
        for path, content in files:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                decoded.append((path, None))
                continue
            decoded.append((path, text))
            valid_files[self._strip_prefix(path)] = text  # last write wins

        commit_error: str | None = None
        if valid_files:
            try:
                self._submit_changes(valid_files)
            except LangSmithError as exc:
                logger.exception("Hub batch upload failed for %r", self._identifier)
                commit_error = f"Hub unavailable: {exc}"

        results: list[FileUploadResponse] = []
        for path, text in decoded:
            if text is None:
                results.append(FileUploadResponse(path=path, error=INVALID_PATH))
            elif commit_error is not None:
                # Backend-specific error string per protocol docs
                # (FileOperationError literal union doesn't cover hub failures).
                results.append(FileUploadResponse(path=path, error=commit_error))
            else:
                results.append(FileUploadResponse(path=path))
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files as raw bytes. Missing paths return `file_not_found`."""
        try:
            cache = self._ensure_cache()
        except LangSmithError as exc:
            logger.exception("Hub pull failed for %r", self._identifier)
            # Backend-specific error string per protocol docs (FileOperationError
            # literal union doesn't cover hub failures).
            return [FileDownloadResponse(path=p, error=f"Hub unavailable: {exc}") for p in paths]
        results: list[FileDownloadResponse] = []
        for path in paths:
            hub_path = self._strip_prefix(path)
            content = cache.get(hub_path)
            if content is not None:
                results.append(FileDownloadResponse(path=path, content=content.encode("utf-8")))
            else:
                results.append(FileDownloadResponse(path=path, error=FILE_NOT_FOUND))
        return results
