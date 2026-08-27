"""`.deepagentsignore` support for Deep Agents Code."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import (
    PERMISSION_DENIED,
    BackendProtocol,
    DeleteResult,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import to_posix_path, validate_path

from deepagents_code._paths import PATHS
from deepagents_code.project_utils import find_project_root

logger = logging.getLogger(__name__)

IGNORE_FILENAME = ".deepagentsignore"
DEFAULT_PATTERNS = (
    ".git/",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "dist/",
    "build/",
)
_EXCLUDED_ERROR = "Path is excluded by .deepagentsignore"
_GREP_SCAN_LIMIT = 10_000


@dataclass(frozen=True, slots=True)
class _Rule:
    matcher: re.Pattern[str]
    exact_matcher: re.Pattern[str]
    negated: bool
    directory_only: bool


@dataclass(frozen=True, slots=True)
class DeepagentsIgnore:
    """Ordered ignore rules for one project root."""

    root: Path
    rules: tuple[_Rule, ...]

    @classmethod
    def from_project(
        cls,
        cwd: str | Path | None = None,
        *,
        project_root: str | Path | None = None,
        profile_root: str | Path | None = None,
    ) -> DeepagentsIgnore:
        """Load defaults, profile rules, and project rules.

        Returns:
            Compiled rules for the project root.
        """
        start = Path(cwd or Path.cwd()).expanduser().resolve()
        root = Path(project_root).expanduser().resolve() if project_root else None
        root = root or find_project_root(start) or start
        profile = Path(profile_root or PATHS.profile.root).expanduser()
        patterns = [*DEFAULT_PATTERNS]
        patterns.extend(_read_patterns(profile / IGNORE_FILENAME))
        patterns.extend(_read_patterns(root / IGNORE_FILENAME))
        return cls(root=root, rules=tuple(_compile_rules(patterns)))

    def is_ignored_relative(self, path: str, *, is_dir: bool = False) -> bool:
        """Return whether a normalized project-relative path is excluded."""
        normalized = _normalize_relative(path)
        if not normalized:
            return False
        ignored = False
        for rule in self.rules:
            if rule.matcher.fullmatch(normalized) and (
                not rule.directory_only
                or is_dir
                or not rule.exact_matcher.fullmatch(normalized)
            ):
                ignored = not rule.negated
        return ignored

    def is_ignored_path(
        self,
        path: str | Path,
        *,
        base: str | Path | None = None,
        is_dir: bool = False,
    ) -> bool:
        """Return whether a local path or its resolved target is excluded."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(base or self.root) / candidate
        if self._is_ignored_candidate(candidate, is_dir=is_dir):
            return True
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        return resolved != candidate and self._is_ignored_candidate(
            resolved, is_dir=is_dir
        )

    def _is_ignored_candidate(self, candidate: Path, *, is_dir: bool) -> bool:
        try:
            relative = candidate.relative_to(self.root)
        except ValueError:
            return False
        return self.is_ignored_relative(relative.as_posix(), is_dir=is_dir)

    def is_ignored_backend_path(
        self,
        path: str,
        *,
        backend_root: Path,
        virtual_mode: bool,
        is_dir: bool = False,
    ) -> bool:
        """Validate a backend path and test its destination.

        Returns:
            Whether the destination is excluded.
        """
        normalized = validate_path(path)
        if virtual_mode:
            candidate = backend_root / normalized.lstrip("/")
        elif Path(path).is_absolute():
            candidate = Path(normalized)
        else:
            candidate = backend_root / normalized.lstrip("/")
        return self.is_ignored_path(candidate, is_dir=is_dir)

    def filter_relative(self, paths: list[str]) -> list[str]:
        """Return relative paths not excluded by this ruleset."""
        return [path for path in paths if not self.is_ignored_relative(path)]


class IgnoringBackend(BackendProtocol):
    """Filter project file operations while preserving backend behavior."""

    def __init__(
        self,
        backend: BackendProtocol,
        ignore: DeepagentsIgnore,
        *,
        backend_root: str | Path,
        virtual_mode: bool,
    ) -> None:
        """Wrap a project backend with ignore filtering."""
        self._backend = backend
        self.ignore = ignore
        self._backend_root = Path(backend_root).resolve()
        self._virtual_mode = virtual_mode

    def _ignored(self, path: str, *, is_dir: bool = False) -> bool:
        return self.ignore.is_ignored_backend_path(
            path,
            backend_root=self._backend_root,
            virtual_mode=self._virtual_mode,
            is_dir=is_dir,
        )

    @staticmethod
    def _error(path: str) -> str:
        return f"{_EXCLUDED_ERROR}: {path}"

    def _ignored_info(self, info: FileInfo) -> bool:
        path = info["path"]
        return self._ignored(
            path,
            is_dir=bool(info.get("is_dir")) or path.endswith(("/", "\\")),
        )

    def _ignored_match(self, match: GrepMatch) -> bool:
        return self._ignored(match["path"])

    def ls(self, path: str) -> LsResult:
        """List non-ignored directory entries.

        Returns:
            Filtered listing result.
        """
        try:
            if self._ignored(path, is_dir=True):
                return LsResult(error=self._error(path))
        except ValueError as exc:
            return LsResult(error=str(exc))
        result = self._backend.ls(path)
        if result.entries is None:
            return result
        try:
            entries = [
                entry for entry in result.entries if not self._ignored_info(entry)
            ]
        except ValueError as exc:
            return LsResult(error=str(exc))
        return LsResult(error=result.error, entries=entries)

    async def als(self, path: str) -> LsResult:
        """List non-ignored directory entries asynchronously.

        Returns:
            Filtered listing result.
        """
        try:
            if self._ignored(path, is_dir=True):
                return LsResult(error=self._error(path))
        except ValueError as exc:
            return LsResult(error=str(exc))
        result = await self._backend.als(path)
        if result.entries is None:
            return result
        try:
            entries = [
                entry for entry in result.entries if not self._ignored_info(entry)
            ]
        except ValueError as exc:
            return LsResult(error=str(exc))
        return LsResult(error=result.error, entries=entries)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a non-ignored file.

        Returns:
            Read result.
        """
        try:
            if self._ignored(file_path):
                return ReadResult(error=self._error(file_path))
        except ValueError as exc:
            return ReadResult(error=str(exc))
        return self._backend.read(file_path, offset, limit)

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Read a non-ignored file asynchronously.

        Returns:
            Read result.
        """
        try:
            if self._ignored(file_path):
                return ReadResult(error=self._error(file_path))
        except ValueError as exc:
            return ReadResult(error=str(exc))
        return await self._backend.aread(file_path, offset, limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search non-ignored files.

        Returns:
            Filtered search result.
        """
        try:
            if path is not None and self._ignored(path, is_dir=True):
                return GrepResult(matches=[])
        except ValueError as exc:
            return GrepResult(error=str(exc))
        if max_count is None:
            return self._filter_grep(self._backend.grep(pattern, path, glob))
        scan_count = max(max_count, 1)
        while True:
            result = self._backend.grep(
                pattern,
                path,
                glob,
                max_count=min(scan_count, max(max_count, _GREP_SCAN_LIMIT)),
            )
            filtered = self._filter_grep(result, max_count=max_count)
            if self._grep_scan_complete(result, filtered, scan_count, max_count):
                return filtered
            scan_count *= 2

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search non-ignored files asynchronously.

        Returns:
            Filtered search result.
        """
        try:
            if path is not None and self._ignored(path, is_dir=True):
                return GrepResult(matches=[])
        except ValueError as exc:
            return GrepResult(error=str(exc))
        if max_count is None:
            return self._filter_grep(await self._backend.agrep(pattern, path, glob))
        scan_count = max(max_count, 1)
        while True:
            result = await self._backend.agrep(
                pattern,
                path,
                glob,
                max_count=min(scan_count, max(max_count, _GREP_SCAN_LIMIT)),
            )
            filtered = self._filter_grep(result, max_count=max_count)
            if self._grep_scan_complete(result, filtered, scan_count, max_count):
                return filtered
            scan_count *= 2

    @staticmethod
    def _grep_scan_complete(
        result: GrepResult,
        filtered: GrepResult,
        scan_count: int,
        max_count: int,
    ) -> bool:
        scan_limit = max(max_count, _GREP_SCAN_LIMIT)
        return (
            result.matches is None
            or not result.truncated
            or len(filtered.matches or []) >= max_count
            or scan_count >= scan_limit
        )

    def _filter_grep(
        self, result: GrepResult, *, max_count: int | None = None
    ) -> GrepResult:
        if result.matches is None:
            return result
        try:
            matches = [
                match for match in result.matches if not self._ignored_match(match)
            ]
        except ValueError as exc:
            return GrepResult(error=str(exc))
        truncated = result.truncated
        if max_count is not None and len(matches) > max_count:
            matches = matches[:max_count]
            truncated = True
        return GrepResult(error=result.error, matches=matches, truncated=truncated)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find non-ignored files.

        Returns:
            Filtered glob result.
        """
        try:
            if path is not None and self._ignored(path, is_dir=True):
                return GlobResult(matches=[])
        except ValueError as exc:
            return GlobResult(error=str(exc))
        return self._filter_glob(self._backend.glob(pattern, path))

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find non-ignored files asynchronously.

        Returns:
            Filtered glob result.
        """
        try:
            if path is not None and self._ignored(path, is_dir=True):
                return GlobResult(matches=[])
        except ValueError as exc:
            return GlobResult(error=str(exc))
        return self._filter_glob(await self._backend.aglob(pattern, path))

    def _filter_glob(self, result: GlobResult) -> GlobResult:
        if result.matches is None:
            return result
        try:
            matches = [
                match for match in result.matches if not self._ignored_info(match)
            ]
        except ValueError as exc:
            return GlobResult(error=str(exc))
        return GlobResult(
            error=result.error,
            matches=matches,
            truncated=result.truncated,
            truncation_reason=result.truncation_reason,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write a non-ignored file.

        Returns:
            Write result.
        """
        try:
            if self._ignored(file_path):
                return WriteResult(error=self._error(file_path))
        except ValueError as exc:
            return WriteResult(error=str(exc))
        return self._backend.write(file_path, content)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Write a non-ignored file asynchronously.

        Returns:
            Write result.
        """
        try:
            if self._ignored(file_path):
                return WriteResult(error=self._error(file_path))
        except ValueError as exc:
            return WriteResult(error=str(exc))
        return await self._backend.awrite(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a non-ignored file.

        Returns:
            Edit result.
        """
        try:
            if self._ignored(file_path):
                return EditResult(error=self._error(file_path))
        except ValueError as exc:
            return EditResult(error=str(exc))
        return self._backend.edit(file_path, old_string, new_string, replace_all)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a non-ignored file asynchronously.

        Returns:
            Edit result.
        """
        try:
            if self._ignored(file_path):
                return EditResult(error=self._error(file_path))
        except ValueError as exc:
            return EditResult(error=str(exc))
        return await self._backend.aedit(file_path, old_string, new_string, replace_all)

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a non-ignored path.

        Returns:
            Delete result.
        """
        try:
            if self._ignored(file_path, is_dir=True):
                return DeleteResult(error=self._error(file_path))
        except ValueError as exc:
            return DeleteResult(error=str(exc))
        return self._backend.delete(file_path)

    async def adelete(self, file_path: str) -> DeleteResult:
        """Delete a non-ignored path asynchronously.

        Returns:
            Delete result.
        """
        try:
            if self._ignored(file_path, is_dir=True):
                return DeleteResult(error=self._error(file_path))
        except ValueError as exc:
            return DeleteResult(error=str(exc))
        return await self._backend.adelete(file_path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload non-ignored files.

        Returns:
            One response per requested file.
        """
        blocked, allowed = self._partition_uploads(files)
        allowed_results = iter(self._backend.upload_files(allowed))
        return [
            FileUploadResponse(path=path, error=PERMISSION_DENIED)
            if index in blocked
            else next(allowed_results)
            for index, (path, _content) in enumerate(files)
        ]

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload non-ignored files asynchronously.

        Returns:
            One response per requested file.
        """
        blocked, allowed = self._partition_uploads(files)
        allowed_results = iter(await self._backend.aupload_files(allowed))
        return [
            FileUploadResponse(path=path, error=PERMISSION_DENIED)
            if index in blocked
            else next(allowed_results)
            for index, (path, _content) in enumerate(files)
        ]

    def _partition_uploads(
        self, files: list[tuple[str, bytes]]
    ) -> tuple[set[int], list[tuple[str, bytes]]]:
        blocked: set[int] = set()
        allowed: list[tuple[str, bytes]] = []
        for index, item in enumerate(files):
            try:
                ignored = self._ignored(item[0])
            except ValueError:
                ignored = True
            if ignored:
                blocked.add(index)
            else:
                allowed.append(item)
        return blocked, allowed

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download non-ignored files.

        Returns:
            One response per requested path.
        """
        blocked, allowed = self._partition_downloads(paths)
        allowed_results = iter(self._backend.download_files(allowed))
        return [
            FileDownloadResponse(path=path, error=PERMISSION_DENIED)
            if index in blocked
            else next(allowed_results)
            for index, path in enumerate(paths)
        ]

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download non-ignored files asynchronously.

        Returns:
            One response per requested path.
        """
        blocked, allowed = self._partition_downloads(paths)
        allowed_results = iter(await self._backend.adownload_files(allowed))
        return [
            FileDownloadResponse(path=path, error=PERMISSION_DENIED)
            if index in blocked
            else next(allowed_results)
            for index, path in enumerate(paths)
        ]

    def _partition_downloads(self, paths: list[str]) -> tuple[set[int], list[str]]:
        blocked: set[int] = set()
        allowed: list[str] = []
        for index, path in enumerate(paths):
            try:
                ignored = self._ignored(path)
            except ValueError:
                ignored = True
            if ignored:
                blocked.add(index)
            else:
                allowed.append(path)
        return blocked, allowed


class IgnoringSandboxBackend(IgnoringBackend, LocalShellBackend):
    """Ignore-filtering wrapper that leaves shell execution unrestricted."""

    def __init__(
        self,
        backend: SandboxBackendProtocol,
        ignore: DeepagentsIgnore,
        *,
        backend_root: str | Path,
        virtual_mode: bool,
    ) -> None:
        """Wrap an executable project backend with ignore filtering."""
        super().__init__(
            backend,
            ignore,
            backend_root=backend_root,
            virtual_mode=virtual_mode,
        )
        self._sandbox = backend

    @property
    def id(self) -> str:
        """The wrapped sandbox identifier."""
        return self._sandbox.id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command without applying path filtering.

        Returns:
            Execution result.
        """
        if timeout is None:
            return self._sandbox.execute(command)
        return self._sandbox.execute(command, timeout=timeout)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """Run a shell command asynchronously without path filtering.

        Returns:
            Execution result.
        """
        if timeout is None:
            return await self._sandbox.aexecute(command)
        return await self._sandbox.aexecute(command, timeout=timeout)


def _read_patterns(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


def _compile_rules(patterns: list[str]) -> list[_Rule]:
    rules: list[_Rule] = []
    for line in patterns:
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        negated = pattern.startswith("!")
        if negated or pattern.startswith((r"\!", r"\#")):
            pattern = pattern[1:]
        pattern = to_posix_path(pattern)
        pattern = re.sub(r"^\./+", "", pattern)
        anchored = pattern.startswith("/") or "/" in pattern.rstrip("/")
        directory_only = pattern.endswith("/")
        pattern = pattern.strip("/")
        if not pattern:
            continue
        source = _glob_source(pattern)
        prefix = "^" if anchored else r"^(?:.*/)?"
        try:
            matcher = re.compile(f"{prefix}{source}(?:/.*)?$")
            exact_matcher = re.compile(f"{prefix}{source}$")
        except re.error:
            logger.warning("Skipping invalid .deepagentsignore pattern %r", line)
            continue
        rules.append(
            _Rule(
                matcher=matcher,
                exact_matcher=exact_matcher,
                negated=negated,
                directory_only=directory_only,
            )
        )
    return rules


def _glob_source(pattern: str) -> str:
    source = ""
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                if index + 2 < len(pattern) and pattern[index + 2] == "/":
                    source += r"(?:.*/)?"
                    index += 3
                else:
                    source += ".*"
                    index += 2
            else:
                source += "[^/]*"
                index += 1
        elif character == "?":
            source += "[^/]"
            index += 1
        elif character == "[":
            end = pattern.find("]", index + 1)
            if end == -1:
                source += r"\["
                index += 1
            else:
                content = pattern[index + 1 : end]
                if content.startswith("!"):
                    content = "^" + content[1:]
                source += "[" + content.replace("\\", r"\\") + "]"
                index = end + 1
        else:
            source += re.escape(character)
            index += 1
    return source


def _normalize_relative(path: str) -> str:
    normalized = validate_path(path).strip("/")
    return "" if normalized == "." else normalized
