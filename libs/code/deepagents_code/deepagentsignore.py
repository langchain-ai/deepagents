"""Project ignore support for Deep Agents Code file context."""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends.protocol import (
    BackendProtocol,
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

from deepagents_code.project_utils import find_project_root

logger = logging.getLogger(__name__)

DEEPAGENTSIGNORE_FILENAME = ".deepagentsignore"
DEFAULT_IGNORE_PATTERNS = (
    ".git/",
    "node_modules/",
    "dist/",
    "build/",
    ".venv/",
    "venv/",
    "__pycache__/",
)
IGNORED_READ_ERROR = "Access denied by .deepagentsignore"


@dataclass(frozen=True)
class IgnoreRule:
    """One parsed `.deepagentsignore` rule."""

    pattern: str
    negated: bool = False
    directory_only: bool = False
    anchored: bool = False


@dataclass(frozen=True)
class DeepagentsIgnore:
    """Compiled ignore rules for one project root."""

    root: Path
    rules: tuple[IgnoreRule, ...]

    @classmethod
    def from_project(cls, cwd: str | Path | None = None) -> DeepagentsIgnore:
        """Load default, global, and project `.deepagentsignore` rules.

        Returns:
            Compiled ignore rules for the detected project root.
        """
        start = _resolve_lenient(Path(cwd or Path.cwd()).expanduser())

        try:
            root = find_project_root(start) or start
        except (OSError, RuntimeError):
            root = start

        rules: list[IgnoreRule] = []
        for pattern in DEFAULT_IGNORE_PATTERNS:
            rule = _parse_ignore_line(pattern)
            if rule is not None:
                rules.append(rule)

        global_file = Path.home() / ".deepagents" / DEEPAGENTSIGNORE_FILENAME
        rules.extend(_load_rules_from_file(global_file))
        rules.extend(_load_rules_from_file(root / DEEPAGENTSIGNORE_FILENAME))
        root = _resolve_lenient(root)
        return cls(root=root, rules=tuple(rules))

    def is_ignored_relative(self, path: str, *, is_dir: bool = False) -> bool:
        """Return whether a project-relative path is ignored."""
        rel_path = _normalize_relative_path(path)
        if not rel_path:
            return False

        ignored = False
        for rule in self.rules:
            if _rule_matches(rule, rel_path, is_dir=is_dir):
                ignored = not rule.negated
        return ignored

    def is_ignored_path(self, path: str | Path, *, is_dir: bool = False) -> bool:
        """Return whether an absolute or relative filesystem path is ignored."""
        rel_path = self.relative_path(path)
        if rel_path is None:
            return False
        return self.is_ignored_relative(rel_path, is_dir=is_dir)

    def relative_path(self, path: str | Path) -> str | None:
        """Map a filesystem path to the configured project-relative path.

        Returns:
            Project-relative path, or `None` when the path is outside the root.
        """
        try:
            raw = str(path)
            trailing_slash = raw.endswith(("/", "\\"))
            candidate = Path(raw.rstrip("/\\")).expanduser()
            if not candidate.is_absolute():
                candidate = self.root / candidate
            resolved = candidate.resolve(strict=False)
            rel = resolved.relative_to(self.root)
        except (OSError, RuntimeError, ValueError):
            return None

        rel_path = rel.as_posix()
        if trailing_slash and rel_path != ".":
            rel_path += "/"
        return rel_path

    def filter_project_files(self, paths: list[str]) -> list[str]:
        """Filter project-relative file paths through the ignore rules.

        Returns:
            Paths that are not ignored.
        """
        return [
            path for path in paths if not self.is_ignored_relative(path, is_dir=False)
        ]


class IgnoringBackend(BackendProtocol):
    """Backend wrapper that hides ignored files from CLI file context."""

    def __init__(self, backend: BackendProtocol, ignore: DeepagentsIgnore) -> None:
        """Wrap a backend with project ignore filtering."""
        self._backend = backend
        self._ignore = ignore

    def ls(self, path: str) -> LsResult:
        """List directory entries, excluding ignored paths.

        Returns:
            Directory listing with ignored entries removed.
        """
        result = self._backend.ls(path)
        if result.error or result.entries is None:
            return result
        return LsResult(
            error=result.error,
            entries=[
                entry
                for entry in result.entries
                if not self._is_ignored_file_info(entry)
            ],
        )

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a file unless it is ignored.

        Returns:
            Read result, or a permission-style error for ignored paths.
        """
        if self._ignore.is_ignored_path(file_path):
            return ReadResult(error=f"{IGNORED_READ_ERROR}: {file_path}")
        return self._backend.read(file_path, offset, limit)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file unless it is ignored.

        Returns:
            Edit result, or a permission-style error for ignored paths.
        """
        if self._ignore.is_ignored_path(file_path):
            return EditResult(error=f"{IGNORED_READ_ERROR}: {file_path}")
        return self._backend.edit(file_path, old_string, new_string, replace_all)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files, excluding ignored matches.

        Returns:
            Glob result with ignored matches removed.
        """
        result = self._backend.glob(pattern, path)
        if result.error or result.matches is None:
            return result
        return GlobResult(
            error=result.error,
            matches=[
                match
                for match in result.matches
                if not self._is_ignored_file_info(match)
            ],
        )

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search files, excluding ignored matches from the result.

        Returns:
            Grep result with ignored matches removed.
        """
        result = self._backend.grep(pattern, path, glob)
        if result.error or result.matches is None:
            return result
        return GrepResult(
            error=result.error,
            matches=[
                match for match in result.matches if not self._is_ignored_grep(match)
            ],
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files, returning permission errors for ignored paths.

        Returns:
            One download response per requested path.
        """
        blocked_indexes = {
            index
            for index, path in enumerate(paths)
            if self._ignore.is_ignored_path(path)
        }
        allowed_paths = [
            path for index, path in enumerate(paths) if index not in blocked_indexes
        ]
        allowed_results = iter(self._backend.download_files(allowed_paths))

        results: list[FileDownloadResponse] = []
        for index, path in enumerate(paths):
            if index in blocked_indexes:
                results.append(
                    FileDownloadResponse(
                        path=path,
                        error="permission_denied",
                    )
                )
            else:
                results.append(next(allowed_results))
        return results

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write a file through to the wrapped backend.

        Returns:
            Wrapped backend write result.
        """
        return self._backend.write(file_path, content)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files through to the wrapped backend.

        Returns:
            Wrapped backend upload results.
        """
        return self._backend.upload_files(files)

    def _is_ignored_file_info(self, info: FileInfo) -> bool:
        path = info.get("path", "")
        is_dir = bool(info.get("is_dir")) or path.endswith("/")
        return self._ignore.is_ignored_path(path, is_dir=is_dir)

    def _is_ignored_grep(self, match: GrepMatch) -> bool:
        return self._ignore.is_ignored_path(match["path"])


class IgnoringSandboxBackend(IgnoringBackend, SandboxBackendProtocol):
    """Ignore-filtering wrapper for backends that can execute shell commands."""

    def __init__(
        self,
        backend: SandboxBackendProtocol,
        ignore: DeepagentsIgnore,
    ) -> None:
        """Wrap a shell-capable backend with project ignore filtering."""
        super().__init__(backend, ignore)
        self._sandbox_backend = backend

    @property
    def id(self) -> str:
        """Return the wrapped backend id.

        Returns:
            Backend identifier.
        """
        return self._sandbox_backend.id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Delegate shell execution to the wrapped backend.

        Returns:
            Wrapped backend execution response.
        """
        if timeout is None:
            return self._sandbox_backend.execute(command)
        return self._sandbox_backend.execute(command, timeout=timeout)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """Delegate async shell execution to the wrapped backend.

        Returns:
            Wrapped backend execution response.
        """
        if timeout is None:
            return await self._sandbox_backend.aexecute(command)
        return await self._sandbox_backend.aexecute(command, timeout=timeout)


def _load_rules_from_file(path: Path) -> list[IgnoreRule]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Could not read %s", path, exc_info=True)
        return []
    return [rule for line in lines if (rule := _parse_ignore_line(line)) is not None]


def _resolve_lenient(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def _parse_ignore_line(line: str) -> IgnoreRule | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        return None

    negated = stripped.startswith("!")
    if negated or stripped.startswith(("\\!", "\\#")):
        stripped = stripped[1:]

    stripped = stripped.strip()
    if not stripped:
        return None

    anchored = stripped.startswith("/")
    if anchored:
        stripped = stripped.lstrip("/")

    directory_only = stripped.endswith("/")
    stripped = stripped.rstrip("/")
    if not stripped:
        return None

    return IgnoreRule(
        pattern=stripped,
        negated=negated,
        directory_only=directory_only,
        anchored=anchored,
    )


def _normalize_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if normalized == ".":
        return ""
    return normalized


def _rule_matches(rule: IgnoreRule, rel_path: str, *, is_dir: bool) -> bool:
    if rule.directory_only:
        return _directory_rule_matches(rule, rel_path, is_dir=is_dir)
    if rule.anchored or "/" in rule.pattern:
        return _path_matches(rule.pattern, rel_path)
    return any(_path_matches(rule.pattern, part) for part in rel_path.split("/"))


def _directory_rule_matches(
    rule: IgnoreRule,
    rel_path: str,
    *,
    is_dir: bool,
) -> bool:
    if rule.anchored or "/" in rule.pattern:
        pattern = rule.pattern.rstrip("/")
        return rel_path == pattern or rel_path.startswith(f"{pattern}/")

    parts = rel_path.split("/")
    directory_parts = parts if is_dir else parts[:-1]
    return any(_path_matches(rule.pattern, part) for part in directory_parts)


def _path_matches(pattern: str, rel_path: str) -> bool:
    return fnmatch.fnmatchcase(rel_path, pattern)
