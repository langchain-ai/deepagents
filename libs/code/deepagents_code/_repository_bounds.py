"""Shared path-safety and size limits for read-only repository inspection.

Both the goal-criteria agent's `_RepositoryToolBudgetMiddleware` and the rubric
grader's read-only tools let an LLM sub-agent inspect working-directory files.
They must apply identical guarantees: reads stay confined to the repository
root, symlink escapes are rejected in a sandbox, and every result is size
bounded so a single tool call cannot blow the sub-agent's context budget.

`RepositoryBounds` centralizes that logic so the middleware and the grader tool
wrappers share one implementation. It is intentionally framework-agnostic: it
operates on tool names and argument dicts and returns either a bounded value or
an error-message string, leaving `ToolMessage`/`Command` construction and the
per-run call budget to the caller.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import SandboxBackendProtocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents.backends.protocol import BackendProtocol, FileInfo

logger = logging.getLogger(__name__)

REPOSITORY_TOOL_CALL_LIMIT = 25
REPOSITORY_READ_LINE_LIMIT = 120
REPOSITORY_READ_BYTE_LIMIT = 256_000
REPOSITORY_DIRECTORY_ENTRY_LIMIT = 200
REPOSITORY_GLOB_MATCH_LIMIT = 200
REPOSITORY_GREP_MATCH_LIMIT = 100
REPOSITORY_TOOL_RESULT_LIMIT = 12_000
REPOSITORY_TOOL_NAMES = frozenset({"ls", "read_file", "glob", "grep"})
REPOSITORY_PATH_RESULT_PREFIX = "__DEEPAGENTS_REPOSITORY_PATH__"

REPOSITORY_PATH_ERROR = "Repository path is unavailable."
REPOSITORY_SIZE_ERROR = "Repository file exceeds the size limit."
REPOSITORY_LISTING_ERROR = "Repository directory exceeds the listing limit."

# Backend faults that should degrade to a bounded, logged "path unavailable"
# error rather than crashing the sub-agent.
_BACKEND_ERRORS: tuple[type[BaseException], ...] = (
    NotImplementedError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class RepositoryBounds:
    """Path-safety and size limits for read-only repository inspection tools."""

    def __init__(self, backend: BackendProtocol, *, root: str = "/") -> None:
        """Initialize repository bounds rooted at an absolute backend path.

        Args:
            backend: Server-side repository backend used by filesystem tools.
            root: Absolute backend path that bounds repository reads.

        Raises:
            ValueError: If `root` is not a safe absolute path.
        """
        normalized = root.replace("\\", "/")
        path = PurePosixPath(normalized)
        if not normalized.startswith("/") or ".." in path.parts or "~" in root:
            msg = f"Repository root must be an absolute contained path: {root!r}"
            raise ValueError(msg)
        self._backend = backend
        self._root = str(path)
        self._sandbox = backend if isinstance(backend, SandboxBackendProtocol) else None

    @property
    def root(self) -> str:
        """Absolute path that bounds repository reads."""
        return self._root

    def safe_path(self, raw_path: str) -> bool:
        """Return whether an explicit repository path is absolute and contained."""
        path = PurePosixPath(raw_path.replace("\\", "/"))
        root = PurePosixPath(self._root)
        return (
            raw_path.startswith("/")
            and ".." not in path.parts
            and "~" not in raw_path
            and (root == PurePosixPath("/") or path == root or root in path.parents)
        )

    @staticmethod
    def safe_pattern(pattern: str) -> bool:
        """Return whether a relative or absolute glob pattern cannot traverse."""
        path = PurePosixPath(pattern.replace("\\", "/"))
        return ".." not in path.parts and "~" not in pattern

    def _containment_command(self, raw_path: str) -> str:
        """Build a sandbox command that checks the canonical repository boundary.

        Returns:
            A command that emits a private success marker only for contained paths.
        """
        payload = base64.b64encode(json.dumps([self._root, raw_path]).encode()).decode()
        return (
            'python3 -c "import base64,json,os;'
            f"values=json.loads(base64.b64decode('{payload}'));"
            "root=os.path.realpath(values[0]);path=os.path.realpath(values[1]);"
            "contained=os.path.commonpath([root,path])==root;"
            f"print('{REPOSITORY_PATH_RESULT_PREFIX}'+str(int(contained)))\""
        )

    def sandbox_contains(self, raw_path: str) -> bool:
        """Return whether the sandbox resolves a path below the repository root."""
        if self._sandbox is None:
            return True
        try:
            result = self._sandbox.execute(self._containment_command(raw_path))
        except _BACKEND_ERRORS:
            logger.warning(
                "Repository containment check failed; treating the path as unavailable",
                exc_info=True,
            )
            return False
        return result.exit_code in {None, 0} and any(
            line == f"{REPOSITORY_PATH_RESULT_PREFIX}1"
            for line in result.output.splitlines()
        )

    async def asandbox_contains(self, raw_path: str) -> bool:
        """Asynchronously check canonical sandbox repository containment.

        Returns:
            `True` when the sandbox resolves the path below the repository root.
        """
        if self._sandbox is None:
            return True
        try:
            result = await self._sandbox.aexecute(self._containment_command(raw_path))
        except _BACKEND_ERRORS:
            logger.warning(
                "Repository containment check failed; treating the path as unavailable",
                exc_info=True,
            )
            return False
        return result.exit_code in {None, 0} and any(
            line == f"{REPOSITORY_PATH_RESULT_PREFIX}1"
            for line in result.output.splitlines()
        )

    @staticmethod
    def entry_size(
        entries: Sequence[FileInfo] | None,
        normalized_path: str,
    ) -> int | None:
        """Return the reported byte size of a backend entry, if present.

        Malformed entries (not a mapping, or missing/non-string `path`) are
        skipped rather than raising, so a single bad entry cannot fail an
        otherwise valid preflight.

        Returns:
            The entry's integer size, or `None` when unknown.
        """
        for item in entries or []:
            raw = item.get("path") if isinstance(item, dict) else None
            if not isinstance(raw, str):
                continue
            if str(PurePosixPath(raw)) == normalized_path:
                size = item.get("size")
                return size if isinstance(size, int) else None
        return None

    def _validate_search_paths(
        self,
        name: str,
        args: dict[str, Any],
    ) -> str | None:
        """Validate optional paths and path-like patterns for search tools.

        Returns:
            A path error message, or `None` when every explicit path is contained.
        """
        path = args.get("path")
        if path is not None and (not isinstance(path, str) or not self.safe_path(path)):
            return REPOSITORY_PATH_ERROR

        patterns = [args.get("pattern")] if name == "glob" else [args.get("glob")]
        if any(
            pattern is not None
            and (not isinstance(pattern, str) or not self.safe_pattern(pattern))
            for pattern in patterns
        ):
            return REPOSITORY_PATH_ERROR
        return None

    def preflight(self, name: str, args: dict[str, Any]) -> str | None:
        """Reject malformed paths and backend entries that exceed hard limits.

        Returns:
            A bounded error message, or `None` when preflight succeeds.
        """
        if name in {"glob", "grep"}:
            error = self._validate_search_paths(name, args)
            if error is not None:
                return error
            raw_path = args.get("path", self._root)
            if isinstance(raw_path, str) and not self.sandbox_contains(raw_path):
                return REPOSITORY_PATH_ERROR
            return None

        key = "file_path" if name == "read_file" else "path"
        raw_path = args.get(key)
        if not isinstance(raw_path, str):
            return None

        path = PurePosixPath(raw_path.replace("\\", "/"))
        if not self.safe_path(raw_path):
            return REPOSITORY_PATH_ERROR
        if not self.sandbox_contains(raw_path):
            return REPOSITORY_PATH_ERROR

        # Scope the guard to the backend call itself: a backend that raises
        # (outage, serialization fault) is otherwise indistinguishable from a
        # genuinely absent path. The size/entry bookkeeping below is deliberately
        # left outside the guard so a defect there surfaces as a real crash
        # rather than silently degrading every run.
        try:
            result = self._backend.ls(raw_path if name == "ls" else str(path.parent))
        except _BACKEND_ERRORS:
            logger.warning(
                "Repository preflight failed for tool %r; treating the path as "
                "unavailable",
                name,
                exc_info=True,
            )
            return REPOSITORY_PATH_ERROR
        if result.error is not None:
            return REPOSITORY_PATH_ERROR
        if name == "ls":
            if len(result.entries or []) > REPOSITORY_DIRECTORY_ENTRY_LIMIT:
                return REPOSITORY_LISTING_ERROR
        else:  # read_file
            size = self.entry_size(result.entries, str(path))
            if size is not None and size > REPOSITORY_READ_BYTE_LIMIT:
                return REPOSITORY_SIZE_ERROR
        return None

    async def apreflight(self, name: str, args: dict[str, Any]) -> str | None:
        """Asynchronously enforce repository path and metadata limits.

        Returns:
            A bounded error message, or `None` when preflight succeeds.
        """
        if name in {"glob", "grep"}:
            error = self._validate_search_paths(name, args)
            if error is not None:
                return error
            raw_path = args.get("path", self._root)
            if isinstance(raw_path, str) and not await self.asandbox_contains(raw_path):
                return REPOSITORY_PATH_ERROR
            return None

        key = "file_path" if name == "read_file" else "path"
        raw_path = args.get(key)
        if not isinstance(raw_path, str):
            return None

        path = PurePosixPath(raw_path.replace("\\", "/"))
        if not self.safe_path(raw_path):
            return REPOSITORY_PATH_ERROR
        if not await self.asandbox_contains(raw_path):
            return REPOSITORY_PATH_ERROR

        try:
            result = await self._backend.als(
                raw_path if name == "ls" else str(path.parent)
            )
        except _BACKEND_ERRORS:
            logger.warning(
                "Repository preflight failed for tool %r; treating the path as "
                "unavailable",
                name,
                exc_info=True,
            )
            return REPOSITORY_PATH_ERROR
        if result.error is not None:
            return REPOSITORY_PATH_ERROR
        if name == "ls":
            if len(result.entries or []) > REPOSITORY_DIRECTORY_ENTRY_LIMIT:
                return REPOSITORY_LISTING_ERROR
        elif name == "read_file":
            size = self.entry_size(result.entries, str(path))
            if size is not None and size > REPOSITORY_READ_BYTE_LIMIT:
                return REPOSITORY_SIZE_ERROR
        return None

    def clamp_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Clamp repository-tool arguments that directly control result size.

        Returns:
            A new args dict with bounded read lines or grep matches, and a
            repository-root default for search paths.
        """
        clamped = dict(args)
        if name == "read_file":
            limit = clamped.get("limit", REPOSITORY_READ_LINE_LIMIT)
            if not isinstance(limit, int) or isinstance(limit, bool):
                limit = REPOSITORY_READ_LINE_LIMIT
            clamped["limit"] = max(1, min(limit, REPOSITORY_READ_LINE_LIMIT))
        elif name in {"glob", "grep"}:
            clamped.setdefault("path", self._root)
        if name == "grep":
            count = clamped.get("max_count", REPOSITORY_GREP_MATCH_LIMIT)
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                count = REPOSITORY_GREP_MATCH_LIMIT
            clamped["max_count"] = min(count, REPOSITORY_GREP_MATCH_LIMIT)
        return clamped

    @staticmethod
    def bounded_glob_content(content: str) -> str:
        """Limit a filesystem glob's rendered path count when it is parseable.

        Returns:
            Glob output containing no more than the configured number of paths.
        """
        body, separator, notes = content.partition("\n\n")
        try:
            paths = ast.literal_eval(body)
        except (SyntaxError, ValueError):
            return content
        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            return content
        if len(paths) <= REPOSITORY_GLOB_MATCH_LIMIT:
            return content
        marker = (
            "[Glob results limited to the first "
            f"{REPOSITORY_GLOB_MATCH_LIMIT} matches.]"
        )
        bounded = str(paths[:REPOSITORY_GLOB_MATCH_LIMIT])
        suffix = f"\n\n{notes}" if separator and notes else ""
        return f"{bounded}\n\n{marker}{suffix}"

    def bound_text(self, name: str, content: str) -> str:
        """Return a size-bounded repository tool result body.

        Returns:
            The bounded content, with glob output additionally match-limited.
        """
        if name == "glob":
            content = self.bounded_glob_content(content)
        if len(content) > REPOSITORY_TOOL_RESULT_LIMIT:
            marker = "\n[Repository tool result shortened to the context limit.]"
            content = content[: REPOSITORY_TOOL_RESULT_LIMIT - len(marker)] + marker
        return content
