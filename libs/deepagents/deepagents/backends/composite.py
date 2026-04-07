"""Composite backend that routes file operations by path prefix.

Routes operations to different backends based on path prefixes. Use this when you
need different storage strategies for different paths (e.g., state for temp files,
persistent store for memories).

Examples:
    ```python
    from deepagents.backends.composite import CompositeBackend, Route, RoutePolicy
    from deepagents.backends.state import StateBackend
    from deepagents.backends.store import StoreBackend

    # Bare backends (no restrictions, backwards compatible)
    composite = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})

    # With policies
    composite = CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": Route(
                backend=StoreBackend(),
                policy=RoutePolicy(allowed_methods={"ls", "read", "glob", "grep"}),
            ),
        },
        default_policy=RoutePolicy(
            allowed_methods={"ls", "read", "write", "edit", "glob", "grep", "execute"},
        ),
    )
    ```
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

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
    execute_accepts_timeout,
)

if TYPE_CHECKING:
    from deepagents.backends.state import StateBackend


@dataclass
class RoutePolicy:
    """Policy that governs which backend methods are allowed on a route.

    Methods are identified by their backend method names: `ls`, `read`, `write`,
    `edit`, `glob`, `grep`, `execute`, `upload_files`, `download_files`.

    Subclass and override `is_allowed` for programmatic policies (e.g.,
    user-based access control).

    Examples:
        ```python
        read_only = RoutePolicy(allowed_methods={"ls", "read", "glob", "grep"})
        read_only.is_allowed("read")  # True
        read_only.is_allowed("write")  # False
        ```
    """

    allowed_methods: set[str]

    def is_allowed(self, method: str, **context: Any) -> bool:  # noqa: ARG002
        """Check whether a backend method is permitted.

        Args:
            method: Backend method name (e.g. `"write"`, `"read"`).
            **context: Reserved for future use (e.g. user identity).

        Returns:
            `True` if the method is allowed, `False` otherwise.
        """
        return method in self.allowed_methods

    def describe(self) -> str:
        """Return a human-readable description of this policy for system prompts.

        Subclasses should override this to describe their custom constraints.

        Returns:
            A short description of the policy.
        """
        return f"allowed methods: {', '.join(sorted(self.allowed_methods))}"


@dataclass
class Route:
    """A backend paired with an optional access policy.

    Use this instead of a bare `BackendProtocol` when you want to attach
    a `RoutePolicy` to a route.

    Examples:
        ```python
        Route(
            backend=StoreBackend(),
            policy=RoutePolicy(allowed_methods={"ls", "read", "glob", "grep"}),
        )
        ```
    """

    backend: BackendProtocol
    policy: RoutePolicy | None = None


def _remap_grep_path(m: GrepMatch, route_prefix: str) -> GrepMatch:
    """Create a new GrepMatch with the route prefix prepended to the path."""
    return cast(
        "GrepMatch",
        {
            **m,
            "path": f"{route_prefix[:-1]}{m['path']}",
        },
    )


def _strip_route_from_pattern(pattern: str, route_prefix: str) -> str:
    """Strip a route prefix from a glob pattern when the pattern targets that route.

    If the pattern (ignoring a leading `/`) starts with the route prefix
    (also ignoring its leading `/`), the overlapping prefix is removed so
    the pattern is relative to the backend's internal root.

    Args:
        pattern: The glob pattern, possibly absolute (e.g. `/memories/**/*.md`).
        route_prefix: The route prefix (e.g. `/memories/`).

    Returns:
        The pattern with the route prefix stripped, or the original pattern
        if it doesn't match the route.
    """
    bare_pattern = pattern.lstrip("/")
    bare_prefix = route_prefix.strip("/") + "/"
    if bare_pattern.startswith(bare_prefix):
        return bare_pattern[len(bare_prefix) :]
    return pattern


def _remap_file_info_path(fi: FileInfo, route_prefix: str) -> FileInfo:
    """Create a new FileInfo with the route prefix prepended to the path."""
    return cast(
        "FileInfo",
        {
            **fi,
            "path": f"{route_prefix[:-1]}{fi['path']}",
        },
    )


def _route_for_path(
    *,
    default: BackendProtocol,
    sorted_routes: list[tuple[str, BackendProtocol]],
    path: str,
) -> tuple[BackendProtocol, str, str | None]:
    """Route a path to a backend and normalize it for that backend.

    Returns the selected backend, the normalized path to pass to that backend,
    and the matched route prefix (or None if the default backend is used).

    Normalization rules:
    - If path is exactly the route root without trailing slash (e.g., "/memories"),
      route to that backend and return backend_path "/".
    - If path starts with the route prefix (e.g., "/memories/notes.txt"), strip the
      route prefix and ensure the result starts with "/".
    - Otherwise return the default backend and the original path.
    """
    for route_prefix, backend in sorted_routes:
        prefix_no_slash = route_prefix.rstrip("/")
        if path == prefix_no_slash:
            return backend, "/", route_prefix

        # Ensure route_prefix ends with / for startswith check to enforce boundary
        normalized_prefix = route_prefix if route_prefix.endswith("/") else f"{route_prefix}/"
        if path.startswith(normalized_prefix):
            suffix = path[len(normalized_prefix) :]
            backend_path = f"/{suffix}" if suffix else "/"
            return backend, backend_path, route_prefix
    return default, path, None


class CompositeBackend(BackendProtocol):
    """Routes file operations to different backends by path prefix.

    Matches paths against route prefixes (longest first) and delegates to the
    corresponding backend. Unmatched paths use the default backend.

    Access policies can be attached to routes via `Route` objects or applied
    globally via `default_policy`.

    Attributes:
        default: Backend for paths that don't match any route.
        routes: Map of path prefixes to backends.
        sorted_routes: Routes sorted by length (longest first) for correct matching.
        default_policy: Policy applied to the default backend and to bare-backend
            routes that lack an explicit policy.
        artifacts_root: Root path for artifacts, such as messages offloaded by middleware.

    Examples:
        ```python
        composite = CompositeBackend(
            default=StateBackend(),
            routes={
                "/memories/": Route(
                    backend=StoreBackend(),
                    policy=RoutePolicy(allowed_methods={"ls", "read", "glob", "grep"}),
                ),
                "/cache/": StoreBackend(),  # bare backend, inherits default_policy
            },
            default_policy=RoutePolicy(allowed_methods={"ls", "read", "write", "edit", "glob", "grep"}),
        )
        ```
    """

    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, Route | BackendProtocol],
        *,
        default_policy: RoutePolicy | None = None,
        artifacts_root: str = "/",
    ) -> None:
        """Initialize composite backend.

        Args:
            default: Backend for paths that don't match any route.
            routes: Map of path prefixes to backends or `Route` objects. Prefixes
                must start with "/" and should end with "/" (e.g., "/memories/").
                Values can be bare `BackendProtocol` instances (backwards
                compatible) or `Route` objects with an optional `RoutePolicy`.
            default_policy: Policy applied to the default backend and to
                bare-backend routes (without an explicit `Route.policy`).
                Does **not** override explicit `Route.policy` values.
            artifacts_root: Root path for artifacts, such as messages offloaded
                by middleware.
        """
        self.default = default
        self.default_policy = default_policy

        backend_routes: dict[str, BackendProtocol] = {}
        self._policies: dict[str, RoutePolicy | None] = {}
        for prefix, value in routes.items():
            if isinstance(value, Route):
                backend_routes[prefix] = value.backend
                self._policies[prefix] = value.policy
            else:
                backend_routes[prefix] = value
                self._policies[prefix] = None

        self.routes = backend_routes
        self.sorted_routes = sorted(backend_routes.items(), key=lambda x: len(x[0]), reverse=True)
        self.artifacts_root = artifacts_root

    def has_any_policy(self) -> bool:
        """Return `True` if any route or the default has a policy configured."""
        return self.default_policy is not None or any(p is not None for p in self._policies.values())

    def policy_for_route(self, route_prefix: str | None) -> RoutePolicy | None:
        """Resolve the effective policy for a matched route.

        Returns the explicit route policy if set, otherwise falls back to
        `default_policy`. The default policy applies to both the default
        backend and bare-backend routes.

        Args:
            route_prefix: The route prefix to look up, or `None` for the
                default backend.
        """
        if route_prefix is None:
            return self.default_policy
        explicit = self._policies.get(route_prefix)
        if explicit is not None:
            return explicit
        return self.default_policy

    def _policy_error(self, method: str, path: str, route_prefix: str | None) -> str | None:
        """Return an error message if the policy blocks the method, else `None`."""
        policy = self.policy_for_route(route_prefix)
        if policy is not None and not policy.is_allowed(method):
            return f"Method '{method}' is not allowed on path '{path}' (route '{route_prefix}'). Allowed methods: {sorted(policy.allowed_methods)}"
        return None

    def globally_blocked_methods(self, methods: set[str]) -> set[str]:
        """Return the subset of methods that are blocked on every route and the default.

        A method is "globally blocked" when every effective policy (the default
        backend's policy *and* every route's resolved policy) disallows it.
        If any route or the default has *no* policy, that path is unrestricted
        and no method can be considered globally blocked.

        Args:
            methods: Candidate backend method names to check
                (e.g. `{"write", "edit", "execute"}`).

        Returns:
            The subset of `methods` that are blocked everywhere. Empty set if
            no policies are configured or if any path is unrestricted.
        """
        if not self.has_any_policy():
            return set()

        effective_policies: list[RoutePolicy | None] = [
            self.default_policy,
            *[self.policy_for_route(prefix) for prefix in self.routes],
        ]

        if any(p is None for p in effective_policies):
            return set()

        policies = cast("list[RoutePolicy]", effective_policies)
        return {m for m in methods if all(not p.is_allowed(m) for p in policies)}

    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        backend, stripped_key, _route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=key,
        )
        return backend, stripped_key

    def _get_backend_key_and_route(self, key: str) -> tuple[BackendProtocol, str, str | None]:
        """Like `_get_backend_and_key` but also returns the matched route prefix."""
        return _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=key,
        )

    @staticmethod
    def _coerce_ls_result(raw: LsResult | list[FileInfo]) -> LsResult:
        """Normalize legacy ``list[FileInfo]`` returns to `LsResult`."""
        if isinstance(raw, LsResult):
            return raw
        return LsResult(entries=raw)

    def ls(self, path: str) -> LsResult:
        """List directory contents (non-recursive).

        If path matches a route, lists only that backend. If path is "/", aggregates
        default backend plus virtual route directories. Otherwise lists default backend.

        Args:
            path: Absolute directory path starting with "/".

        Returns:
            LsResult with directory entries or error.

        Examples:
            ```python
            result = composite.ls("/")
            result = composite.ls("/memories/")
            ```
        """
        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            err = self._policy_error("ls", path, route_prefix)
            if err:
                return LsResult(error=err)
            ls_result = self._coerce_ls_result(backend.ls(backend_path))
            if ls_result.error:
                return ls_result
            return LsResult(entries=[_remap_file_info_path(fi, route_prefix) for fi in (ls_result.entries or [])])

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            err = self._policy_error("ls", path, None)
            if not err:
                default_result = self._coerce_ls_result(self.default.ls(path))
                results.extend(default_result.entries or [])
            for route_prefix, _backend in self.sorted_routes:
                if self._policy_error("ls", route_prefix, route_prefix):
                    continue
                # Add the route itself as a directory (e.g., /memories/)
                results.append(
                    FileInfo(
                        path=route_prefix,
                        is_dir=True,
                        size=0,
                        modified_at="",
                    )
                )

            results.sort(key=lambda x: x.get("path", ""))
            return LsResult(entries=results)

        # Path doesn't match a route: query only default backend
        err = self._policy_error("ls", path, None)
        if err:
            return LsResult(error=err)
        return self._coerce_ls_result(self.default.ls(path))

    async def als(self, path: str) -> LsResult:
        """Async version of ls."""
        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            err = self._policy_error("ls", path, route_prefix)
            if err:
                return LsResult(error=err)
            ls_result = self._coerce_ls_result(await backend.als(backend_path))
            if ls_result.error:
                return ls_result
            return LsResult(entries=[_remap_file_info_path(fi, route_prefix) for fi in (ls_result.entries or [])])

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            err = self._policy_error("ls", path, None)
            if not err:
                default_result = self._coerce_ls_result(await self.default.als(path))
                results.extend(default_result.entries or [])
            for route_prefix, _backend in self.sorted_routes:
                if self._policy_error("ls", route_prefix, route_prefix):
                    continue
                # Add the route itself as a directory (e.g., /memories/)
                results.append(
                    {
                        "path": route_prefix,
                        "is_dir": True,
                        "size": 0,
                        "modified_at": "",
                    }
                )

            results.sort(key=lambda x: x.get("path", ""))
            return LsResult(entries=results)

        # Path doesn't match a route: query only default backend
        err = self._policy_error("ls", path, None)
        if err:
            return LsResult(error=err)
        return self._coerce_ls_result(await self.default.als(path))

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content, routing to appropriate backend.

        Args:
            file_path: Absolute file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            ReadResult
        """
        backend, stripped_key, route_prefix = self._get_backend_key_and_route(file_path)
        err = self._policy_error("read", file_path, route_prefix)
        if err:
            return ReadResult(error=err)
        return backend.read(stripped_key, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Async version of read."""
        backend, stripped_key, route_prefix = self._get_backend_key_and_route(file_path)
        err = self._policy_error("read", file_path, route_prefix)
        if err:
            return ReadResult(error=err)
        return await backend.aread(stripped_key, offset=offset, limit=limit)

    @staticmethod
    def _coerce_grep_result(raw: GrepResult | list[GrepMatch] | str) -> GrepResult:
        """Normalize legacy ``list[GrepMatch] | str`` returns to `GrepResult`."""
        if isinstance(raw, GrepResult):
            return raw
        if isinstance(raw, str):
            return GrepResult(error=raw)
        return GrepResult(matches=raw)

    def grep(  # noqa: C901, PLR0911
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search files for literal text pattern.

        Routes to backends based on path: specific route searches one backend,
        "/" or None searches all backends, otherwise searches default backend.

        Args:
            pattern: Literal text to search for (NOT regex).
            path: Directory to search. None searches all backends.
            glob: Glob pattern to filter files (e.g., "*.py", "**/*.txt").
                Filters by filename, not content.

        Returns:
            GrepResult with matches or error.

        Examples:
            ```python
            result = composite.grep("TODO", path="/memories/")
            result = composite.grep("error", path="/")
            result = composite.grep("import", path="/", glob="*.py")
            ```
        """
        if path is not None:
            backend, backend_path, route_prefix = _route_for_path(
                default=self.default,
                sorted_routes=self.sorted_routes,
                path=path,
            )
            if route_prefix is not None:
                err = self._policy_error("grep", path, route_prefix)
                if err:
                    return GrepResult(error=err)
                grep_result = self._coerce_grep_result(backend.grep(pattern, backend_path, glob))
                if grep_result.error:
                    return grep_result
                return GrepResult(matches=[_remap_grep_path(m, route_prefix) for m in (grep_result.matches or [])])

        # If path is None or "/", search default and all routed backends and merge
        # Otherwise, search only the default backend
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            err = self._policy_error("grep", path or "/", None)
            if not err:
                default_result = self._coerce_grep_result(self.default.grep(pattern, path, glob))
                if default_result.error:
                    return default_result
                all_matches.extend(default_result.matches or [])

            for route_prefix, backend in self.routes.items():
                err = self._policy_error("grep", path or "/", route_prefix)
                if err:
                    continue
                grep_result = self._coerce_grep_result(backend.grep(pattern, "/", glob))
                if grep_result.error:
                    return grep_result
                all_matches.extend(_remap_grep_path(m, route_prefix) for m in (grep_result.matches or []))

            return GrepResult(matches=all_matches)
        # Path specified but doesn't match a route - search only default
        err = self._policy_error("grep", path, None)
        if err:
            return GrepResult(error=err)
        return self._coerce_grep_result(self.default.grep(pattern, path, glob))

    async def agrep(  # noqa: C901, PLR0911
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Async version of grep.

        See grep() for detailed documentation on routing behavior and parameters.
        """
        if path is not None:
            backend, backend_path, route_prefix = _route_for_path(
                default=self.default,
                sorted_routes=self.sorted_routes,
                path=path,
            )
            if route_prefix is not None:
                err = self._policy_error("grep", path, route_prefix)
                if err:
                    return GrepResult(error=err)
                grep_result = self._coerce_grep_result(await backend.agrep(pattern, backend_path, glob))
                if grep_result.error:
                    return grep_result
                return GrepResult(matches=[_remap_grep_path(m, route_prefix) for m in (grep_result.matches or [])])

        # If path is None or "/", search default and all routed backends and merge
        # Otherwise, search only the default backend
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            err = self._policy_error("grep", path or "/", None)
            if not err:
                default_result = self._coerce_grep_result(await self.default.agrep(pattern, path, glob))
                if default_result.error:
                    return default_result
                all_matches.extend(default_result.matches or [])

            for route_prefix, backend in self.routes.items():
                err = self._policy_error("grep", path or "/", route_prefix)
                if err:
                    continue
                grep_result = self._coerce_grep_result(await backend.agrep(pattern, "/", glob))
                if grep_result.error:
                    return grep_result
                all_matches.extend(_remap_grep_path(m, route_prefix) for m in (grep_result.matches or []))

            return GrepResult(matches=all_matches)
        # Path specified but doesn't match a route - search only default
        err = self._policy_error("grep", path, None)
        if err:
            return GrepResult(error=err)
        return self._coerce_grep_result(await self.default.agrep(pattern, path, glob))

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching a glob pattern, routing by path prefix."""
        results: list[FileInfo] = []

        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            err = self._policy_error("glob", path, route_prefix)
            if err:
                return GlobResult(error=err)
            glob_result = backend.glob(pattern, backend_path)
            matches = glob_result.matches if isinstance(glob_result, GlobResult) else glob_result
            if isinstance(glob_result, GlobResult) and glob_result.error:
                return glob_result
            return GlobResult(matches=[_remap_file_info_path(fi, route_prefix) for fi in (matches or [])])

        # Path doesn't match any specific route - search default backend AND all routed backends
        err = self._policy_error("glob", path, None)
        if not err:
            default_result = self.default.glob(pattern, path)
            default_matches = default_result.matches if isinstance(default_result, GlobResult) else default_result
            results.extend(default_matches or [])

        for route_prefix, backend in self.routes.items():
            err = self._policy_error("glob", path, route_prefix)
            if err:
                continue
            route_pattern = _strip_route_from_pattern(pattern, route_prefix)
            sub_result = backend.glob(route_pattern, "/")
            sub_matches = sub_result.matches if isinstance(sub_result, GlobResult) else sub_result
            results.extend(_remap_file_info_path(fi, route_prefix) for fi in (sub_matches or []))

        # Deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return GlobResult(matches=results)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        """Async version of glob."""
        results: list[FileInfo] = []

        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            err = self._policy_error("glob", path, route_prefix)
            if err:
                return GlobResult(error=err)
            glob_result = await backend.aglob(pattern, backend_path)
            matches = glob_result.matches if isinstance(glob_result, GlobResult) else glob_result
            if isinstance(glob_result, GlobResult) and glob_result.error:
                return glob_result
            return GlobResult(matches=[_remap_file_info_path(fi, route_prefix) for fi in (matches or [])])

        # Path doesn't match any specific route - search default backend AND all routed backends
        err = self._policy_error("glob", path, None)
        if not err:
            default_result = await self.default.aglob(pattern, path)
            default_matches = default_result.matches if isinstance(default_result, GlobResult) else default_result
            results.extend(default_matches or [])

        for route_prefix, backend in self.routes.items():
            err = self._policy_error("glob", path, route_prefix)
            if err:
                continue
            route_pattern = _strip_route_from_pattern(pattern, route_prefix)
            sub_result = await backend.aglob(route_pattern, "/")
            sub_matches = sub_result.matches if isinstance(sub_result, GlobResult) else sub_result
            results.extend(_remap_file_info_path(fi, route_prefix) for fi in (sub_matches or []))

        # Deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return GlobResult(matches=results)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file, routing to appropriate backend.

        Args:
            file_path: Absolute file path.
            content: File content as a string.

        Returns:
            Success message or Command object, or error if file already exists.
        """
        backend, stripped_key, route_prefix = self._get_backend_key_and_route(file_path)
        err = self._policy_error("write", file_path, route_prefix)
        if err:
            return WriteResult(error=err)
        res = backend.write(stripped_key, content)
        if res.path is not None:
            res = replace(res, path=file_path)
        return res

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Async version of write."""
        backend, stripped_key, route_prefix = self._get_backend_key_and_route(file_path)
        err = self._policy_error("write", file_path, route_prefix)
        if err:
            return WriteResult(error=err)
        res = await backend.awrite(stripped_key, content)
        if res.path is not None:
            res = replace(res, path=file_path)
        return res

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file, routing to appropriate backend.

        Args:
            file_path: Absolute file path.
            old_string: String to find and replace.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences.

        Returns:
            Success message or Command object, or error message on failure.
        """
        backend, stripped_key, route_prefix = self._get_backend_key_and_route(file_path)
        err = self._policy_error("edit", file_path, route_prefix)
        if err:
            return EditResult(error=err)
        res = backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.path is not None:
            res = replace(res, path=file_path)
        return res

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Async version of edit."""
        backend, stripped_key, route_prefix = self._get_backend_key_and_route(file_path)
        err = self._policy_error("edit", file_path, route_prefix)
        if err:
            return EditResult(error=err)
        res = await backend.aedit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.path is not None:
            res = replace(res, path=file_path)
        return res

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command via the default backend.

        Unlike file operations, execution is not path-routable — it always
        delegates to the default backend.

        Args:
            command: Shell command to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

        Returns:
            ExecuteResponse with output, exit code, and truncation flag.

        Raises:
            NotImplementedError: If the default backend is not a
                `SandboxBackendProtocol` (i.e., it doesn't support execution).
        """
        err = self._policy_error("execute", "/", None)
        if err:
            return ExecuteResponse(output=err, exit_code=1, truncated=False)

        if isinstance(self.default, SandboxBackendProtocol):
            if timeout is not None and execute_accepts_timeout(type(self.default)):
                return self.default.execute(command, timeout=timeout)
            return self.default.execute(command)

        # This shouldn't be reached if the runtime check in the execute tool works correctly,
        # but we include it as a safety fallback.
        msg = (
            "Default backend doesn't support command execution (SandboxBackendProtocol). "
            "To enable execution, provide a default backend that implements SandboxBackendProtocol."
        )
        raise NotImplementedError(msg)

    async def aexecute(
        self,
        command: str,
        *,
        # ASYNC109 - timeout is a semantic parameter forwarded to the underlying
        # backend's implementation, not an asyncio.timeout() contract.
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """Async version of execute.

        See `execute()` for detailed documentation on parameters and behavior.
        """
        err = self._policy_error("execute", "/", None)
        if err:
            return ExecuteResponse(output=err, exit_code=1, truncated=False)

        if isinstance(self.default, SandboxBackendProtocol):
            if timeout is not None and execute_accepts_timeout(type(self.default)):
                return await self.default.aexecute(command, timeout=timeout)
            return await self.default.aexecute(command)

        # This shouldn't be reached if the runtime check in the execute tool works correctly,
        # but we include it as a safety fallback.
        msg = (
            "Default backend doesn't support command execution (SandboxBackendProtocol). "
            "To enable execution, provide a default backend that implements SandboxBackendProtocol."
        )
        raise NotImplementedError(msg)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files, batching by backend for efficiency.

        Groups files by their target backend, calls each backend's upload_files
        once with all files for that backend, then merges results in original order.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        results: list[FileUploadResponse | None] = [None] * len(files)
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path, route_prefix = self._get_backend_key_and_route(path)
            if self._policy_error("upload_files", path, route_prefix):
                results[idx] = FileUploadResponse(path=path, error="permission_denied")
            else:
                backend_batches[backend].append((idx, stripped_path, content))

        for backend, batch in backend_batches.items():
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))
            batch_responses = backend.upload_files(batch_files)
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileUploadResponse]", results)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Async version of upload_files."""
        results: list[FileUploadResponse | None] = [None] * len(files)
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path, route_prefix = self._get_backend_key_and_route(path)
            if self._policy_error("upload_files", path, route_prefix):
                results[idx] = FileUploadResponse(path=path, error="permission_denied")
            else:
                backend_batches[backend].append((idx, stripped_path, content))

        for backend, batch in backend_batches.items():
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))
            batch_responses = await backend.aupload_files(batch_files)
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileUploadResponse]", results)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files, batching by backend for efficiency.

        Groups paths by their target backend, calls each backend's download_files
        once with all paths for that backend, then merges results in original order.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.
        """
        results: list[FileDownloadResponse | None] = [None] * len(paths)
        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path, route_prefix = self._get_backend_key_and_route(path)
            if self._policy_error("download_files", path, route_prefix):
                results[idx] = FileDownloadResponse(path=path, error="permission_denied")
            else:
                backend_batches[backend].append((idx, stripped_path))

        for backend, batch in backend_batches.items():
            indices, stripped_paths = zip(*batch, strict=False)
            batch_responses = backend.download_files(list(stripped_paths))
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileDownloadResponse]", results)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        results: list[FileDownloadResponse | None] = [None] * len(paths)
        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path, route_prefix = self._get_backend_key_and_route(path)
            if self._policy_error("download_files", path, route_prefix):
                results[idx] = FileDownloadResponse(path=path, error="permission_denied")
            else:
                backend_batches[backend].append((idx, stripped_path))

        for backend, batch in backend_batches.items():
            indices, stripped_paths = zip(*batch, strict=False)
            batch_responses = await backend.adownload_files(list(stripped_paths))
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileDownloadResponse]", results)
