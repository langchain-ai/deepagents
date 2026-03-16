"""Composite backend that routes file operations by path prefix.

Routes operations to different backends based on path prefixes. Use this when you
need different storage strategies for different paths (e.g., state for temp files,
persistent store for memories).

Examples:
    ```python
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.state import StateBackend
    from deepagents.backends.store import StoreBackend

    runtime = make_runtime()
    composite = CompositeBackend(default=StateBackend(runtime), routes={"/memories/": StoreBackend(runtime)})

    composite.write("/temp.txt", "ephemeral")
    composite.write("/memories/note.md", "persistent")
    ```
"""

from collections import defaultdict
from dataclasses import replace
from typing import cast

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
from deepagents.backends.state import StateBackend


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

    Attributes:
        default: Backend for paths that don't match any route.
        routes: Map of path prefixes to backends (e.g., {"/memories/": store_backend}).
        sorted_routes: Routes sorted by length (longest first) for correct matching.

    Examples:
        ```python
        composite = CompositeBackend(default=StateBackend(runtime), routes={"/memories/": StoreBackend(runtime), "/cache/": StoreBackend(runtime)})

        composite.write("/temp.txt", "data")
        composite.write("/memories/note.txt", "data")
        ```
    """

    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, BackendProtocol],
    ) -> None:
        """Initialize composite backend.

        Args:
            default: Backend for paths that don't match any route.
            routes: Map of path prefixes to backends. Prefixes must start with "/"
                and should end with "/" (e.g., "/memories/").
        """
        # Default backend
        self.default = default

        # Virtual routes
        self.routes = routes

        # Sort routes by length (longest first) for correct prefix matching
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        backend, stripped_key, _route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=key,
        )
        return backend, stripped_key

    @staticmethod
    def _coerce_ls_result(raw: LsResult | list[FileInfo]) -> LsResult:
        """Normalize legacy ``list[FileInfo]`` returns to `LsResult`."""
        if isinstance(raw, LsResult):
            return raw
        return LsResult(entries=raw)

    def ls_info(self, path: str) -> LsResult:
        """List directory contents (non-recursive).

        If path matches a route, lists only that backend. If path is "/", aggregates
        default backend plus virtual route directories. Otherwise lists default backend.

        Args:
            path: Absolute directory path starting with "/".

        Returns:
            LsResult with directory entries or error.

        Examples:
            ```python
            result = composite.ls_info("/")
            result = composite.ls_info("/memories/")
            ```
        """
        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            ls_result = self._coerce_ls_result(backend.ls_info(backend_path))
            if ls_result.error:
                return ls_result
            return LsResult(entries=[_remap_file_info_path(fi, route_prefix) for fi in (ls_result.entries or [])])

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            default_result = self._coerce_ls_result(self.default.ls_info(path))
            results.extend(default_result.entries or [])
            for route_prefix, _backend in self.sorted_routes:
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
        return self._coerce_ls_result(self.default.ls_info(path))

    async def als_info(self, path: str) -> LsResult:
        """Async version of ls_info."""
        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            ls_result = self._coerce_ls_result(await backend.als_info(backend_path))
            if ls_result.error:
                return ls_result
            return LsResult(entries=[_remap_file_info_path(fi, route_prefix) for fi in (ls_result.entries or [])])

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            default_result = self._coerce_ls_result(await self.default.als_info(path))
            results.extend(default_result.entries or [])
            for route_prefix, _backend in self.sorted_routes:
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
        return self._coerce_ls_result(await self.default.als_info(path))

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
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Async version of read."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        return await backend.aread(stripped_key, offset=offset, limit=limit)

    @staticmethod
    def _coerce_grep_result(raw: GrepResult | list[GrepMatch] | str) -> GrepResult:
        """Normalize legacy ``list[GrepMatch] | str`` returns to `GrepResult`."""
        if isinstance(raw, GrepResult):
            return raw
        if isinstance(raw, str):
            return GrepResult(error=raw)
        return GrepResult(matches=raw)

    def grep_raw(
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
            result = composite.grep_raw("TODO", path="/memories/")
            result = composite.grep_raw("error", path="/")
            result = composite.grep_raw("import", path="/", glob="*.py")
            ```
        """
        if path is not None:
            backend, backend_path, route_prefix = _route_for_path(
                default=self.default,
                sorted_routes=self.sorted_routes,
                path=path,
            )
            if route_prefix is not None:
                grep_result = self._coerce_grep_result(backend.grep_raw(pattern, backend_path, glob))
                if grep_result.error:
                    return grep_result
                return GrepResult(matches=[_remap_grep_path(m, route_prefix) for m in (grep_result.matches or [])])

        # If path is None or "/", search default and all routed backends and merge
        # Otherwise, search only the default backend
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            default_result = self._coerce_grep_result(self.default.grep_raw(pattern, path, glob))
            if default_result.error:
                return default_result
            all_matches.extend(default_result.matches or [])

            for route_prefix, backend in self.routes.items():
                grep_result = self._coerce_grep_result(backend.grep_raw(pattern, "/", glob))
                if grep_result.error:
                    return grep_result
                all_matches.extend(_remap_grep_path(m, route_prefix) for m in (grep_result.matches or []))

            return GrepResult(matches=all_matches)
        # Path specified but doesn't match a route - search only default
        return self._coerce_grep_result(self.default.grep_raw(pattern, path, glob))

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Async version of grep_raw.

        See grep_raw() for detailed documentation on routing behavior and parameters.
        """
        if path is not None:
            backend, backend_path, route_prefix = _route_for_path(
                default=self.default,
                sorted_routes=self.sorted_routes,
                path=path,
            )
            if route_prefix is not None:
                grep_result = self._coerce_grep_result(await backend.agrep_raw(pattern, backend_path, glob))
                if grep_result.error:
                    return grep_result
                return GrepResult(matches=[_remap_grep_path(m, route_prefix) for m in (grep_result.matches or [])])

        # If path is None or "/", search default and all routed backends and merge
        # Otherwise, search only the default backend
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            default_result = self._coerce_grep_result(await self.default.agrep_raw(pattern, path, glob))
            if default_result.error:
                return default_result
            all_matches.extend(default_result.matches or [])

            for route_prefix, backend in self.routes.items():
                grep_result = self._coerce_grep_result(await backend.agrep_raw(pattern, "/", glob))
                if grep_result.error:
                    return grep_result
                all_matches.extend(_remap_grep_path(m, route_prefix) for m in (grep_result.matches or []))

            return GrepResult(matches=all_matches)
        # Path specified but doesn't match a route - search only default
        return self._coerce_grep_result(await self.default.agrep_raw(pattern, path, glob))

    def glob_info(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching a glob pattern, routing by path prefix."""
        results: list[FileInfo] = []

        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            glob_result = backend.glob_info(pattern, backend_path)
            matches = glob_result.matches if isinstance(glob_result, GlobResult) else glob_result
            if isinstance(glob_result, GlobResult) and glob_result.error:
                return glob_result
            return GlobResult(matches=[_remap_file_info_path(fi, route_prefix) for fi in (matches or [])])

        # Path doesn't match any specific route - search default backend AND all routed backends
        default_result = self.default.glob_info(pattern, path)
        default_matches = default_result.matches if isinstance(default_result, GlobResult) else default_result
        results.extend(default_matches or [])

        for route_prefix, backend in self.routes.items():
            route_pattern = _strip_route_from_pattern(pattern, route_prefix)
            sub_result = backend.glob_info(route_pattern, "/")
            sub_matches = sub_result.matches if isinstance(sub_result, GlobResult) else sub_result
            results.extend(_remap_file_info_path(fi, route_prefix) for fi in (sub_matches or []))

        # Deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return GlobResult(matches=results)

    async def aglob_info(self, pattern: str, path: str = "/") -> GlobResult:
        """Async version of glob_info."""
        results: list[FileInfo] = []

        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            glob_result = await backend.aglob_info(pattern, backend_path)
            matches = glob_result.matches if isinstance(glob_result, GlobResult) else glob_result
            if isinstance(glob_result, GlobResult) and glob_result.error:
                return glob_result
            return GlobResult(matches=[_remap_file_info_path(fi, route_prefix) for fi in (matches or [])])

        # Path doesn't match any specific route - search default backend AND all routed backends
        default_result = await self.default.aglob_info(pattern, path)
        default_matches = default_result.matches if isinstance(default_result, GlobResult) else default_result
        results.extend(default_matches or [])

        for route_prefix, backend in self.routes.items():
            route_pattern = _strip_route_from_pattern(pattern, route_prefix)
            sub_result = await backend.aglob_info(route_pattern, "/")
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
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.write(stripped_key, content)
        if res.path is not None:
            res = replace(res, path=file_path)
        # If this is a state-backed update and default has state, merge so listings reflect changes
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:  # noqa: BLE001, S110  # Intentional for best-effort state sync
                pass
        return res

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Async version of write."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.awrite(stripped_key, content)
        if res.path is not None:
            res = replace(res, path=file_path)
        # If this is a state-backed update and default has state, merge so listings reflect changes
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:  # noqa: BLE001, S110  # Intentional for best-effort state sync
                pass
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
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.path is not None:
            res = replace(res, path=file_path)
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:  # noqa: BLE001, S110  # Intentional for best-effort state sync
                pass
        return res

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Async version of edit."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.aedit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.path is not None:
            res = replace(res, path=file_path)
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:  # noqa: BLE001, S110  # Intentional for best-effort state sync
                pass
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
        # Pre-allocate result list
        results: list[FileUploadResponse | None] = [None] * len(files)

        # Group files by backend, tracking original indices
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # Call backend once with all its files
            batch_responses = backend.upload_files(batch_files)

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # Original path
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileUploadResponse]", results)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Async version of upload_files."""
        # Pre-allocate result list
        results: list[FileUploadResponse | None] = [None] * len(files)

        # Group files by backend, tracking original indices
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # Call backend once with all its files
            batch_responses = await backend.aupload_files(batch_files)

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # Original path
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
        # Pre-allocate result list
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths = zip(*batch, strict=False)

            # Call backend once with all its paths
            batch_responses = backend.download_files(list(stripped_paths))

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # Original path
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileDownloadResponse]", results)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        # Pre-allocate result list
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths = zip(*batch, strict=False)

            # Call backend once with all its paths
            batch_responses = await backend.adownload_files(list(stripped_paths))

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # Original path
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileDownloadResponse]", results)
