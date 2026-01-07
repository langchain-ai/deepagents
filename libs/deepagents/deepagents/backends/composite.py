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

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.state import StateBackend


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
        """Get backend for path and strip route prefix.

        Args:
            key: File path to route.

        Returns:
            Tuple of (backend, stripped_path). The stripped path has the route
            prefix removed but keeps the leading slash.
        """
        # Check routes in order of length (longest first)
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # Strip full prefix and ensure a leading slash remains
                # e.g., "/memories/notes.txt" → "/notes.txt"; "/memories/" → "/"
                suffix = key[len(prefix) :]
                stripped_key = f"/{suffix}" if suffix else "/"
                return backend, stripped_key

        return self.default, key

    def ls_info(self, path: str) -> list[FileInfo]:
        """List directory contents (non-recursive).

        If path matches a route, lists only that backend. If path is "/", aggregates
        default backend plus virtual route directories. Otherwise lists default backend.

        Args:
            path: Absolute directory path starting with "/".

        Returns:
            List of FileInfo dicts. Directories have trailing "/" and is_dir=True.
            Route prefixes are restored in returned paths.

        Examples:
            ```python
            infos = composite.ls_info("/")
            infos = composite.ls_info("/memories/")
            ```
        """
        # Check if path matches a specific route
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # Query only the matching routed backend
                suffix = path[len(route_prefix) :]
                search_path = f"/{suffix}" if suffix else "/"
                infos = backend.ls_info(search_path)
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi)
                return prefixed

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            results.extend(self.default.ls_info(path))
            for route_prefix, backend in self.sorted_routes:
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
            return results

        # Path doesn't match a route: query only default backend
        return self.default.ls_info(path)

    async def als_info(self, path: str) -> list[FileInfo]:
        """Async version of ls_info."""
        # Check if path matches a specific route
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # Query only the matching routed backend
                suffix = path[len(route_prefix) :]
                search_path = f"/{suffix}" if suffix else "/"
                infos = await backend.als_info(search_path)
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi)
                return prefixed

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            results.extend(await self.default.als_info(path))
            for route_prefix, backend in self.sorted_routes:
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
            return results

        # Path doesn't match a route: query only default backend
        return await self.default.als_info(path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content, routing to appropriate backend.

        Args:
            file_path: Absolute file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or error message.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Async version of read."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        return await backend.aread(stripped_key, offset=offset, limit=limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search files for regex pattern.

        Routes to backends based on path: specific route searches one backend,
        "/" or None searches all backends, otherwise searches default backend.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search. None searches all backends.
            glob: Glob pattern to filter files (e.g., "*.py", "**/*.txt").
                Filters by filename, not content.

        Returns:
            List of GrepMatch dicts with path (route prefix restored), line
            (1-indexed), and text. Returns error string on failure.

        Examples:
            ```python
            matches = composite.grep_raw("TODO", path="/memories/")
            matches = composite.grep_raw("error", path="/")
            matches = composite.grep_raw("import", path="/", glob="*.py")
            ```
        """
        # If path targets a specific route, search only that backend
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                raw = backend.grep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

        # If path is None or "/", search default and all routed backends and merge
        # Otherwise, search only the default backend
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            raw_default = self.default.grep_raw(pattern, path, glob)  # type: ignore[attr-defined]
            if isinstance(raw_default, str):
                # This happens if error occurs
                return raw_default
            all_matches.extend(raw_default)

            for route_prefix, backend in self.routes.items():
                raw = backend.grep_raw(pattern, "/", glob)
                if isinstance(raw, str):
                    # This happens if error occurs
                    return raw
                all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

            return all_matches
        # Path specified but doesn't match a route - search only default
        return self.default.grep_raw(pattern, path, glob)  # type: ignore[attr-defined]

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Async version of grep_raw.

        See grep_raw() for detailed documentation on routing behavior and parameters.
        """
        # If path targets a specific route, search only that backend
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                raw = await backend.agrep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

        # If path is None or "/", search default and all routed backends and merge
        # Otherwise, search only the default backend
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            raw_default = await self.default.agrep_raw(pattern, path, glob)  # type: ignore[attr-defined]
            if isinstance(raw_default, str):
                # This happens if error occurs
                return raw_default
            all_matches.extend(raw_default)

            for route_prefix, backend in self.routes.items():
                raw = await backend.agrep_raw(pattern, "/", glob)
                if isinstance(raw, str):
                    # This happens if error occurs
                    return raw
                all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

            return all_matches
        # Path specified but doesn't match a route - search only default
        return await self.default.agrep_raw(pattern, path, glob)  # type: ignore[attr-defined]

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        results: list[FileInfo] = []

        # Route based on path, not pattern
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                infos = backend.glob_info(pattern, search_path if search_path else "/")
                return [{**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos]

        # Path doesn't match any specific route - search default backend AND all routed backends
        results.extend(self.default.glob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = backend.glob_info(pattern, "/")
            results.extend({**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos)

        # Deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return results

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Async version of glob_info."""
        results: list[FileInfo] = []

        # Route based on path, not pattern
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                infos = await backend.aglob_info(pattern, search_path if search_path else "/")
                return [{**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos]

        # Path doesn't match any specific route - search default backend AND all routed backends
        results.extend(await self.default.aglob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = await backend.aglob_info(pattern, "/")
            results.extend({**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos)

        # Deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return results

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
        # If this is a state-backed update and default has state, merge so listings reflect changes
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
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
        # If this is a state-backed update and default has state, merge so listings reflect changes
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
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
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Async version of edit."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.aedit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute shell command via default backend.

        Args:
            command: Shell command to execute.

        Returns:
            ExecuteResponse with output, exit code, and truncation flag.

        Raises:
            NotImplementedError: If default backend doesn't implement SandboxBackendProtocol.

        Examples:
            ```python
            composite = CompositeBackend(default=FilesystemBackend(root_dir="/tmp"), routes={"/memories/": StoreBackend(runtime)})

            result = composite.execute("ls -la")
            ```
        """
        if isinstance(self.default, SandboxBackendProtocol):
            return self.default.execute(command)

        # This shouldn't be reached if the runtime check in the execute tool works correctly,
        # but we include it as a safety fallback.
        raise NotImplementedError(
            "Default backend doesn't support command execution (SandboxBackendProtocol). "
            "To enable execution, provide a default backend that implements SandboxBackendProtocol."
        )

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Async version of execute."""
        if isinstance(self.default, SandboxBackendProtocol):
            return await self.default.aexecute(command)

        # This shouldn't be reached if the runtime check in the execute tool works correctly,
        # but we include it as a safety fallback.
        raise NotImplementedError(
            "Default backend doesn't support command execution (SandboxBackendProtocol). "
            "To enable execution, provide a default backend that implements SandboxBackendProtocol."
        )

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
        from collections import defaultdict

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

        return results  # type: ignore[return-value]

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

        return results  # type: ignore[return-value]

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

        return results  # type: ignore[return-value]

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

        return results  # type: ignore[return-value]
