"""CompositeBackend: Route operations to different backends based on path prefix."""

from typing import Any, Literal, Optional, TYPE_CHECKING

from langchain.tools import ToolRuntime

from deepagents.backends.protocol import BackendProtocol, BackendProvider, StateBackendProvider, StateBackendProtocol
from deepagents.backends.state import StateBackend, StateBackendProvider
from langgraph.types import Command


class _CompositeBackend:
    
    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, BackendProtocol],
    ) -> None:
        # Default backend
        self.default = default

        # Virtual routes
        self.routes = routes
        
        # Sort routes by length (longest first) for correct prefix matching
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        """Determine which backend handles this key and strip prefix.
        
        Args:
            key: Original file path
        
        Returns:
            Tuple of (backend, stripped_key) where stripped_key has the route
            prefix removed (but keeps leading slash).
        """
        # Check routes in order of length (longest first)
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # Strip prefix but keep leading slash
                # e.g., "/memories/notes.txt" → "/notes.txt"
                stripped_key = key[len(prefix) - 1:] if key[len(prefix) - 1:] else "/"
                return backend, stripped_key
        
        return self.default, key
    
    def ls(self, path: str) -> list[str]:
        """List files from backends, with appropriate prefixes.
        
        Args:
            path: Absolute path to directory.
        
        Returns:
            List of file paths with route prefixes added.
        """
        # Check if path matches a specific route
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # Query only the matching routed backend
                search_path = path[len(route_prefix) - 1:]
                keys = backend.ls(search_path if search_path else "/")
                return [f"{route_prefix[:-1]}{key}" for key in keys]
        
        # Path doesn't match a route: query only default backend
        return self.default.ls(path)
    
    def read(
        self, 
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content, routing to appropriate backend.
        
        Args:
            file_path: Absolute file path
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to readReturns:
            Formatted file content with line numbers, or error message.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)
    
    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files, routing to appropriate backend(s).
        
        Args:
            pattern: String pattern to search for
            path: Path to search in (default "/")
            glob: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"Returns:
            Formatted search results based on output_mode.
        """
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1:]
                result = backend.grep(pattern, search_path if search_path else "/", glob, output_mode)
                if result.startswith("No matches found"):
                    return result
                
                lines = result.split("\n")
                prefixed_lines = []
                for line in lines:
                    if output_mode == "files_with_matches" or line.endswith(":") or ": " in line.split(":", 1)[0]:
                        if line and not line.startswith(" "):
                            prefixed_lines.append(f"{route_prefix[:-1]}{line}")
                        else:
                            prefixed_lines.append(line)
                    else:
                        prefixed_lines.append(line)
                return "\n".join(prefixed_lines)

        all_results = []
        
        default_result = self.default.grep(pattern, path, glob, output_mode)
        if not default_result.startswith("No matches found"):
            all_results.append(default_result)
        
        for route_prefix, backend in self.routes.items():
            result = backend.grep(pattern, None, glob, output_mode)
            if not result.startswith("No matches found"):
                lines = result.split("\n")
                prefixed_lines = []
                for line in lines:
                    if output_mode == "files_with_matches" or line.endswith(":") or (": " in line and not line.startswith(" ")):
                        if line and not line.startswith(" "):
                            prefixed_lines.append(f"{route_prefix[:-1]}{line}")
                        else:
                            prefixed_lines.append(line)
                    else:
                        prefixed_lines.append(line)
                all_results.append("\n".join(prefixed_lines))
        
        if not all_results:
            return f"No matches found for pattern: '{pattern}'"
        
        return "\n".join(all_results)
    
    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern across all backends.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")Returns:
            List of absolute file paths matching the pattern.
        """
        results = []

        # Route based on path, not pattern
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # Path matches a specific route - search only that backend
                search_path = path[len(route_prefix) - 1:]
                matches = backend.glob(pattern, search_path if search_path else "/")
                results.extend(f"{route_prefix[:-1]}{match}" for match in matches)
                return sorted(results)

        # Path doesn't match any specific route - search default backend AND all routed backends
        default_matches = self.default.glob(pattern, path)
        results.extend(default_matches)

        # Also search in all routed backends and prefix results
        for route_prefix, backend in self.routes.items():
            matches = backend.glob(pattern, "/")
            results.extend(f"{route_prefix[:-1]}{match}" for match in matches)

        return sorted(results)


class CompositeStateBacked(_CompositeBackend):

    def __init__(
            self,
            default: StateBackend,
            routes: dict[str, BackendProtocol],
    ) -> None:
        self.default = default
        self.routes = routes

        # Sort routes by length (longest first) for correct prefix matching
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def write(
            self,
            file_path: str,
            content: str,
    ) -> Command | str:
        """Create a new file, routing to appropriate backend.

        Args:
            file_path: Absolute file path
            content: File content as a stringReturns:
            Success message or Command object, or error if file already exists.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.write(stripped_key, content)

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> Command | str:
        """Edit a file, routing to appropriate backend.

        Args:
            file_path: Absolute file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrencesReturns:
            Success message or Command object, or error message on failure.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)


class CompositeBackend(_CompositeBackend):
    def write(
            self,
            file_path: str,
            content: str,
    ) -> str:
        """Create a new file, routing to appropriate backend.

        Args:
            file_path: Absolute file path
            content: File content as a stringReturns:
            Success message or Command object, or error if file already exists.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.write(stripped_key, content)

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> str:
        """Edit a file, routing to appropriate backend.

        Args:
            file_path: Absolute file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrencesReturns:
            Success message or Command object, or error message on failure.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)

class CompositeStateBackendProvider(StateBackendProvider):

    def __init__(self, routes: dict[str, BackendProtocol | BackendProvider]):
        self.routes = routes

    def get_backend(self, runtime: ToolRuntime) -> StateBackendProtocol:
        return CompositeStateBacked(
            default=StateBackendProvider.get_backend(runtime),
            routes={
                k: v if isinstance(v, BackendProtocol) else v.get_backend(runtime) for k,v in self.routes.items()
            }
        )