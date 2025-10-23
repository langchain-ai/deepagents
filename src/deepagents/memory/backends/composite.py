"""CompositeBackend: Route operations to different backends based on path prefix."""

from typing import Any, Optional

from deepagents.memory.protocol import MemoryBackend


class CompositeBackend:
    """Backend that routes operations to different backends based on path prefix.
    
    This backend enables hybrid storage strategies, such as:
    - Short-term files (/*) → StateBackend (ephemeral)
    - Long-term files (/memories/*) → StoreBackend or FilesystemBackend (persistent)
    
    The routing is transparent to tools - they just call backend.get(path) and
    CompositeBackend handles the routing internally.
    
    Example:
        ```python
        backend = CompositeBackend(
            default=StateBackend(runtime),
            routes={"/memories/": FilesystemBackend("/data/memories")}
        )
        
        # Routes to StateBackend
        backend.get("/temp.txt")
        
        # Routes to FilesystemBackend, strips prefix
        backend.get("/memories/notes.txt")  # → FilesystemBackend.get("/notes.txt")
        ```
    """
    
    def __init__(
        self,
        default: MemoryBackend,
        routes: dict[str, MemoryBackend],
    ) -> None:
        """Initialize composite backend with routing rules.
        
        Args:
            default: Default backend for paths that don't match any route.
            routes: Dict mapping path prefixes to backends. Keys should include
                   trailing slash (e.g., "/memories/").
        """
        self.default = default
        self.routes = routes
        
        # Sort routes by length (longest first) for correct prefix matching
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)
    
    @property
    def uses_state(self) -> bool:
        """True if any sub-backend uses state.
        
        This is important for tools to know whether to expect Command returns.
        """
        if getattr(self.default, "uses_state", False):
            return True
        return any(getattr(backend, "uses_state", False) for backend in self.routes.values())
    
    def _get_backend_and_key(self, key: str) -> tuple[MemoryBackend, str]:
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
    
    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get file, routing to appropriate backend.
        
        Args:
            key: File path (e.g., "/memories/notes.txt")
        
        Returns:
            FileData dict or None if not found.
        """
        backend, stripped_key = self._get_backend_and_key(key)
        return backend.get(stripped_key)
    
    def put(self, key: str, value: dict[str, Any]) -> Any:
        """Store file, routing to appropriate backend.
        
        Args:
            key: File path
            value: FileData dict
        
        Returns:
            Return value from backend (None, Command, or str).
        """
        backend, stripped_key = self._get_backend_and_key(key)
        return backend.put(stripped_key, value)
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List files from all backends, with appropriate prefixes.
        
        Args:
            prefix: Optional path prefix to filter results.
        
        Returns:
            List of file paths with route prefixes added.
        """
        result: list[str] = []
        
        # If prefix matches a route, only query that backend
        if prefix is not None:
            for route_prefix, backend in self.sorted_routes:
                if prefix.startswith(route_prefix):
                    # Strip route prefix from search prefix
                    search_prefix = prefix[len(route_prefix) - 1:]
                    keys = backend.ls(search_prefix if search_prefix != "/" else None)
                    # Add route prefix back to results
                    result.extend(f"{route_prefix[:-1]}{key}" for key in keys)
                    return result
        
        # Query default backend
        default_keys = self.default.ls(prefix)
        result.extend(default_keys)
        
        # Query each routed backend
        for route_prefix, backend in self.routes.items():
            # Skip if prefix filter doesn't match this route
            if prefix is not None and not route_prefix.startswith(prefix):
                continue
            
            # Get keys from backend (without prefix in search)
            keys = backend.ls(None)
            # Add route prefix to each key
            result.extend(f"{route_prefix[:-1]}{key}" for key in keys)
        
        return result
    
    def delete(self, key: str) -> Any:
        """Delete file, routing to appropriate backend.
        
        Args:
            key: File path to delete
        
        Returns:
            Return value from backend (None, Command, or str).
        """
        backend, stripped_key = self._get_backend_and_key(key)
        return backend.delete(stripped_key)
