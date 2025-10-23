"""Concrete implementations of MemoryBackend protocol."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol


class StateAccessor(Protocol):
    """Protocol for accessing agent state in a backend."""
    
    def get_files(self) -> dict[str, dict[str, Any]]:
        """Get the current files dictionary from state."""
        ...
    
    def update_files(self, updates: dict[str, dict[str, Any] | None]) -> None:
        """Update files in state. None values indicate deletion."""
        ...


class StateBackend:
    """Backend that stores files in agent state/checkpointer.
    
    This backend is READ-ONLY from the backend's perspective. Writes must
    be handled by the tools returning Command objects to update state.
    
    This maintains backward compatibility with the original state-based
    virtual filesystem while working with the pluggable backend system.
    """
    
    def __init__(self, state_accessor: StateAccessor):
        """Initialize the state backend.
        
        Args:
            state_accessor: Object that provides read access to state
        """
        self.state_accessor = state_accessor
        self.uses_state = True  # Flag for tools to know to use Commands
    
    def get(self, key: str) -> dict[str, Any] | None:
        files = self.state_accessor.get_files()
        return files.get(key)
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        # This is a no-op - actual writes happen via Command objects
        # The tools check uses_state flag and handle this appropriately
        pass
    
    def ls(self) -> list[str]:
        files = self.state_accessor.get_files()
        return list(files.keys())


class StoreBackend:
    """Adapter for LangGraph's BaseStore to MemoryBackend protocol.
    
    This enables backward compatibility with existing code that uses
    LangGraph's store infrastructure.
    """
    
    def __init__(self, store: Any, agent_id: str | None = None):
        """Initialize the store backend adapter.
        
        Args:
            store: LangGraph BaseStore instance
            agent_id: Optional agent ID for multi-tenancy isolation
        """
        self.store = store
        self.namespace = (agent_id, "filesystem") if agent_id else ("filesystem",)
    
    def get(self, key: str) -> dict[str, Any] | None:
        item = self.store.get(self.namespace, key)
        return item.value if item else None
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        self.store.put(self.namespace, key, value)
    
    def ls(self) -> list[str]:
        items = self.store.search(self.namespace)
        return [item.key for item in items]


class FilesystemBackend:
    """Store memory in local filesystem as JSON files.
    
    Files are stored in a directory structure matching the key paths.
    For example, key "/agent.md" becomes "<base_path>/agent.md.json"
    """
    
    def __init__(self, base_path: Path | str = "~/.deepagents/memory", agent_id: str | None = None):
        """Initialize the filesystem backend.
        
        Args:
            base_path: Base directory for storing memory files
            agent_id: Optional agent ID for multi-tenancy isolation.
                     If provided, files are stored in <base_path>/<agent_id>/
        """
        self.base_path = Path(base_path).expanduser()
        if agent_id:
            self.base_path = self.base_path / agent_id
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        """Convert key to filesystem path."""
        # Remove leading slash and convert to Path
        key_path = key.lstrip("/")
        file_path = self.base_path / f"{key_path}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path
    
    def get(self, key: str) -> dict[str, Any] | None:
        path = self._get_path(key)
        if not path.exists():
            return None
        with path.open("r") as f:
            return json.load(f)
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._get_path(key)
        with path.open("w") as f:
            json.dump(value, f, indent=2)
    
    def ls(self) -> list[str]:
        keys = []
        for json_file in self.base_path.rglob("*.json"):
            # Convert path back to key format
            rel_path = json_file.relative_to(self.base_path)
            key = "/" + str(rel_path.with_suffix("")).replace("\\", "/")
            keys.append(key)
        return sorted(keys)


class SQLiteBackend:
    """Store memory in SQLite database.
    
    All agents can share the same database file, with isolation
    provided by the agent_id column.
    """
    
    def __init__(self, db_path: Path | str = "~/.deepagents/memory.db", agent_id: str | None = None):
        """Initialize the SQLite backend.
        
        Args:
            db_path: Path to SQLite database file
            agent_id: Optional agent ID for multi-tenancy isolation
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id or "default"
        self._init_db()
    
    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    agent_id TEXT,
                    key TEXT,
                    value TEXT,
                    PRIMARY KEY (agent_id, key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_key 
                ON memory(agent_id, key)
            """)
    
    def get(self, key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM memory WHERE agent_id = ? AND key = ?",
                (self.agent_id, key)
            )
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory (agent_id, key, value) VALUES (?, ?, ?)",
                (self.agent_id, key, json.dumps(value))
            )
    
    def ls(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT key FROM memory WHERE agent_id = ? ORDER BY key",
                (self.agent_id,)
            )
            return [row[0] for row in cursor.fetchall()]


class CompositeBackend:
    """Composite backend that routes between virtual and long-term storage.
    
    Routes file operations based on path prefix:
    - Paths starting with prefix go to long_term_backend
    - All other paths go to virtual_backend
    
    This enables the common pattern of having ephemeral files in state
    and persistent files in a separate backend.
    
    Important: If virtual_backend is a StateBackend, this backend will also
    have uses_state=True to indicate tools should use Command objects for writes.
    """
    
    def __init__(
        self,
        virtual_backend: Any,
        long_term_backend: Any,
        long_term_prefix: str = "/memories/"
    ):
        """Initialize the composite backend.
        
        Args:
            virtual_backend: Backend for ephemeral files (e.g., StateBackend)
            long_term_backend: Backend for persistent files (e.g., FilesystemBackend)
            long_term_prefix: Path prefix that routes to long_term_backend
        """
        self.virtual_backend = virtual_backend
        self.long_term_backend = long_term_backend
        self.long_term_prefix = long_term_prefix
        # Inherit uses_state flag from virtual backend if it has one
        self.uses_state = getattr(virtual_backend, 'uses_state', False)
    
    def _is_long_term(self, key: str) -> bool:
        """Check if key should be routed to long-term backend."""
        return key.startswith(self.long_term_prefix)
    
    def _strip_prefix(self, key: str) -> str:
        """Remove long-term prefix from key."""
        if key.startswith(self.long_term_prefix):
            return key[len(self.long_term_prefix) - 1:]  # Keep leading slash
        return key
    
    def _add_prefix(self, key: str) -> str:
        """Add long-term prefix to key."""
        return f"{self.long_term_prefix.rstrip('/')}{key}"
    
    def get(self, key: str) -> dict[str, Any] | None:
        if self._is_long_term(key):
            return self.long_term_backend.get(self._strip_prefix(key))
        return self.virtual_backend.get(key)
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        if self._is_long_term(key):
            self.long_term_backend.put(self._strip_prefix(key), value)
        else:
            self.virtual_backend.put(key, value)
    
    def get_backend_for_key(self, key: str) -> Any:
        """Get the actual backend that handles this key."""
        if self._is_long_term(key):
            return self.long_term_backend
        return self.virtual_backend
    
    def ls(self) -> list[str]:
        virtual_files = self.virtual_backend.ls()
        long_term_files = [self._add_prefix(k) for k in self.long_term_backend.ls()]
        return virtual_files + long_term_files
