"""SQLiteBackend: Store files in SQLite database."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


class SQLiteBackend:
    """Backend that stores files in a SQLite database.
    
    Creates a table with columns:
    - key (TEXT PRIMARY KEY): File path
    - content (TEXT): JSON-serialized list of lines
    - created_at (TEXT): ISO format timestamp
    - modified_at (TEXT): ISO format timestamp
    - agent_id (TEXT): Optional agent identifier for isolation
    
    The database file is created automatically if it doesn't exist.
    """
    
    def __init__(self, db_path: str, agent_id: Optional[str] = None) -> None:
        """Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database file.
            agent_id: Optional agent ID for isolation. When provided, only files
                     belonging to this agent are accessible.
        """
        self.db_path = Path(db_path)
        self.agent_id = agent_id
        
        # Create parent directories if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    @property
    def uses_state(self) -> bool:
        """False for SQLiteBackend - stores directly to database."""
        return False
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    key TEXT NOT NULL,
                    agent_id TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    PRIMARY KEY (key, agent_id)
                )
            """)
            
            # Create index for faster ls operations
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_agent_key 
                ON files(agent_id, key)
            """)
            
            conn.commit()
    
    def _get_agent_id(self) -> str:
        """Get agent_id for queries (empty string if None)."""
        return self.agent_id or ""
    
    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get file from database.
        
        Args:
            key: File path (e.g., "/notes.txt")
        
        Returns:
            FileData dict or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT content, created_at, modified_at FROM files WHERE key = ? AND agent_id = ?",
                (key, self._get_agent_id())
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            try:
                content = json.loads(row["content"])
                return {
                    "content": content,
                    "created_at": row["created_at"],
                    "modified_at": row["modified_at"],
                }
            except json.JSONDecodeError:
                return None
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store file to database.
        
        Args:
            key: File path (e.g., "/notes.txt")
            value: FileData dict
        """
        content_json = json.dumps(value["content"])
        agent_id = self._get_agent_id()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO files (key, agent_id, content, created_at, modified_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key, agent_id) DO UPDATE SET
                    content = excluded.content,
                    modified_at = excluded.modified_at
                """,
                (key, agent_id, content_json, value["created_at"], value["modified_at"])
            )
            conn.commit()
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List files from database.
        
        Args:
            prefix: Optional path prefix to filter results.
        
        Returns:
            List of file paths.
        """
        with sqlite3.connect(self.db_path) as conn:
            if prefix is not None:
                # Use LIKE for prefix matching (escape special chars)
                like_pattern = prefix.replace("%", "\\%").replace("_", "\\_") + "%"
                cursor = conn.execute(
                    "SELECT key FROM files WHERE agent_id = ? AND key LIKE ? ESCAPE '\\\\'",
                    (self._get_agent_id(), like_pattern)
                )
            else:
                cursor = conn.execute(
                    "SELECT key FROM files WHERE agent_id = ?",
                    (self._get_agent_id(),)
                )
            
            return sorted([row[0] for row in cursor.fetchall()])
    
    def delete(self, key: str) -> None:
        """Delete file from database.
        
        Args:
            key: File path to delete
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM files WHERE key = ? AND agent_id = ?",
                (key, self._get_agent_id())
            )
            conn.commit()
