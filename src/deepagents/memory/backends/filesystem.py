"""FilesystemBackend: Store files as JSON on disk."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class FilesystemBackend:
    """Backend that stores files as JSON on disk.
    
    Each file is stored as a JSON file containing the FileData structure:
    {
        "content": ["line1", "line2", ...],
        "created_at": "2024-01-01T00:00:00Z",
        "modified_at": "2024-01-01T00:00:00Z"
    }
    
    Files are organized in a directory structure that mirrors the virtual
    file paths. Optional agent_id parameter provides multi-agent isolation
    by creating separate subdirectories per agent.
    """
    
    def __init__(self, base_dir: str, agent_id: Optional[str] = None) -> None:
        """Initialize filesystem backend.
        
        Args:
            base_dir: Base directory for storing files.
            agent_id: Optional agent ID for isolation. If provided, files are
                     stored in base_dir/agent_id/ subdirectory.
        """
        self.base_dir = Path(base_dir)
        self.agent_id = agent_id
        
        # Create base directory structure
        if agent_id:
            self.root_dir = self.base_dir / agent_id
        else:
            self.root_dir = self.base_dir
        
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def uses_state(self) -> bool:
        """False for FilesystemBackend - stores directly to disk."""
        return False
    
    def _key_to_path(self, key: str) -> Path:
        """Convert virtual path to actual filesystem path.
        
        Args:
            key: Virtual file path (e.g., "/notes.txt" or "/subdir/file.txt")
        
        Returns:
            Actual filesystem path where data is stored.
        """
        # Remove leading slash and convert to path
        rel_path = key.lstrip("/")
        if not rel_path:
            rel_path = "root"
        
        # Add .json extension
        return self.root_dir / f"{rel_path}.json"
    
    def _path_to_key(self, path: Path) -> str:
        """Convert filesystem path back to virtual path.
        
        Args:
            path: Filesystem path
        
        Returns:
            Virtual file path (e.g., "/notes.txt")
        """
        # Get relative path from root_dir
        rel_path = path.relative_to(self.root_dir)
        
        # Remove .json extension
        path_str = str(rel_path)
        if path_str.endswith(".json"):
            path_str = path_str[:-5]
        
        # Handle root case
        if path_str == "root":
            return "/"
        
        # Add leading slash
        return f"/{path_str}"
    
    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get file from filesystem.
        
        Args:
            key: File path (e.g., "/notes.txt")
        
        Returns:
            FileData dict or None if not found.
        """
        file_path = self._key_to_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict):
                return None
            if "content" not in data or not isinstance(data["content"], list):
                return None
            if "created_at" not in data or not isinstance(data["created_at"], str):
                return None
            if "modified_at" not in data or not isinstance(data["modified_at"], str):
                return None
            
            return data
        except (json.JSONDecodeError, OSError):
            return None
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store file to filesystem.
        
        Args:
            key: File path (e.g., "/notes.txt")
            value: FileData dict
        """
        file_path = self._key_to_path(key)
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List files from filesystem.
        
        Args:
            prefix: Optional path prefix to filter results.
        
        Returns:
            List of file paths.
        """
        results: list[str] = []
        
        # Walk the directory tree
        for path in self.root_dir.rglob("*.json"):
            if path.is_file():
                key = self._path_to_key(path)
                
                # Filter by prefix if provided
                if prefix is None or key.startswith(prefix):
                    results.append(key)
        
        return sorted(results)
    
    def delete(self, key: str) -> None:
        """Delete file from filesystem.
        
        Args:
            key: File path to delete
        """
        file_path = self._key_to_path(key)
        
        if file_path.exists():
            file_path.unlink()
            
            # Clean up empty parent directories
            try:
                parent = file_path.parent
                while parent != self.root_dir and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
            except OSError:
                pass  # Ignore errors during cleanup
