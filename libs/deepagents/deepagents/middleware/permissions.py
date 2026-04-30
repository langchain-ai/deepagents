"""Backward-compatible re-export for filesystem permissions."""

from deepagents.middleware.filesystem import FilesystemPermission as FilePermission  # Re-exported for backwards compatibility.

__all__ = ["FilePermission"]
