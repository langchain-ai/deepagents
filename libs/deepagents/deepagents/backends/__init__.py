"""Memory backends for pluggable file storage."""

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.state import StateBackend
from deepagents.backends.store import (
    BackendContext,
    NamespaceFactory,
    StoreBackend,
)

__all__ = [
    "BackendContext",
    "BackendProtocol",
    "CompositeBackend",
    "FilesystemBackend",
    "LocalShellBackend",
    "NamespaceFactory",
    "StateBackend",
    "StoreBackend",
]
