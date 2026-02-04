"""Memory backends for pluggable file storage."""

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.state import StateBackend
from deepagents.backends.store import (
    BackendContext,
    ContextT,
    NamespaceFactory,
    StateT,
    StoreBackend,
)

__all__ = [
    "BackendContext",
    "BackendProtocol",
    "CompositeBackend",
    "ContextT",
    "FilesystemBackend",
    "LocalShellBackend",
    "NamespaceFactory",
    "StateBackend",
    "StateT",
    "StoreBackend",
]
