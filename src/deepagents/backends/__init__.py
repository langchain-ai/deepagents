"""Memory backends for pluggable file storage."""

from deepagents.backends.composite import CompositeBackend, CompositeStateBackendProvider
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend, StateBackendProvider
from deepagents.backends.store import StoreBackend, StoreBackendProvider
from deepagents.backends.protocol import BackendProtocol

__all__ = [
    "BackendProtocol",
    "CompositeBackend",
    "CompositeStateBackendProvider",
    "FilesystemBackend",
    "StateBackend",
    "StateBackendProvider",
    "StoreBackend",
    "StoreBackendProvider",
]
