"""Persistence checks for file backends.

Utilities to determine whether a [`BackendProtocol`][deepagents.backends.protocol.BackendProtocol]
writes files to the real filesystem (visible to external processes like a test
verifier) or stores them in ephemeral, in-process state that vanishes when the
graph execution ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES


def persists_to_filesystem(backend: BACKEND_TYPES) -> bool:
    """Return ``True`` if *backend* writes files to the real filesystem.

    A backend "persists to the filesystem" when file writes reach the actual
    disk of the host environment, making them visible to external processes
    (e.g. a test verifier that runs ``test.sh`` against the working directory
    after the agent finishes).

    Backends that store files in LangGraph state channels
    ([`StateBackend`][deepagents.backends.state.StateBackend]), LangGraph's
    [`BaseStore`][langgraph.store.base.BaseStore]
    ([`StoreBackend`][deepagents.backends.store.StoreBackend]), or remote
    services ([`ContextHubBackend`][deepagents.backends.context_hub.ContextHubBackend],
    sandbox backends) do **not** persist to the local filesystem and return
    ``False``.

    For [`CompositeBackend`][deepagents.backends.composite.CompositeBackend],
    returns ``True`` only when **every** route (including the default backend)
    persists to the filesystem. If any route is ephemeral, files written to
    that route's paths will not reach disk, so the composite as a whole is
    considered non-persistent.

    Backend factories (callables that return a backend at runtime) cannot be
    checked at compile time and return ``False``.

    Args:
        backend: The backend to check — either a resolved instance or a
            factory callable. May be a single backend or a
            [`CompositeBackend`][deepagents.backends.composite.CompositeBackend]
            with nested routes.

    Returns:
        ``True`` if all file writes reach the real filesystem, ``False``
        otherwise.
    """
    if isinstance(backend, CompositeBackend):
        return persists_to_filesystem(backend.default) and all(
            persists_to_filesystem(route_backend) for _prefix, route_backend in backend.sorted_routes
        )
    return isinstance(backend, (FilesystemBackend, LocalShellBackend))
