"""Backend configuration for Meta-Agent Builder.

This module provides the CompositeBackend configuration with routing
for different storage zones (memories, docs, templates, etc.).
"""

from typing import Optional

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.base import BaseStore


def create_meta_agent_backend(store: Optional[BaseStore] = None) -> CompositeBackend:
    """Create the composite backend for Meta-Agent Builder.

    This backend routes different path prefixes to appropriate storage backends:
    - /memories/ -> StoreBackend (persistent knowledge base)
    - /docs/ -> StoreBackend (cached documentation)
    - /templates/ -> StoreBackend (reusable project templates)
    - /project_specs/ -> StateBackend (current project outputs)
    - /validation/ -> StateBackend (validation artifacts)
    - default -> StateBackend (temporary scratch space)

    Args:
        store: Optional LangGraph BaseStore instance for persistent storage.
               If None, falls back to StateBackend for all routes.

    Returns:
        CompositeBackend configured with appropriate routing.

    Example:
        >>> from langgraph.store.memory import InMemoryStore
        >>> store = InMemoryStore()
        >>> backend = create_meta_agent_backend(store)
        >>> # Now use backend with create_deep_agent
    """
    # Determine backends based on store availability
    if store is not None:
        memory_backend = StoreBackend()
        docs_backend = StoreBackend()
        templates_backend = StoreBackend()
    else:
        # Fallback to ephemeral storage if no store provided
        memory_backend = StateBackend()
        docs_backend = StateBackend()
        templates_backend = StateBackend()

    return CompositeBackend(
        default=StateBackend(),  # Ephemeral default for scratch space
        routes={
            # Persistent knowledge base (agent learnings)
            "/memories/": memory_backend,
            # Cached documentation
            "/docs/": docs_backend,
            # Reusable project templates
            "/templates/": templates_backend,
            # Current project specifications (ephemeral)
            "/project_specs/": StateBackend(),
            # Validation artifacts (ephemeral)
            "/validation/": StateBackend(),
        },
    )


def create_backend_with_sandbox(
    store: Optional[BaseStore] = None,
    sandbox_backend=None,
) -> CompositeBackend:
    """Create backend with optional sandbox for code execution.

    Args:
        store: Optional persistent store
        sandbox_backend: Optional SandboxBackend instance for code execution

    Returns:
        CompositeBackend with sandbox as default if provided

    Example:
        >>> from deepagents.backends.sandbox import SandboxBackend
        >>> sandbox = SandboxBackend()  # Your sandbox implementation
        >>> backend = create_backend_with_sandbox(store, sandbox)
    """
    # Determine storage backends
    if store is not None:
        memory_backend = StoreBackend()
        docs_backend = StoreBackend()
        templates_backend = StoreBackend()
    else:
        memory_backend = StateBackend()
        docs_backend = StateBackend()
        templates_backend = StateBackend()

    # Use sandbox as default if provided, otherwise StateBackend
    default_backend = sandbox_backend if sandbox_backend is not None else StateBackend()

    return CompositeBackend(
        default=default_backend,
        routes={
            "/memories/": memory_backend,
            "/docs/": docs_backend,
            "/templates/": templates_backend,
            "/project_specs/": StateBackend(),
            "/validation/": StateBackend(),
        },
    )
