"""Deep agent configured as a living knowledge base with Context Hub memories.

This graph mounts `/memories/` to Context Hub (durable) while keeping the default
state backend thread-scoped.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, ContextHubBackend, StateBackend

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
DEFAULT_MEMORIES_HUB_IDENTIFIER = "-/my-agent"


def resolve_model_name() -> str:
    """Return model id from env, falling back to a stable default."""
    return os.getenv("DEEPAGENT_MODEL", DEFAULT_MODEL)


def resolve_memories_identifier() -> str:
    """Return Context Hub identifier from env or default.

    Identifier format must be `owner/repo` or `-/repo`.
    """
    return os.getenv("MEMORIES_HUB_IDENTIFIER", DEFAULT_MEMORIES_HUB_IDENTIFIER)


def build_agent() -> CompiledStateGraph:
    """Create a deep agent with durable `/memories/` in Context Hub."""
    memories_identifier = resolve_memories_identifier()

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": ContextHubBackend(memories_identifier),
        },
    )

    return create_deep_agent(
        model=init_chat_model(model=resolve_model_name(), temperature=0),
        backend=backend,
    )


agent = build_agent()
