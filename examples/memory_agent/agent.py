"""Memory Agent — a deep agent that improves over time.

Demonstrates:
- Global system prompt learned across all users (stored in shared namespace)
- Per-user system prompt with personalized context (stored in user namespace)
- LangGraph Store for persistent cross-thread memory
- Checkpointer for conversation continuity

Deploy with: `langgraph up` or run standalone with `python agent.py`
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from memory_agent.prompts import (
    AGENT_INSTRUCTIONS,
    GLOBAL_MEMORY_PROMPT,
    USER_MEMORY_PROMPT,
)

GLOBAL_MEMORY_NAMESPACE = ("memory", "global")
USER_MEMORY_NAMESPACE_PREFIX = ("memory", "users")
MEMORY_KEY = "system_prompt"


def _load_memory(store: BaseStore, namespace: tuple[str, ...], key: str) -> str:
    """Load a memory document from the store, returning empty string if not found."""
    item = store.get(namespace, key)
    if item is None:
        return ""
    return item.value.get("content", "")


def _build_system_prompt(
    store: BaseStore,
    user_id: str,
) -> str:
    """Build the full system prompt with global + user memory injected."""
    global_memory = _load_memory(store, GLOBAL_MEMORY_NAMESPACE, MEMORY_KEY)
    user_namespace = (*USER_MEMORY_NAMESPACE_PREFIX, user_id)
    user_memory = _load_memory(store, user_namespace, MEMORY_KEY)

    parts = [AGENT_INSTRUCTIONS]

    if global_memory:
        parts.append(GLOBAL_MEMORY_PROMPT.format(global_memory=global_memory))
    if user_memory:
        parts.append(USER_MEMORY_PROMPT.format(user_memory=user_memory))

    return "\n\n".join(parts)


def create_memory_agent(
    *,
    model: str = "anthropic:claude-sonnet-4-6",
    store: BaseStore | None = None,
    user_id: str = "default",
    tools: list[Any] | None = None,
) -> Any:
    """Create a memory-enhanced deep agent.

    Args:
        model: Model identifier in provider:model format.
        store: LangGraph BaseStore for persistent memory. Defaults to InMemoryStore.
        user_id: Identifier for per-user memory namespace.
        tools: Additional tools to give the agent.
    """
    if store is None:
        store = InMemoryStore()

    system_prompt = _build_system_prompt(store, user_id)
    chat_model = init_chat_model(model)
    checkpointer = MemorySaver()

    return create_deep_agent(
        model=chat_model,
        tools=tools or [],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        store=store,
    )


# Default agent instance for LangGraph deployment.
# In production, the store would be backed by a database (e.g., Postgres via
# langgraph-checkpoint-postgres). For local dev, InMemoryStore is sufficient.
store = InMemoryStore()
model = init_chat_model("anthropic:claude-sonnet-4-6")
checkpointer = MemorySaver()

agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt=AGENT_INSTRUCTIONS,
    checkpointer=checkpointer,
    store=store,
)
