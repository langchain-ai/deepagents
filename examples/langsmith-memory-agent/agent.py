"""A Deep Agent that demonstrates each kind of memory LangSmith can back.

| Memory     | Agent-facing surface             | Storage                          | Scope      |
| ---------- | -------------------------------- | -------------------------------- | ---------- |
| Short-term | Messages and scratch files       | Checkpointer (thread state)      | One thread |
| Semantic   | `/memories/user/profile.md`      | LangGraph `BaseStore`            | One user   |
| Episodic   | `search_past_conversations` tool | LangGraph `BaseStore`            | One user   |
| Procedural | `/memories/agent/AGENTS.md`      | LangSmith Context Hub (commits)  | Every user |
| Skills     | `/memories/agent/skills/`        | LangSmith Context Hub, read-only | Every user |
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    ContextHubBackend,
    StateBackend,
    StoreBackend,
)
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.config import get_config

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime
    from langgraph.store.base import BaseStore, SearchItem
    from langgraph.types import Checkpointer
    from langsmith import Client

MODEL = os.getenv("MEMORY_AGENT_MODEL", "anthropic:claude-sonnet-5-5")
CONTEXT_HUB_REPO = os.getenv("CONTEXT_HUB_REPO", "-/langsmith-memory-agent")
APP_NAMESPACE = "memory-agent"
EPISODE_RESULTS = 5
EPISODE_SCAN_LIMIT = 50
EPISODE_TEXT_LIMIT = 500

INSTRUCTIONS = """You are a helpful assistant with several kinds of memory.

## Your memory

- `/memories/agent/AGENTS.md`: conventions and procedures that apply to every user of this agent.
  It is stored in LangSmith Context Hub, so every edit becomes a reviewable commit.
- `/memories/agent/skills/`: curated skills maintained by humans in Context Hub. They are read-only to you.
- `/memories/user/profile.md`: durable facts and preferences about the current user only.
  It is stored in the LangGraph store under this user's namespace.
- `search_past_conversations`: searches summaries of your earlier conversations with the current user.
- Any other path, such as `/scratch/`: working files for this conversation only. They disappear when the conversation ends.

## How to use it

- When the user shares something lasting about themselves (name, role, preferences), update `/memories/user/profile.md`.
  Create it with `write_file` if it does not exist yet.
- When the user teaches a convention or procedure that should apply to everyone, add it to `/memories/agent/AGENTS.md`.
  Never put personal or private details there, because every user reads it.
- When the user refers to an earlier conversation, call `search_past_conversations` before answering.
- Keep memory files short. Update existing bullets instead of appending duplicates.
- Tell the user briefly when you save something to memory and where you saved it.
"""


@dataclass
class UserContext:
    """Run-scoped context identifying the person the agent is talking to."""

    user_id: str = "anonymous"


def user_namespace(runtime: Runtime[UserContext] | ToolRuntime[UserContext], kind: str) -> tuple[str, ...]:
    """Return the store namespace that holds one kind of memory for the current user."""
    context = runtime.context
    user_id = context.user_id if context is not None else "anonymous"
    return (APP_NAMESPACE, user_id, kind)


def build_backend(hub_client: Client | None = None) -> CompositeBackend:
    """Route each memory path to the storage that owns it."""
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/user/": StoreBackend(namespace=lambda rt: user_namespace(rt, "files")),
            "/memories/agent/": ContextHubBackend(CONTEXT_HUB_REPO, client=hub_client),
        },
    )


def _last_text(messages: list[AnyMessage], kind: type[AnyMessage]) -> str:
    message = next((m for m in reversed(messages) if isinstance(m, kind)), None)
    return message.text[:EPISODE_TEXT_LIMIT] if message is not None else ""


def build_episode(messages: list[AnyMessage]) -> tuple[str, dict[str, str]] | None:
    """Return the store key and value that record the turn that just finished."""
    user_message = _last_text(messages, HumanMessage)
    assistant_message = _last_text(messages, AIMessage)
    if not user_message or not assistant_message:
        return None
    thread_id = str(get_config().get("configurable", {}).get("thread_id", "local"))
    value = {
        "thread_id": thread_id,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    return f"{thread_id}:{len(messages)}", value


class EpisodeRecorder(AgentMiddleware[AgentState, UserContext]):
    """Write each finished turn to the store so later threads can search it."""

    def after_agent(self, state: AgentState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        """Record the turn with the sync store API."""
        episode = build_episode(state["messages"])
        if episode is not None and runtime.store is not None:
            runtime.store.put(user_namespace(runtime, "episodes"), *episode)
        return None

    async def aafter_agent(self, state: AgentState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        """Record the turn with the async store API."""
        episode = build_episode(state["messages"])
        if episode is not None and runtime.store is not None:
            await runtime.store.aput(user_namespace(runtime, "episodes"), *episode)
        return None


def rank_episodes(items: list[SearchItem], current_thread: str | None) -> list[SearchItem]:
    """Drop the current thread and order by relevance, or by recency without semantic search."""
    past = [item for item in items if item.value.get("thread_id") != current_thread]
    if all(item.score is None for item in past):
        past.sort(key=lambda item: item.updated_at, reverse=True)
    return past[:EPISODE_RESULTS]


@tool
def search_past_conversations(query: str, runtime: ToolRuntime[UserContext]) -> str:  # noqa: D417  # `runtime` is injected and hidden from the model
    """Search earlier conversations with the current user.

    Args:
        query: What to look for, for example "on-call handoff checklist".

    """
    if runtime.store is None:
        return "No store is configured, so past conversations are unavailable."
    items = runtime.store.search(user_namespace(runtime, "episodes"), query=query, limit=EPISODE_SCAN_LIMIT)
    current_thread = runtime.config.get("configurable", {}).get("thread_id")
    episodes = rank_episodes(items, current_thread)
    if not episodes:
        return "No earlier conversations found."
    return "\n\n".join(
        f"[{item.value['recorded_at']}] User: {item.value['user_message']}\nAssistant: {item.value['assistant_message']}" for item in episodes
    )


def build_agent(
    *,
    model: str | BaseChatModel = MODEL,
    hub_client: Client | None = None,
    store: BaseStore | None = None,
    checkpointer: Checkpointer | None = None,
) -> CompiledStateGraph:
    """Assemble the memory agent.

    On LangSmith Deployment and `langgraph dev`, leave `store` and `checkpointer`
    unset: the server provides both.
    """
    return create_deep_agent(
        model=model,
        system_prompt=INSTRUCTIONS,
        tools=[search_past_conversations],
        memory=["/memories/agent/AGENTS.md", "/memories/user/profile.md"],
        skills=["/memories/agent/skills/"],
        permissions=[
            FilesystemPermission(operations=["write"], paths=["/memories/agent/skills/**"], mode="deny"),
        ],
        backend=build_backend(hub_client),
        middleware=[EpisodeRecorder()],
        context_schema=UserContext,
        store=store,
        checkpointer=checkpointer,
        name="memory-agent",
    )


def make_agent() -> CompiledStateGraph:
    """Graph factory for Agent Server, which calls it for every run.

    A fresh `ContextHubBackend` per run means each run reads the latest Context
    Hub commit, including edits made in the LangSmith UI.
    """
    return build_agent()
