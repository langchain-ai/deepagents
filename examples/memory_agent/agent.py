"""Memory Agent — a deep agent that improves over time.

Demonstrates:
- Global memory learned across all users (persistent via StoreBackend)
- Per-user memory with personalized context (persistent via StoreBackend)
- CompositeBackend routing /memories/ to Store, everything else to State
- Agent can read AND write memories live during conversations
- Sleep-time cron for background memory consolidation

Deploy with: `langgraph up` or `langgraph dev`
"""

from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.chat_models import init_chat_model

from memory_agent.prompts import AGENT_INSTRUCTIONS


def create_backend(runtime):
    """Create a CompositeBackend: StoreBackend for /memories/, StateBackend for the rest.

    Files under /memories/ persist across threads via LangGraph Store.
    The agent can read and edit these files at any time to update its
    own memory. Everything else is ephemeral per-thread state.
    """
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(
                runtime,
                namespace=lambda ctx: ("memories",),
            ),
        },
    )


def create_memory_agent(
    *,
    model: str = "anthropic:claude-sonnet-4-6",
    tools: list | None = None,
    enable_live_memory: bool = True,
):
    """Create a memory-enhanced deep agent.

    Args:
        model: Model identifier in provider:model format.
        tools: Additional tools to give the agent.
        enable_live_memory: If True, agent can read/write /memories/ live.
            Set to False to disable live memory editing (cron-only updates).
    """
    chat_model = init_chat_model(model)

    instructions = (
        AGENT_INSTRUCTIONS if enable_live_memory else AGENT_INSTRUCTIONS_NO_LIVE
    )
    backend = create_backend if enable_live_memory else None

    return create_deep_agent(
        model=chat_model,
        tools=tools or [],
        system_prompt=instructions,
        backend=backend,
    )


AGENT_INSTRUCTIONS_NO_LIVE = """\
You are a helpful assistant that learns and improves over time.

Your system prompt includes learned context from prior conversations,
updated during background memory consolidation. Use this context to
provide better, more personalized responses.

## Guidelines

- Reference learned context naturally — don't announce "I remember that..."
- If global memory conflicts with user memory, prefer user-specific preferences
- When uncertain about a preference, ask rather than assume
- Be concise and direct
"""

# Default agent for LangGraph deployment (langgraph up / langgraph dev).
# No checkpointer — LangGraph adds one automatically.
model = init_chat_model("anthropic:claude-sonnet-4-6")

agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt=AGENT_INSTRUCTIONS,
    backend=create_backend,
)
