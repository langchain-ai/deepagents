from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from langgraph.store.memory import InMemoryStore

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from tests.evals.utils import TrajectoryExpectations, run_agent

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime


@pytest.mark.langsmith
def test_memory_middleware_composite_backend(model: str) -> None:
    """Test that agent can access memory from store backend via composite backend routing."""
    store = InMemoryStore()
    now = datetime.now(UTC).isoformat()
    store.put(
        ("filesystem",),
        "/AGENTS.md",
        {
            "content": ["Your name is Jackson"],
            "created_at": now,
            "modified_at": now,
        },
    )

    def sample_backend(rt: ToolRuntime) -> CompositeBackend:
        return CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/memories/": StoreBackend(rt),
            },
        )

    agent = create_deep_agent(
        model=model,
        backend=sample_backend,
        memory=["/memories/AGENTS.md"],
        store=store,
    )

    # Agent should be able to answer based on memory file
    run_agent(
        agent,
        model=model,
        query="What is your name?",
        expect=TrajectoryExpectations(
            num_agent_steps=1,
            num_tool_call_requests=0,
        ).require_final_text_contains("Jackson"),
    )
