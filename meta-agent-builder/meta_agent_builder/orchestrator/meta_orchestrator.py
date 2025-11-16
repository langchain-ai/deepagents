"""Meta-Orchestrator for coordinating specialist agents."""

import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from meta_agent_builder.backends import create_meta_agent_backend
from meta_agent_builder.specialists import (
    ArchitectureSpecialist,
    DocumentationSpecialist,
)


class MetaOrchestrator:
    """Main orchestrator for Meta-Agent Builder.

    Coordinates specialist agents to generate complete project specifications
    from user descriptions.

    The orchestrator manages:
    - Project intake and brief creation
    - Specialist coordination
    - Result aggregation
    - Final delivery

    Attributes:
        backend: CompositeBackend with routing
        specialists: List of specialist agents
        agent: Main orchestrator deep agent
    """

    def __init__(
        self,
        store: Optional[BaseStore] = None,
        checkpointer: Optional[Any] = None,
    ):
        """Initialize the Meta-Orchestrator.

        Args:
            store: Optional persistent store (defaults to InMemoryStore)
            checkpointer: Optional checkpointer (defaults to MemorySaver)
        """
        # Create backend with routing
        self.backend = create_meta_agent_backend(store or InMemoryStore())

        # Initialize specialists
        self.specialists = self._create_specialists()

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

        # Create orchestrator agent
        self.agent = create_deep_agent(
            model="claude-sonnet-4-5-20250929",
            system_prompt=self.system_prompt,
            subagents=[s.to_subagent_config() for s in self.specialists],
            backend=self.backend,
            checkpointer=checkpointer or MemorySaver(),
            store=store or InMemoryStore(),
        ).with_config({"recursion_limit": 1500})

    def _create_specialists(self) -> list:
        """Create all specialist agents.

        Returns:
            List of specialist instances
        """
        return [
            DocumentationSpecialist(),
            ArchitectureSpecialist(),
        ]

    def _load_system_prompt(self) -> str:
        """Load the orchestrator system prompt.

        Returns:
            Prompt content as string
        """
        prompt_path = (
            Path(__file__).parent.parent / "prompts" / "meta_orchestrator_prompt.md"
        )
        return prompt_path.read_text()

    async def process_project_request(
        self,
        user_request: str,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process a project request and generate specifications.

        Args:
            user_request: User's project description
            thread_id: Optional thread ID for resumption

        Yields:
            Events from the agent execution containing messages and state updates
        """
        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

        async for event in self.agent.astream(
            {"messages": [{"role": "user", "content": user_request}]},
            config=config,
            stream_mode="values",
        ):
            yield event

    def get_deliverables(self, thread_id: str) -> dict[str, Optional[str]]:
        """Extract deliverables from a completed execution.

        Args:
            thread_id: Thread ID of the execution

        Returns:
            Dictionary with paths to generated specifications
        """
        # This would read from the backend
        # For now, return structure
        return {
            "project_brief": "/project_specs/project_brief.md",
            "architecture": "/project_specs/architecture/architecture.md",
            "agents_hierarchy": "/project_specs/architecture/agents_hierarchy.md",
            "data_flows": "/project_specs/architecture/data_flows.md",
            "backend_strategy": "/project_specs/architecture/backend_strategy.md",
            "executive_summary": "/project_specs/executive_summary.md",
        }
