"""Architecture Specialist - Multi-agent system architect."""

from meta_agent_builder.specialists.base import BaseSpecialist
from meta_agent_builder.tools import (
    create_mermaid_diagram,
    suggest_middleware_stack,
    validate_agent_hierarchy,
)


class ArchitectureSpecialist(BaseSpecialist):
    """Specialist for designing multi-agent architectures.

    This specialist designs:
    - Agent hierarchies (orchestrator + specialists)
    - Tool distribution across agents
    - Middleware stack configuration
    - Backend and storage strategies
    - Communication and orchestration patterns
    - Context engineering approaches

    Outputs complete architecture specifications with diagrams.
    """

    def __init__(self):
        """Initialize the Architecture Specialist."""
        super().__init__(
            name="architecture-specialist",
            description=(
                "Expert in designing multi-agent architectures using Deep Agents. "
                "Use this specialist to design agent hierarchies, determine tool "
                "distribution, design middleware stacks, select backend strategies, "
                "plan communication patterns, and design context management approaches. "
                "Creates detailed architecture specifications with diagrams."
            ),
            prompt_file="architecture_specialist_prompt.md",
            tools=[
                create_mermaid_diagram,
                validate_agent_hierarchy,
                suggest_middleware_stack,
            ],
            model="claude-sonnet-4-5-20250929",
        )
