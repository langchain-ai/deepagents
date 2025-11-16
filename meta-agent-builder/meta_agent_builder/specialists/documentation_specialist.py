"""Documentation Specialist - Deep Agents expert."""

from meta_agent_builder.specialists.base import BaseSpecialist
from meta_agent_builder.tools import (
    extract_code_examples,
    internet_search,
    summarize_documentation,
)


class DocumentationSpecialist(BaseSpecialist):
    """Specialist for researching Deep Agents documentation.

    This specialist is an expert in:
    - Deep Agents framework capabilities
    - LangChain and LangGraph patterns
    - Middleware configurations
    - Backend strategies
    - Best practices

    It maintains a persistent knowledge base in /memories/documentation/
    and saves research results to /docs/ for other specialists.
    """

    def __init__(self):
        """Initialize the Documentation Specialist."""
        super().__init__(
            name="documentation-specialist",
            description=(
                "Expert in Deep Agents framework with persistent knowledge base. "
                "Use this specialist when you need information about Deep Agents "
                "capabilities, LangChain/LangGraph patterns, middleware configurations, "
                "backend strategies, or best practices. "
                "The specialist maintains a knowledge base and learns from each query."
            ),
            prompt_file="documentation_specialist_prompt.md",
            tools=[
                internet_search,
                extract_code_examples,
                summarize_documentation,
            ],
            model="claude-sonnet-4-5-20250929",
        )
