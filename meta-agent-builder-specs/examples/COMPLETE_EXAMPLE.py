"""
META-AGENT BUILDER - Complete Working Example

This example demonstrates a fully functional Meta-Agent Builder
that generates project specifications.

Requirements:
- pip install deepagents langchain langchain-anthropic tavily-python
- ANTHROPIC_API_KEY in environment
- TAVILY_API_KEY in environment
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from tavily import TavilyClient

# ============================================================================
# TOOLS
# ============================================================================

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))


@tool
def internet_search(query: str, max_results: int = 5) -> dict:
    """Search the web for documentation and information.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        Dictionary with search results
    """
    try:
        return tavily_client.search(
            query, max_results=max_results, include_raw_content=True
        )
    except:
        return {"error": "Search failed - check API key"}


@tool
def create_diagram(diagram_type: str, content: str, title: str) -> str:
    """Create a Mermaid diagram.

    Args:
        diagram_type: Type of diagram (graph, sequence, flowchart)
        content: Mermaid diagram syntax
        title: Diagram title

    Returns:
        Formatted Mermaid diagram
    """
    return f"""```mermaid
---
title: {title}
---
{content}
```"""


# ============================================================================
# BACKEND CONFIGURATION
# ============================================================================


def create_backend(use_persistent_store: bool = False) -> CompositeBackend:
    """Create composite backend with routing.

    Args:
        use_persistent_store: Use persistent storage (requires Store)

    Returns:
        Configured CompositeBackend
    """
    # Determine backend types
    if use_persistent_store:
        # Note: Would need actual Store instance in production
        memory_backend = StateBackend()  # Fallback to State for example
        docs_backend = StateBackend()
        templates_backend = StateBackend()
    else:
        memory_backend = StateBackend()
        docs_backend = StateBackend()
        templates_backend = StateBackend()

    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": memory_backend,
            "/docs/": docs_backend,
            "/templates/": templates_backend,
            "/project_specs/": StateBackend(),
            "/validation/": StateBackend(),
        },
    )


# ============================================================================
# SPECIALIST CONFIGURATIONS
# ============================================================================

# Simplified prompts for example
DOC_SPECIALIST_PROMPT = """You are a Deep Agents documentation expert.

When asked to research:
1. Use internet_search to find information
2. Save findings to /docs/research.md
3. Return concise summary

Focus on Deep Agents capabilities: middleware, backends, subagents, tools."""

ARCH_SPECIALIST_PROMPT = """You are a software architect specializing in Deep Agents.

When asked to design architecture:
1. Read project brief from /project_specs/project_brief.md
2. Read research from /docs/research.md
3. Design agent hierarchy
4. Design middleware stacks
5. Design backend strategy
6. Create diagrams using create_diagram tool
7. Save to /project_specs/architecture/architecture.md

Include:
- Agent hierarchy
- Tool distribution
- Middleware configuration
- Backend routing
- Communication patterns"""

PRD_SPECIALIST_PROMPT = """You are a product manager creating PRDs.

When asked to create PRD:
1. Read project brief
2. Read architecture
3. Write comprehensive PRD
4. Save to /project_specs/prd.md

Include:
- Executive summary
- User stories
- Functional requirements
- Non-functional requirements
- Technical requirements
- Acceptance criteria"""

IMPL_SPECIALIST_PROMPT = """You are an implementation specialist.

When asked to create implementation guide:
1. Read all specs in /project_specs/
2. Create detailed implementation guide
3. Generate code templates
4. Create checklist
5. Save to /project_specs/implementation/

Provide:
- Step-by-step implementation guide
- Code examples
- Configuration files
- Testing approach"""


# ============================================================================
# SPECIALIST DEFINITIONS
# ============================================================================


def create_specialists(backend: CompositeBackend) -> list[dict[str, Any]]:
    """Create all specialist subagent configurations.

    Args:
        backend: Backend to use

    Returns:
        List of SubAgent configurations
    """
    return [
        {
            "name": "documentation-specialist",
            "description": (
                "Expert in Deep Agents documentation. "
                "Use for researching Deep Agents capabilities, patterns, and best practices."
            ),
            "system_prompt": DOC_SPECIALIST_PROMPT,
            "tools": [internet_search],
            "model": "claude-sonnet-4-5-20250929",
        },
        {
            "name": "architecture-specialist",
            "description": (
                "Expert in designing multi-agent architectures. "
                "Use for designing agent hierarchies, tool distribution, and system architecture."
            ),
            "system_prompt": ARCH_SPECIALIST_PROMPT,
            "tools": [create_diagram],
            "model": "claude-sonnet-4-5-20250929",
        },
        {
            "name": "prd-specialist",
            "description": (
                "Expert in creating Product Requirements Documents. "
                "Use for writing comprehensive PRDs with requirements and specs."
            ),
            "system_prompt": PRD_SPECIALIST_PROMPT,
            "tools": [],
            "model": "claude-sonnet-4-5-20250929",
        },
        {
            "name": "implementation-specialist",
            "description": (
                "Expert in creating implementation guides and code. "
                "Use for generating implementation guides, code templates, and checklists."
            ),
            "system_prompt": IMPL_SPECIALIST_PROMPT,
            "tools": [],
            "model": "claude-sonnet-4-5-20250929",
        },
    ]


# ============================================================================
# META-ORCHESTRATOR
# ============================================================================

META_ORCHESTRATOR_PROMPT = """You are the Meta-Orchestrator for project specification generation.

## Workflow

1. **Create Project Brief**
   - Understand user request
   - Write /project_specs/project_brief.md with:
     * Project goal
     * Key requirements
     * Success criteria

2. **Invoke Documentation Specialist**
   task(subagent_type="documentation-specialist", prompt="Research Deep Agents for: [project]")

3. **Invoke Architecture Specialist**
   task(subagent_type="architecture-specialist", prompt="Design architecture for: [project]")

4. **Invoke PRD Specialist** (can run parallel with next)
   task(subagent_type="prd-specialist", prompt="Create PRD for: [project]")

5. **Invoke Implementation Specialist**
   task(subagent_type="implementation-specialist", prompt="Create implementation guide")

6. **Create Executive Summary**
   - Summarize all outputs
   - Save to /project_specs/executive_summary.md
   - Report to user

## Tools

- task: Invoke specialists
- write_todos: Track progress
- read_file, write_file: Manage files

## Remember

- Use write_todos to plan workflow
- Invoke specialists in optimal order
- Check outputs for completeness
- Provide clear, concise updates to user
"""


class MetaOrchestrator:
    """Meta-Agent Builder Orchestrator."""

    def __init__(self, use_persistent: bool = False):
        """Initialize orchestrator.

        Args:
            use_persistent: Use persistent storage
        """
        # Create backend
        self.backend = create_backend(use_persistent)

        # Create specialists
        specialists = create_specialists(self.backend)

        # Create orchestrator agent
        self.agent = create_deep_agent(
            model="claude-sonnet-4-5-20250929",
            system_prompt=META_ORCHESTRATOR_PROMPT,
            subagents=specialists,
            backend=self.backend,
            checkpointer=MemorySaver(),
            store=InMemoryStore() if use_persistent else None,
        ).with_config({"recursion_limit": 1500})

    async def generate_specs(self, user_request: str):
        """Generate project specifications.

        Args:
            user_request: User's project description

        Yields:
            Events from execution
        """
        config = {"configurable": {"thread_id": "example-session"}}

        async for event in self.agent.astream(
            {"messages": [{"role": "user", "content": user_request}]},
            config=config,
            stream_mode="values",
        ):
            if "messages" in event:
                last_message = event["messages"][-1]
                if hasattr(last_message, "content"):
                    yield last_message.content


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


async def main():
    """Main example execution."""

    print("🧠 META-AGENT BUILDER - Example\n")

    # Create orchestrator
    orchestrator = MetaOrchestrator(use_persistent=False)

    # Example project request
    user_request = """
    Create a research agent system that:
    - Takes a research query from the user
    - Searches the web for relevant information
    - Analyzes and synthesizes findings
    - Generates a comprehensive research report

    The system should use Deep Agents with multiple specialists
    for different aspects of research.
    """

    print("📝 User Request:")
    print(user_request)
    print("\n" + "=" * 80 + "\n")

    print("🚀 Generating Specifications...\n")

    # Generate specs
    async for output in orchestrator.generate_specs(user_request):
        print(output)
        print("\n" + "-" * 80 + "\n")

    print("\n✅ Specification generation complete!")
    print("\nGenerated files would be in /project_specs/ (virtual filesystem)")


# ============================================================================
# RUN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Check for API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)

    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY not set (search will fail)")
        print("   Set it with: export TAVILY_API_KEY='your-key-here'")

    # Run the example
    asyncio.run(main())
