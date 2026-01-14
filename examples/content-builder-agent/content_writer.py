#!/usr/bin/env python3
"""
Content Builder Agent

A content writer agent configured entirely through files on disk:
- AGENTS.md defines brand voice and style guide
- skills/ provides specialized workflows (blog posts, social media)
- subagents handle research and other delegated tasks

Usage:
    uv run python content_writer.py "Write a blog post about AI agents"
    uv run python content_writer.py "Create a LinkedIn post about prompt engineering"
"""

import asyncio
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Get the directory containing this script
EXAMPLE_DIR = Path(__file__).parent


def create_content_writer():
    """Create a content writer agent configured by filesystem files.

    The agent's behavior is defined by:
    - AGENTS.md: Brand voice, style guide, writing standards
    - skills/: Specialized workflows for different content types
    - subagents: Task delegation (e.g., research before writing)
    """
    return create_deep_agent(
        # AGENTS.md provides persistent context - always loaded into system prompt
        # This defines the agent's "personality" and standards
        memory=["./AGENTS.md"],
        # Skills are loaded on-demand when the agent needs them
        # Each skill provides a specialized workflow
        skills=["./skills/"],
        # Subagents handle delegated tasks
        # The main agent can spawn these for specific work
        subagents=[
            {
                "name": "researcher",
                "description": "Research a topic thoroughly before writing. Use this to gather sources, statistics, and background information.",
                "system_prompt": """You are a research assistant. Your job is to:
1. Search for authoritative sources on the given topic
2. Gather key statistics, quotes, and examples
3. Identify recent developments or trends
4. Save your findings to a markdown file

Be thorough but focused. Aim for 3-5 high-quality sources rather than many low-quality ones.
Always cite your sources with URLs.""",
            }
        ],
        # FilesystemBackend reads files from disk relative to root_dir
        backend=FilesystemBackend(root_dir=EXAMPLE_DIR),
    )


async def main():
    """Run the content writer agent."""
    # Get task from command line or use default
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "Write a blog post about how AI agents are transforming software development"

    print(f"Task: {task}\n")
    print("Creating content writer agent...")
    print(f"  - Loading AGENTS.md from: {EXAMPLE_DIR / 'AGENTS.md'}")
    print(f"  - Loading skills from: {EXAMPLE_DIR / 'skills'}")
    print()

    agent = create_content_writer()

    # Run the agent
    result = await agent.ainvoke(
        {"messages": [("user", task)]},
        config={"configurable": {"thread_id": "content-writer-demo"}},
    )

    # Print the final response
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60 + "\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
