"""Meta-Agent Builder - MVP Example

This example demonstrates the Meta-Agent Builder system with
Documentation and Architecture specialists.

Requirements:
- pip install -r requirements.txt
- ANTHROPIC_API_KEY in environment
- TAVILY_API_KEY in environment
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_agent_builder.orchestrator import MetaOrchestrator


async def main():
    """Run the Meta-Agent Builder example."""

    print("🧠 META-AGENT BUILDER - MVP Example\n")
    print("=" * 80)

    # Check API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY not set (search may fail)")
        print("   Set it with: export TAVILY_API_KEY='your-key-here'\n")

    # Project request
    user_request = """
    Create a research agent system that:
    - Takes a research query from the user
    - Searches the web for relevant information
    - Analyzes and synthesizes findings
    - Generates a comprehensive research report

    The system should use Deep Agents with multiple specialists
    for different aspects of research (search, analysis, writing).
    """

    print("\n📝 User Request:")
    print(user_request)
    print("\n" + "=" * 80 + "\n")

    # Create orchestrator
    print("🚀 Initializing Meta-Orchestrator...\n")
    orchestrator = MetaOrchestrator()

    # Generate specifications
    print("⚙️  Generating Specifications...\n")
    print("-" * 80 + "\n")

    async for event in orchestrator.process_project_request(user_request):
        if "messages" in event:
            last_message = event["messages"][-1]

            # Print AI messages
            if hasattr(last_message, "content") and last_message.content:
                print(f"🤖 {last_message.content}\n")
                print("-" * 80 + "\n")

    print("\n" + "=" * 80)
    print("\n✅ Specification generation complete!")
    print("\nGenerated specifications are in the virtual filesystem:")
    print("- /project_specs/project_brief.md")
    print("- /project_specs/architecture/architecture.md")
    print("- /project_specs/architecture/agents_hierarchy.md")
    print("- /project_specs/architecture/data_flows.md")
    print("- /project_specs/executive_summary.md")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
