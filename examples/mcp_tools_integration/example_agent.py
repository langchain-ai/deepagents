"""Example: Using MCP tools with DeepAgents.

This example demonstrates how to create a DeepAgents agent with custom MCP tools
for Google search, RAG, weather forecast, and Sentinel satellite search.

Usage:
    # Set environment variables first
    export OPENWEATHER_API_KEY=your_api_key
    export OPENAI_API_KEY=your_api_key  # or ANTHROPIC_API_KEY

    # Run the example
    python example_agent.py
"""

import asyncio
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from mcp_tools import (
    ALL_MCP_TOOLS,
    google_search_and_summarize,
    rag_search,
    sentinel_search,
    weather_forecast,
)


def create_mcp_agent(
    model: str | None = None,
    working_dir: str | None = None,
):
    """Create a DeepAgents agent with MCP tools.

    Args:
        model: Model name (e.g., "openai:gpt-4o", "anthropic:claude-sonnet-4-5-20250929")
        working_dir: Working directory for file operations

    Returns:
        Compiled agent graph
    """
    working_dir = working_dir or str(Path.cwd())

    # Create backend for file operations
    backend = FilesystemBackend(root=working_dir)

    # System prompt with tool usage instructions
    system_prompt = """You are a helpful AI assistant with access to multiple specialized tools.

## Available Tools

1. **google_search_and_summarize**: Search the web for current information
   - Use for: news, current events, general web research
   - Returns: Search results with page contents

2. **rag_search**: Search local document/paper database
   - Use for: research papers, technical documentation, indexed content
   - Returns: Relevant document chunks with metadata

3. **weather_forecast**: Get weather forecasts
   - Use for: weather queries, outdoor planning, satellite imaging feasibility
   - Returns: 5-day weather forecast data

4. **sentinel_search**: Search for satellite imagery
   - Use for: Sentinel-1/2/3 satellite scene queries
   - Returns: Available satellite scenes with metadata

## Usage Guidelines

- For weather + satellite queries, check weather first to determine optical vs SAR imaging
- Combine tools when needed (e.g., weather + sentinel for imaging planning)
- Always summarize results in natural Korean language
- Cite sources when using google_search results
- If information is missing, ask clarifying questions

## Response Format

- Answer in Korean (unless asked otherwise)
- Be concise but thorough
- Include relevant citations/references
"""

    # Create agent with MCP tools
    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=ALL_MCP_TOOLS,  # Register all MCP tools
        backend=backend,
    )

    return agent


async def run_example():
    """Run example queries with the MCP agent."""
    # Create agent
    agent = create_mcp_agent(
        model="openai:gpt-4o",  # or "anthropic:claude-sonnet-4-5-20250929"
    )

    # Example queries
    queries = [
        "서울 내일 날씨 어때?",
        "트럼프 임기 기간을 알려줘",
        "이번달 대전 유성구를 찍은 위성영상이 있어?",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        # Invoke agent
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": "example-thread"}},
        )

        # Print response
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            print(f"\nResponse:\n{last_message.content}")


async def run_interactive():
    """Run interactive chat with the MCP agent."""
    agent = create_mcp_agent(
        model="openai:gpt-4o",
    )

    print("MCP Tools Agent - Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 40)

    thread_id = "interactive-session"

    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"configurable": {"thread_id": thread_id}},
            )

            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                print(f"\nAssistant: {last_message.content}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Tools Agent Example")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to run"
    )

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(run_interactive())
    elif args.query:
        async def single_query():
            agent = create_mcp_agent(model="openai:gpt-4o")
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": args.query}]},
                config={"configurable": {"thread_id": "single-query"}},
            )
            if "messages" in result and result["messages"]:
                print(result["messages"][-1].content)

        asyncio.run(single_query())
    else:
        asyncio.run(run_example())


if __name__ == "__main__":
    main()
