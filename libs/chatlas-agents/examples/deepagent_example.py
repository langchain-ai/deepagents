"""Example of using ChATLAS agents with DeepAgents framework."""

import asyncio
import os
from chatlas_agents.config import AgentConfig, LLMConfig, LLMProvider, MCPServerConfig
from chatlas_agents.agents import create_deep_agent


async def main():
    """Run DeepAgent example."""
    # Get API key from environment or use placeholder
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    # Create configuration
    config = AgentConfig(
        name="chatlas-deepagent",
        description="ChATLAS agent with DeepAgents framework capabilities",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            api_key=api_key,  # Set via CHATLAS_LLM_API_KEY environment variable
        ),
        mcp=MCPServerConfig(
            url="https://chatlas-mcp.app.cern.ch/mcp",
        ),
        verbose=True,
    )

    # Create and initialize DeepAgent
    # DeepAgents come with built-in capabilities:
    # - Planning: TODO list management for complex tasks
    # - File system: Mock file system operations
    # - Sub-agents: Spawn isolated sub-agents for subtasks
    # - Plus any custom tools from MCP server
    agent = await create_deep_agent(config)

    try:
        # Run queries with conversation persistence
        thread_id = "example-session"
        
        queries = [
            "Hello, can you introduce yourself and list your capabilities?",
            "Can you help me break down a complex research task into steps?",
        ]

        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            # Run with conversation memory
            result = await agent.run(query, thread_id=thread_id)
            print(f"\nAgent: {result.get('output', result)}\n")
    finally:
        await agent.close()


async def streaming_example():
    """Example with streaming responses."""
    # Get API key from environment
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    config = AgentConfig(
        name="streaming-deepagent",
        description="DeepAgent with streaming capabilities",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key=api_key,  # Set via CHATLAS_LLM_API_KEY environment variable
            streaming=True,
        ),
    )

    agent = await create_deep_agent(config)

    try:
        print("\n" + "="*60)
        print("Streaming Example with DeepAgent")
        print("="*60 + "\n")
        
        print("Agent: ", end="", flush=True)
        async for event in agent.stream("Tell me about DeepAgents features", thread_id="stream-session"):
            # Process streaming events
            if isinstance(event, dict):
                for node, data in event.items():
                    if "messages" in data:
                        for msg in data["messages"]:
                            if hasattr(msg, "content"):
                                print(msg.content, end="", flush=True)
        print("\n")
    finally:
        await agent.close()


async def complex_task_example():
    """Example demonstrating DeepAgent's planning capabilities."""
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    config = AgentConfig(
        name="planning-deepagent",
        description="DeepAgent for complex multi-step tasks with planning",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key=api_key,
        ),
    )

    agent = await create_deep_agent(config)

    try:
        print("\n" + "="*60)
        print("Complex Task Planning Example")
        print("="*60 + "\n")
        
        # DeepAgents can break down complex tasks and maintain a TODO list
        complex_task = """
        Research the latest developments in quantum computing and:
        1. Identify the top 3 breakthrough papers from 2024
        2. Summarize their key findings
        3. Write a brief report comparing their approaches
        """
        
        result = await agent.run(complex_task, thread_id="planning-session")
        print(f"\nAgent response:\n{result.get('output', result)}\n")
        
    finally:
        await agent.close()


if __name__ == "__main__":
    # Run basic example
    asyncio.run(main())
    
    # Uncomment to run streaming example
    # asyncio.run(streaming_example())
    
    # Uncomment to run complex task example
    # asyncio.run(complex_task_example())
