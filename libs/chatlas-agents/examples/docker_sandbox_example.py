"""Example of using ChATLAS agents with Docker sandbox backend.

This example demonstrates two approaches:
1. Direct instantiation of DockerSandboxBackend (original approach)
2. Using create_docker_sandbox factory function (recommended for new code)
"""

import asyncio
import os
from chatlas_agents.config import AgentConfig, LLMConfig, LLMProvider
from chatlas_agents.agents import create_deep_agent
from chatlas_agents.sandbox import create_docker_sandbox


async def main_with_factory():
    """Run DeepAgent example with Docker sandbox using factory function.
    
    This is the recommended approach as it follows deepagents-cli patterns
    and provides automatic lifecycle management.
    """
    # Get API key from environment or use placeholder
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    # Create configuration
    config = AgentConfig(
        name="chatlas-sandbox-agent",
        description="ChATLAS agent with Docker sandbox for secure code execution",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            api_key=api_key,  # Set via CHATLAS_LLM_API_KEY environment variable
        ),
        verbose=True,
    )

    print("Creating DeepAgent with Docker sandbox (factory approach)...")
    
    # Use factory function with context manager for automatic cleanup
    with create_docker_sandbox(
        image="python:3.13-slim",
        working_dir="/workspace",
        auto_remove=True,
    ) as backend:
        # Create agent with the sandbox backend
        agent = await create_deep_agent(
            config,
            use_docker_sandbox=True,
            docker_image="python:3.13-slim",  # Can use any Docker image
        )

        try:
            # Run queries that may involve file operations or command execution
            thread_id = "sandbox-session"
            
            queries = [
                "Create a Python script that prints 'Hello from Docker sandbox!'",
                "Execute the Python script you just created",
                "List all files in the current directory",
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
    
    print("✓ Sandbox cleaned up automatically")


async def main():
    """Run DeepAgent example with Docker sandbox (original approach).
    
    This demonstrates direct instantiation without factory functions.
    Kept for backwards compatibility.
    """
    # Get API key from environment or use placeholder
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    # Create configuration
    config = AgentConfig(
        name="chatlas-sandbox-agent",
        description="ChATLAS agent with Docker sandbox for secure code execution",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            api_key=api_key,  # Set via CHATLAS_LLM_API_KEY environment variable
        ),
        verbose=True,
    )

    # Create and initialize DeepAgent with Docker sandbox
    # This enables the agent to execute shell commands safely in an isolated container
    print("Creating DeepAgent with Docker sandbox...")
    agent = await create_deep_agent(
        config,
        use_docker_sandbox=True,
        docker_image="python:3.13-slim",  # Can use any Docker image
    )

    try:
        # Run queries that may involve file operations or command execution
        thread_id = "sandbox-session"
        
        queries = [
            "Create a Python script that prints 'Hello from Docker sandbox!'",
            "Execute the Python script you just created",
            "List all files in the current directory",
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


async def code_execution_example():
    """Example demonstrating safe code execution in Docker sandbox."""
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    config = AgentConfig(
        name="code-execution-agent",
        description="Agent for safe code execution in isolated Docker environment",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key=api_key,
        ),
    )

    print("Creating agent with Docker sandbox for code execution...")
    agent = await create_deep_agent(config, use_docker_sandbox=True)

    try:
        print("\n" + "="*60)
        print("Code Execution Example with Docker Sandbox")
        print("="*60 + "\n")
        
        # The agent can safely execute potentially dangerous commands
        # because they run in an isolated Docker container
        task = """
        Write a Python script that:
        1. Creates a file called data.json
        2. Writes some sample JSON data to it
        3. Reads it back and prints the contents
        4. Then executes the script to show it works
        """
        
        result = await agent.run(task, thread_id="code-exec-session")
        print(f"\nAgent response:\n{result.get('output', result)}\n")
        
    finally:
        await agent.close()


async def data_analysis_example():
    """Example showing data analysis with Docker sandbox."""
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    config = AgentConfig(
        name="data-analysis-agent",
        description="Agent for data analysis with secure execution environment",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key=api_key,
        ),
    )

    print("Creating agent with Docker sandbox for data analysis...")
    # Use a data science Docker image with pre-installed packages
    agent = await create_deep_agent(
        config,
        use_docker_sandbox=True,
        docker_image="python:3.13-slim",  # Could use jupyter/datascience-notebook
    )

    try:
        print("\n" + "="*60)
        print("Data Analysis Example")
        print("="*60 + "\n")
        
        task = """
        Create a Python script that generates some random data,
        performs basic statistics (mean, median, std dev),
        and saves the results to a file. Then execute it.
        """
        
        result = await agent.run(task, thread_id="analysis-session")
        print(f"\nAgent response:\n{result.get('output', result)}\n")
        
    finally:
        await agent.close()


if __name__ == "__main__":
    # Run factory-based example (recommended)
    print("\n🆕 Running factory-based example (recommended)...")
    asyncio.run(main_with_factory())
    
    # Uncomment to run original approach
    # print("\n📦 Running original approach...")
    # asyncio.run(main())
    
    # Uncomment to run code execution example
    # asyncio.run(code_execution_example())
    
    # Uncomment to run data analysis example
    # asyncio.run(data_analysis_example())
