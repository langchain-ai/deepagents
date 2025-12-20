"""Example of using ChATLAS agents with Apptainer sandbox backend.

This example demonstrates two approaches:
1. Direct instantiation of ApptainerSandboxBackend (original approach)
2. Using create_apptainer_sandbox factory function (recommended for new code)

Apptainer is particularly useful in HPC environments like CERN lxplus where
it doesn't require root privileges or a daemon.
"""

import asyncio
import os
from deepagents import create_deep_agent as create_deep_agent_sync
from langchain.chat_models import init_chat_model
from chatlas_agents.config import AgentConfig, LLMConfig, LLMProvider
from chatlas_agents.agents import create_deep_agent
from chatlas_agents.sandbox import SandboxBackendType, create_apptainer_sandbox


def main_with_factory():
    """Run DeepAgent example with Apptainer sandbox using factory function.
    
    This is the recommended approach as it follows deepagents-cli patterns
    and provides automatic lifecycle management.
    """
    print("Creating DeepAgent with Apptainer sandbox (factory approach)...")
    
    # Use factory function with context manager for automatic cleanup
    with create_apptainer_sandbox(
        image="docker://python:3.13-slim",
        working_dir="/workspace",
        auto_remove=True,
    ) as backend:
        # Create agent directly with the backend
        # Note: Using synchronous deepagents.create_deep_agent
        # Initialize model using init_chat_model to avoid AttributeError
        model = init_chat_model("openai:gpt-5-mini")
        
        agent = create_deep_agent_sync(
            model=model,
            backend=backend,
            system_prompt="You are a helpful coding assistant with access to an Apptainer sandbox.",
        )

        # Run queries that may involve file operations or command execution
        print("\n" + "="*60)
        print("Querying agent...")
        print("="*60)
        
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Create a Python script that prints 'Hello from Apptainer sandbox!' and execute it"
            }]
        })
        
        # Extract the final response
        final_message = result["messages"][-1]
        print(f"\nAgent: {final_message.content}\n")
    
    print("✓ Sandbox cleaned up automatically")


async def main():
    """Run DeepAgent example with Apptainer sandbox (original approach).
    
    This demonstrates direct instantiation without factory functions.
    Kept for backwards compatibility.
    """
    # Get API key from environment or use placeholder
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    # Create configuration
    config = AgentConfig(
        name="chatlas-apptainer-agent",
        description="ChATLAS agent with Apptainer sandbox for secure code execution",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            api_key=api_key,  # Set via CHATLAS_LLM_API_KEY environment variable
        ),
        verbose=True,
    )

    # Create and initialize DeepAgent with Apptainer sandbox
    # This enables the agent to execute shell commands safely in an isolated container
    # Apptainer is particularly useful in HPC environments like CERN lxplus
    print("Creating DeepAgent with Apptainer sandbox...")
    agent = await create_deep_agent(
        config,
        use_docker_sandbox=True,
        docker_image="python:3.13-slim",  # Can use Docker images with docker:// prefix
        sandbox_backend=SandboxBackendType.APPTAINER,
    )

    try:
        # Run queries that may involve file operations or command execution
        thread_id = "apptainer-session"
        
        queries = [
            "Create a Python script that prints 'Hello from Apptainer sandbox!'",
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
    """Example demonstrating safe code execution in Apptainer sandbox."""
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    config = AgentConfig(
        name="code-execution-agent",
        description="Agent for safe code execution in isolated Apptainer environment",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key=api_key,
        ),
    )

    print("Creating agent with Apptainer sandbox for code execution...")
    agent = await create_deep_agent(
        config, 
        use_docker_sandbox=True,
        sandbox_backend=SandboxBackendType.APPTAINER,
    )

    try:
        print("\n" + "="*60)
        print("Code Execution Example with Apptainer Sandbox")
        print("="*60 + "\n")
        
        # The agent can safely execute potentially dangerous commands
        # because they run in an isolated Apptainer container
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
    """Example showing data analysis with Apptainer sandbox."""
    api_key = os.getenv("CHATLAS_LLM_API_KEY", "your-api-key-here")
    
    config = AgentConfig(
        name="data-analysis-agent",
        description="Agent for data analysis with secure Apptainer execution environment",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key=api_key,
        ),
    )

    print("Creating agent with Apptainer sandbox for data analysis...")
    # Use a container image (Apptainer can pull Docker images)
    agent = await create_deep_agent(
        config,
        use_docker_sandbox=True,
        docker_image="docker://python:3.13-slim",  # Explicit docker:// prefix
        sandbox_backend=SandboxBackendType.APPTAINER,
    )

    try:
        print("\n" + "="*60)
        print("Data Analysis Example with Apptainer")
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
    main_with_factory()
    
    # Uncomment to run original approach
    # print("\n📦 Running original approach...")
    # asyncio.run(main())
    
    # Uncomment to run code execution example
    # asyncio.run(code_execution_example())
    
    # Uncomment to run data analysis example
    # asyncio.run(data_analysis_example())
