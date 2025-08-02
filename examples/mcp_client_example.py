#!/usr/bin/env python
"""Example MCP client for DeepAgents.

This demonstrates how to connect to and use DeepAgents via MCP.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def run_deepagents_client():
    """Example client that connects to DeepAgents MCP server."""
    
    # Configure server parameters to run DeepAgents in MCP mode
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "deepagents", "--mode", "mcp", "--transport", "stdio"],
        cwd=str(Path(__file__).parent.parent)  # Run from project root
    )
    
    print("Starting DeepAgents MCP client...")
    print("-" * 50)
    
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        # Initialize session
        await session.initialize()
        print("✓ Connected to DeepAgents MCP server")
        
        # List available tools
        tools_result = await session.list_tools()
        tools = tools_result.tools
        print(f"\n✓ Available tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # List available resources (sub-agents)
        resources = await session.list_resources()
        print(f"\n✓ Available resources ({len(resources.resources)}):")
        for resource in resources.resources:
            print(f"  - {resource.uri}: {resource.name}")
        
        # List available prompts
        prompts = await session.list_prompts()
        print(f"\n✓ Available prompts ({len(prompts.prompts)}):")
        for prompt in prompts.prompts:
            print(f"  - {prompt.name}: {prompt.description}")
        
        # Example 1: Create a todo list
        print("\n" + "=" * 50)
        print("Example 1: Task Planning")
        print("=" * 50)
        
        session_id = "example-session-1"
        
        result = await session.call_tool(
            "write_todos",
            {
                "session_id": session_id,
                "todos": [
                    {
                        "id": "1",
                        "content": "Research quantum computing applications",
                        "status": "pending",
                        "priority": "high"
                    },
                    {
                        "id": "2",
                        "content": "Write summary report",
                        "status": "pending",
                        "priority": "medium"
                    },
                    {
                        "id": "3",
                        "content": "Review and edit report",
                        "status": "pending",
                        "priority": "low"
                    }
                ]
            }
        )
        print(f"Todo list created: {result.content}")
        
        # Example 2: File operations
        print("\n" + "=" * 50)
        print("Example 2: File Operations")
        print("=" * 50)
        
        # Write a file
        result = await session.call_tool(
            "write_file",
            {
                "session_id": session_id,
                "file_path": "research_notes.md",
                "content": """# Quantum Computing Research Notes

## Introduction
Quantum computing represents a fundamental shift in computational paradigm...

## Key Concepts
- Superposition
- Entanglement
- Quantum gates

## Applications
1. Cryptography
2. Drug discovery
3. Financial modeling
"""
            }
        )
        print(f"File written: {result.content}")
        
        # List files
        result = await session.call_tool(
            "ls",
            {"session_id": session_id}
        )
        print(f"\nFiles in virtual filesystem: {result.content}")
        
        # Read the file
        result = await session.call_tool(
            "read_file",
            {
                "session_id": session_id,
                "file_path": "research_notes.md",
                "limit": 5  # Read first 5 lines
            }
        )
        print(f"\nFile content (first 5 lines):\n{result.content}")
        
        # Example 3: Using sub-agents
        print("\n" + "=" * 50)
        print("Example 3: Sub-Agent Invocation")
        print("=" * 50)
        
        # Get available sub-agents
        subagents_resource = await session.read_resource("subagents://available")
        print(f"Available sub-agents: {subagents_resource.contents[0].text}")
        
        # Invoke a sub-agent for a complex task
        result = await session.call_tool(
            "invoke_subagent",
            {
                "session_id": session_id,
                "agent_type": "general-purpose",
                "task_description": "Analyze the research notes and create a summary"
            }
        )
        print(f"\nSub-agent result: {result.content}")
        
        # Example 4: Using prompts
        print("\n" + "=" * 50)
        print("Example 4: Using Prompts")
        print("=" * 50)
        
        # Get a research prompt
        prompt_result = await session.get_prompt(
            "deepagents_research",
            {"query": "What are the latest breakthroughs in quantum computing?"}
        )
        
        print("Research prompt generated:")
        for msg in prompt_result.messages:
            print(f"  [{msg.role}]: {msg.content[:100]}...")
        
        print("\n✓ All examples completed successfully!")


async def interactive_client():
    """Interactive client for testing DeepAgents MCP server."""
    
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "deepagents", "--mode", "mcp", "--transport", "stdio"],
        cwd=str(Path(__file__).parent.parent)
    )
    
    print("DeepAgents MCP Interactive Client")
    print("Type 'help' for available commands, 'exit' to quit")
    print("-" * 50)
    
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        print("✓ Connected to DeepAgents MCP server\n")
        
        session_id = "interactive-session"
        
        while True:
            try:
                command = input("> ").strip().lower()
                
                if command == "exit":
                    break
                elif command == "help":
                    print("""
Available commands:
  todo <content>     - Add a todo item
  todos             - List current todos
  write <path>      - Write a file (will prompt for content)
  read <path>       - Read a file
  ls                - List files
  agent <task>      - Run a sub-agent task
  help              - Show this help
  exit              - Exit the client
""")
                elif command.startswith("todo "):
                    content = command[5:]
                    # Get current todos first
                    # (In a real implementation, we'd track this client-side)
                    result = await session.call_tool(
                        "write_todos",
                        {
                            "session_id": session_id,
                            "todos": [{
                                "id": str(asyncio.get_event_loop().time()),
                                "content": content,
                                "status": "pending",
                                "priority": "medium"
                            }]
                        }
                    )
                    print(f"✓ Todo added: {content}")
                    
                elif command == "ls":
                    result = await session.call_tool(
                        "ls",
                        {"session_id": session_id}
                    )
                    files = eval(result.content[0].text)["result"]
                    if files:
                        print("Files:")
                        for f in files:
                            print(f"  - {f}")
                    else:
                        print("No files in virtual filesystem")
                        
                elif command.startswith("read "):
                    path = command[5:]
                    result = await session.call_tool(
                        "read_file",
                        {
                            "session_id": session_id,
                            "file_path": path
                        }
                    )
                    content = eval(result.content[0].text)["result"]
                    print(f"\nContent of {path}:\n{content}")
                    
                elif command.startswith("write "):
                    path = command[6:]
                    print("Enter content (type END on a new line to finish):")
                    lines = []
                    while True:
                        line = input()
                        if line == "END":
                            break
                        lines.append(line)
                    content = "\n".join(lines)
                    
                    result = await session.call_tool(
                        "write_file",
                        {
                            "session_id": session_id,
                            "file_path": path,
                            "content": content
                        }
                    )
                    print(f"✓ File written: {path}")
                    
                elif command.startswith("agent "):
                    task = command[6:]
                    print(f"Running sub-agent for: {task}")
                    result = await session.call_tool(
                        "invoke_subagent",
                        {
                            "session_id": session_id,
                            "agent_type": "general-purpose",
                            "task_description": task
                        }
                    )
                    print(f"Result: {result.content}")
                    
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_client())
    else:
        asyncio.run(run_deepagents_client())