#!/usr/bin/env python3
"""Example showing DeepAgents using MCP tools.

This example demonstrates how to:
1. Create a simple MCP server (math operations)
2. Configure DeepAgents to use MCP tools
3. Run the agent with both native and MCP tools
"""

import asyncio
import os
from pathlib import Path

# Simple MCP server for demonstration
def create_math_server():
    """Create a simple math MCP server file."""
    server_code = '''#!/usr/bin/env python3
"""Simple math MCP server for demonstration."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@mcp.tool()
def calculate_factorial(n: int) -> int:
    """Calculate the factorial of a number."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")
'''
    
    server_path = Path(__file__).parent / "math_server.py"
    with open(server_path, 'w') as f:
        f.write(server_code)
    
    # Make it executable
    os.chmod(server_path, 0o755)
    return str(server_path)


async def main():
    """Main example function."""
    print("DeepAgents + MCP Tools Example")
    print("=" * 40)
    
    # Create the math server
    math_server_path = create_math_server()
    print(f"Created math server at: {math_server_path}")
    
    # Import after creating the server
    from deepagents import create_deep_agent_async
    
    # Define MCP connections
    mcp_connections = {
        "math": {
            "command": "python",
            "args": [math_server_path],
            "transport": "stdio"
        }
    }
    
    # Define some native tools
    def get_system_info() -> str:
        """Get basic system information."""
        import platform
        return f"System: {platform.system()} {platform.release()}"
    
    def generate_report(title: str, content: str) -> str:
        """Generate a formatted report."""
        return f"""
# {title}

{content}

Generated at: {asyncio.get_event_loop().time()}
"""
    
    # Create the agent with both native and MCP tools
    print("\nCreating DeepAgent with MCP tools...")
    agent = await create_deep_agent_async(
        tools=[get_system_info, generate_report],
        instructions="""You are a helpful assistant with access to:
- Native tools for system info and report generation
- MCP tools for mathematical calculations
- Built-in file system and todo management tools

When asked to perform calculations, use the MCP math tools.
When asked for system info, use the native get_system_info tool.
Always be helpful and explain what tools you're using.""",
        mcp_connections=mcp_connections
    )
    
    print("Agent created successfully!")
    
    # Example interactions
    test_queries = [
        "What's 15 + 27?",
        "Calculate the factorial of 5",
        "What's the system information?",
        "Calculate (12 + 8) * 3 and create a report with the result"
    ]
    
    print("\n" + "=" * 50)
    print("Testing DeepAgent with MCP tools:")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: {query}")
        print("-" * 30)
        
        try:
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": query}]
            })
            
            # Get the last message from the agent
            if result.get("messages"):
                last_message = result["messages"][-1]
                print(f"Response: {last_message.content}")
            else:
                print("No response received")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nThe agent successfully used:")
    print("- MCP math tools (add, multiply, factorial)")
    print("- Native Python tools (system info, reports)")
    print("- Built-in DeepAgents tools (file system, todos)")


def run_config_example():
    """Show how to use configuration files."""
    print("\n" + "=" * 50)
    print("Configuration Example:")
    print("=" * 50)
    
    from deepagents_mcp import create_mcp_config_from_file
    
    # Show loading from JSON config
    config_path = Path(__file__).parent / "mcp_config.json"
    if config_path.exists():
        print(f"Loading MCP config from: {config_path}")
        try:
            config = create_mcp_config_from_file(str(config_path))
            print("Available MCP servers:")
            for name, connection in config["mcp_servers"].items():
                print(f"  - {name}: {connection}")
        except Exception as e:
            print(f"Error loading config: {e}")
    else:
        print("No mcp_config.json found in examples directory")


if __name__ == "__main__":
    try:
        # Run the main async example
        asyncio.run(main())
        
        # Show configuration example
        run_config_example()
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure to install dependencies:")
        print("  uv sync")
        print("  # or")
        print("  pip install langchain-mcp-adapters")
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()