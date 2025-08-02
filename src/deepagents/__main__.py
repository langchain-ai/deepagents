"""Main entry point for DeepAgents.

Supports traditional LangGraph mode with optional MCP tool integration.
"""

import argparse
import sys
import os
import asyncio
from typing import Optional, List

from deepagents import create_deep_agent, create_deep_agent_async

# Check if MCP client is available
try:
    from deepagents_mcp.mcp_client import create_mcp_config_from_file
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False


def run_langgraph_mode(args):
    """Run DeepAgents in traditional LangGraph mode."""
    print("Running DeepAgents in LangGraph mode...")
    
    # Load MCP configuration if provided
    mcp_connections = None
    if args.mcp_config and MCP_CLIENT_AVAILABLE:
        try:
            config = create_mcp_config_from_file(args.mcp_config)
            mcp_connections = config.get("mcp_servers", {})
            if mcp_connections:
                print(f"Loaded MCP configuration with {len(mcp_connections)} servers")
        except Exception as e:
            print(f"Warning: Could not load MCP config: {e}")
    elif args.mcp_config and not MCP_CLIENT_AVAILABLE:
        print("Warning: MCP config specified but langchain-mcp-adapters not available")
    
    # Simple example tool
    def example_tool(query: str) -> str:
        """Example tool for demonstration."""
        return f"Processed query: {query}"
    
    # Create agent (async if MCP tools, sync otherwise)
    if mcp_connections:
        # Use async version for MCP tool loading
        async def run_async():
            agent = await create_deep_agent_async(
                tools=[example_tool],
                instructions="You are a helpful assistant with planning capabilities and access to MCP tools.",
                mcp_connections=mcp_connections
            )
            
            # Interactive loop
            print("\nDeepAgent ready! Type 'exit' to quit.")
            print("-" * 50)
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        break
                        
                    # Invoke agent
                    result = await agent.ainvoke({
                        "messages": [{"role": "user", "content": user_input}]
                    })
                    
                    # Print response
                    if result.get("messages"):
                        last_message = result["messages"][-1]
                        print(f"\nAgent: {last_message.content}")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\nError: {e}")
            
            print("\nGoodbye!")
        
        # Run the async version
        asyncio.run(run_async())
    else:
        # Use sync version
        agent = create_deep_agent(
            tools=[example_tool],
            instructions="You are a helpful assistant with planning capabilities.",
        )
        
        # Interactive loop
        print("\nDeepAgent ready! Type 'exit' to quit.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                    
                # Invoke agent
                result = agent.invoke({
                    "messages": [{"role": "user", "content": user_input}]
                })
                
                # Print response
                if result.get("messages"):
                    last_message = result["messages"][-1]
                    print(f"\nAgent: {last_message.content}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
        
        print("\nGoodbye!")




def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="DeepAgents - Advanced AI agents with planning and sub-agent capabilities"
    )
    
    # MCP configuration
    parser.add_argument(
        "--mcp-config",
        help="Path to MCP configuration file (JSON/YAML format)"
    )
    
    # Tool configuration
    parser.add_argument(
        "--tools-module",
        help="Python module containing additional tool functions"
    )
    
    # Instructions
    parser.add_argument(
        "--instructions",
        help="Agent instructions (prompt prefix)"
    )
    parser.add_argument(
        "--instructions-file",
        help="File containing agent instructions"
    )
    
    # Common arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging if verbose
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Run the agent
    run_langgraph_mode(args)


if __name__ == "__main__":
    main()