"""Example: Using MCPMiddleware with DeepAgents.

This example demonstrates how to use the MCPMiddleware to add MCP server
support to a deepagents agent without modifying the upstream packages.
"""

import asyncio
import logging
import os

from deepagents import create_deep_agent

from chatlas_agents.config import MCPServerConfig
from chatlas_agents.middleware import MCPMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run example of MCPMiddleware with DeepAgents."""
    
    # Configure MCP server
    # You can use environment variables or hardcode for testing
    mcp_url = os.getenv("CHATLAS_MCP_URL", "https://chatlas-mcp.app.cern.ch/mcp")
    mcp_timeout = int(os.getenv("CHATLAS_MCP_TIMEOUT", "60"))
    
    mcp_config = MCPServerConfig(
        url=mcp_url,
        timeout=mcp_timeout,
    )
    
    logger.info(f"Connecting to MCP server at {mcp_url}")
    
    # Create MCP middleware (loads tools from server)
    try:
        mcp_middleware = await MCPMiddleware.create(mcp_config)
        logger.info(f"Loaded {len(mcp_middleware.tools)} tools from MCP server")
        
        # List the loaded tools
        print("\n" + "="*60)
        print("Available MCP Tools:")
        print("="*60)
        for tool in mcp_middleware.tools:
            tool_name = getattr(tool, "name", "unknown")
            tool_desc = getattr(tool, "description", "No description")
            print(f"\n• {tool_name}")
            print(f"  {tool_desc}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        logger.info("This example requires a running MCP server.")
        return
    
    # Create a deep agent with MCP middleware
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        middleware=[mcp_middleware],
        system_prompt="""You are a helpful assistant with access to ChATLAS tools.
        
You can search ChATLAS documentation and data sources.
When the user asks about ATLAS or ChATLAS, use the available MCP tools to find information.
""",
    )
    
    logger.info("Agent created successfully with MCP support")
    
    # Example interaction
    print("\n" + "="*60)
    print("Example Query: What is ChATLAS?")
    print("="*60 + "\n")
    
    try:
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is ChATLAS? Search the documentation."}]
        })
        
        # Print the response
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            print("\nAgent Response:")
            print("-" * 60)
            print(last_message.content)
            print("-" * 60)
        
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
