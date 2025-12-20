"""Example: Integrating MCPMiddleware with DeepAgents-CLI.

This example demonstrates how to extend deepagents-cli with MCP support
using the MCPMiddleware without modifying the upstream package.

There are two approaches:
1. Pass MCP tools via the tools parameter (simple)
2. Create a wrapper function that adds MCPMiddleware (advanced)
"""

import asyncio
import logging
import os
from pathlib import Path

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import create_model

from chatlas_agents.config import MCPServerConfig
from chatlas_agents.middleware import MCPMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def approach1_tools_parameter():
    """Approach 1: Pass MCP tools via tools parameter.
    
    This is the simplest approach - load MCP tools and pass them
    as regular tools to create_cli_agent().
    """
    logger.info("\n" + "="*60)
    logger.info("Approach 1: MCP Tools via tools parameter")
    logger.info("="*60)
    
    # Configure MCP server
    mcp_url = os.getenv("CHATLAS_MCP_URL", "https://chatlas-mcp.app.cern.ch/mcp")
    mcp_config = MCPServerConfig(url=mcp_url, timeout=60)
    
    # Load MCP tools using middleware
    mcp_middleware = await MCPMiddleware.create(mcp_config)
    mcp_tools = mcp_middleware.tools
    
    logger.info(f"Loaded {len(mcp_tools)} MCP tools")
    
    # Create CLI agent with MCP tools
    model = create_model("anthropic:claude-sonnet-4-5-20250929")
    agent, backend = create_cli_agent(
        model=model,
        assistant_id="mcp-cli-agent",
        tools=mcp_tools,  # Pass MCP tools here
        auto_approve=False,
        enable_memory=True,
        enable_skills=True,
        enable_shell=True,
    )
    
    logger.info("CLI agent created with MCP tools")
    logger.info("Agent has access to:")
    logger.info("  - MCP server tools")
    logger.info("  - Skills system")
    logger.info("  - Memory system")
    logger.info("  - Shell commands")
    
    return agent, backend


async def approach2_middleware_composition():
    """Approach 2: Create a wrapper that composes middleware.
    
    This approach creates a custom wrapper around create_cli_agent
    that properly composes MCPMiddleware with other CLI middleware.
    
    Note: This requires that create_cli_agent supports passing
    additional middleware via kwargs or we modify our local copy.
    """
    logger.info("\n" + "="*60)
    logger.info("Approach 2: Middleware Composition (Conceptual)")
    logger.info("="*60)
    
    # Configure MCP server
    mcp_url = os.getenv("CHATLAS_MCP_URL", "https://chatlas-mcp.app.cern.ch/mcp")
    mcp_config = MCPServerConfig(url=mcp_url, timeout=60)
    
    # Create MCP middleware
    mcp_middleware = await MCPMiddleware.create(mcp_config)
    
    logger.info(f"Created MCP middleware with {len(mcp_middleware.tools)} tools")
    
    # In a real implementation, we would:
    # 1. Either modify create_cli_agent to accept external middleware
    # 2. Or create our own version that calls the upstream with our middleware
    # 3. For now, we show the conceptual approach
    
    logger.info("Conceptual approach:")
    logger.info("  1. Create MCPMiddleware")
    logger.info("  2. Create CLI agent with standard middleware")
    logger.info("  3. Compose MCPMiddleware with CLI middleware")
    logger.info("  4. This gives full integration with all CLI features")
    
    # Since we can't easily modify the CLI agent's middleware stack without
    # modifying deepagents-cli, Approach 1 is recommended for now
    logger.info("\nNote: Approach 1 (tools parameter) is recommended")
    logger.info("      as it requires no modifications to deepagents-cli")


async def create_chatlas_cli_agent(
    model,
    assistant_id: str,
    mcp_config: MCPServerConfig | None = None,
    **kwargs
):
    """Helper function to create CLI agent with optional MCP support.
    
    This is a convenience wrapper that adds MCP tools if config is provided.
    
    Args:
        model: LLM model to use
        assistant_id: Agent identifier
        mcp_config: Optional MCP server configuration
        **kwargs: Additional arguments passed to create_cli_agent
        
    Returns:
        Tuple of (agent, backend) from create_cli_agent
    """
    tools = kwargs.get('tools', [])
    
    # Add MCP tools if config provided
    if mcp_config:
        logger.info("Loading MCP tools...")
        mcp_middleware = await MCPMiddleware.create(mcp_config)
        tools.extend(mcp_middleware.tools)
        logger.info(f"Added {len(mcp_middleware.tools)} MCP tools")
        kwargs['tools'] = tools
    
    # Create CLI agent with all tools
    return create_cli_agent(
        model=model,
        assistant_id=assistant_id,
        **kwargs
    )


async def demo_usage():
    """Demonstrate usage of the helper function."""
    logger.info("\n" + "="*60)
    logger.info("Demo: Using create_chatlas_cli_agent() helper")
    logger.info("="*60)
    
    # Configure MCP
    mcp_url = os.getenv("CHATLAS_MCP_URL", "https://chatlas-mcp.app.cern.ch/mcp")
    mcp_config = MCPServerConfig(url=mcp_url, timeout=60)
    
    # Create agent with MCP support in one line
    model = create_model("anthropic:claude-sonnet-4-5-20250929")
    agent, backend = await create_chatlas_cli_agent(
        model=model,
        assistant_id="demo-agent",
        mcp_config=mcp_config,  # This adds MCP support
        auto_approve=False,
        enable_memory=True,
        enable_skills=True,
        enable_shell=True,
    )
    
    logger.info("Agent created successfully with MCP + CLI features")
    
    # Example query
    logger.info("\nExample: Running a simple query...")
    try:
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "List available tools"}]
        })
        
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            logger.info(f"\nAgent response: {last_message.content[:200]}...")
    except Exception as e:
        logger.error(f"Error during demo: {e}")
        logger.info("(This is expected if MCP server is not available)")


async def main():
    """Run all examples."""
    logger.info("DeepAgents-CLI + MCP Integration Examples")
    logger.info("=" * 60)
    
    try:
        # Approach 1: Simple tools parameter
        await approach1_tools_parameter()
        
        # Approach 2: Conceptual middleware composition
        await approach2_middleware_composition()
        
        # Demo of helper function
        await demo_usage()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        logger.info("\nNote: These examples require:")
        logger.info("  1. A running MCP server (set CHATLAS_MCP_URL)")
        logger.info("  2. Valid LLM credentials")
        logger.info("  3. All dependencies installed")


if __name__ == "__main__":
    asyncio.run(main())
