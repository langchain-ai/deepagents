"""MCP Tool Optimization Example.

This example demonstrates how to use the MCP (Model Context Protocol) tool
optimization feature in deepagents to achieve ~47% token reduction through
progressive disclosure.

## Overview

Instead of loading all MCP tool schemas into the system prompt upfront,
this approach:
1. Stores tool metadata in a folder-per-server structure under `/.mcp/`
2. Agents discover tools using filesystem operations (`ls`, `read_file`)
3. Only the schemas needed for the current task are loaded

## Requirements

1. Install langchain-mcp-adapters:
   ```
   pip install langchain-mcp-adapters
   ```

2. Install MCP servers (example with Brave Search):
   ```
   npm install -g @modelcontextprotocol/server-brave-search
   ```

3. Set up API keys:
   ```
   export BRAVE_API_KEY="your-api-key"
   ```

## Running the Example

```bash
python examples/mcp_tools_optimization.py
```
"""

import asyncio
import os
from typing import Any

# Import deepagents
from deepagents import create_deep_agent


async def basic_mcp_usage() -> None:
    """Basic example of MCP tool optimization.

    This example shows the simplest way to use MCP with progressive disclosure.
    """
    print("=" * 60)
    print("Basic MCP Tool Optimization Example")
    print("=" * 60)

    # Configure MCP servers
    # Each server will get its own folder under /.mcp/
    mcp_servers = [
        {
            "name": "brave-search",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")},
        },
    ]

    # Create agent with MCP optimization
    # Tools are discovered progressively via filesystem operations
    agent = create_deep_agent(
        mcp_servers=mcp_servers,
        # Optional: customize MCP metadata storage location
        # mcp_root="/custom/mcp/path",
    )

    # The agent will:
    # 1. See MCP discovery instructions in its system prompt
    # 2. Use `ls /.mcp/` to discover available servers
    # 3. Use `ls /.mcp/brave-search/` to see available tools
    # 4. Use `read_file /.mcp/brave-search/search.json` to read tool schemas
    # 5. Use `mcp_invoke(tool_name="search", arguments={...})` to execute

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for recent news about quantum computing breakthroughs.",
                }
            ]
        }
    )

    print("\nAgent Response:")
    print(response["messages"][-1].content)


async def multiple_mcp_servers() -> None:
    """Example with multiple MCP servers.

    Shows how to configure multiple MCP servers and let the agent
    discover and use tools from different providers.
    """
    print("\n" + "=" * 60)
    print("Multiple MCP Servers Example")
    print("=" * 60)

    # Configure multiple MCP servers
    mcp_servers = [
        # Web search capability
        {
            "name": "brave-search",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")},
        },
        # Filesystem access (sandboxed to /tmp)
        {
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        },
        # GitHub integration
        {
            "name": "github",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", "")},
        },
    ]

    agent = create_deep_agent(mcp_servers=mcp_servers)

    # Agent can now discover and use tools from all three servers
    # Example query that might use multiple tools
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Search for the latest LangChain releases, "
                        "then save a summary to /tmp/langchain_releases.txt"
                    ),
                }
            ]
        }
    )

    print("\nAgent Response:")
    print(response["messages"][-1].content)


async def custom_mcp_root() -> None:
    """Example with custom MCP metadata storage location.

    Shows how to specify a custom path for MCP tool metadata storage.
    """
    print("\n" + "=" * 60)
    print("Custom MCP Root Example")
    print("=" * 60)

    mcp_servers = [
        {
            "name": "brave-search",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")},
        },
    ]

    # Use a custom path for MCP metadata
    # This is useful for:
    # - Persisting metadata across sessions
    # - Sharing metadata between multiple agents
    # - Debugging (easier to inspect files)
    agent = create_deep_agent(
        mcp_servers=mcp_servers,
        mcp_root="/var/lib/deepagents/.mcp",  # Custom path
    )

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What MCP tools are available to you?",
                }
            ]
        }
    )

    print("\nAgent Response:")
    print(response["messages"][-1].content)


async def mcp_with_other_middleware() -> None:
    """Example combining MCP with skills and memory middleware.

    Shows that MCP optimization works alongside other deepagent features.
    """
    print("\n" + "=" * 60)
    print("MCP with Skills and Memory Example")
    print("=" * 60)

    from deepagents.backends.filesystem import FilesystemBackend

    # Use filesystem backend for all middleware
    backend = FilesystemBackend(root_dir="/tmp/deepagent_workspace", virtual_mode=True)

    mcp_servers = [
        {
            "name": "brave-search",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")},
        },
    ]

    # Create agent with multiple features
    agent = create_deep_agent(
        backend=backend,
        mcp_servers=mcp_servers,
        skills=["/skills/user/"],  # Skills middleware
        memory=["/memory/AGENTS.md"],  # Memory middleware
    )

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use your available tools and skills to help me research AI trends.",
                }
            ]
        }
    )

    print("\nAgent Response:")
    print(response["messages"][-1].content)


def explain_token_reduction() -> None:
    """Explain how MCP optimization achieves token reduction."""
    print("\n" + "=" * 60)
    print("Token Reduction Explanation")
    print("=" * 60)

    explanation = """
## How MCP Tool Optimization Works

### Traditional Approach (Without Optimization)
- All MCP tool schemas loaded into system prompt upfront
- Each tool schema can be 500-2000 tokens
- With 10 tools: ~5,000-20,000 tokens in every request
- Most tools often unused in any given conversation

### Optimized Approach (Progressive Disclosure)
- Only discovery instructions in system prompt (~200 tokens)
- Tool metadata stored in filesystem (/.mcp/<server>/<tool>.json)
- Agent discovers tools on-demand using existing filesystem tools
- Only loads schemas for tools actually needed

### Token Savings Calculation
Example with 10 MCP tools:
- Traditional: ~10,000 tokens (10 tools × 1,000 tokens average)
- Optimized: ~200 tokens (discovery prompt) + ~1,000 tokens (1-2 tools used)
- Savings: ~8,800 tokens (~47% reduction)

### Benefits
1. Lower costs (fewer tokens = lower API costs)
2. Faster responses (smaller prompts = faster processing)
3. More context available for actual task work
4. Scales better with more MCP tools

### Discovery Pattern
1. `ls /.mcp/` - See available servers
2. `ls /.mcp/<server>/` - See tools from a server
3. `read_file /.mcp/<server>/<tool>.json` - Get full schema
4. `mcp_invoke(tool_name="...", arguments={...})` - Execute tool
"""
    print(explanation)


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MCP Tool Optimization Examples")
    print("=" * 60)
    print("\nThese examples demonstrate how to use MCP tool optimization")
    print("in deepagents to achieve ~47% token reduction.\n")

    # First explain the concept
    explain_token_reduction()

    # Check for required environment variables
    if not os.getenv("BRAVE_API_KEY"):
        print("\n" + "!" * 60)
        print("WARNING: BRAVE_API_KEY not set")
        print("Set this environment variable to run the examples:")
        print("  export BRAVE_API_KEY='your-api-key'")
        print("!" * 60)
        print("\nSkipping live examples. Showing configuration only.\n")

        # Show configuration examples without running
        print("\n## Example Configurations\n")

        print("### Basic MCP Configuration:")
        print("""
mcp_servers = [
    {
        "name": "brave-search",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
    }
]

agent = create_deep_agent(mcp_servers=mcp_servers)
""")

        print("\n### Multiple MCP Servers:")
        print("""
mcp_servers = [
    {"name": "brave-search", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"]},
    {"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
    {"name": "github", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
]

agent = create_deep_agent(mcp_servers=mcp_servers)
""")

        print("\n### With Custom MCP Root:")
        print("""
agent = create_deep_agent(
    mcp_servers=mcp_servers,
    mcp_root="/var/lib/my_app/.mcp"
)
""")
        return

    # Run live examples
    try:
        await basic_mcp_usage()
    except Exception as e:
        print(f"Basic example error: {e}")

    try:
        await multiple_mcp_servers()
    except Exception as e:
        print(f"Multiple servers example error: {e}")

    try:
        await custom_mcp_root()
    except Exception as e:
        print(f"Custom root example error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
