#!/usr/bin/env python3
"""Demo showing MCP tool discovery without LLM execution.

This demonstrates the successful MCP integration by showing:
1. Available tools from filesystem MCP server
2. Available tools from DuckDuckGo MCP server  
3. Tool descriptions and parameters
"""

import asyncio
import json
from deepagents_mcp import MCPToolProvider

async def main():
    """Demonstrate MCP tool discovery."""
    print("🔍 DeepAgents MCP Tool Discovery Demo")
    print("=" * 50)
    
    # Define MCP connections
    mcp_connections = {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "./"
            ],
            "transport": "stdio"
        },
        "duckduckgo": {
            "command": "uvx",
            "args": ["duckduckgo-mcp-server"],
            "transport": "stdio"
        }
    }
    
    print(f"📡 Connecting to {len(mcp_connections)} MCP servers...")
    print("- Filesystem MCP Server")
    print("- DuckDuckGo Search MCP Server")
    print()
    
    try:
        # Create MCP provider
        provider = MCPToolProvider(mcp_connections)
        
        # Load tools
        tools = await provider.get_tools()
        
        print(f"✅ Successfully loaded {len(tools)} MCP tools!")
        print()
        
        # Group tools by server (based on tool name patterns)
        filesystem_tools = [t for t in tools if any(keyword in t.name.lower() 
                           for keyword in ['read', 'write', 'list', 'create', 'delete', 'move'])]
        search_tools = [t for t in tools if any(keyword in t.name.lower() 
                       for keyword in ['search', 'query', 'find'])]
        other_tools = [t for t in tools if t not in filesystem_tools and t not in search_tools]
        
        # Show filesystem tools
        if filesystem_tools:
            print("📁 Filesystem MCP Tools:")
            print("-" * 25)
            for tool in filesystem_tools:
                print(f"  • {tool.name}")
                print(f"    Description: {tool.description}")
                if hasattr(tool, 'args') and tool.args:
                    print(f"    Parameters: {list(tool.args.keys())}")
                print()
        
        # Show search tools  
        if search_tools:
            print("🔍 DuckDuckGo Search MCP Tools:")
            print("-" * 30)
            for tool in search_tools:
                print(f"  • {tool.name}")
                print(f"    Description: {tool.description}")
                if hasattr(tool, 'args') and tool.args:
                    print(f"    Parameters: {list(tool.args.keys())}")
                print()
        
        # Show other tools
        if other_tools:
            print("🛠️  Other MCP Tools:")
            print("-" * 20)
            for tool in other_tools:
                print(f"  • {tool.name}")
                print(f"    Description: {tool.description}")
                if hasattr(tool, 'args') and tool.args:
                    print(f"    Parameters: {list(tool.args.keys())}")
                print()
        
        print("🎯 Integration Status:")
        print("-" * 20)
        print("✅ MCP client connection: SUCCESS")
        print("✅ Tool discovery: SUCCESS")  
        print("✅ Tool loading: SUCCESS")
        print(f"✅ Total tools available: {len(tools)}")
        print()
        print("🚀 DeepAgents can now use these MCP tools alongside:")
        print("  • Native Python tools")
        print("  • Built-in DeepAgents tools (file system, todos)")
        print("  • LangGraph task orchestration")
        
    except Exception as e:
        print(f"❌ Error connecting to MCP servers: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())