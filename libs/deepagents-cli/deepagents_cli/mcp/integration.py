"""MCP tool integration for deepagents CLI.

This module provides functions to integrate MCP servers as tools
in the deepagents framework.
"""

import asyncio
import concurrent.futures
from typing import Any, List

from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool

from deepagents_cli.config import console, COLORS
from deepagents_cli.mcp.manager import get_mcp_manager


async def create_mcp_tool_wrapper(
    original_tool: BaseTool, client: Any, server_name: str
) -> BaseTool:
    """Create a wrapped MCP tool that manages its own session.

    Args:
        original_tool: The original MCP tool from load_mcp_tools
        client: MultiServerMCPClient instance
        server_name: Name of the MCP server

    Returns:
        A wrapped BaseTool that creates a new session for each invocation
    """
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # Store the tool name and description
    tool_name = original_tool.name
    tool_description = original_tool.description

    async def wrapped_tool_func(**kwargs):
        """Wrapper function that creates a session before calling the tool."""
        async with client.session(server_name) as session:
            # Reload tools within this session
            session_tools = await load_mcp_tools(session)
            # Find the specific tool by name
            for session_tool in session_tools:
                if session_tool.name == tool_name:
                    # Call the tool with the provided arguments
                    return await session_tool.ainvoke(kwargs)
            raise ValueError(f"Tool '{tool_name}' not found in MCP server '{server_name}'")

    # Create a new StructuredTool with our wrapper function
    from langchain_core.tools import StructuredTool

    # Try to copy the args_schema from the original tool if it exists
    # This ensures the agent knows the required parameters
    tool_kwargs = {
        "name": tool_name,
        "func": None,  # No sync function
        "coroutine": wrapped_tool_func,
        "description": tool_description,
    }

    # Copy args_schema from original tool if available
    if hasattr(original_tool, "args_schema") and original_tool.args_schema is not None:
        tool_kwargs["args_schema"] = original_tool.args_schema

    return StructuredTool.from_function(**tool_kwargs)


async def get_mcp_tools_async(server_name: str) -> List[BaseTool]:
    """Get tools from an MCP server asynchronously.

    Args:
        server_name: Name of the MCP server

    Returns:
        List of BaseTool objects from the MCP server
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        # Get server configuration from manager
        manager = get_mcp_manager()
        server_config = manager.get_server_config(server_name)
        if not server_config:
            raise ValueError(f"MCP server '{server_name}' not configured")

        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        if not command:
            raise ValueError(f"No command specified for MCP server '{server_name}'")

        # Create connection dict for MultiServerMCPClient
        connection = {
            "transport": "stdio",
            "command": command,
            "args": args,
            "env": env,
        }

        # Create a MultiServerMCPClient with just this server
        client = MultiServerMCPClient({server_name: connection})  # type: ignore

        # Get tools using the client's session
        # We need to create tools that use the client to get a session on each call
        async with client.session(server_name) as session:
            tools = await load_mcp_tools(session)

        # Create wrapped tools that use the client for session management
        wrapped_tools = []
        for tool in tools:
            wrapped_tool = await create_mcp_tool_wrapper(tool, client, server_name)
            wrapped_tools.append(wrapped_tool)

        return wrapped_tools

    except ImportError:
        # Fall back to mcp library if langchain_mcp_adapters not available
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            # Get server configuration
            manager = get_mcp_manager()
            server_config = manager.get_server_config(server_name)
            if not server_config:
                raise ValueError(f"MCP server '{server_name}' not configured")

            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env", {})

            if not command:
                raise ValueError(f"No command specified for MCP server '{server_name}'")

            # Prepare server parameters
            server_params = StdioServerParameters(command=command, args=args, env=env)

            # Connect to server and get tools
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()

                    # Convert MCP tools to LangChain tools
                    # This is a simplified conversion - in production you'd need
                    # to properly handle the tool schemas and create proper BaseTool objects
                    console.print(
                        f"[yellow]Warning:[/yellow] Using basic MCP tool conversion for '{server_name}'. "
                        f"For full functionality, install langchain_mcp_adapters.",
                        style=COLORS["dim"],
                    )

                    # Create simple tools (this is a placeholder - real implementation would be more complex)
                    simple_tools = []
                    for tool in tools_result.tools:
                        # Create a simple tool that just reports it's an MCP tool
                        # In a real implementation, you'd create proper async tools
                        def create_tool_func(tool_name, tool_desc):
                            async def tool_func(**kwargs):
                                return f"MCP tool '{tool_name}' ({tool_desc[:50]}...) called with args: {kwargs}. "
                                "Note: Full MCP tool integration requires langchain_mcp_adapters."

                            return tool_func

                        tool_func = create_tool_func(tool.name, tool.description)
                        simple_tool = StructuredTool.from_function(
                            name=tool.name,
                            func=None,  # Sync function not implemented
                            coroutine=tool_func,
                            description=tool.description or f"MCP tool {tool.name}",
                        )
                        simple_tools.append(simple_tool)

                    return simple_tools

        except ImportError:
            raise ImportError(
                "Neither langchain_mcp_adapters nor mcp library is installed. "
                "Install at least one of them to use MCP tools."
            )


def get_mcp_tools(server_name: str) -> List[BaseTool]:
    """Get tools from an MCP server (synchronous wrapper).

    Args:
        server_name: Name of the MCP server

    Returns:
        List of BaseTool objects from the MCP server
    """
    try:
        # Check if we're already in an event loop
        asyncio.get_running_loop()
        # We're in an async context, need to run in a thread to avoid nesting loops
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.new_event_loop().run_until_complete(
                    get_mcp_tools_async(server_name)
                )
            )
            return future.result()
    except RuntimeError:
        # No running event loop, safe to create one
        return asyncio.new_event_loop().run_until_complete(get_mcp_tools_async(server_name))
