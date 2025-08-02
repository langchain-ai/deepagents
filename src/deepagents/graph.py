from deepagents.sub_agent import _create_task_tool, SubAgent
from deepagents.model import get_default_model
from deepagents.tools import write_todos, write_file, read_file, ls, edit_file
from deepagents.state import DeepAgentState
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional, Dict, List
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike

from langgraph.prebuilt import create_react_agent

# Optional MCP client import
try:
    from deepagents_mcp.mcp_client import MCPToolProvider
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPToolProvider = None

StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = Type[StateSchema]

base_prompt = """You have access to a number of standard tools

## `write_todos`

You have access to the `write_todos` tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
## `task`

- When doing web search, prefer to use the `task` tool in order to reduce context usage."""


def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    mcp_connections: Optional[Dict[str, Any]] = None,
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and then four file editing tools: write_file, ls, read_file, edit_file.

    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
        mcp_connections: Optional dictionary of MCP server connections. Format:
            {
                "server_name": {
                    "command": "python",
                    "args": ["/path/to/server.py"],
                    "transport": "stdio"
                }  # or {"url": "http://...", "transport": "streamable_http"}
            }
    """
    prompt = instructions + base_prompt
    built_in_tools = [write_todos, write_file, read_file, ls, edit_file]
    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    
    # Load MCP tools if connections provided
    mcp_tools = []
    if mcp_connections and MCP_AVAILABLE:
        try:
            import asyncio
            # Create MCP provider and load tools
            provider = MCPToolProvider(mcp_connections)
            # Use asyncio.run to handle async tool loading
            mcp_tools = asyncio.run(provider.get_tools())
            if mcp_tools:
                print(f"Loaded {len(mcp_tools)} MCP tools from {len(mcp_connections)} servers")
        except Exception as e:
            print(f"Warning: Failed to load MCP tools: {e}")
    elif mcp_connections and not MCP_AVAILABLE:
        print("Warning: MCP connections specified but langchain-mcp-adapters not available")
    
    # Combine all tools
    all_user_tools = list(tools) + mcp_tools
    task_tool = _create_task_tool(
        all_user_tools + built_in_tools,
        instructions,
        subagents or [],
        model,
        state_schema
    )
    all_tools = built_in_tools + all_user_tools + [task_tool]
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
    )


async def create_deep_agent_async(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    mcp_connections: Optional[Dict[str, Any]] = None,
):
    """Async version of create_deep_agent for better MCP tool loading.
    
    This version properly handles async MCP tool loading without blocking.
    Recommended when using MCP connections.
    """
    prompt = instructions + base_prompt
    built_in_tools = [write_todos, write_file, read_file, ls, edit_file]
    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    
    # Load MCP tools if connections provided
    mcp_tools = []
    if mcp_connections and MCP_AVAILABLE:
        try:
            provider = MCPToolProvider(mcp_connections)
            mcp_tools = await provider.get_tools()
            if mcp_tools:
                print(f"Loaded {len(mcp_tools)} MCP tools from {len(mcp_connections)} servers")
        except Exception as e:
            print(f"Warning: Failed to load MCP tools: {e}")
    elif mcp_connections and not MCP_AVAILABLE:
        print("Warning: MCP connections specified but langchain-mcp-adapters not available")
    
    # Combine all tools
    all_user_tools = list(tools) + mcp_tools
    task_tool = _create_task_tool(
        all_user_tools + built_in_tools,
        instructions,
        subagents or [],
        model,
        state_schema
    )
    all_tools = built_in_tools + all_user_tools + [task_tool]
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
    )
