"""MCP (Model Context Protocol) middleware for ChATLAS agents.

This middleware provides native MCP server integration for deepagents and deepagents-cli,
enabling dynamic tool loading from MCP servers without modifying the upstream packages.

The middleware follows the standard deepagents middleware pattern:
1. Loads MCP tools at initialization
2. Exposes tools to the agent via the tools property
3. Can be composed with other middleware

Example usage with deepagents:
    from chatlas_agents.middleware import MCPMiddleware
    from chatlas_agents.config import MCPServerConfig
    from deepagents import create_deep_agent
    
    # Create MCP middleware
    mcp_config = MCPServerConfig(
        url="https://chatlas-mcp.app.cern.ch/mcp",
        timeout=30
    )
    mcp_middleware = await MCPMiddleware.create(mcp_config)
    
    # Use with deepagents
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        middleware=[mcp_middleware],
    )

For deepagents-cli integration:
    from deepagents_cli.agent import create_cli_agent
    from chatlas_agents.middleware import MCPMiddleware
    
    # Create MCP middleware and extract tools
    mcp_middleware = await MCPMiddleware.create(mcp_config)
    
    # Pass tools to CLI agent
    agent, backend = create_cli_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        assistant_id="my-agent",
        tools=mcp_middleware.tools,  # Use MCP tools
    )
"""

from __future__ import annotations

import logging
from typing import Any, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

from chatlas_agents.config import MCPServerConfig
from chatlas_agents.mcp import create_mcp_client_and_load_tools

logger = logging.getLogger(__name__)


class MCPState(AgentState):
    """State extension for MCP middleware."""

    mcp_tools: NotRequired[list[BaseTool]]
    """List of tools loaded from MCP server."""


class MCPStateUpdate(TypedDict):
    """State update for MCP middleware."""

    mcp_tools: list[BaseTool]
    """List of tools loaded from MCP server."""


DEFAULT_MCP_SYSTEM_PROMPT = """

## MCP Server Tools

You have access to tools from a Model Context Protocol (MCP) server.
These tools provide access to specialized capabilities and data sources.

{tools_description}

**Important Notes:**
- MCP tools are dynamically loaded from the server
- Each tool has its own documentation describing its purpose and parameters
- Use the tool descriptions to understand what each tool does
- Some MCP tools may have longer execution times - be patient when waiting for results
"""


class MCPMiddleware(AgentMiddleware):
    """Middleware for loading and exposing MCP server tools.

    This middleware provides native MCP server integration by:
    - Loading tools from an MCP server at initialization
    - Exposing tools to the agent via the tools property
    - Optionally injecting tool descriptions into the system prompt

    The middleware is designed to work with both deepagents and deepagents-cli
    without requiring any modifications to those packages.

    Args:
        config: MCP server configuration (url, timeout, headers)
        inject_prompt: Whether to inject MCP tool descriptions into system prompt (default: True)
        tools: Pre-loaded MCP tools (optional, loaded from server if not provided)
        system_prompt_template: Custom template for MCP section in system prompt (optional)
    """

    state_schema = MCPState

    def __init__(
        self,
        *,
        config: MCPServerConfig,
        inject_prompt: bool = True,
        tools: list[BaseTool] | None = None,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize MCP middleware.

        Note: Use MCPMiddleware.create() async factory method instead of
        calling this constructor directly.

        Args:
            config: MCP server configuration
            inject_prompt: Whether to inject tool descriptions into system prompt
            tools: Pre-loaded MCP tools (if None, must call load_tools())
            system_prompt_template: Custom template for MCP section. Must contain
                {tools_description} placeholder. If None, uses DEFAULT_MCP_SYSTEM_PROMPT.
        """
        super().__init__()
        self.config = config
        self.inject_prompt = inject_prompt
        self._tools = tools or []
        self._tools_loaded = tools is not None
        self.system_prompt_template = system_prompt_template or DEFAULT_MCP_SYSTEM_PROMPT

    @classmethod
    async def create(
        cls,
        config: MCPServerConfig,
        inject_prompt: bool = True,
        system_prompt_template: str | None = None,
    ) -> MCPMiddleware:
        """Create and initialize MCP middleware with tools loaded from server.

        This is the recommended way to create MCPMiddleware instances.

        Args:
            config: MCP server configuration
            inject_prompt: Whether to inject tool descriptions into system prompt
            system_prompt_template: Custom template for MCP section (optional)

        Returns:
            Initialized MCPMiddleware with tools loaded from the MCP server

        Raises:
            ConnectionError: If unable to connect to MCP server
            TimeoutError: If connection to MCP server times out
            ValueError: If MCP server returns invalid tool definitions
        """
        logger.info(f"Creating MCP middleware for server: {config.url}")
        try:
            tools = await create_mcp_client_and_load_tools(config)
            logger.info(f"Successfully loaded {len(tools)} tools from MCP server")
            return cls(
                config=config, 
                inject_prompt=inject_prompt, 
                tools=tools,
                system_prompt_template=system_prompt_template
            )
        except Exception as e:
            error_msg = (
                f"Failed to connect to MCP server at {config.url}. "
                f"Error: {type(e).__name__}: {str(e)}. "
                f"Please verify:\n"
                f"  1. The MCP server URL is correct: {config.url}\n"
                f"  2. The MCP server is running and accessible\n"
                f"  3. Network connectivity is available\n"
                f"  4. The server timeout ({config.timeout}s) is sufficient"
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    async def load_tools(self) -> None:
        """Load tools from MCP server.

        This is called automatically by create() but can be called manually
        if tools need to be reloaded.
        
        Raises:
            ConnectionError: If unable to connect to MCP server
            TimeoutError: If connection to MCP server times out
        """
        if not self._tools_loaded:
            logger.info(f"Loading tools from MCP server at {self.config.url}...")
            try:
                self._tools = await create_mcp_client_and_load_tools(self.config)
                self._tools_loaded = True
                logger.info(f"Successfully loaded {len(self._tools)} tools from MCP server")
            except Exception as e:
                error_msg = (
                    f"Failed to load tools from MCP server at {self.config.url}. "
                    f"Error: {type(e).__name__}: {str(e)}"
                )
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e

    @property
    def tools(self) -> list[BaseTool]:
        """Get the list of MCP tools.

        Returns:
            List of BaseTool instances loaded from the MCP server
        """
        return self._tools

    def before_agent(self, state: MCPState, runtime: Runtime) -> MCPStateUpdate | None:
        """Hook that runs before agent execution.
        
        Note: Tools are provided via the `tools` property and don't need to be stored in state.
        Storing tools in state causes serialization issues with msgpack.

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            None - no state updates needed
        """
        # Tools are automatically discovered via the `tools` property
        # No need to store them in state (causes serialization issues)
        return None

    def _format_tools_description(self) -> str:
        """Format MCP tools for system prompt.

        Returns:
            Formatted string describing available MCP tools
        """
        if not self._tools:
            return "No MCP tools are currently available."

        lines = ["**Available MCP Tools:**", ""]
        for tool in self._tools:
            tool_name = getattr(tool, "name", "unknown")
            tool_desc = getattr(tool, "description", "No description available")
            lines.append(f"- **{tool_name}**: {tool_desc}")

        return "\n".join(lines)

    def wrap_model_call(
        self,
        request: Any,
        handler: Any,
    ) -> Any:
        """Inject MCP tools documentation into the system prompt.

        This runs on model calls to ensure MCP tools are documented
        in the system prompt.

        Args:
            request: The model request being processed
            handler: The handler function to call with the modified request

        Returns:
            The model response from the handler
        """
        if not self.inject_prompt or not self._tools:
            return handler(request)

        # Format tools description
        tools_description = self._format_tools_description()

        # Format the MCP section
        mcp_section = self.system_prompt_template.format(
            tools_description=tools_description,
        )

        # Inject into system prompt
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + mcp_section
        else:
            system_prompt = mcp_section

        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: Any,
        handler: Any,
    ) -> Any:
        """(async) Inject MCP tools documentation into the system prompt.

        Args:
            request: The model request being processed
            handler: The handler function to call with the modified request

        Returns:
            The model response from the handler
        """
        if not self.inject_prompt or not self._tools:
            return await handler(request)

        # Format tools description
        tools_description = self._format_tools_description()

        # Format the MCP section
        mcp_section = self.system_prompt_template.format(
            tools_description=tools_description,
        )

        # Inject into system prompt
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + mcp_section
        else:
            system_prompt = mcp_section

        return await handler(request.override(system_prompt=system_prompt))


__all__ = ["MCPMiddleware", "MCPState", "MCPStateUpdate"]
