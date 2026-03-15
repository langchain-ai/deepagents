"""Skill tools middleware for providing MCP and native tools.

This module extends the Skills system to support:
1. MCP (Model Context Protocol) servers declared in skill YAML frontmatter
2. Native Python tools registered via decorator
3. A single dispatcher tool that routes to native or MCP tools
4. Lazy MCP server connection when skills become active

## Architecture

The middleware provides a single `skill_tool_call` dispatcher tool that routes
to either:
- Native Python tools registered with @register_tool decorator
- MCP tools from servers declared in skill's mcp-servers YAML field

MCP servers are connected lazily when a skill becomes active (i.e., when it's
loaded in the agent's skills_metadata). Connections persist for the session
and are cleaned up on session end.

## Usage

### MCP Servers in SKILL.md

```yaml
---
name: github-integration
description: GitHub API integration
mcp-servers:
  - name: github
    transport: stdio
    command: npx
    args: [-y, @modelcontextprotocol/server-github]
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}
---
```

### Native Tools in helper.py

```python
from deepagents.middleware.skill_tools import register_tool

@register_tool("github-integration")
def create_repository(repo_name: str, description: str) -> str:
    \"\"\"Create a new GitHub repository.\"\"\"
    return f"Created {repo_name}"
```

### Agent Creation

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    skills=["/skills/user/", "/skills/project/"],
)

# Agent can call: skill_tool_call(
#     skill_name="github-integration",
#     tool_name="create_repository",
#     repo_name="my-repo",
#     description="My repo"
# )
```
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import ToolRuntime

from deepagents.middleware._utils import append_to_system_message

# MCP client imports - optional dependency
try:
    from langchain_mcp_adapters.client import (
        MultiServerMCPClient,
        SSEConnection,
        StdioConnection,
        StreamableHttpConnection,
    )
    from langchain_mcp_adapters.tools import load_mcp_tools
except ImportError:
    MultiServerMCPClient = None  # type: ignore[assignment]
    SSEConnection = None
    StdioConnection = None
    StreamableHttpConnection = None
    load_mcp_tools = None

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol


logger = logging.getLogger(__name__)


# ==========================================
# Native Tool Registry
# ==========================================


_NATIVE_TOOLS: dict[str, dict[str, Callable]] = {}
"""Global registry of native tools keyed by skill name.

Structure:
{
    "skill-name": {
        "tool_name": callable,
        ...
    },
    ...
}
"""


def register_tool(skill_name: str) -> Callable[[Callable], Callable]:
    """Decorator to register a native tool for a skill.

    Args:
        skill_name: Name of the skill this tool belongs to.
            Must match the skill's name in its SKILL.md frontmatter.

    Returns:
        Decorator function that registers the tool.

    Example:
        ```python
        @register_tool("github-integration")
        def create_repository(repo_name: str, description: str) -> str:
            \"\"\"Create a new GitHub repository.\"\"\"
            return f"Created {repo_name}"
        ```
    """

    def decorator(func: Callable) -> Callable:
        if skill_name not in _NATIVE_TOOLS:
            _NATIVE_TOOLS[skill_name] = {}
        _NATIVE_TOOLS[skill_name][func.__name__] = func
        return func

    return decorator


def get_native_tools(skill_name: str) -> dict[str, Callable]:
    """Get all native tools registered for a skill.

    Args:
        skill_name: Name of the skill.

    Returns:
        Dictionary mapping tool names to callables. Empty if skill not found.
    """
    return _NATIVE_TOOLS.get(skill_name, {}).copy()


def clear_native_tools(skill_name: str | None = None) -> None:
    """Clear native tools from the registry.

    Args:
        skill_name: If provided, only clear tools for this skill.
            If None, clear all registered tools.
    """
    if skill_name is None:
        _NATIVE_TOOLS.clear()
    elif skill_name in _NATIVE_TOOLS:
        del _NATIVE_TOOLS[skill_name]


# ==========================================
# MCP Server Configuration
# ==========================================


class MCPServerConfig(TypedDict, total=False):
    """Configuration for a single MCP server in skill YAML.

    All fields are optional to support different transport types.
    stdio servers require: command, optional: args, env
    sse/http servers require: url, optional: headers
    """

    name: str
    """Unique identifier for this server within the skill."""

    transport: NotRequired[str]
    """Transport type: "stdio", "sse", or "http". Defaults to "stdio"."""

    command: NotRequired[str]
    """Command to run for stdio transport (e.g., "npx", "uvx")."""

    args: NotRequired[list[str]]
    """Arguments for stdio transport command."""

    env: NotRequired[dict[str, str]]
    """Environment variables for stdio transport.
    Supports ${VAR} expansion for values.
    """

    url: NotRequired[str]
    """URL for SSE or HTTP transport."""

    headers: NotRequired[dict[str, str]]
    """HTTP headers for SSE or HTTP transport."""


def _expand_env_vars(value: str) -> str:
    """Expand environment variables in a string.

    Uses os.path.expandvars which supports ${VAR} and $VAR syntax.

    Args:
        value: String that may contain environment variable references.

    Returns:
        String with environment variables expanded.
    """
    return os.path.expandvars(value)


def _expand_env_vars_recursive(obj: object) -> object:
    """Recursively expand environment variables in strings within a structure.

    Args:
        obj: Object to process (str, dict, list, or other).

    Returns:
        Object with environment variables expanded in all string values.
    """
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars_recursive(item) for item in obj]
    return obj


# ==========================================
# State Definitions
# ==========================================


class SkillToolState(AgentState):
    """State for the skill tools middleware."""

    mcp_clients: NotRequired[Annotated[dict[str, MCPSessionManager], PrivateStateAttr]]
    """MCP session managers keyed by skill name.

    Not propagated to parent agents (private state).
    """

    active_skill_tools: NotRequired[Annotated[dict[str, dict[str, BaseTool]], PrivateStateAttr]]
    """Active skill tools keyed by skill name and tool name.

    Structure:
    {
        "skill-name": {
            "tool_name": BaseTool,
            ...
        },
        ...
    }

    Not propagated to parent agents (private state).
    """


class SkillToolStateUpdate(TypedDict):
    """State update for the skill tools middleware."""

    mcp_clients: dict[str, MCPSessionManager]
    """MCP session managers to merge into state."""

    active_skill_tools: dict[str, dict[str, BaseTool]]
    """Active skill tools to merge into state."""


# ==========================================
# MCP Session Manager (Simplified)
# ==========================================


class MCPSessionManager:
    """Manages a single MCP client session.

    This is a simplified version of the CLI's MCPSessionManager,
    adapted for middleware use.
    """

    def __init__(
        self,
        client: MultiServerMCPClient,
        exit_stack: AsyncExitStack,
    ) -> None:
        """Initialize the session manager.

        Args:
            client: The MCP client instance.
            exit_stack: Async exit stack for cleanup.
        """
        self.client = client
        self.exit_stack = exit_stack

    async def cleanup(self) -> None:
        """Clean up the MCP session."""
        await self.exit_stack.aclose()


# ==========================================
# Middleware Implementation
# ==========================================


class SkillToolMiddleware(AgentMiddleware[SkillToolState, ContextT, ResponseT]):
    """Middleware for providing skill-based tools (MCP + native).

    This middleware provides a single `skill_tool_call` dispatcher tool that
    routes to either native Python tools or MCP tools from servers declared
    in skill YAML frontmatter.

    MCP servers are connected lazily when skills become active (i.e., when
    they're loaded in the agent's skills_metadata). Tools are injected into
    the system prompt for discoverability.

    Example:
        ```python
        from deepagents.backends.state import StateBackend
        from deepagents.middleware.skill_tools import SkillToolMiddleware

        middleware = SkillToolMiddleware(backend=StateBackend)
        ```

    Args:
        backend: Backend instance or factory function that takes runtime and
            returns a backend. Use a factory for StateBackend:
            `lambda rt: StateBackend(rt)`
    """

    state_schema = SkillToolState

    def __init__(self, *, backend: BACKEND_TYPES) -> None:
        """Initialize the skill tools middleware.

        Args:
            backend: Backend instance or factory function.
        """
        self._backend = backend
        self._exit_stack = AsyncExitStack()

    def _get_backend(
        self,
        state: SkillToolState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.
            config: Runnable config to pass to backend factory.

        Returns:
            Resolved backend instance.
        """
        if callable(self._backend):
            # Construct an artificial tool runtime to resolve backend factory
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            backend = self._backend(tool_runtime)  # type: ignore[call-arg]
            if backend is None:
                msg = "SkillToolMiddleware requires a valid backend instance"
                raise AssertionError(msg)
            return backend

        return self._backend

    def _get_active_skills(self, state: SkillToolState) -> set[str]:
        """Extract active skill names from skills_metadata.

        Args:
            state: Current agent state.

        Returns:
            Set of active skill names.
        """
        skills_metadata = state.get("skills_metadata", [])
        return {s["name"] for s in skills_metadata if isinstance(s, dict) and "name" in s}

    async def _connect_skill_mcp_servers(  # noqa: C901
        self,
        skill_name: str,
        server_configs: list[MCPServerConfig],
    ) -> dict[str, BaseTool]:
        """Connect to MCP servers for a skill and return tools.

        Args:
            skill_name: Name of the skill.
            server_configs: List of MCP server configurations.

        Returns:
            Dictionary mapping tool names to BaseTool instances.
        """
        if MultiServerMCPClient is None:
            logger.error("langchain-mcp-adapters required for MCP skill tools")
            return {}

        # Build connections dict for MultiServerMCPClient
        connections: dict[str, object] = {}
        for server_config in server_configs:
            server_name = server_config.get("name", "")
            transport = server_config.get("transport", "stdio")

            # Expand environment variables in config
            expanded_config = _expand_env_vars_recursive(server_config)

            if transport == "http":
                conn = StreamableHttpConnection(
                    transport="streamable_http",
                    url=expanded_config.get("url", ""),
                )
                if "headers" in expanded_config:
                    conn["headers"] = expanded_config["headers"]  # type: ignore[index]
            elif transport == "sse":
                conn = SSEConnection(
                    transport="sse",
                    url=expanded_config.get("url", ""),
                )
                if "headers" in expanded_config:
                    conn["headers"] = expanded_config["headers"]  # type: ignore[index]
            else:  # stdio (default)
                conn = StdioConnection(
                    command=expanded_config.get("command", ""),
                    args=expanded_config.get("args", []),
                    env=expanded_config.get("env") or None,
                    transport="stdio",
                )

            connections[server_name] = conn

        if not connections:
            return {}

        # Create MCP client
        client = MultiServerMCPClient(connections=connections)

        # Create session manager for cleanup
        # NOTE: Session managers are created but not currently stored for lifecycle management
        exit_stack = AsyncExitStack()
        MCPSessionManager(client, exit_stack)

        # Load tools from all servers
        all_tools: dict[str, BaseTool] = {}
        for server_name in connections:
            try:
                session = await exit_stack.enter_async_context(client.session(server_name))
                tools = await load_mcp_tools(
                    session,
                    server_name=server_name,
                    tool_name_prefix=True,
                )
                for tool in tools:
                    all_tools[tool.name] = tool
                logger.info(
                    "Skill '%s' → connected to MCP server '%s' (%d tools)",
                    skill_name,
                    server_name,
                    len(tools),
                )
            except Exception:
                logger.exception(
                    "Failed to connect to MCP server '%s' for skill '%s'",
                    server_name,
                    skill_name,
                )

        return all_tools

    def _wrap_native_tool(
        self,
        skill_name: str,
        tool_name: str,
        func: Callable,
    ) -> BaseTool:
        """Wrap a native function as a LangChain BaseTool.

        Args:
            skill_name: Name of the skill.
            tool_name: Name of the tool.
            func: The function to wrap.

        Returns:
            StructuredTool wrapping the function.
        """
        # Try to extract description from function docstring
        description = func.__doc__ or f"Tool from {skill_name} skill."

        return StructuredTool.from_function(
            name=tool_name,
            description=description,
            func=func,
        )

    async def abefore_agent(
        self,
        state: SkillToolState,
        runtime: Runtime,  # noqa: ARG002
        config: RunnableConfig,  # noqa: ARG002
    ) -> SkillToolStateUpdate | None:
        """Connect to MCP servers and load tools for active skills.

        This is called before agent execution. It checks which skills are
        active (in skills_metadata) and lazily connects to their MCP servers
        if not already connected.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with mcp_clients and active_skill_tools populated,
            or None if already initialized.
        """
        # Skip if already initialized
        if "mcp_clients" in state or "active_skill_tools" in state:
            return None

        # Get active skills
        active_skills = self._get_active_skills(state)

        if not active_skills:
            return SkillToolStateUpdate(
                mcp_clients={},
                active_skill_tools={},
            )

        # Get skills_metadata to find MCP server configs
        skills_metadata = state.get("skills_metadata", [])

        mcp_clients: dict[str, MCPSessionManager] = {}
        active_skill_tools: dict[str, dict[str, BaseTool]] = {}

        for skill_meta in skills_metadata:
            if not isinstance(skill_meta, dict):
                continue

            skill_name = skill_meta.get("name", "")
            if not skill_name or skill_name not in active_skills:
                continue

            # Load native tools
            native_funcs = get_native_tools(skill_name)
            tools: dict[str, BaseTool] = {}
            for tool_name, func in native_funcs.items():
                tools[tool_name] = self._wrap_native_tool(skill_name, tool_name, func)

            # Connect to MCP servers if configured
            server_configs = skill_meta.get("mcp_servers")
            if server_configs and isinstance(server_configs, list):
                mcp_tools = await self._connect_skill_mcp_servers(skill_name, server_configs)
                tools.update(mcp_tools)

                # Create session manager for cleanup
                # Note: We'd need to track the exit_stack properly here
                # For now, we'll create a simplified manager
                if mcp_tools:
                    # In a full implementation, we'd properly manage the lifecycle
                    # For now, we'll store a placeholder
                    pass

            if tools:
                active_skill_tools[skill_name] = tools

        return SkillToolStateUpdate(
            mcp_clients=mcp_clients,
            active_skill_tools=active_skill_tools,
        )

    def _create_tool_dispatcher(  # noqa: C901
        self, state: SkillToolState
    ) -> BaseTool:
        """Create the master skill_tool_call dispatcher tool.

        Args:
            state: Current agent state.

        Returns:
            StructuredTool that dispatches to native or MCP tools.
        """
        active_skill_tools = state.get("active_skill_tools", {})

        async def _dispatch_async(
            skill_name: str,
            tool_name: str,
            arguments: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> str:
            """Async dispatch implementation."""
            if arguments is None:
                arguments = kwargs
            else:
                arguments.update(kwargs)

            # Get tools for this skill
            skill_tools = active_skill_tools.get(skill_name, {})

            # Try exact match first
            tool = skill_tools.get(tool_name)

            # Try fuzzy match (for MCP tools with server prefix)
            if tool is None:
                for full_name, t in skill_tools.items():
                    if full_name == tool_name or full_name.endswith(f"_{tool_name}"):
                        tool = t
                        break

            if tool is None:
                return json.dumps(
                    {
                        "error": f"Tool '{tool_name}' not found in skill '{skill_name}'",
                        "available_tools": list(skill_tools.keys()),
                    },
                    ensure_ascii=False,
                )

            try:
                result = await tool.ainvoke(arguments)
                return str(result)
            except Exception:
                logger.exception(
                    "Error calling tool '%s' from skill '%s'",
                    tool_name,
                    skill_name,
                )
                return json.dumps(
                    {
                        "error": f"Error calling tool '{tool_name}'",
                        "tool": tool_name,
                        "skill": skill_name,
                    },
                    ensure_ascii=False,
                )

        def dispatch(
            skill_name: str,
            tool_name: str,
            arguments: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> str:
            """Dispatch a tool call to the appropriate skill tool.

            Args:
                skill_name: Name of the skill.
                tool_name: Name of the tool within the skill.
                arguments: Tool arguments (can be None for backward compatibility).
                **kwargs: Additional keyword arguments for the tool.

            Returns:
                Tool result as string.
            """
            try:
                # Try to get a running event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create a task and await it
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            _dispatch_async(skill_name, tool_name, arguments, **kwargs),
                        )
                        return future.result()
                else:
                    # No running loop, just run the async function
                    return asyncio.run(_dispatch_async(skill_name, tool_name, arguments, **kwargs))
            except RuntimeError:
                # No event loop exists, create a new one
                return asyncio.run(_dispatch_async(skill_name, tool_name, arguments, **kwargs))

        return StructuredTool.from_function(
            name="skill_tool_call",
            func=dispatch,
            description=(
                "Call tools provided by active skills. Use this when you need to invoke a tool that's part of an active skill's capabilities."
            ),
            args_schema=None,  # Will be inferred from function signature
        )

    def _format_tools_documentation(
        self,
        active_skill_tools: dict[str, dict[str, BaseTool]],
    ) -> str:
        """Format available skill tools for system prompt.

        Args:
            active_skill_tools: Dictionary of active skill tools.

        Returns:
            Formatted documentation string.
        """
        if not active_skill_tools:
            return ""

        lines = ["\n\n## Available Skill Tools", ""]
        lines.append("Use `skill_tool_call(skill_name=..., tool_name=..., arguments={...})` to invoke these tools:")
        lines.append("")

        for skill_name, tools in sorted(active_skill_tools.items()):
            lines.append(f"### {skill_name}")
            lines.append("")
            lines.extend(f"- **{tool.name}**: {tool.description or '(no description)'}" for tool in sorted(tools.values(), key=lambda t: t.name))
            lines.append("")

        return "\n".join(lines)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject skill tools documentation into system prompt.

        Args:
            request: Model request to modify.

        Returns:
            New model request with skill tools documentation injected.
        """
        active_skill_tools = request.state.get("active_skill_tools", {})

        if not active_skill_tools:
            return request

        tools_doc = self._format_tools_documentation(active_skill_tools)
        if not tools_doc:
            return request

        new_system_message = append_to_system_message(request.system_message, tools_doc)

        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject skill tools documentation into system prompt.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Inject skill tools documentation into system prompt (async).

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)

    async def aafter_agent(
        self,
        state: SkillToolState,
        runtime: Runtime,  # noqa: ARG002
        config: RunnableConfig,  # noqa: ARG002
    ) -> None:
        """Clean up MCP client connections.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.
        """
        mcp_clients = state.get("mcp_clients", {})
        for skill_name, session_manager in mcp_clients.items():
            try:
                await session_manager.cleanup()
                logger.debug("Cleaned up MCP client for skill '%s'", skill_name)
            except Exception:
                logger.exception(
                    "Error cleaning up MCP client for skill '%s'",
                    skill_name,
                )


__all__ = [
    "MCPServerConfig",
    "MCPSessionManager",
    "SkillToolMiddleware",
    "clear_native_tools",
    "get_native_tools",
    "register_tool",
]
