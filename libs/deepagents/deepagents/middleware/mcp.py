"""MCP (Model Context Protocol) middleware for progressive tool disclosure."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)


class MCPServerConfig(TypedDict):
    """Configuration for an MCP server."""

    name: str
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]


class MCPToolMetadata(TypedDict):
    """Metadata for an MCP tool stored in the filesystem."""

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    status: NotRequired[str]


class MCPState(AgentState):
    """State for the MCP middleware."""

    mcp_metadata: NotRequired[Annotated[dict[str, MCPToolMetadata], PrivateStateAttr]]
    mcp_initialized: NotRequired[Annotated[bool, PrivateStateAttr]]


MCP_SYSTEM_PROMPT = """
## MCP Tools (Model Context Protocol)

You have access to external tools provided by MCP servers. Tool metadata is organized
in the `/.mcp/` directory using a folder-per-server structure.

**Discovery Pattern:**

1. **List available servers**: Use `ls /.mcp/` to see all MCP servers
2. **Browse server tools**: Use `ls /.mcp/<server>/` to see tools from a specific server
3. **Read tool details**: Use `read_file /.mcp/<server>/<tool>.json` to see full schema
4. **Search capabilities**: Use `grep` to search tool descriptions across servers

**Tool Execution:**

To invoke an MCP tool, use the `mcp_invoke` tool:
- `tool_name`: The name of the tool to invoke
- `arguments`: JSON object with tool parameters

**Example Workflow:**

User: "Search the web for latest AI news"

1. ls /.mcp/                          # See: brave-search/, filesystem/
2. ls /.mcp/brave-search/             # See: search.json, suggest.json
3. read_file /.mcp/brave-search/search.json  # Read full schema
4. mcp_invoke(tool_name="search", arguments={"query": "latest AI news"})

**Best Practices:**

- Only read tool schemas when needed (progressive disclosure)
- Use `grep` to search across all tool descriptions
- Check tool `status` field for availability
"""


class MCPMiddleware(AgentMiddleware):
    """Middleware for progressive MCP tool discovery and execution."""

    state_schema = MCPState

    def __init__(
        self,
        *,
        servers: list[MCPServerConfig],
        mcp_root: str = "/tmp/.mcp",
        sync_on_startup: bool = True,
    ) -> None:
        self._validate_servers(servers)
        self.servers = servers
        self.mcp_root = mcp_root
        self.sync_on_startup = sync_on_startup
        self._mcp_client: Any | None = None

        from deepagents.backends.filesystem import FilesystemBackend

        self._mcp_backend = FilesystemBackend(root_dir=mcp_root, virtual_mode=True)
        self.tools: list[BaseTool] = [self._create_mcp_invoke_tool()]

    def _validate_servers(self, servers: list[MCPServerConfig]) -> None:
        if not servers:
            raise ValueError("At least one MCP server must be configured")

        names: set[str] = set()
        for server in servers:
            if "name" not in server or "command" not in server:
                raise ValueError("MCP server requires 'name' and 'command'")
            if server["name"] in names:
                raise ValueError(f"Duplicate server name: {server['name']}")
            names.add(server["name"])

    async def connect(self) -> None:
        """Connect to MCP servers and sync metadata."""
        if self._mcp_client is not None:
            return

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as e:
            raise ImportError(
                "langchain-mcp-adapters is required for MCP support. "
                "Install with: pip install langchain-mcp-adapters"
            ) from e

        server_configs = {
            server["name"]: {
                "command": server["command"],
                "args": server.get("args", []),
                "env": server.get("env", {}),
            }
            for server in self.servers
        }

        self._mcp_client = MultiServerMCPClient(server_configs)
        await self._mcp_client.__aenter__()

        if self.sync_on_startup:
            await self._sync_metadata()

    async def close(self) -> None:
        """Cleanup MCP connections."""
        if self._mcp_client is not None:
            try:
                await self._mcp_client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error closing MCP client: %s", e)
            finally:
                self._mcp_client = None

    async def __aenter__(self) -> "MCPMiddleware":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _sync_metadata(self) -> None:
        """Sync MCP tool metadata to filesystem."""
        if self._mcp_client is None:
            logger.warning("Cannot sync metadata: MCP client not connected")
            return

        try:
            mcp_tools = self._mcp_client.get_tools()
            tools_by_server: dict[str, list[Any]] = {}
            for tool in mcp_tools:
                server_name = self._extract_server_name(tool)
                tools_by_server.setdefault(server_name, []).append(tool)

            files_to_upload: list[tuple[str, bytes]] = []
            for server_name, server_tools in tools_by_server.items():
                for tool in server_tools:
                    tool_name = self._extract_tool_name(tool, server_name)
                    metadata: MCPToolMetadata = {
                        "server": server_name,
                        "name": tool_name,
                        "description": getattr(tool, "description", "") or "",
                        "input_schema": self._get_tool_schema(tool),
                        "status": "available",
                    }
                    file_path = f"/{server_name}/{tool_name}.json"
                    content = json.dumps(metadata, indent=2)
                    files_to_upload.append((file_path, content.encode("utf-8")))

            if files_to_upload:
                await self._mcp_backend.aupload_files(files_to_upload)
                logger.info("Synced %d MCP tool metadata files to %s", len(files_to_upload), self.mcp_root)
        except Exception as e:
            logger.error("Error syncing MCP metadata: %s", e)

    def _extract_server_name(self, tool: Any) -> str:
        tool_name = getattr(tool, "name", "")
        for server in self.servers:
            if tool_name.startswith(f"{server['name']}_"):
                return server["name"]
        return self.servers[0]["name"] if self.servers else "unknown"

    def _extract_tool_name(self, tool: Any, server_name: str) -> str:
        tool_name = getattr(tool, "name", "unknown")
        prefix = f"{server_name}_"
        if tool_name.startswith(prefix):
            return tool_name[len(prefix) :]
        return tool_name

    def _get_tool_schema(self, tool: Any) -> dict[str, Any]:
        try:
            if hasattr(tool, "args_schema") and tool.args_schema is not None:
                return tool.args_schema.schema()
            if hasattr(tool, "get_input_schema"):
                return tool.get_input_schema().schema()
        except Exception as e:
            logger.debug("Could not extract schema from tool: %s", e)
        return {"type": "object", "properties": {}}

    def _create_mcp_invoke_tool(self) -> BaseTool:
        middleware = self

        async def mcp_invoke(
            tool_name: str,
            arguments: dict[str, Any],
            runtime: ToolRuntime,  # type: ignore[type-arg]
        ) -> str:
            if middleware._mcp_client is None:
                return (
                    "Error: MCP not connected. "
                    "Use agent.ainvoke() to auto-connect, or call middleware.connect() explicitly."
                )

            try:
                mcp_tools = middleware._mcp_client.get_tools()
                target_tool = None
                for tool in mcp_tools:
                    name = getattr(tool, "name", "")
                    if name == tool_name or name.endswith(f"_{tool_name}"):
                        target_tool = tool
                        break

                if target_tool is None:
                    available = [getattr(t, "name", "") for t in mcp_tools]
                    return f"Error: Tool '{tool_name}' not found. Available tools: {available}"

                result = await target_tool.ainvoke(arguments)
                return str(result)
            except Exception as e:
                logger.error("Error executing MCP tool '%s': %s", tool_name, e)
                return f"Error executing '{tool_name}': {e!s}"

        return StructuredTool.from_function(
            name="mcp_invoke",
            description=(
                "Invoke an MCP tool. First discover tools via `ls /.mcp/` and "
                "read schemas from `/.mcp/<server>/<tool>.json`."
            ),
            func=None,
            coroutine=mcp_invoke,
        )

    def before_agent(
        self,
        state: MCPState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> dict[str, Any] | None:
        if state.get("mcp_initialized"):
            return None
        return {"mcp_initialized": True}

    async def abefore_agent(
        self,
        state: MCPState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> dict[str, Any] | None:
        if state.get("mcp_initialized"):
            return None

        if self._mcp_client is None:
            try:
                await self.connect()
            except Exception as e:
                logger.error("Failed to connect to MCP servers: %s", e)

        return {"mcp_initialized": True}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + MCP_SYSTEM_PROMPT
        else:
            system_prompt = MCP_SYSTEM_PROMPT
        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + MCP_SYSTEM_PROMPT
        else:
            system_prompt = MCP_SYSTEM_PROMPT
        return await handler(request.override(system_prompt=system_prompt))


__all__ = ["MCPMiddleware", "MCPServerConfig", "MCPState", "MCPToolMetadata"]
