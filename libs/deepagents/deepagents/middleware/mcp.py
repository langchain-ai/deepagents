"""MCP (Model Context Protocol) middleware for progressive tool disclosure."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, ClassVar, NotRequired, TypedDict

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

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol
from deepagents.middleware.filesystem import FileData, FilesystemState, _file_data_reducer

logger = logging.getLogger(__name__)


class MCPServerConfig(TypedDict):
    """Configuration for an MCP server (HTTP transport)."""

    name: str
    url: str
    headers: NotRequired[dict[str, str]]


class MCPToolMetadata(TypedDict):
    """Metadata for an MCP tool stored in the filesystem."""

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    status: NotRequired[str]


class MCPState(AgentState):
    """State for the MCP middleware.

    Works with any backend implementing BackendProtocol:
    - StateBackend: Stores metadata in agent state (ephemeral)
    - FilesystemBackend: Stores metadata as files (persistent)
    - StoreBackend: Stores metadata in LangGraph store (persistent)
    - Any custom backend implementing BackendProtocol

    The 'files' field is only used when StateBackend is the backend.
    For external backends, files are persisted directly and this field remains empty.
    """

    mcp_metadata: NotRequired[Annotated[dict[str, MCPToolMetadata], PrivateStateAttr]]
    mcp_initialized: NotRequired[Annotated[bool, PrivateStateAttr]]
    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]


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

    # Global client cache: maps cache_key -> (client, ref_count)
    _client_cache: ClassVar[dict[str, tuple[Any, int]]] = {}
    _client_lock: ClassVar[asyncio.Lock | None] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._client_lock is None:
            cls._client_lock = asyncio.Lock()
        return cls._client_lock

    def __init__(
        self,
        *,
        servers: list[MCPServerConfig],
        backend: BACKEND_TYPES | None = None,
        mcp_prefix: str = "/.mcp",
        sync_on_startup: bool = True,
    ) -> None:
        """Initialize MCP middleware with progressive tool disclosure.

        Args:
            servers: List of MCP server configurations (HTTP transport).
            backend: Backend for storing MCP metadata. Supports any BackendProtocol:
                - None (default): Uses StateBackend factory (ephemeral, in agent state)
                - BackendProtocol instance: Direct backend instance
                - Callable: Factory function (runtime) -> BackendProtocol

                Examples:
                - StateBackend: Metadata in agent state (ephemeral)
                - FilesystemBackend: Metadata as files on disk (persistent)
                - StoreBackend: Metadata in LangGraph store (persistent, cross-thread)
                - CompositeBackend: Route different paths to different backends

            mcp_prefix: Virtual directory for MCP metadata (default: "/.mcp").
            sync_on_startup: Whether to discover and sync tool metadata on first agent run.
        """
        self._validate_servers(servers)
        self.servers = servers
        self.mcp_prefix = mcp_prefix
        self.sync_on_startup = sync_on_startup
        self._cache_key: str | None = None
        self.tools: list[BaseTool] = [self._create_mcp_invoke_tool()]
        # Use provided backend or default to StateBackend factory
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

    def _validate_servers(self, servers: list[MCPServerConfig]) -> None:
        if not servers:
            raise ValueError("At least one MCP server must be configured")

        names: set[str] = set()
        for server in servers:
            if "name" not in server:
                raise ValueError("MCP server requires 'name'")
            if "url" not in server:
                raise ValueError(f"MCP server '{server['name']}' requires 'url'")
            if server["name"] in names:
                raise ValueError(f"Duplicate server name: {server['name']}")
            names.add(server["name"])

    def _get_backend(
        self,
        state: MCPState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Constructs an artificial ToolRuntime to support backend factories
        that need runtime context (pattern from MemoryMiddleware).
        """
        if callable(self.backend):
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self.backend(tool_runtime)
        return self.backend

    def _compute_cache_key(self) -> str:
        key_parts = []
        for server in sorted(self.servers, key=lambda s: s["name"]):
            key_parts.append(f"{server['name']}:{server['url']}")
        return "|".join(key_parts)

    @property
    def _mcp_client(self) -> Any | None:
        if self._cache_key is None:
            return None
        entry = MCPMiddleware._client_cache.get(self._cache_key)
        return entry[0] if entry else None

    async def connect(self) -> None:
        """Connect to MCP servers via HTTP."""
        cache_key = self._compute_cache_key()

        async with self._get_lock():
            if cache_key in MCPMiddleware._client_cache:
                client, ref_count = MCPMiddleware._client_cache[cache_key]
                MCPMiddleware._client_cache[cache_key] = (client, ref_count + 1)
                self._cache_key = cache_key
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
                    "transport": "http",
                    "url": server["url"],
                    "headers": server.get("headers", {}),
                }
                for server in self.servers
            }

            client = MultiServerMCPClient(server_configs)
            MCPMiddleware._client_cache[cache_key] = (client, 1)
            self._cache_key = cache_key

    async def close(self) -> None:
        """Cleanup MCP connections."""
        if self._cache_key is None:
            return

        async with self._get_lock():
            if self._cache_key not in MCPMiddleware._client_cache:
                self._cache_key = None
                return

            client, ref_count = MCPMiddleware._client_cache[self._cache_key]
            if ref_count > 1:
                MCPMiddleware._client_cache[self._cache_key] = (client, ref_count - 1)
            else:
                del MCPMiddleware._client_cache[self._cache_key]

            self._cache_key = None

    async def __aenter__(self) -> "MCPMiddleware":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _sync_metadata(self, backend: BackendProtocol) -> dict[str, Any]:
        """Build MCP tool metadata and write to backend.

        Args:
            backend: Backend to write metadata files to.

        Returns:
            Dictionary of files_update for state merge (from checkpoint backends),
            or empty dict (for external backends that persist directly).
        """
        if self._mcp_client is None:
            logger.warning("Cannot sync metadata: MCP client not connected")
            return {}

        files_update: dict[str, Any] = {}
        files_written = 0

        try:
            mcp_tools = await self._mcp_client.get_tools()
            tools_by_server: dict[str, list[Any]] = {}
            for tool in mcp_tools:
                server_name = self._extract_server_name(tool)
                tools_by_server.setdefault(server_name, []).append(tool)

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
                    file_path = f"{self.mcp_prefix}/{server_name}/{tool_name}.json"
                    content = json.dumps(metadata, indent=2)

                    # Use backend interface instead of direct state manipulation
                    result = await backend.awrite(file_path, content)
                    if result.error:
                        logger.error("Failed to write MCP metadata %s: %s", file_path, result.error)
                        continue
                    files_written += 1
                    # Accumulate state updates for checkpoint backends
                    if result.files_update:
                        files_update.update(result.files_update)

            if files_written:
                logger.info(
                    "Prepared %d MCP tool metadata files for %s",
                    files_written,
                    self.mcp_prefix,
                )
        except Exception as e:
            logger.error("Error building MCP metadata: %s", e)

        return files_update

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
                mcp_tools = await middleware._mcp_client.get_tools()
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

        files_update: dict[str, Any] = {}

        # Build metadata files if connected and sync_on_startup is enabled
        if self._mcp_client is not None and self.sync_on_startup:
            try:
                # Resolve backend using helper
                backend = self._get_backend(state, runtime, config)
                files_update = await self._sync_metadata(backend)
            except Exception as e:
                logger.error("Failed to build MCP metadata: %s", e)

        result: dict[str, Any] = {"mcp_initialized": True}
        if files_update:
            result["files"] = files_update

        return result

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
