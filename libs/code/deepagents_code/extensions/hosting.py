"""Host startup and runtime extension registrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code.extensions.registry import ExtensionError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Sequence

    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
        ToolCallRequest,
    )
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import BaseTool
    from langgraph.types import Command

    from deepagents_code.extensions.registry import ExtensionRegistry, RegisteredUnit


def _tool_name(tool: object) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


class ExtensionRuntimeMiddleware(AgentMiddleware):
    """Expose tools registered after the agent graph was built."""

    name = "__deepagents_extension_runtime__"

    def __init__(self, registry: ExtensionRegistry) -> None:
        """Bind dynamic model and tool dispatch to `registry`."""
        self._registry = registry

    def _tools(
        self, existing: Sequence[BaseTool | dict[str, Any]]
    ) -> list[BaseTool | dict[str, Any]]:
        tools = list(existing)
        indexes = {_tool_name(tool): index for index, tool in enumerate(tools)}
        for registered in self._registry.tool_units():
            index = indexes.get(registered.name)
            if index is None:
                indexes[registered.name] = len(tools)
                tools.append(registered.unit)
            else:
                tools[index] = registered.unit
        return tools

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Inject the latest extension-tool snapshot into a sync model call.

        Returns:
            The wrapped model response.
        """
        return handler(request.override(tools=self._tools(request.tools)))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Inject the latest extension-tool snapshot into an async model call.

        Returns:
            The wrapped model response.
        """
        return await handler(request.override(tools=self._tools(request.tools)))

    def _tool_request(self, request: ToolCallRequest) -> ToolCallRequest:
        registered = self._registry.find_tool(request.tool_call["name"])
        return request if registered is None else request.override(tool=registered.unit)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Execute a sync runtime tool through LangChain's normal handler.

        Returns:
            The wrapped tool result.
        """
        return handler(self._tool_request(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Execute an async runtime tool through LangChain's normal handler.

        Returns:
            The wrapped tool result.
        """
        return await handler(self._tool_request(request))


def validate_backend_route(
    item: RegisteredUnit[Any],
    protected_routes: Collection[str],
    *,
    sandbox_active: bool,
) -> None:
    """Reject a backend route that violates host storage boundaries.

    The sandbox check is deliberately shallow. Wrapping a filesystem backend
    in another `BackendProtocol` implementation changes who owns that safety
    contract, so dcode does not inspect arbitrary backend object graphs.

    Args:
        item: Backend route registration to validate.
        protected_routes: Internal route prefixes unavailable to extensions.
        sandbox_active: Whether the default execution backend is sandboxed.

    Raises:
        ExtensionError: If the route overlaps internal storage or directly
            exposes a host filesystem backend to a sandboxed agent.
    """
    if any(
        item.name.startswith(prefix) or prefix.startswith(item.name)
        for prefix in protected_routes
    ):
        msg = (
            f"Extension backend route {item.name!r} from {item.source.label} "
            "overlaps an internal route"
        )
        raise ExtensionError(msg)
    if sandbox_active and isinstance(item.unit, FilesystemBackend):
        msg = (
            f"Extension backend route {item.name!r} from {item.source.label} "
            f"cannot mount {type(item.unit).__name__} in sandbox mode"
        )
        raise ExtensionError(msg)


def bind_runtime_host_policy(
    registry: ExtensionRegistry,
    protected_routes: Collection[str],
    *,
    sandbox_active: bool = False,
) -> None:
    """Validate late routes and flag graph-bound registrations for restart."""

    def apply(kind: str, item: RegisteredUnit[Any]) -> None:
        if kind == "middleware":
            registry.require_restart()
            return
        if kind == "backend_route":
            validate_backend_route(
                item, protected_routes, sandbox_active=sandbox_active
            )
            registry.require_restart()

    registry.subscribe_to_registrations(apply)
