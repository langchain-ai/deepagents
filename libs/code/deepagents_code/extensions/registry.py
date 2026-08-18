"""In-memory registrations contributed by Python extensions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deepagents_code.extensions.models import RegisteredUnit

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool

    from deepagents_code.extensions.models import SourceInfo

logger = logging.getLogger(__name__)
RegistrySnapshot = tuple[int, int, int, int]


class ExtensionRegistry:
    """Hold extension registrations in load order."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self.middleware: list[RegisteredUnit[AgentMiddleware[Any, Any]]] = []
        self.tools: list[RegisteredUnit[BaseTool]] = []
        self.backend_routes: list[RegisteredUnit[BackendProtocol]] = []
        self.shutdown_hooks: list[RegisteredUnit[Callable[[], Any]]] = []

    def _snapshot(self) -> RegistrySnapshot:
        return (
            len(self.middleware),
            len(self.tools),
            len(self.backend_routes),
            len(self.shutdown_hooks),
        )

    def _rollback(self, snapshot: RegistrySnapshot) -> None:
        middleware, tools, routes, hooks = snapshot
        del self.middleware[middleware:]
        del self.tools[tools:]
        del self.backend_routes[routes:]
        del self.shutdown_hooks[hooks:]

    @staticmethod
    def _add[T](
        items: list[RegisteredUnit[T]],
        name: str,
        unit: T,
        source: SourceInfo,
    ) -> None:
        existing = next((item for item in items if item.name == name), None)
        if existing is not None:
            logger.warning(
                "Ignoring %r from %s: already registered by %s",
                name,
                source.label,
                existing.source.label,
            )
            return
        items.append(RegisteredUnit(name, unit, source))

    def add_middleware(
        self, middleware: AgentMiddleware[Any, Any], source: SourceInfo
    ) -> None:
        """Register uniquely named middleware."""
        self._add(
            self.middleware,
            getattr(middleware, "name", type(middleware).__name__),
            middleware,
            source,
        )

    def add_tool(self, tool: BaseTool, source: SourceInfo) -> None:
        """Register a uniquely named tool."""
        self._add(self.tools, tool.name, tool, source)

    def add_backend_route(
        self, prefix: str, backend: BackendProtocol, source: SourceInfo
    ) -> None:
        """Register a unique virtual filesystem route."""
        self._add(self.backend_routes, prefix, backend, source)

    def add_shutdown_hook(self, hook: Callable[[], Any], source: SourceInfo) -> None:
        """Register a teardown callback."""
        self.shutdown_hooks.append(RegisteredUnit(source.label, hook, source))
