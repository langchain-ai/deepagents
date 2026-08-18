"""In-memory registry for units contributed by Python extensions."""

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
"""Registration counts captured before an extension factory runs."""


class ExtensionRegistry:
    """Hold extension registrations in deterministic load order."""

    def __init__(self) -> None:
        """Create an empty extension registry."""
        self._middleware: list[RegisteredUnit[AgentMiddleware[Any, Any]]] = []
        self._tools: list[RegisteredUnit[BaseTool]] = []
        self._backend_routes: list[RegisteredUnit[BackendProtocol]] = []
        self._shutdown_hooks: list[RegisteredUnit[Callable[[], Any]]] = []

    def _snapshot(self) -> RegistrySnapshot:
        """Capture registration counts for factory-failure rollback.

        Returns:
            Per-kind registration counts for the current registry state.
        """
        return (
            len(self._middleware),
            len(self._tools),
            len(self._backend_routes),
            len(self._shutdown_hooks),
        )

    def _rollback(self, snapshot: RegistrySnapshot) -> None:
        """Remove registrations made after `snapshot`.

        Args:
            snapshot: Counts returned by `_snapshot` before factory execution.
        """
        middleware, tools, backend_routes, shutdown_hooks = snapshot
        del self._middleware[middleware:]
        del self._tools[tools:]
        del self._backend_routes[backend_routes:]
        del self._shutdown_hooks[shutdown_hooks:]

    @property
    def middleware(self) -> list[RegisteredUnit[AgentMiddleware[Any, Any]]]:
        """Registered middleware in load order."""
        return list(self._middleware)

    @property
    def tools(self) -> list[RegisteredUnit[BaseTool]]:
        """Registered tools in load order."""
        return list(self._tools)

    @property
    def backend_routes(self) -> list[RegisteredUnit[BackendProtocol]]:
        """Registered backend routes in load order."""
        return list(self._backend_routes)

    @property
    def shutdown_hooks(self) -> list[RegisteredUnit[Callable[[], Any]]]:
        """Registered teardown callbacks in load order."""
        return list(self._shutdown_hooks)

    def add_middleware(
        self,
        middleware: AgentMiddleware[Any, Any],
        source: SourceInfo,
    ) -> None:
        """Record uniquely named middleware.

        Args:
            middleware: Middleware instance to install on the agent.
            source: Extension provenance.
        """
        name = getattr(middleware, "name", type(middleware).__name__)
        if any(existing.name == name for existing in self._middleware):
            logger.warning(
                "Ignoring duplicate middleware %r from %s", name, source.label
            )
            return
        self._middleware.append(
            RegisteredUnit(name=name, unit=middleware, source=source)
        )

    def add_tool(self, tool: BaseTool, source: SourceInfo) -> None:
        """Record a tool, keeping the first registration of a name.

        Args:
            tool: Tool to expose to the model.
            source: Extension provenance.
        """
        existing = next((unit for unit in self._tools if unit.name == tool.name), None)
        if existing is not None:
            logger.warning(
                "Ignoring tool %r from %s: already registered by %s",
                tool.name,
                source.label,
                existing.source.label,
            )
            return
        self._tools.append(RegisteredUnit(name=tool.name, unit=tool, source=source))

    def add_backend_route(
        self,
        prefix: str,
        backend: BackendProtocol,
        source: SourceInfo,
    ) -> None:
        """Record a backend route, keeping the first registration of a prefix.

        Args:
            prefix: Virtual path prefix mounted by the backend.
            backend: Backend serving matching paths.
            source: Extension provenance.
        """
        existing = next(
            (unit for unit in self._backend_routes if unit.name == prefix), None
        )
        if existing is not None:
            logger.warning(
                "Ignoring backend route %r from %s: already registered by %s",
                prefix,
                source.label,
                existing.source.label,
            )
            return
        self._backend_routes.append(
            RegisteredUnit(name=prefix, unit=backend, source=source)
        )

    def add_shutdown_hook(
        self,
        hook: Callable[[], Any],
        source: SourceInfo,
    ) -> None:
        """Record a deterministic session teardown callback.

        Args:
            hook: Sync or async callback invoked during shutdown.
            source: Extension provenance.
        """
        self._shutdown_hooks.append(
            RegisteredUnit(name=source.label, unit=hook, source=source)
        )

    def is_empty(self) -> bool:
        """Return whether the registry contains no extension units."""
        return not (self._middleware or self._tools or self._backend_routes)

    def units(self) -> list[RegisteredUnit[Any]]:
        """Return all model-visible registrations in display order."""
        return [*self._middleware, *self._tools, *self._backend_routes]
