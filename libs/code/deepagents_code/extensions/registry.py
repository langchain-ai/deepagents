"""In-memory registry of units contributed by extensions.

Collisions resolve by simple per-kind rules applied in load order, never by
scope: a tool name is claimed by its first registration, and a duplicate command
name coexists under a suffixed name. Provenance (`SourceInfo`) is recorded for
display and filtering only.

The registry accepts registrations after startup, so `register_*` called from a
command handler or another extension's callback behaves like one called from the
factory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deepagents_code.extensions.models import (
    RegisteredUnit,
    UnitSource,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool

    from deepagents_code.extensions.models import (
        CommandHandler,
        SourceInfo,
    )

logger = logging.getLogger(__name__)


class ExtensionRegistry:
    """Holds every unit registered by loaded extensions."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._middleware: list[RegisteredUnit[AgentMiddleware[Any, Any]]] = []
        self._tools: list[RegisteredUnit[BaseTool]] = []
        self._commands: list[RegisteredUnit[CommandHandler]] = []
        self._backend_routes: list[RegisteredUnit[BackendProtocol]] = []
        self._shutdown_hooks: list[RegisteredUnit[Callable[[], Any]]] = []
        self._descriptions: dict[str, str] = {}

    def _snapshot(self) -> tuple[int, int, int, int, int]:
        """Capture list lengths so a failed factory can be rolled back.

        Returns:
            Per-kind registration counts for the current registry state.
        """
        return (
            len(self._middleware),
            len(self._tools),
            len(self._commands),
            len(self._backend_routes),
            len(self._shutdown_hooks),
        )

    def _rollback(self, snapshot: tuple[int, int, int, int, int]) -> None:
        """Remove every registration made after `snapshot`.

        Args:
            snapshot: Per-kind list lengths returned by `_snapshot`.
        """
        middleware, tools, commands, backend_routes, shutdown_hooks = snapshot
        removed_commands = self._commands[commands:]
        del self._middleware[middleware:]
        del self._tools[tools:]
        del self._commands[commands:]
        del self._backend_routes[backend_routes:]
        del self._shutdown_hooks[shutdown_hooks:]
        for command in removed_commands:
            self._descriptions.pop(command.name, None)

    @property
    def middleware(self) -> list[RegisteredUnit[AgentMiddleware[Any, Any]]]:
        """Registered middleware in load order."""
        return list(self._middleware)

    @property
    def tools(self) -> list[RegisteredUnit[BaseTool]]:
        """Registered tools in load order."""
        return list(self._tools)

    @property
    def commands(self) -> list[RegisteredUnit[CommandHandler]]:
        """Registered commands in load order."""
        return list(self._commands)

    @property
    def backend_routes(self) -> list[RegisteredUnit[BackendProtocol]]:
        """Registered backend routes in load order, keyed by path prefix."""
        return list(self._backend_routes)

    @property
    def shutdown_hooks(self) -> list[RegisteredUnit[Callable[[], Any]]]:
        """Registered teardown callbacks in load order."""
        return list(self._shutdown_hooks)

    def add_shutdown_hook(
        self,
        hook: Callable[[], Any],
        source: SourceInfo,
    ) -> None:
        """Record a teardown callback for deterministic session cleanup.

        Args:
            hook: Callable invoked at session shutdown; may be async.
            source: Provenance of the registering extension.
        """
        self._shutdown_hooks.append(
            RegisteredUnit(name=source.label, unit=hook, source=source)
        )

    def command_description(self, name: str) -> str:
        """Return the description recorded for a command name.

        Returns:
            The registered description, or an empty string when none was given.
        """
        return self._descriptions.get(name, "")

    def add_middleware(
        self,
        middleware: AgentMiddleware[Any, Any],
        source: SourceInfo,
    ) -> None:
        """Record a middleware instance.

        Middleware names must be unique for `create_agent`, so a duplicate name
        is dropped with a warning rather than tripping the SDK's assertion.

        Args:
            middleware: Middleware instance to install on the agent.
            source: Provenance of the registering extension.
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
            source: Provenance of the registering extension.
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
        if source.source is UnitSource.EXTENSION:
            logger.debug("Extension %s registered tool %r", source.label, tool.name)
        self._tools.append(RegisteredUnit(name=tool.name, unit=tool, source=source))

    def add_command(
        self,
        name: str,
        handler: CommandHandler,
        source: SourceInfo,
        *,
        description: str = "",
    ) -> str:
        """Record a command, suffixing the name when it is already taken.

        Args:
            name: Command name without the leading slash.
            handler: Callable invoked when the command runs.
            source: Provenance of the registering extension.
            description: Short user-facing description.

        Returns:
            The name the command was registered under, which may carry a
                numeric suffix when the requested name collided.
        """
        resolved = name
        suffix = 2
        while any(unit.name == resolved for unit in self._commands):
            resolved = f"{name}-{suffix}"
            suffix += 1
        if resolved != name:
            logger.warning(
                "Command %r from %s renamed to %r to avoid a collision",
                name,
                source.label,
                resolved,
            )
        self._commands.append(
            RegisteredUnit(name=resolved, unit=handler, source=source)
        )
        self._descriptions[resolved] = description
        return resolved

    def add_backend_route(
        self,
        prefix: str,
        backend: BackendProtocol,
        source: SourceInfo,
    ) -> None:
        """Record a backend route, keeping the first registration of a prefix.

        Args:
            prefix: Virtual path prefix the backend is mounted at.
            backend: Backend that handles paths under the prefix.
            source: Provenance of the registering extension.
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

    def find_command(self, name: str) -> RegisteredUnit[CommandHandler] | None:
        """Look up a registered command by name.

        Returns:
            The matching command, or `None` when no command uses that name.
        """
        return next((unit for unit in self._commands if unit.name == name), None)

    def is_empty(self) -> bool:
        """Return whether nothing has been registered.

        Returns:
            `True` when no middleware, tools, commands, or backend routes are
                registered.
        """
        return not (
            self._middleware or self._tools or self._commands or self._backend_routes
        )

    def units(self) -> list[RegisteredUnit[Any]]:
        """Return every registered unit for provenance display.

        Returns:
            Middleware, tools, commands, and backend routes in that order.
        """
        return [
            *self._middleware,
            *self._tools,
            *self._commands,
            *self._backend_routes,
        ]
