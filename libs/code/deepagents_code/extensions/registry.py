"""In-memory registrations contributed by Python extensions."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    """Registration counts used to restore registry state."""

    middleware: int
    tools: int
    backend_routes: int
    shutdown_hooks: int


class SourceScope(StrEnum):
    """Authority scope that supplied an extension; temporary means one invocation."""

    USER = "user"
    PROJECT = "project"
    TEMPORARY = "temporary"


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """An extension entry file, ownership, and import shape."""

    path: Path
    is_package: bool = False
    source_id: str | None = None
    scope: SourceScope = SourceScope.USER
    version: str | None = None
    installed_root: Path | None = None

    @property
    def label(self) -> str:
        """Short extension label."""
        if self.source_id is not None:
            return self.source_id
        return self.path.parent.name if self.is_package else self.path.stem

    def as_dict(self) -> dict[str, str | bool | None]:
        """Return JSON-safe provenance without reading extension contents."""
        return {
            "path": str(self.path),
            "source": "extension",
            "scope": self.scope.value,
            "origin": "package" if self.is_package else "top-level",
            "is_package": self.is_package,
            "source_id": self.source_id,
            "version": self.version,
            "installed_root": (
                str(self.installed_root) if self.installed_root is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class RegisteredUnit[T]:
    """A registered unit paired with its source."""

    name: str
    unit: T
    source: SourceInfo


class ExtensionError(Exception):
    """Raised when an extension cannot be loaded."""


class ExtensionRegistry:
    """Hold extension registrations in load order."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self.middleware: list[RegisteredUnit[AgentMiddleware[Any, Any]]] = []
        self.tools: list[RegisteredUnit[BaseTool]] = []
        self.backend_routes: list[RegisteredUnit[BackendProtocol]] = []
        self.shutdown_hooks: list[RegisteredUnit[Callable[[], Any]]] = []
        self._lock = threading.RLock()
        self._registration_listeners: list[
            Callable[[str, RegisteredUnit[Any]], None]
        ] = []
        self._restart_required = False

    @property
    def restart_required(self) -> bool:
        """Whether a late registration needs a new agent graph."""
        with self._lock:
            return self._restart_required

    def require_restart(self) -> None:
        """Record that registrations have changed graph construction inputs."""
        with self._lock:
            self._restart_required = True

    def _snapshot(self) -> RegistrySnapshot:
        with self._lock:
            return RegistrySnapshot(
                middleware=len(self.middleware),
                tools=len(self.tools),
                backend_routes=len(self.backend_routes),
                shutdown_hooks=len(self.shutdown_hooks),
            )

    def _rollback(self, snapshot: RegistrySnapshot) -> None:
        with self._lock:
            del self.middleware[snapshot.middleware :]
            del self.tools[snapshot.tools :]
            del self.backend_routes[snapshot.backend_routes :]
            del self.shutdown_hooks[snapshot.shutdown_hooks :]

    def subscribe_to_registrations(
        self, listener: Callable[[str, RegisteredUnit[Any]], None]
    ) -> None:
        """Observe registrations and allow the callback to reject them.

        Args:
            listener: Called with the registration kind and registered unit. Raising
                rolls back that registration.
        """
        with self._lock:
            self._registration_listeners.append(listener)

    def find_tool(self, name: str) -> RegisteredUnit[BaseTool] | None:
        """Return the current extension tool named `name`."""
        with self._lock:
            return next((item for item in self.tools if item.name == name), None)

    def tool_units(self) -> tuple[RegisteredUnit[BaseTool], ...]:
        """Return a stable snapshot for one model request."""
        with self._lock:
            return tuple(self.tools)

    def registrations(self) -> tuple[tuple[str, RegisteredUnit[Any]], ...]:
        """Return every registration in display order."""
        with self._lock:
            return tuple(
                (kind, item)
                for kind, items in (
                    ("middleware", self.middleware),
                    ("tool", self.tools),
                    ("backend_route", self.backend_routes),
                    ("shutdown", self.shutdown_hooks),
                )
                for item in items
            )

    def _add[T](
        self,
        items: list[RegisteredUnit[T]],
        kind: str,
        name: str,
        unit: T,
        source: SourceInfo,
    ) -> None:
        with self._lock:
            existing = next((item for item in items if item.name == name), None)
            if existing is not None:
                logger.warning(
                    "Ignoring %r from %s: already registered by %s",
                    name,
                    source.label,
                    existing.source.label,
                )
                return
            registered = RegisteredUnit(name, unit, source)
            items.append(registered)
            listeners = tuple(self._registration_listeners)
        try:
            for listener in listeners:
                listener(kind, registered)
        except Exception:
            with self._lock:
                items.remove(registered)
            raise

    def add_middleware(
        self, middleware: AgentMiddleware[Any, Any], source: SourceInfo
    ) -> None:
        """Register uniquely named middleware."""
        self._add(
            self.middleware,
            "middleware",
            getattr(middleware, "name", type(middleware).__name__),
            middleware,
            source,
        )

    def add_tool(self, tool: BaseTool, source: SourceInfo) -> None:
        """Register a uniquely named tool."""
        self._add(self.tools, "tool", tool.name, tool, source)

    def add_backend_route(
        self, prefix: str, backend: BackendProtocol, source: SourceInfo
    ) -> None:
        """Register a unique virtual filesystem route."""
        self._add(self.backend_routes, "backend_route", prefix, backend, source)

    def add_shutdown_hook(self, hook: Callable[[], Any], source: SourceInfo) -> None:
        """Register a teardown callback."""
        registered = RegisteredUnit(source.label, hook, source)
        with self._lock:
            self.shutdown_hooks.append(registered)
            listeners = tuple(self._registration_listeners)
        try:
            for listener in listeners:
                listener("shutdown", registered)
        except Exception:
            with self._lock:
                self.shutdown_hooks.remove(registered)
            raise
