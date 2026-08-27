"""Public registrar passed to Python extension factories."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from deepagents_code.extensions.registry import ExtensionError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool

    from deepagents_code.extensions.registry import ExtensionRegistry, SourceInfo

_ROUTE_PREFIX = re.compile(r"^/(?:[a-z0-9][a-z0-9_-]*/)+$")


class ExtensionMode(StrEnum):
    """Runtime mode exposed to extensions."""

    INTERACTIVE = "interactive"
    HEADLESS = "headless"


class ExtensionAPI:
    """Factory-scoped registrar and read-only session context.

    Each extension receives a dedicated instance, which attributes every
    registration to the extension's source without exposing mutable dcode
    application state.
    """

    def __init__(
        self,
        registry: ExtensionRegistry,
        source: SourceInfo,
        *,
        cwd: Path,
        mode: ExtensionMode,
    ) -> None:
        """Bind a registrar to one extension's provenance.

        Args:
            registry: Shared registry receiving extension units.
            source: Provenance recorded on each registration.
            cwd: Working directory for this session.
            mode: Runtime mode, either `interactive` or `headless`.
        """
        self._registry = registry
        self._source = source
        self._cwd = cwd
        self._mode = ExtensionMode(mode)
        self._active = True

    def _deactivate(self) -> None:
        """Close this registrar after failed initialization or shutdown."""
        self._active = False

    def _ensure_active(self) -> None:
        if not self._active:
            msg = "Extension registration is closed for this session"
            raise ExtensionError(msg)

    @property
    def cwd(self) -> Path:
        """Working directory for this session."""
        return self._cwd

    @property
    def mode(self) -> ExtensionMode:
        """Runtime mode, either `interactive` or `headless`."""
        return self._mode

    @property
    def has_ui(self) -> bool:
        """Whether this session has an interactive terminal UI."""
        return self._mode == ExtensionMode.INTERACTIVE

    @property
    def path(self) -> Path:
        """Entry file for this extension."""
        return self._source.path

    def register_middleware(
        self,
        middleware: AgentMiddleware[Any, Any] | type[AgentMiddleware[Any, Any]],
    ) -> None:
        """Install LangChain middleware on the agent.

        A class is instantiated without arguments. Pass an instance when
        construction requires configuration.

        Args:
            middleware: Middleware class or instance.

        Raises:
            ExtensionError: If `middleware` is invalid or cannot be constructed.
        """
        self._ensure_active()
        from langchain.agents.middleware.types import AgentMiddleware

        try:
            instance = middleware() if isinstance(middleware, type) else middleware
        except Exception as exc:
            msg = f"Could not construct middleware {middleware!r}: {exc}"
            raise ExtensionError(msg) from exc
        if not isinstance(instance, AgentMiddleware):
            kind = type(instance).__name__
            msg = f"Registered middleware must be an AgentMiddleware, got {kind}"
            raise ExtensionError(msg)
        self._registry.add_middleware(instance, self._source)

    def register_tool(self, tool: BaseTool | Callable[..., Any]) -> None:
        """Expose an LLM-callable tool.

        Plain callables are converted using LangChain's tool schema inference.

        Args:
            tool: Tool instance or plain callable.

        Raises:
            ExtensionError: If a callable cannot be converted into a tool.
        """
        self._ensure_active()
        from langchain_core.tools import BaseTool, tool as as_tool

        if isinstance(tool, BaseTool):
            self._registry.add_tool(tool, self._source)
            return
        try:
            converted = as_tool(tool)
        except Exception as exc:
            name = getattr(tool, "__name__", repr(tool))
            msg = f"Could not convert {name} into a tool: {exc}"
            raise ExtensionError(msg) from exc
        self._registry.add_tool(converted, self._source)  # type: ignore[arg-type]  # callable overload returns BaseTool

    def register_backend_route(self, prefix: str, backend: BackendProtocol) -> None:
        """Mount a backend under a virtual filesystem path.

        File operations under `prefix` are routed to `backend` by the agent's
        `CompositeBackend`. Shell execution remains on the default backend and
        cannot access routed virtual content.

        Args:
            prefix: Lowercase absolute path ending in `/`, such as `/memories/`.
            backend: Backend serving file operations under the prefix.

        Raises:
            ExtensionError: If the prefix or backend is invalid.
        """
        self._ensure_active()
        from deepagents.backends.protocol import BackendProtocol

        if _ROUTE_PREFIX.fullmatch(prefix) is None:
            msg = (
                f"Invalid backend route prefix {prefix!r}: use lowercase path "
                "segments and include leading and trailing slashes"
            )
            raise ExtensionError(msg)
        if not isinstance(backend, BackendProtocol):
            msg = (
                f"Backend route {prefix!r} got {type(backend).__name__}, "
                "which is not a BackendProtocol"
            )
            raise ExtensionError(msg)
        self._registry.add_backend_route(prefix, backend, self._source)

    def on_shutdown(self, hook: Callable[[], Any]) -> None:
        """Register a deterministic session teardown callback.

        Args:
            hook: Sync or async zero-argument callback.

        Raises:
            ExtensionError: If `hook` is not callable.
        """
        self._ensure_active()
        if not callable(hook):
            msg = "Shutdown hook is not callable"
            raise ExtensionError(msg)
        self._registry.add_shutdown_hook(hook, self._source)
