"""DeepAgentsRuntime and the private _DeepAgentsRuntimeMixin.

`DeepAgentsRuntime` extends the LangGraph `Runtime` dataclass with a pre-resolved
`backend` field so middleware hooks no longer have to synthesize a fake `ToolRuntime`
from a bare `Runtime` just to call a backend factory.

`_DeepAgentsRuntimeMixin` is a private mixin for every deepagents middleware that
holds ``self._backend``.  It provides ``_resolve_backend_for_runtime(runtime)`` which
replaces the repetitive ToolRuntime-synthesis pattern found in the old ``_get_backend``
helpers, and a forward-compatible ``_build_runtime`` hook that will be called
automatically once LangChain ships ``AgentMiddleware._build_runtime`` support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

    from deepagents.backends.protocol import BackendProtocol

__all__ = ["DeepAgentsRuntime", "_DeepAgentsRuntimeMixin"]


@dataclass(frozen=True)
class DeepAgentsRuntime(Runtime):
    """LangGraph ``Runtime`` enriched with a pre-resolved backend.

    Middleware hooks that receive a ``DeepAgentsRuntime`` can access
    ``runtime.backend`` directly instead of synthesising a ``ToolRuntime``
    from the bare ``Runtime`` fields just to call a backend factory.

    Attributes:
        backend: A resolved ``BackendProtocol`` instance.  Always set before
            any middleware hook is dispatched when ``_DeepAgentsRuntimeMixin``
            is in the class hierarchy.
    """

    backend: BackendProtocol | None = field(default=None)


class _DeepAgentsRuntimeMixin:
    """Internal mixin that resolves ``self._backend`` into a ``DeepAgentsRuntime``.

    Attach this mixin to any deepagents middleware that holds ``self._backend``
    (which may be a concrete ``BackendProtocol`` instance *or* a factory
    callable that requires a ``ToolRuntime`` to resolve).

    The mixin provides two helpers:

    * ``_resolve_backend_for_runtime(runtime)`` — synthesises the ``ToolRuntime``
      when necessary and returns a concrete ``BackendProtocol``.  Call this from
      ``wrap_model_call`` / ``before_agent`` hooks that already have a ``Runtime``
      in scope.

    * ``_build_runtime(runtime)`` — forward-compatible hook.  Once LangChain adds
      ``AgentMiddleware._build_runtime`` support this will be called automatically
      before every hook dispatch, populating ``runtime.backend`` so that
      ``_resolve_backend_for_runtime`` becomes a trivial attribute access.
    """

    # Subclasses must set this at construction time.
    _backend: Any  # BackendProtocol | BackendFactory

    def _resolve_backend_for_runtime(
        self,
        runtime: Runtime,
    ) -> BackendProtocol:
        """Return a resolved backend from ``self._backend``.

        If ``self._backend`` is callable (a factory), a ``ToolRuntime`` is
        constructed from the ``Runtime`` fields so the factory can resolve.
        Otherwise the instance is returned directly.

        Args:
            runtime: The LangGraph ``Runtime`` (or any subclass) that is
                currently in scope for the middleware hook.

        Returns:
            Resolved ``BackendProtocol`` instance.
        """
        if isinstance(runtime, DeepAgentsRuntime) and runtime.backend is not None:
            # Already resolved by a prior _build_runtime call.
            return runtime.backend

        if callable(self._backend):
            from langchain.tools import ToolRuntime  # noqa: PLC0415

            config = cast("RunnableConfig", getattr(runtime, "config", {}))
            tool_rt = ToolRuntime(
                state=None,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_rt)

        return self._backend

    def _build_runtime(self, runtime: Runtime) -> DeepAgentsRuntime:
        """Forward-compatible hook: enrich a bare ``Runtime`` into a ``DeepAgentsRuntime``.

        Once ``AgentMiddleware._build_runtime`` is wired into the framework
        dispatch loop this method will be called automatically before every
        middleware hook.  Until then it is a no-op from the framework's
        perspective but can be called explicitly if needed.

        Args:
            runtime: The incoming ``Runtime`` (may already be a
                ``DeepAgentsRuntime``).

        Returns:
            A ``DeepAgentsRuntime`` with ``backend`` populated.
        """
        # Avoid redundant resolution if we already have a DeepAgentsRuntime.
        if isinstance(runtime, DeepAgentsRuntime) and runtime.backend is not None:
            return runtime

        backend = self._resolve_backend_for_runtime(runtime)
        return DeepAgentsRuntime(
            context=runtime.context,
            store=runtime.store,
            stream_writer=runtime.stream_writer,
            previous=runtime.previous,
            backend=backend,
        )
