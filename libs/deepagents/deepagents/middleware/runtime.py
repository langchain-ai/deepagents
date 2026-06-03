"""DeepAgentsRuntime and _DeepAgentsRuntimeMixin for middleware backend access."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import TYPE_CHECKING, Any, cast

from langgraph.runtime import Runtime
from langgraph.types import _DC_KWARGS

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

    from deepagents.backends.protocol import BackendProtocol

__all__ = ["DeepAgentsRuntime", "_DeepAgentsRuntimeMixin"]


@dataclass(**_DC_KWARGS)
class DeepAgentsRuntime(Runtime):
    """Runtime with a pre-resolved backend for deepagents middleware."""

    backend: BackendProtocol | None = field(default=None)


class _DeepAgentsRuntimeMixin:
    """Private mixin that resolves self._backend into a DeepAgentsRuntime."""

    _backend: Any  # BackendProtocol | BackendFactory

    def _resolve_backend_for_runtime(self, runtime: Runtime) -> BackendProtocol:
        """Return a resolved backend, using runtime.backend if already populated."""
        if isinstance(runtime, DeepAgentsRuntime) and runtime.backend is not None:
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
        """Enrich a Runtime into a DeepAgentsRuntime with backend populated."""
        if isinstance(runtime, DeepAgentsRuntime) and runtime.backend is not None:
            return runtime

        backend = self._resolve_backend_for_runtime(runtime)
        inherited = {f.name: getattr(runtime, f.name) for f in dc_fields(Runtime)}
        return DeepAgentsRuntime(**inherited, backend=backend)
