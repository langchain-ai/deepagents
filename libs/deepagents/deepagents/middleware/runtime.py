"""DeepAgents AgentRuntime and BackendMiddleware."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware, AgentRuntime as _LangchainAgentRuntime  # type: ignore[attr-defined]
from langgraph.types import _DC_KWARGS
from langgraph.typing import ContextT

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

__all__ = ["AgentRuntime", "BackendMiddleware"]


@dataclass(**_DC_KWARGS)
class AgentRuntime(_LangchainAgentRuntime[ContextT]):
    """AgentRuntime with a resolved backend for deepagents middleware.

    Extends langchain's ``AgentRuntime`` with a typed ``backend`` field.
    Injected by ``BackendMiddleware`` (prepended automatically by
    ``create_deep_agent``).  Middleware authors should type their hook
    parameters as ``runtime: AgentRuntime`` (importing from ``deepagents``)
    to get a fully typed ``runtime.backend``.
    """

    backend: BackendProtocol = field(default=None)  # type: ignore[assignment]
    """Resolved backend instance, always set when middleware receives this runtime."""


class BackendMiddleware(AgentMiddleware):
    """Injects a typed backend into the AgentRuntime for all downstream middleware.

    Must appear before any middleware that accesses ``runtime.backend``.
    ``create_deep_agent`` prepends this automatically; use it directly only
    when composing middleware manually with ``create_agent``.

    Example::

        agent = create_agent(
            model,
            middleware=[
                BackendMiddleware(backend=my_backend),
                MemoryMiddleware(sources=["AGENTS.md"]),
            ],
        )
    """

    name = "__deepagents_backend__"

    def __init__(self, backend: BackendProtocol) -> None:  # noqa: D107
        self._backend = backend

    def _build_runtime(self, runtime: _LangchainAgentRuntime) -> AgentRuntime:
        if isinstance(runtime, AgentRuntime):
            return runtime
        kwargs = {f.name: getattr(runtime, f.name) for f in dc_fields(runtime)}
        kwargs["backend"] = self._backend
        return AgentRuntime(**kwargs)
