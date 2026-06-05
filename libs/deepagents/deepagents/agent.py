"""create_deep_agent — convenience wrapper around create_agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

    from deepagents.backends.protocol import BackendProtocol

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware  # noqa: TC002

from deepagents.middleware.runtime import BackendMiddleware

__all__ = ["create_deep_agent"]


def create_deep_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    backend: BackendProtocol,
    middleware: Sequence[AgentMiddleware] = (),
    **kwargs: Any,
) -> CompiledStateGraph:
    """Create an agent with a backend bound to all deepagents middleware.

    Thin wrapper around :func:`langchain.agents.create_agent` that prepends
    :class:`~deepagents.middleware.runtime.BackendMiddleware` so every
    middleware hook receives a typed ``runtime.backend``.  All other
    ``create_agent`` keyword arguments are forwarded unchanged.

    Args:
        model: The language model for the agent.
        tools: Tools available to the agent.
        backend: Backend instance made available as ``runtime.backend`` in
            all middleware hooks.
        middleware: Middleware to apply.  ``BackendMiddleware`` is prepended
            automatically; do not add it again.
        **kwargs: Forwarded to :func:`~langchain.agents.create_agent`
            (``system_prompt``, ``response_format``, ``checkpointer``, etc.).
    """
    return create_agent(
        model,
        tools,
        middleware=[BackendMiddleware(backend), *middleware],
        **kwargs,
    )
