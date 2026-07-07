"""Type-checking fixture for context-aware middleware on `create_deep_agent`.

This module is not executed; it is type-checked by `ty` from
`test_graph.py::TestMiddlewareTyping` to guard against regressions where the
`middleware` parameter pins `ContextT` to `None` (see issue #4051).
"""

from dataclasses import dataclass

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware

from deepagents import create_deep_agent


@dataclass
class Context:
    """Runtime context schema used by the context-aware middleware."""

    user_name: str


class ContextAwareMiddleware(AgentMiddleware[AgentState, Context]):
    """Middleware parameterized with a concrete context type."""


create_deep_agent(
    context_schema=Context,
    middleware=[ContextAwareMiddleware()],
)
