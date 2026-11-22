"""Memory loading for long-lived Talon conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.middleware.memory import MemoryMiddleware

if TYPE_CHECKING:
    from deepagents.middleware.memory import MemoryState, MemoryStateUpdate
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime


class TalonMemoryMiddleware(MemoryMiddleware):
    """Reload configured memory sources before every turn."""

    @property
    def name(self) -> str:
        """Replace the SDK memory middleware in its existing stack position."""
        return "MemoryMiddleware"

    def before_agent(
        self, state: MemoryState, runtime: Runtime, config: RunnableConfig
    ) -> MemoryStateUpdate | None:
        """Read current sources without changing the incoming checkpoint state."""
        state = state.copy()
        state.pop("memory_contents", None)
        return super().before_agent(state, runtime, config)

    async def abefore_agent(
        self, state: MemoryState, runtime: Runtime, config: RunnableConfig
    ) -> MemoryStateUpdate | None:
        """Read current sources without changing the incoming checkpoint state."""
        state = state.copy()
        state.pop("memory_contents", None)
        return await super().abefore_agent(state, runtime, config)
